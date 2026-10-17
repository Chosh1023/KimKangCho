"""Geometry-first validation and candidate selection metrics."""

from __future__ import annotations

import math
from typing import Mapping

import torch

from .config import HopePathConfig
from .geometry import CURVATURE_DISCRETIZATION_MARGIN
from .losses import footprint_clearance_samples


def candidate_trackability_metrics(
    output: Mapping[str, torch.Tensor], config: HopePathConfig
) -> dict[str, torch.Tensor]:
    """Measure controller-facing steering quality for every candidate."""

    poses = output["poses"]
    _batch_size, _candidates, steps, _ = poses.shape
    lengths = output["lengths"].clamp(2, steps)
    token = torch.arange(steps, device=poses.device)[None, None]
    active = token < lengths[..., None]
    edge_active = token[..., :-1] < (lengths - 1)[..., None]

    steering = output.get("steering")
    if steering is None and "curvature" in output:
        steering = torch.atan(
            config.vehicle.wheelbase_m * output["curvature"]
        )
    if steering is None:
        # Compatibility fallback for external/legacy candidate producers.
        distance = torch.linalg.vector_norm(
            torch.diff(poses[..., :2], dim=2), dim=-1
        ).clamp_min(1.0e-4)
        yaw_delta = torch.atan2(
            torch.sin(torch.diff(poses[..., 2], dim=2)),
            torch.cos(torch.diff(poses[..., 2], dim=2)),
        )
        curvature = yaw_delta / distance
        curvature = torch.cat((curvature, curvature[..., -1:]), dim=-1)
        steering = torch.atan(config.vehicle.wheelbase_m * curvature)

    if "gear" in output:
        gear = output["gear"]
    elif "gear_logits" in output:
        gear = output["gear_logits"].argmax(dim=-1)
    else:
        gear = torch.ones_like(steering, dtype=torch.long)

    spatial_step = torch.linalg.vector_norm(
        torch.diff(poses[..., :2], dim=2), dim=-1
    )
    path_length = (spatial_step * edge_active.to(spatial_step.dtype)).sum(dim=-1)
    steering_delta_deg = torch.rad2deg(torch.abs(torch.diff(steering, dim=-1)))
    active_delta = steering_delta_deg * edge_active.to(steering_delta_deg.dtype)
    steering_variation = active_delta.sum(dim=-1) / path_length.clamp_min(0.1)
    same_gear = (gear[..., 1:] == gear[..., :-1]) & edge_active
    same_gear_jump = steering_delta_deg.masked_fill(~same_gear, 0.0).amax(dim=-1)

    endpoint_index = (lengths - 1)[..., None]
    terminal_steering = torch.rad2deg(
        torch.abs(steering.gather(-1, endpoint_index).squeeze(-1))
    )
    maximum_steering = math.atan(
        config.vehicle.wheelbase_m
        * config.vehicle.max_curvature_m_inv
        * CURVATURE_DISCRETIZATION_MARGIN
    )
    saturated = torch.abs(steering) >= 0.90 * maximum_steering
    saturation_fraction = (
        (saturated & active).to(steering.dtype).sum(dim=-1)
        / active.to(steering.dtype).sum(dim=-1).clamp_min(1.0)
    )

    data = config.data
    trackable = (
        (steering_variation <= data.trackability_max_steering_variation_deg_per_m)
        & (same_gear_jump <= data.trackability_max_same_gear_steer_jump_deg)
        & (terminal_steering <= data.trackability_max_terminal_steer_deg)
        & (saturation_fraction <= data.trackability_max_saturation_fraction)
    )
    return {
        "candidate_trackable": trackable,
        "candidate_path_length_m": path_length,
        "candidate_steering_variation_deg_per_m": steering_variation,
        "candidate_max_same_gear_steer_jump_deg": same_gear_jump,
        "candidate_terminal_steer_deg": terminal_steering,
        "candidate_steering_saturation_fraction": saturation_fraction,
    }


def evaluate_inference_batch(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    config: HopePathConfig,
    collision_free_override: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    poses = output["poses"]
    batch_size, candidates, steps, _ = poses.shape
    lengths = output["lengths"].clamp(2, steps)
    token = torch.arange(steps, device=poses.device)[None, None]
    predicted_mask = token < lengths[..., None]
    clearance = footprint_clearance_samples(poses, batch["bev"][:, 2], config)
    clearance = clearance.masked_fill(~predicted_mask[..., None], float("inf"))
    minimum_clearance = clearance.amin(dim=(-1, -2))
    collision_free = minimum_clearance >= 0.0
    if collision_free_override is not None:
        collision_free = collision_free_override.to(
            device=collision_free.device, dtype=torch.bool
        )

    endpoint_index = (lengths - 1)[..., None, None].expand(-1, -1, 1, 3)
    endpoint = poses.gather(2, endpoint_index).squeeze(2)
    goal_yaw = torch.atan2(batch["goal"][:, 2], batch["goal"][:, 3])
    goal_position_error = torch.linalg.vector_norm(
        endpoint[..., :2] - batch["goal"][:, None, :2], dim=-1
    )
    goal_yaw_error = torch.abs(torch.atan2(
        torch.sin(endpoint[..., 2] - goal_yaw[:, None]),
        torch.cos(endpoint[..., 2] - goal_yaw[:, None]),
    ))
    geometry_success = (
        collision_free
        & (goal_position_error <= config.data.goal_position_tolerance_m)
        & (goal_yaw_error <= math.radians(config.data.goal_yaw_tolerance_deg))
    )
    anchor_position_correction = output.get(
        "terminal_anchor_position_correction_m",
        torch.zeros_like(goal_position_error),
    )
    anchor_yaw_correction = output.get(
        "terminal_anchor_yaw_correction_rad",
        torch.zeros_like(goal_yaw_error),
    )
    anchor_correction_valid = (
        anchor_position_correction
        <= float(config.model.terminal_anchor_max_position_correction_m)
    ) & (
        anchor_yaw_correction
        <= math.radians(config.model.terminal_anchor_max_yaw_correction_deg)
    )
    if config.model.hard_goal_terminal_pose:
        geometry_success = geometry_success & anchor_correction_valid
    trackability = candidate_trackability_metrics(output, config)
    candidate_trackable = trackability["candidate_trackable"]
    candidate_success = geometry_success
    admissible = collision_free
    if config.data.trackability_gate_enabled:
        candidate_success = candidate_success & candidate_trackable
        admissible = admissible & candidate_trackable
    # Goal pose and tolerances are runtime inputs, not hidden labels.  The old
    # selector discarded that information after exact collision filtering and
    # trusted only the learned ranking score.  Consequently it could reject a
    # candidate that was already known to satisfy every success constraint.
    # Use a lexicographic feasibility selector:
    #   1) successful candidates, ranked by the learned quality score;
    #   2) otherwise collision-free candidates closest to the goal envelope;
    #   3) otherwise the least goal-violating candidate.
    # This guarantees selected_success == oracle_success whenever exact
    # collision state is available, while preserving the learned score as the
    # quality tie-breaker among equally feasible paths.
    normalized_goal_error = torch.maximum(
        goal_position_error / float(config.data.goal_position_tolerance_m),
        goal_yaw_error / math.radians(config.data.goal_yaw_tolerance_deg),
    )
    success_score = output["scores"].masked_fill(~candidate_success, -1.0e9)
    safe_goal_score = (-normalized_goal_error).masked_fill(~admissible, -1.0e9)
    collision_free_goal_score = (-normalized_goal_error).masked_fill(
        ~collision_free, -1.0e9
    )
    no_collision_free = ~collision_free.any(dim=1)
    no_admissible = ~admissible.any(dim=1)
    any_success = candidate_success.any(dim=1)
    selected = success_score.argmax(dim=1)
    selected = torch.where(any_success, selected, safe_goal_score.argmax(dim=1))
    selected = torch.where(
        no_admissible, collision_free_goal_score.argmax(dim=1), selected
    )
    selected = torch.where(no_collision_free, (-normalized_goal_error).argmax(dim=1), selected)
    selected_success = candidate_success.gather(1, selected[:, None]).squeeze(1)
    top1 = candidate_success.gather(
        1, output["scores"].argmax(dim=1, keepdim=True)
    ).squeeze(1)
    selected_position = goal_position_error.gather(1, selected[:, None]).squeeze(1)
    selected_yaw = goal_yaw_error.gather(1, selected[:, None]).squeeze(1)
    selected_anchor_position = anchor_position_correction.gather(
        1, selected[:, None]
    ).squeeze(1)
    selected_anchor_yaw = anchor_yaw_correction.gather(
        1, selected[:, None]
    ).squeeze(1)
    selected_clearance = minimum_clearance.gather(1, selected[:, None]).squeeze(1)
    selected_trackable = candidate_trackable.gather(
        1, selected[:, None]
    ).squeeze(1)
    selected_geometry_success = geometry_success.gather(
        1, selected[:, None]
    ).squeeze(1)
    selected_variation = trackability[
        "candidate_steering_variation_deg_per_m"
    ].gather(1, selected[:, None]).squeeze(1)
    selected_jump = trackability[
        "candidate_max_same_gear_steer_jump_deg"
    ].gather(1, selected[:, None]).squeeze(1)
    selected_terminal = trackability[
        "candidate_terminal_steer_deg"
    ].gather(1, selected[:, None]).squeeze(1)
    selected_saturation = trackability[
        "candidate_steering_saturation_fraction"
    ].gather(1, selected[:, None]).squeeze(1)
    return {
        "selected": selected,
        "candidate_success": candidate_success,
        "candidate_geometry_success": geometry_success,
        "candidate_collision_free": collision_free,
        "candidate_goal_position_error_m": goal_position_error,
        "candidate_goal_yaw_error_deg": torch.rad2deg(goal_yaw_error),
        "candidate_anchor_position_correction_m": anchor_position_correction,
        "candidate_anchor_yaw_correction_deg": torch.rad2deg(
            anchor_yaw_correction
        ),
        "candidate_anchor_correction_valid": anchor_correction_valid,
        "candidate_minimum_clearance_m": minimum_clearance,
        **trackability,
        "selected_success": selected_success,
        "selected_geometry_success": selected_geometry_success,
        "selected_trackable": selected_trackable,
        "top1_success": top1,
        "oracle_success": candidate_success.any(dim=1),
        "geometry_oracle_success": geometry_success.any(dim=1),
        "goal_position_error_m": selected_position,
        "goal_yaw_error_deg": torch.rad2deg(selected_yaw),
        "anchor_position_correction_m": selected_anchor_position,
        "anchor_yaw_correction_deg": torch.rad2deg(selected_anchor_yaw),
        "minimum_clearance_m": selected_clearance,
        "steering_variation_deg_per_m": selected_variation,
        "max_same_gear_steer_jump_deg": selected_jump,
        "terminal_steer_deg": selected_terminal,
        "steering_saturation_fraction": selected_saturation,
        "no_collision_free_candidate": no_collision_free,
        "no_trackable_candidate": ~candidate_trackable.any(dim=1),
    }
