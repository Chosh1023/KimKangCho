"""Best-of-K imitation, feasibility, diversity, and ranking objectives."""

from __future__ import annotations

import math
from typing import Mapping

import torch
import torch.nn.functional as F

from .config import HopePathConfig
from .geometry import CURVATURE_DISCRETIZATION_MARGIN


def _masked_average(value: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    weight = mask.to(value.dtype)
    return (value * weight).sum(dim=dim) / weight.sum(dim=dim).clamp_min(1.0)


def _candidate_gather(value: torch.Tensor, winner: torch.Tensor) -> torch.Tensor:
    shape = [len(winner), 1] + [1] * (value.ndim - 2)
    index = winner.view(*shape).expand(-1, 1, *value.shape[2:])
    return value.gather(1, index).squeeze(1)


def _coarse_target(batch: Mapping[str, torch.Tensor], count: int) -> torch.Tensor:
    target = batch["target_pose"]
    length = batch["target_length"].clamp_min(1)
    progress = torch.linspace(0.0, 1.0, count, device=target.device)
    indices = torch.round(progress[None] * (length[:, None] - 1)).long()
    return target.gather(1, indices[..., None].expand(-1, -1, 3))


def _normalized_goal_envelope_cost(
    position_error: torch.Tensor,
    yaw_error: torch.Tensor,
    config: HopePathConfig,
) -> torch.Tensor:
    """Differentiable cost aligned with the runtime success envelope."""

    position = position_error / float(config.data.goal_position_tolerance_m)
    yaw = yaw_error / math.radians(config.data.goal_yaw_tolerance_deg)
    # Success requires both normalized errors <= 1.  Blend the worst
    # constraint with their mean so the active boundary receives most of the
    # gradient without starving the already-better coordinate.
    return 0.5 * torch.maximum(position, yaw) + 0.25 * (position + yaw)


def _goal_envelope_violation(
    position_error: torch.Tensor,
    yaw_error: torch.Tensor,
    config: HopePathConfig,
) -> torch.Tensor:
    """Exact hinge surrogate for the runtime AND goal tolerance."""

    normalized_worst = torch.maximum(
        position_error / float(config.data.goal_position_tolerance_m),
        yaw_error / math.radians(config.data.goal_yaw_tolerance_deg),
    )
    return F.relu(normalized_worst - 1.0)


def footprint_clearance_samples(
    poses: torch.Tensor,
    sdf_normalized: torch.Tensor,
    config: HopePathConfig,
) -> torch.Tensor:
    """Return differentiable clearance [B,K,T,9] in metres."""

    vehicle, data = config.vehicle, config.data
    longitudinal = poses.new_tensor(
        (-vehicle.rear_overhang_m,
         0.5 * (vehicle.wheelbase_m + vehicle.front_overhang_m - vehicle.rear_overhang_m),
         vehicle.wheelbase_m + vehicle.front_overhang_m)
    )
    lateral = poses.new_tensor((-0.5 * vehicle.width_m, 0.0, 0.5 * vehicle.width_m))
    ox, oy = torch.meshgrid(longitudinal, lateral, indexing="ij")
    ox, oy = ox.reshape(1, 1, 1, -1), oy.reshape(1, 1, 1, -1)
    x, y, yaw = poses[..., 0:1], poses[..., 1:2], poses[..., 2:3]
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    world_x = x + cosine * ox - sine * oy
    world_y = y + sine * ox + cosine * oy
    grid = torch.stack(
        (
            2.0 * (world_x - data.raster_center_x_m) / data.extent_m,
            -2.0 * (world_y - data.raster_center_y_m) / data.extent_m,
        ),
        dim=-1,
    )
    batch, candidates, steps, points, _ = grid.shape
    sampled = F.grid_sample(
        sdf_normalized[:, None],
        grid.reshape(batch, candidates * steps * points, 1, 2),
        mode="bilinear", padding_mode="border", align_corners=False,
    ).reshape(batch, candidates, steps, points)
    return sampled * data.sdf_clip_m


def footprint_collision_cost(
    poses: torch.Tensor,
    sdf_normalized: torch.Tensor,
    mask: torch.Tensor,
    config: HopePathConfig,
) -> torch.Tensor:
    """Differentiable [B,K] swept-footprint collision risk.

    Runtime rejects a path when *any* part of the full vehicle intersects an
    obstacle.  A uniform mean over nine footprint samples and all path tokens
    made those local contacts almost disappear from the training objective.
    Sample inter-token midpoints (0.1 m for the 0.2 m dataset), retain the most
    exposed footprint points, and combine mean risk with the worst path steps.
    ``mask`` may be [B,T] or a soft/per-candidate [B,K,T] activity mask.
    """

    candidates = poses.shape[1]
    if mask.ndim == 2:
        valid = mask[:, None].expand(-1, candidates, -1)
    elif mask.ndim == 3:
        valid = mask.expand(-1, candidates, -1)
    else:
        raise ValueError("collision mask must have shape [B,T] or [B,K,T]")
    valid = valid.to(dtype=poses.dtype)

    if poses.shape[2] > 1:
        yaw_delta = torch.atan2(
            torch.sin(poses[..., 1:, 2] - poses[..., :-1, 2]),
            torch.cos(poses[..., 1:, 2] - poses[..., :-1, 2]),
        )
        midpoint = torch.cat(
            (
                0.5 * (poses[..., 1:, :2] + poses[..., :-1, :2]),
                (poses[..., :-1, 2] + 0.5 * yaw_delta).unsqueeze(-1),
            ),
            dim=-1,
        )
        swept_poses = torch.cat((poses, midpoint), dim=2)
        midpoint_valid = torch.minimum(valid[..., 1:], valid[..., :-1])
        swept_valid = torch.cat((valid, midpoint_valid), dim=2)
    else:
        swept_poses, swept_valid = poses, valid

    sdf_m = footprint_clearance_samples(swept_poses, sdf_normalized, config)
    point_penalty = F.relu(config.loss.collision_margin_m - sdf_m)
    point_count = min(int(config.loss.collision_topk_points), point_penalty.shape[-1])
    pose_risk = point_penalty.topk(point_count, dim=-1).values.mean(dim=-1)
    mean_risk = (pose_risk * swept_valid).sum(dim=-1) / swept_valid.sum(dim=-1).clamp_min(1.0)

    step_count = min(int(config.loss.collision_topk_steps), pose_risk.shape[-1])
    ranked_risk = pose_risk.masked_fill(swept_valid <= 1.0e-6, -1.0e6)
    top_values, top_indices = ranked_risk.topk(step_count, dim=-1)
    top_weights = swept_valid.gather(-1, top_indices)
    top_risk = (top_values.clamp_min(0.0) * top_weights).sum(dim=-1) / top_weights.sum(
        dim=-1
    ).clamp_min(1.0e-6)
    mean_weight = float(config.loss.collision_mean_weight)
    return mean_weight * mean_risk + (1.0 - mean_weight) * top_risk


def compute_path_loss(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    config: HopePathConfig,
    *,
    enable_diversity: bool = True,
    enable_ranking: bool = True,
    collision_scale: float = 1.0,
    ranking_scale: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    poses = output["poses"]
    batch_size, candidates, steps, _ = poses.shape
    target_pose = batch["target_pose"][:, None].expand(-1, candidates, -1, -1)
    target_curvature = batch["target_curvature"][:, None].expand(-1, candidates, -1)
    target_step = batch["target_step"][:, None].expand(-1, candidates, -1)
    target_gear = batch["target_gear"][:, None].expand(-1, candidates, -1)
    valid = batch["target_mask"][:, None].expand(-1, candidates, -1)

    position_per_step = F.smooth_l1_loss(
        poses[..., :2], target_pose[..., :2], reduction="none"
    ).sum(dim=-1)
    yaw_per_step = 1.0 - torch.cos(poses[..., 2] - target_pose[..., 2])
    curvature_per_step = F.smooth_l1_loss(
        output["curvature"], target_curvature, reduction="none"
    )
    step_per_step = F.smooth_l1_loss(
        output["step_magnitude"] / config.data.primitive_step_m,
        target_step / config.data.primitive_step_m,
        reduction="none",
    )
    gear_per_step = F.cross_entropy(
        output["gear_logits"].reshape(-1, 2), target_gear.reshape(-1), reduction="none"
    ).reshape(batch_size, candidates, steps)
    position_cost = _masked_average(position_per_step, valid)
    yaw_cost = _masked_average(yaw_per_step, valid)
    curvature_cost = _masked_average(curvature_per_step, valid)
    motion_valid = valid.clone()
    motion_valid[..., -1] = False
    step_cost = _masked_average(step_per_step, motion_valid)
    gear_cost = _masked_average(gear_per_step, valid)
    dense_position_cost = torch.zeros_like(position_cost)
    dense_yaw_cost = torch.zeros_like(yaw_cost)
    dense_steering_cost = torch.zeros_like(curvature_cost)
    if config.data.dense_substeps > 1:
        required_dense = {
            "dense_poses", "dense_steering",
        }
        required_batch = {
            "dense_target_pose", "dense_target_steering", "dense_target_mask",
        }
        missing_output = required_dense.difference(output)
        missing_batch = required_batch.difference(batch)
        if missing_output or missing_batch:
            raise KeyError(
                "dense supervision is incomplete (output={}, batch={})".format(
                    sorted(missing_output), sorted(missing_batch)
                )
            )
        dense_poses = output["dense_poses"]
        dense_target_pose = batch["dense_target_pose"][:, None].expand(
            -1, candidates, -1, -1
        )
        dense_target_steering = batch["dense_target_steering"][:, None].expand(
            -1, candidates, -1
        )
        dense_valid = batch["dense_target_mask"][:, None].expand(
            -1, candidates, -1
        )
        if dense_poses.shape[2] != dense_target_pose.shape[2]:
            raise ValueError(
                "dense output/target length mismatch: {} != {}".format(
                    dense_poses.shape[2], dense_target_pose.shape[2]
                )
            )
        dense_position_per_step = F.smooth_l1_loss(
            dense_poses[..., :2], dense_target_pose[..., :2], reduction="none"
        ).sum(dim=-1)
        dense_yaw_per_step = 1.0 - torch.cos(
            dense_poses[..., 2] - dense_target_pose[..., 2]
        )
        dense_steering_per_step = F.smooth_l1_loss(
            output["dense_steering"] / float(config.vehicle.max_steer_rad),
            dense_target_steering / float(config.vehicle.max_steer_rad),
            reduction="none",
        )
        dense_position_cost = _masked_average(
            dense_position_per_step, dense_valid
        )
        dense_yaw_cost = _masked_average(dense_yaw_per_step, dense_valid)
        dense_steering_cost = _masked_average(
            dense_steering_per_step, dense_valid
        )
    # Best-of-K assignment must include the controls that are ultimately
    # executed.  Position/yaw alone allowed a spatially close but violently
    # oscillating candidate to win, leaving curvature and step supervision on
    # the wrong mode.
    imitation_cost = (
        position_cost
        + 0.5 * yaw_cost
        + 0.5 * curvature_cost
        + 0.2 * step_cost
        + 0.2 * gear_cost
        + 0.5 * dense_position_cost
        + 0.25 * dense_yaw_cost
        + 0.25 * dense_steering_cost
    )
    best_winner = imitation_cost.detach().argmin(dim=1)
    topology_weight = float(config.loss.topology_assignment_weight)
    topology_active = bool(
        topology_weight > 0.0
        and candidates == config.model.candidate_count
        and candidates == 8
    )
    if topology_active:
        if "target_topology_mode" not in batch:
            raise KeyError(
                "target_topology_mode is required for topology-conditioned training"
            )
        topology_winner = batch["target_topology_mode"].to(
            device=poses.device, dtype=torch.long
        )
        if bool(((topology_winner < 0) | (topology_winner >= candidates)).any()):
            raise ValueError("target_topology_mode is outside the candidate range")
    else:
        topology_winner = best_winner
        topology_weight = 0.0
    winner = topology_winner if topology_weight >= 0.5 else best_winner
    quality = batch["quality_weight"].to(poses.dtype)

    def assigned_candidate_cost(cost: torch.Tensor) -> torch.Tensor:
        best = _candidate_gather(cost, best_winner)
        assigned = _candidate_gather(cost, topology_winner)
        return (1.0 - topology_weight) * best + topology_weight * assigned

    def assigned_imitation_loss(cost: torch.Tensor) -> torch.Tensor:
        return (assigned_candidate_cost(cost) * quality).mean()

    position_loss = assigned_imitation_loss(position_cost)
    yaw_loss = assigned_imitation_loss(yaw_cost)
    curvature_loss = assigned_imitation_loss(curvature_cost)
    step_loss = assigned_imitation_loss(step_cost)
    gear_loss = assigned_imitation_loss(gear_cost)
    dense_position_loss = assigned_imitation_loss(dense_position_cost)
    dense_yaw_loss = assigned_imitation_loss(dense_yaw_cost)
    dense_steering_loss = assigned_imitation_loss(dense_steering_cost)

    length_target = (batch["target_length"] - 1).clamp(0, steps - 1)
    length_logits = output["length_logits"]
    length_probability = torch.softmax(length_logits, dim=-1)
    length_loss_all = F.cross_entropy(
        length_logits.reshape(batch_size * candidates, steps),
        length_target[:, None].expand(-1, candidates).reshape(-1),
        reduction="none",
    ).reshape(batch_size, candidates)
    # Anonymous Best-of-K needs calibrated lengths on all candidates.  Under
    # explicit topology assignment, forcing every alternative to use the GT
    # route's token count would collapse long and short detours back together;
    # supervise the assigned route length and let every other proposal learn
    # its own endpoint through the predicted-length goal objective below.
    all_candidate_stop = length_loss_all.mean(dim=1)
    assigned_stop = _candidate_gather(length_loss_all, topology_winner)
    stop_loss = (
        ((1.0 - topology_weight) * all_candidate_stop
         + topology_weight * assigned_stop)
        * quality
    ).mean()

    # The hard terminal bridge guarantees the executed endpoint. Continue to
    # supervise the unanchored rollout here so the learned controls approach
    # the goal naturally and the deterministic correction remains small.
    goal_supervision_poses = output.get("unanchored_poses", poses)
    indices = length_target[:, None, None, None].expand(-1, candidates, 1, 3)
    final_pose = goal_supervision_poses.gather(2, indices).squeeze(2)
    goal_yaw = torch.atan2(batch["goal"][:, 2], batch["goal"][:, 3])
    goal_position = torch.linalg.vector_norm(final_pose[..., :2] - batch["goal"][:, None, :2], dim=-1)
    goal_yaw_error = torch.abs(torch.atan2(
        torch.sin(final_pose[..., 2] - goal_yaw[:, None]),
        torch.cos(final_pose[..., 2] - goal_yaw[:, None]),
    ))
    target_goal_cost = _normalized_goal_envelope_cost(
        goal_position, goal_yaw_error, config
    )
    target_goal_training_cost = (
        _goal_envelope_violation(goal_position, goal_yaw_error, config)
        if config.loss.goal_tolerance_hinge
        else target_goal_cost
    )

    # Also supervise the endpoint implied by the model's own length
    # distribution.  Previously goal loss always indexed the GT length while
    # runtime stopped at argmax(length_logits), leaving an avoidable endpoint
    # train/inference mismatch.
    predicted_final_xy = (
        goal_supervision_poses[..., :2] * length_probability.unsqueeze(-1)
    ).sum(dim=2)
    predicted_final_yaw = torch.atan2(
        (torch.sin(goal_supervision_poses[..., 2]) * length_probability).sum(dim=2),
        (torch.cos(goal_supervision_poses[..., 2]) * length_probability).sum(dim=2),
    )
    predicted_goal_position = torch.linalg.vector_norm(
        predicted_final_xy - batch["goal"][:, None, :2], dim=-1
    )
    predicted_goal_yaw_error = torch.abs(torch.atan2(
        torch.sin(predicted_final_yaw - goal_yaw[:, None]),
        torch.cos(predicted_final_yaw - goal_yaw[:, None]),
    ))
    soft_predicted_goal_cost = _normalized_goal_envelope_cost(
        predicted_goal_position, predicted_goal_yaw_error, config
    )
    soft_predicted_goal_training_cost = (
        _goal_envelope_violation(
            predicted_goal_position, predicted_goal_yaw_error, config
        )
        if config.loss.goal_tolerance_hinge
        else soft_predicted_goal_cost
    )

    # Match the hard endpoint used by model.infer(). The argmax index is
    # discrete, but gathering the pose still provides a direct gradient to the
    # executed endpoint; all-candidate length CE above trains the index itself.
    predicted_length = length_logits.detach().argmax(dim=-1)
    hard_index = predicted_length[..., None, None].expand(-1, -1, 1, 3)
    hard_final_pose = goal_supervision_poses.gather(2, hard_index).squeeze(2)
    hard_goal_position = torch.linalg.vector_norm(
        hard_final_pose[..., :2] - batch["goal"][:, None, :2], dim=-1
    )
    hard_goal_yaw_error = torch.abs(torch.atan2(
        torch.sin(hard_final_pose[..., 2] - goal_yaw[:, None]),
        torch.cos(hard_final_pose[..., 2] - goal_yaw[:, None]),
    ))
    hard_predicted_goal_cost = _normalized_goal_envelope_cost(
        hard_goal_position, hard_goal_yaw_error, config
    )
    hard_predicted_goal_training_cost = (
        _goal_envelope_violation(hard_goal_position, hard_goal_yaw_error, config)
        if config.loss.goal_tolerance_hinge
        else hard_predicted_goal_cost
    )
    predicted_goal_cost = 0.5 * (
        soft_predicted_goal_cost + hard_predicted_goal_cost
    )
    predicted_goal_training_cost = 0.5 * (
        soft_predicted_goal_training_cost + hard_predicted_goal_training_cost
    )
    predicted_goal_weight = float(config.loss.predicted_endpoint_goal_weight)
    all_target_goal = target_goal_training_cost.mean(dim=1)
    assigned_target_goal = _candidate_gather(
        target_goal_training_cost, topology_winner
    )
    target_goal_loss = (
        (1.0 - topology_weight) * all_target_goal
        + topology_weight * assigned_target_goal
    ).mean()
    goal_loss = (
        target_goal_loss
        + predicted_goal_weight * predicted_goal_training_cost.mean()
    ) / (1.0 + predicted_goal_weight)

    predicted_active = torch.flip(
        torch.cumsum(torch.flip(length_probability, dims=(-1,)), dim=-1),
        dims=(-1,),
    )
    if topology_active:
        assigned_candidate = F.one_hot(
            topology_winner, num_classes=candidates
        ).to(poses.dtype)[..., None]
        collision_valid = torch.maximum(
            valid.to(poses.dtype) * assigned_candidate, predicted_active
        )
    else:
        collision_valid = torch.maximum(valid.to(poses.dtype), predicted_active)
    collision_cost = footprint_collision_cost(
        poses, batch["bev"][:, 2], collision_valid, config
    )
    collision_loss = collision_cost.mean()

    # Optimize the set-level success criterion rather than pulling every
    # already-successful endpoint toward the exact goal. Pick the candidate
    # with the best detached combination of collision risk and goal-envelope
    # violation, then backpropagate only its violation. This preserves path
    # diversity/clearance while directly training the oracle success gap.
    hard_goal_violation = _goal_envelope_violation(
        hard_goal_position, hard_goal_yaw_error, config
    )
    feasibility_winner = (
        hard_goal_violation + 4.0 * collision_cost
    ).detach().argmin(dim=1)
    feasibility_loss = (
        _candidate_gather(hard_goal_violation, feasibility_winner) * quality
    ).mean()

    same_gear = (target_gear[..., 1:] == target_gear[..., :-1]) & valid[..., 1:] & valid[..., :-1]
    curvature_delta = torch.abs(output["curvature"][..., 1:] - output["curvature"][..., :-1])
    smoothness_cost = _masked_average(curvature_delta, same_gear)
    smoothness_loss = assigned_candidate_cost(smoothness_cost).mean()

    # Trackability is a set property: every candidate may be selected at
    # runtime, so these terms intentionally supervise all K paths instead of
    # only the imitation winner.
    steering = output.get("steering")
    if steering is None:
        steering = torch.atan(config.vehicle.wheelbase_m * output["curvature"])
    maximum_steering = math.atan(
        config.vehicle.wheelbase_m
        * config.vehicle.max_curvature_m_inv
        * CURVATURE_DISCRETIZATION_MARGIN
    )
    maximum_delta = math.radians(config.model.max_steering_delta_deg)
    steering_delta = torch.diff(steering, dim=-1)
    edge_active = torch.minimum(predicted_active[..., 1:], predicted_active[..., :-1])
    steering_rate_cost = _masked_average(
        torch.abs(steering_delta) / maximum_delta, edge_active
    )
    if steps > 2:
        steering_jerk = torch.diff(steering_delta, dim=-1)
        jerk_active = torch.minimum(edge_active[..., 1:], edge_active[..., :-1])
        steering_jerk_cost = _masked_average(
            torch.abs(steering_jerk) / maximum_delta, jerk_active
        )
    else:
        steering_jerk_cost = torch.zeros_like(steering_rate_cost)

    soft_terminal_steering = (
        torch.abs(steering) * length_probability
    ).sum(dim=-1)
    hard_terminal_index = predicted_length[..., None]
    hard_terminal_steering = torch.abs(steering).gather(
        -1, hard_terminal_index
    ).squeeze(-1)
    terminal_steering_cost = 0.5 * (
        soft_terminal_steering + hard_terminal_steering
    ) / maximum_steering
    saturation_per_step = F.relu(
        torch.abs(steering) / maximum_steering - 0.90
    ) / 0.10
    steering_saturation_cost = _masked_average(
        saturation_per_step, predicted_active
    )

    forward_probability = torch.softmax(output["gear_logits"], dim=-1)[..., 1]
    gear_probability_delta = torch.abs(torch.diff(forward_probability, dim=-1))
    same_target_gear = (
        (target_gear[..., 1:] == target_gear[..., :-1])
        & valid[..., 1:]
        & valid[..., :-1]
    )
    gear_flicker_cost = _masked_average(
        gear_probability_delta,
        same_target_gear.to(edge_active.dtype) * edge_active,
    )

    def all_candidate_loss(cost: torch.Tensor) -> torch.Tensor:
        return (cost.mean(dim=1) * quality).mean()

    steering_rate_loss = all_candidate_loss(steering_rate_cost)
    steering_jerk_loss = all_candidate_loss(steering_jerk_cost)
    terminal_steering_loss = all_candidate_loss(terminal_steering_cost)
    steering_saturation_loss = all_candidate_loss(steering_saturation_cost)
    gear_flicker_loss = all_candidate_loss(gear_flicker_cost)
    trackability_cost = (
        steering_rate_cost
        + 0.5 * steering_jerk_cost
        + terminal_steering_cost
        + steering_saturation_cost
        + gear_flicker_cost
    )

    # Reference-point supervision makes deformable sampling useful before the
    # kinematic head has learned stable curvature and gear sequences.
    refined_position = F.smooth_l1_loss(
        output["refined_xy"], target_pose[..., :2], reduction="none"
    ).sum(dim=-1)
    refined_loss = assigned_candidate_cost(
        _masked_average(refined_position, valid)
    ).mean()
    coarse_target = _coarse_target(batch, config.data.coarse_count)
    coarse_expected = coarse_target[:, None].expand(-1, candidates, -1, -1)
    coarse_position = F.smooth_l1_loss(
        output["coarse_pose"][..., :2], coarse_expected[..., :2], reduction="none"
    ).sum(dim=-1).mean(dim=-1)
    coarse_yaw = (1.0 - torch.cos(
        output["coarse_pose"][..., 2] - coarse_expected[..., 2]
    )).mean(dim=-1)
    coarse_loss = assigned_candidate_cost(
        coarse_position + 0.5 * coarse_yaw
    ).mean()

    diversity_loss = poses.new_zeros(())
    if enable_diversity and candidates > 1:
        pair_losses = []
        for left in range(candidates):
            for right in range(left + 1, candidates):
                distance = torch.linalg.vector_norm(
                    poses[:, left, :, :2] - poses[:, right, :, :2], dim=-1
                )
                if topology_active:
                    pair_mask = torch.minimum(
                        predicted_active[:, left], predicted_active[:, right]
                    )
                else:
                    pair_mask = batch["target_mask"].to(poses.dtype)
                average = (distance * pair_mask).sum(-1) / pair_mask.sum(-1).clamp_min(1.0)
                pair_losses.append(torch.exp(-average / 2.0))
        diversity_loss = torch.stack(pair_losses, dim=1).mean()

    ranking_loss = poses.new_zeros(())
    if enable_ranking and candidates > 1:
        detached_cost = (
            predicted_goal_cost
            + 4.0 * collision_cost
            + imitation_cost
            + config.loss.ranking_trackability * trackability_cost
        ).detach()
        target_distribution = torch.softmax(-detached_cost, dim=1)
        ranking_loss = -(target_distribution * torch.log_softmax(output["scores"], dim=1)).sum(1).mean()

    weights = config.loss
    base_total = (
        weights.position * (position_loss + 0.25 * refined_loss + 0.25 * coarse_loss)
        + weights.yaw * yaw_loss
        + weights.curvature * curvature_loss
        + weights.step * step_loss
        + weights.gear * gear_loss
        + weights.stop * stop_loss
        + weights.goal * goal_loss
        + weights.collision * float(collision_scale) * collision_loss
        + weights.feasibility * feasibility_loss
        + weights.smoothness * smoothness_loss
        + weights.steering_rate * steering_rate_loss
        + weights.steering_jerk * steering_jerk_loss
        + weights.terminal_steering * terminal_steering_loss
        + weights.steering_saturation * steering_saturation_loss
        + weights.gear_flicker * gear_flicker_loss
        + weights.dense_position * dense_position_loss
        + weights.dense_yaw * dense_yaw_loss
        + weights.dense_steering * dense_steering_loss
    )
    diversity_contribution = weights.diversity * diversity_loss
    ranking_contribution = weights.ranking * float(ranking_scale) * ranking_loss
    total = base_total + diversity_contribution + ranking_contribution
    metrics = {
        "loss": total.detach(), "position": position_loss.detach(),
        "yaw": yaw_loss.detach(), "curvature": curvature_loss.detach(),
        "step": step_loss.detach(),
        "gear": gear_loss.detach(), "stop": stop_loss.detach(),
        "goal": goal_loss.detach(), "collision": collision_loss.detach(),
        "feasibility": feasibility_loss.detach(),
        "smoothness": smoothness_loss.detach(), "diversity": diversity_loss.detach(),
        "steering_rate": steering_rate_loss.detach(),
        "steering_jerk": steering_jerk_loss.detach(),
        "terminal_steering": terminal_steering_loss.detach(),
        "steering_saturation": steering_saturation_loss.detach(),
        "gear_flicker": gear_flicker_loss.detach(),
        "dense_position": dense_position_loss.detach(),
        "dense_yaw": dense_yaw_loss.detach(),
        "dense_steering": dense_steering_loss.detach(),
        "ranking": ranking_loss.detach(), "coarse": coarse_loss.detach(),
        "refined": refined_loss.detach(), "winner": winner.detach().float().mean(),
        "best_winner": best_winner.detach().float().mean(),
        "topology_winner": topology_winner.detach().float().mean(),
        "topology_assignment_active": poses.new_tensor(float(topology_active)),
        "topology_best_match": (
            (best_winner == topology_winner).to(poses.dtype).mean().detach()
        ),
        "base_loss": base_total.detach(),
        "diversity_contribution": diversity_contribution.detach(),
        "ranking_contribution": ranking_contribution.detach(),
    }
    return total, metrics
