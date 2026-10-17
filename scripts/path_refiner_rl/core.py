"""Path-refinement environment and physically aligned reward."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Mapping

import torch
import torch.nn.functional as F

from hope_path_transformer.metrics import evaluate_inference_batch
from hope_path_transformer.model import (
    anchor_terminal_pose_trace,
    controls_from_pose_trace,
    integrate_primitives,
    terminal_pose_error,
)


@dataclass(frozen=True)
class PathRefinerSpec:
    action_knots: int = 12
    topology_count: int = 1
    topology_control_dims: int = 0
    max_steering_residual_deg: float = 18.0
    topology_steering_bias_deg: float = 8.0
    terminal_anchor_steps: int = 96
    minimum_clearance_m: float = 0.10
    raster_size: int = 64
    raster_channels: int = 4
    vector_dim: int = 32
    anchor_position_limit_m: float = 0.75
    anchor_yaw_limit_deg: float = 10.0

    def validate(self) -> None:
        if int(self.action_knots) < 4:
            raise ValueError("action_knots must be at least four")
        if int(self.topology_count) < 1:
            raise ValueError("topology_count must be positive")
        if int(self.topology_control_dims) not in (0, 2):
            raise ValueError("topology_control_dims must be zero or two")
        if int(self.topology_count) > 1 and int(self.topology_control_dims) != 2:
            raise ValueError("topology mode requires two high-level controls")
        if int(self.topology_count) == 1 and int(self.topology_control_dims) != 0:
            raise ValueError("legacy refiner cannot have topology controls")
        if float(self.topology_steering_bias_deg) < 0.0:
            raise ValueError("topology steering bias cannot be negative")
        if float(self.max_steering_residual_deg) <= 0.0:
            raise ValueError("maximum steering residual must be positive")
        if int(self.terminal_anchor_steps) < 2:
            raise ValueError("terminal_anchor_steps must be at least two")
        if int(self.vector_dim) != 32 or int(self.raster_channels) != 4:
            raise ValueError("path-refiner observation contract changed unexpectedly")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def action_dim(self) -> int:
        return int(self.action_knots) + int(self.topology_control_dims)


def _candidate_gather(value: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    shape = [len(selected), 1] + [1] * (value.ndim - 2)
    index = selected.view(*shape).expand(-1, 1, *value.shape[2:])
    return value.gather(1, index).squeeze(1)


@torch.no_grad()
def select_base_path(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    config,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Choose the runtime candidate, or the least-bad fixable collision path."""

    geometry = evaluate_inference_batch(output, batch, config)
    selected = geometry["selected"]
    no_safe = ~geometry["candidate_collision_free"].any(dim=1)
    goal_cost = torch.maximum(
        geometry["candidate_goal_position_error_m"]
        / float(config.data.goal_position_tolerance_m),
        geometry["candidate_goal_yaw_error_deg"]
        / float(config.data.goal_yaw_tolerance_deg),
    )
    fallback_score = (
        12.0
        * geometry["candidate_minimum_clearance_m"].clamp(-1.5, 1.0)
        + 3.0 * geometry["candidate_trackable"].float()
        - goal_cost
    )
    selected = torch.where(no_safe, fallback_score.argmax(dim=1), selected)
    keys = (
        "poses",
        "steering",
        "curvature",
        "step_magnitude",
        "raw_step_magnitude",
        "gear",
        "lengths",
    )
    base = {
        key: _candidate_gather(output[key], selected)
        for key in keys
        if key in output
    }
    base["selected"] = selected
    return base, geometry


def _interpolate_action_knots(
    action: torch.Tensor, lengths: torch.Tensor, steps: int
) -> torch.Tensor:
    """Piecewise-linear spline with exactly zero residual at both endpoints."""

    batch, groups, knots = action.shape
    zero = torch.zeros((batch, groups, 1), device=action.device, dtype=action.dtype)
    control = torch.cat((zero, action, zero), dim=-1)
    intervals = knots + 1
    token = torch.arange(steps, device=action.device, dtype=action.dtype)
    denominator = (lengths.to(action.dtype) - 1.0).clamp_min(1.0)
    progress = token.view(1, 1, steps) / denominator[:, None, None]
    progress = progress.clamp(0.0, 1.0) * float(intervals)
    left = progress.floor().long().clamp(0, knots)
    right = (left + 1).clamp_max(knots + 1)
    weight = progress - left.to(progress.dtype)
    expanded = control[:, :, None, :].expand(-1, -1, steps, -1)
    left_value = expanded.gather(-1, left.expand(-1, groups, -1).unsqueeze(-1)).squeeze(-1)
    right_value = expanded.gather(-1, right.expand(-1, groups, -1).unsqueeze(-1)).squeeze(-1)
    residual = left_value + weight.expand(-1, groups, -1) * (right_value - left_value)
    active = token.view(1, 1, steps) < lengths[:, None, None]
    return torch.where(active, residual, torch.zeros_like(residual))


@torch.no_grad()
def refine_paths(
    base: Mapping[str, torch.Tensor],
    goal: torch.Tensor,
    action: torch.Tensor,
    planner_config,
    spec: PathRefinerSpec,
) -> dict[str, torch.Tensor]:
    """Apply the compact action, bicycle-integrate, then hard-anchor the goal."""

    spec.validate()
    if action.ndim != 3 or action.shape[-1] != int(spec.action_dim):
        raise ValueError("action must have shape [B,G,spec.action_dim]")
    batch, groups, _ = action.shape
    base_steering = base["steering"]
    steps = int(base_steering.shape[-1])
    if base_steering.shape[0] != batch:
        raise ValueError("base path and action batch sizes differ")
    lengths = base["lengths"].long().clamp(2, steps)
    if int(spec.topology_count) > 1:
        topology_control = action[..., : int(spec.topology_control_dims)]
        steering_action = action[..., int(spec.topology_control_dims) :]
    else:
        topology_control = action.new_zeros((batch, groups, 2))
        steering_action = action
    residual = _interpolate_action_knots(steering_action, lengths, steps)
    residual = residual * math.radians(float(spec.max_steering_residual_deg))

    token = torch.arange(steps, device=action.device, dtype=action.dtype)
    denominator = (lengths.to(action.dtype) - 1.0).clamp_min(1.0)
    progress = (
        token.view(1, 1, steps) / denominator[:, None, None]
    ).clamp(0.0, 1.0)
    topology_ids = torch.remainder(
        torch.arange(groups, device=action.device),
        max(1, int(spec.topology_count)),
    ).view(1, groups, 1)
    if int(spec.topology_count) > 1:
        # Families 1/2 begin from opposite S-shaped staging corridors.  The
        # profile has zero endpoint steering and approximately zero net yaw,
        # but translates the middle of the route to opposite sides of the
        # obstacle field.  Family 3 receives a gentler double-S profile for a
        # forward/reverse/forward/reverse correction manoeuvre.
        amplitude_scale = 0.65 + 0.35 * (
            topology_control[..., 1:2] + 1.0
        ) * 0.5
        base_bias = math.radians(float(spec.topology_steering_bias_deg))
        single_s = torch.sin(2.0 * math.pi * progress)
        double_s = torch.sin(4.0 * math.pi * progress)
        left = (topology_ids == 1).to(action.dtype)
        right = (topology_ids == 2).to(action.dtype)
        multi = (topology_ids == 3).to(action.dtype)
        multi_side = torch.where(
            topology_control[..., 1:2] >= 0.0,
            torch.ones_like(topology_control[..., 1:2]),
            -torch.ones_like(topology_control[..., 1:2]),
        )
        residual = residual + base_bias * amplitude_scale * (
            (left - right) * single_s
            + 0.55 * multi * multi_side * double_s
        )
    maximum = float(planner_config.vehicle.max_steer_rad)
    steering = (
        base_steering[:, None].expand(-1, groups, -1) + residual
    ).clamp(-maximum, maximum)
    gear = base["gear"].long()[:, None].expand(-1, groups, -1).clone()
    if int(spec.topology_count) > 1:
        # Route family 0 preserves the mature guide.  Families 1/2 keep a
        # simple F->R topology but move the staging boundary; family 3 always
        # explores an executable F->R->F->R sequence.  All changes are masked
        # after the predicted stop token, so padding cannot create gear flicker.
        active = token.view(1, 1, steps) < lengths[:, None, None]
        base_gear = base["gear"].long()
        edge_change = base_gear[:, 1:] != base_gear[:, :-1]
        has_switch = edge_change.any(dim=-1)
        first_switch = edge_change.float().argmax(dim=-1) + 1
        fallback_switch = torch.round(0.32 * lengths.to(action.dtype)).long()
        first_switch = torch.where(has_switch, first_switch, fallback_switch)
        base_fraction = first_switch.to(action.dtype) / denominator
        stage_shift = (
            0.05 + 0.10 * topology_control[..., 0]
        )
        staged_fraction = (
            base_fraction[:, None] + stage_shift
        ).clamp(0.18, 0.55)
        staged_forward = progress < staged_fraction[..., None]
        simple_family = (topology_ids == 1) | (topology_ids == 2)
        simple_gear = torch.where(
            staged_forward,
            torch.ones_like(gear),
            torch.zeros_like(gear),
        )
        gear = torch.where(simple_family & active, simple_gear, gear)

        first_boundary = (
            base_fraction[:, None]
            + 0.03
            + 0.08 * topology_control[..., 0]
        ).clamp(0.18, 0.48)
        correction_center = (
            0.64 + 0.08 * topology_control[..., 1]
        ).clamp(0.52, 0.76)
        correction_half_width = 0.055
        second_boundary = torch.maximum(
            first_boundary + 0.08,
            correction_center - correction_half_width,
        )
        third_boundary = torch.minimum(
            torch.maximum(
                second_boundary + 0.06,
                correction_center + correction_half_width,
            ),
            torch.full_like(second_boundary, 0.88),
        )
        multi_gear = torch.where(
            progress < first_boundary[..., None],
            torch.ones_like(gear),
            torch.where(
                progress < second_boundary[..., None],
                torch.zeros_like(gear),
                torch.where(
                    progress < third_boundary[..., None],
                    torch.ones_like(gear),
                    torch.zeros_like(gear),
                ),
            ),
        )
        gear = torch.where((topology_ids == 3) & active, multi_gear, gear)
    direction = gear.to(steering.dtype).mul(2.0).sub(1.0)
    step = base["step_magnitude"][:, None].expand(-1, groups, -1)
    curvature = torch.tan(steering) / float(planner_config.vehicle.wheelbase_m)
    raw_poses = integrate_primitives(
        curvature,
        direction,
        float(planner_config.data.primitive_step_m)
        / float(max(1, planner_config.data.dense_substeps)),
        step,
    )
    expanded_lengths = lengths[:, None].expand(-1, groups)
    anchor_position, anchor_yaw = terminal_pose_error(
        raw_poses, goal, expanded_lengths
    )
    poses = anchor_terminal_pose_trace(
        raw_poses,
        goal,
        expanded_lengths,
        int(spec.terminal_anchor_steps),
    )
    controls = controls_from_pose_trace(
        poses,
        direction,
        expanded_lengths,
        float(planner_config.vehicle.wheelbase_m),
    )
    return {
        "poses": poses,
        "curvature": controls["curvature"],
        "steering": controls["steering"],
        "steering_increment": controls["steering_increment"],
        "step_magnitude": controls["step_magnitude"],
        "raw_step_magnitude": controls["raw_step_magnitude"],
        "gear": gear,
        "lengths": expanded_lengths,
        "scores": torch.zeros((batch, groups), device=poses.device, dtype=poses.dtype),
        "terminal_anchor_position_correction_m": anchor_position,
        "terminal_anchor_yaw_correction_rad": anchor_yaw,
        "residual_steering": residual,
        "topology_ids": topology_ids.expand(batch, -1, -1).squeeze(-1),
    }


@torch.no_grad()
def score_paths(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    planner_config,
    spec: PathRefinerSpec,
    action: torch.Tensor | None = None,
    collision_free_override: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    geometry = evaluate_inference_batch(
        output,
        batch,
        planner_config,
        collision_free_override=collision_free_override,
    )
    clearance = geometry["candidate_minimum_clearance_m"]
    collision_free = geometry["candidate_collision_free"]
    clearance_safe = collision_free & (clearance >= float(spec.minimum_clearance_m))
    trackable = geometry["candidate_trackable"]
    anchor_position = output["terminal_anchor_position_correction_m"]
    anchor_yaw_deg = torch.rad2deg(output["terminal_anchor_yaw_correction_rad"])
    anchor_valid = (
        (anchor_position <= float(spec.anchor_position_limit_m))
        & (anchor_yaw_deg <= float(spec.anchor_yaw_limit_deg))
    )
    maximum_steering = output["steering"].abs().amax(dim=-1)
    steering_valid = maximum_steering <= float(planner_config.vehicle.max_steer_rad) + 1.0e-4
    success = clearance_safe & trackable & anchor_valid & steering_valid

    path_length = geometry["candidate_path_length_m"]
    variation = geometry["candidate_steering_variation_deg_per_m"]
    jump = geometry["candidate_max_same_gear_steer_jump_deg"]
    saturation = geometry["candidate_steering_saturation_fraction"]
    reward = (
        100.0 * success.float()
        + 25.0 * collision_free.float()
        - 100.0 * (~collision_free).float()
        + 15.0 * clearance_safe.float()
        + 10.0 * trackable.float()
        - 25.0 * (~trackable).float()
        + 10.0 * anchor_valid.float()
        - 25.0 * (~anchor_valid).float()
        + 12.0 * (clearance / 0.50).clamp(-1.0, 1.0)
        - 0.08 * path_length
        - 0.08 * variation.clamp_max(150.0)
        - 0.10 * jump.clamp_max(180.0)
        - 8.0 * saturation
        - 8.0 * (anchor_position / float(spec.anchor_position_limit_m)).clamp_max(3.0)
        - 5.0 * (anchor_yaw_deg / float(spec.anchor_yaw_limit_deg)).clamp_max(3.0)
    )
    if action is not None:
        reward = reward - 1.5 * action.square().mean(dim=-1)
    token = torch.arange(
        output["gear"].shape[-1], device=output["gear"].device
    ).view(1, 1, -1)
    active_edge = token[..., 1:] < output["lengths"][..., None]
    gear_switches = (
        (output["gear"][..., 1:] != output["gear"][..., :-1]) & active_edge
    ).sum(dim=-1)
    # One or three deliberate transitions are both valid parking topologies.
    # Penalize only excessive switching; otherwise RL would suppress exactly
    # the recovery family needed by the far benchmark poses.
    reward = reward - 0.5 * (gear_switches - 3).clamp_min(0).to(reward.dtype)
    return {
        "reward": reward,
        "success": success,
        "collision_free": collision_free,
        "clearance_safe": clearance_safe,
        "trackable": trackable,
        "anchor_valid": anchor_valid,
        "minimum_clearance_m": clearance,
        "path_length_m": path_length,
        "steering_variation_deg_per_m": variation,
        "max_same_gear_steer_jump_deg": jump,
        "anchor_position_correction_m": anchor_position,
        "anchor_yaw_correction_deg": anchor_yaw_deg,
        "gear_switches": gear_switches,
        "topology_ids": output.get("topology_ids"),
    }


def _rasterize_center_path(
    poses: torch.Tensor, lengths: torch.Tensor, planner_config, size: int
) -> torch.Tensor:
    batch, _groups, steps, _ = poses.shape
    path = torch.zeros((batch, 1, size, size), device=poses.device, dtype=torch.float32)
    rows = poses[:, 0]
    data = planner_config.data
    col = (
        (rows[..., 0] - float(data.raster_center_x_m)) / float(data.extent_m)
        + 0.5
    ) * float(size - 1)
    row = (
        0.5
        - (rows[..., 1] - float(data.raster_center_y_m)) / float(data.extent_m)
    ) * float(size - 1)
    col = col.round().long().clamp(0, size - 1)
    row = row.round().long().clamp(0, size - 1)
    token = torch.arange(steps, device=poses.device)[None]
    active = token < lengths[:, None]
    for sample in range(batch):
        path[sample, 0, row[sample, active[sample]], col[sample, active[sample]]] = 1.0
    return F.max_pool2d(path, kernel_size=3, stride=1, padding=1)


@torch.no_grad()
def build_policy_observation(
    batch: Mapping[str, torch.Tensor],
    base_output: Mapping[str, torch.Tensor],
    base_score: Mapping[str, torch.Tensor],
    planner_config,
    spec: PathRefinerSpec,
) -> dict[str, torch.Tensor]:
    size = int(spec.raster_size)
    occupancy = F.interpolate(
        batch["bev"][:, 0:1].float(), size=(size, size), mode="nearest"
    )
    signed_distance = F.interpolate(
        batch["bev"][:, 2:3].float(),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )
    goal_mask = F.interpolate(
        batch["bev"][:, 4:5].float(), size=(size, size), mode="nearest"
    )
    path = _rasterize_center_path(
        base_output["poses"], base_output["lengths"][:, 0], planner_config, size
    )
    raster = torch.cat((occupancy, signed_distance, path, goal_mask), dim=1)

    sample_count = 12
    progress = torch.linspace(0.0, 1.0, sample_count, device=raster.device)
    lengths = base_output["lengths"][:, 0]
    indices = torch.round(progress[None] * (lengths[:, None] - 1)).long()
    steering = base_output["steering"][:, 0].gather(1, indices)
    steering = steering / float(planner_config.vehicle.max_steer_rad)
    gear = base_output["gear"][:, 0].gather(1, indices).float().mul(2.0).sub(1.0)
    goal = batch["goal"].float().clone()
    goal[:, :2] /= 10.0
    scalar = torch.stack(
        (
            base_score["path_length_m"][:, 0] / 20.0,
            base_score["minimum_clearance_m"][:, 0].clamp(-1.0, 1.0),
            base_score["anchor_position_correction_m"][:, 0]
            / float(spec.anchor_position_limit_m),
            base_score["anchor_yaw_correction_deg"][:, 0]
            / float(spec.anchor_yaw_limit_deg),
        ),
        dim=-1,
    )
    vector = torch.cat((goal, steering, gear, scalar), dim=-1)
    if vector.shape[-1] != int(spec.vector_dim):
        raise AssertionError("unexpected path-refiner vector size {}".format(vector.shape))
    return {"raster": raster.contiguous(), "vector": vector.contiguous()}
