"""Checkpoint loading and exact-filtered candidate selection."""

from __future__ import annotations

from pathlib import Path
import math
from typing import Mapping, Sequence

import torch

from .config import HopePathConfig, config_from_dict
from .engine import ExponentialMovingAverage
from .geometry import swept_path_poses
from .metrics import evaluate_inference_batch
from .model import HopePathPlanner


GoalTTAOffset = tuple[float, float, float]


def _goal_pose_masks(
    goal: torch.Tensor,
    config: HopePathConfig,
) -> torch.Tensor:
    """Rasterize rear-axle goal poses as binary masks on the model grid."""

    data, vehicle = config.data, config.vehicle
    size = int(data.raster_size)
    half = 0.5 * float(data.extent_m)
    x = torch.linspace(
        float(data.raster_center_x_m) - half,
        float(data.raster_center_x_m) + half,
        size,
        device=goal.device,
        dtype=goal.dtype,
    )
    y = torch.linspace(
        float(data.raster_center_y_m) + half,
        float(data.raster_center_y_m) - half,
        size,
        device=goal.device,
        dtype=goal.dtype,
    )
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    yaw = torch.atan2(goal[:, 2], goal[:, 3])
    dx = grid_x[None] - goal[:, 0, None, None]
    dy = grid_y[None] - goal[:, 1, None, None]
    cosine = torch.cos(yaw)[:, None, None]
    sine = torch.sin(yaw)[:, None, None]
    longitudinal = cosine * dx + sine * dy
    lateral = -sine * dx + cosine * dy
    inside = (
        (longitudinal >= -float(vehicle.rear_overhang_m))
        & (longitudinal <= float(vehicle.wheelbase_m + vehicle.front_overhang_m))
        & (lateral.abs() <= 0.5 * float(vehicle.width_m))
    )
    return inside.to(dtype=goal.dtype)


def perturb_goal_inference_batch(
    batch: Mapping[str, torch.Tensor],
    config: HopePathConfig,
    offset: GoalTTAOffset,
) -> dict[str, torch.Tensor]:
    """Perturb a goal in its own frame and keep vector/raster inputs aligned.

    Offsets are ``(longitudinal_m, lateral_m, yaw_deg)``.  This is inference
    augmentation only: candidates are still selected against the untouched
    original goal and exact collision geometry.
    """

    longitudinal_m, lateral_m, yaw_deg = map(float, offset)
    original = batch["goal"]
    yaw = torch.atan2(original[:, 2], original[:, 3])
    cosine, sine = torch.cos(yaw), torch.sin(yaw)
    perturbed = original.clone()
    perturbed[:, 0] += cosine * longitudinal_m - sine * lateral_m
    perturbed[:, 1] += sine * longitudinal_m + cosine * lateral_m
    perturbed_yaw = yaw + math.radians(yaw_deg)
    perturbed[:, 2] = torch.sin(perturbed_yaw)
    perturbed[:, 3] = torch.cos(perturbed_yaw)

    result = dict(batch)
    result["goal"] = perturbed
    result["bev"] = batch["bev"].clone()
    result["bev"][:, 4] = _goal_pose_masks(perturbed, config)
    return result


def mirror_inference_batch(
    batch: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Reflect a local-frame inference batch across the vehicle x-axis."""

    mirrored = dict(batch)
    mirrored["bev"] = torch.flip(batch["bev"], dims=(-2,))
    mirrored["goal"] = batch["goal"].clone()
    mirrored["goal"][..., 1] *= -1.0
    mirrored["goal"][..., 2] *= -1.0
    mirrored["obstacle_vertices"] = batch["obstacle_vertices"].clone()
    mirrored["obstacle_vertices"][..., 1] *= -1.0
    mirrored["map_polygon"] = batch["map_polygon"].clone()
    mirrored["map_polygon"][..., 1] *= -1.0
    return mirrored


def unmirror_planner_output(
    output: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return mirrored predictions in the original local coordinate frame."""

    restored = dict(output)
    restored["poses"] = output["poses"].clone()
    restored["poses"][..., 1] *= -1.0
    restored["poses"][..., 2] *= -1.0
    if "curvature" in output:
        restored["curvature"] = -output["curvature"]
    if "steering" in output:
        restored["steering"] = -output["steering"]
    if "steering_increment" in output:
        restored["steering_increment"] = -output["steering_increment"]
    if "coarse_pose" in output:
        restored["coarse_pose"] = output["coarse_pose"].clone()
        restored["coarse_pose"][..., 1] *= -1.0
        restored["coarse_pose"][..., 2] *= -1.0
    if "refined_xy" in output:
        restored["refined_xy"] = output["refined_xy"].clone()
        restored["refined_xy"][..., 1] *= -1.0
    return restored


def concatenate_candidate_outputs(
    left: Mapping[str, torch.Tensor],
    right: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Concatenate model outputs whose second dimension is candidate K."""

    left_candidates = int(left["poses"].shape[1])
    right_candidates = int(right["poses"].shape[1])
    combined: dict[str, torch.Tensor] = {}
    for key, value in left.items():
        other = right.get(key)
        if (
            torch.is_tensor(value)
            and torch.is_tensor(other)
            and value.ndim >= 2
            and other.ndim == value.ndim
            and value.shape[1] == left_candidates
            and other.shape[1] == right_candidates
        ):
            combined[key] = torch.cat((value, other), dim=1)
        else:
            combined[key] = value
    return combined


def exact_candidate_collision_free(
    output: Mapping[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    config: HopePathConfig,
) -> torch.Tensor:
    """CPU exact polygon validation at 0.1 m-or-finer swept samples."""

    import numpy as np
    import shapely
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    poses = output["poses"].detach().cpu().numpy()
    lengths = output["lengths"].detach().cpu().numpy()
    vertices = batch["obstacle_vertices"].detach().cpu().numpy()
    vertex_mask = batch["obstacle_vertex_mask"].detach().cpu().numpy()
    obstacle_mask = batch["obstacle_mask"].detach().cpu().numpy()
    boundaries = batch["map_polygon"].detach().cpu().numpy()
    valid = np.ones(lengths.shape, dtype=np.bool_)
    vehicle = config.vehicle
    footprint_local = np.asarray(
        [(-vehicle.rear_overhang_m, -0.5 * vehicle.width_m),
         (vehicle.wheelbase_m + vehicle.front_overhang_m, -0.5 * vehicle.width_m),
         (vehicle.wheelbase_m + vehicle.front_overhang_m, 0.5 * vehicle.width_m),
         (-vehicle.rear_overhang_m, 0.5 * vehicle.width_m)], dtype=np.float64,
    )
    for sample in range(len(poses)):
        obstacles = []
        for polygon_index in np.flatnonzero(obstacle_mask[sample]):
            polygon_vertices = vertices[sample, polygon_index][vertex_mask[sample, polygon_index]]
            if len(polygon_vertices) >= 3:
                obstacles.append(Polygon(polygon_vertices))
        obstacle_union = unary_union(obstacles) if obstacles else None
        map_polygon = Polygon(boundaries[sample])
        for candidate in range(poses.shape[1]):
            length = int(lengths[sample, candidate])
            path = poses[sample, candidate, :length]
            swept = swept_path_poses(path, max_spacing_m=0.1)
            cosine, sine = np.cos(swept[:, 2]), np.sin(swept[:, 2])
            local_x, local_y = footprint_local[:, 0], footprint_local[:, 1]
            corners = np.stack(
                (swept[:, None, 0] + cosine[:, None] * local_x - sine[:, None] * local_y,
                 swept[:, None, 1] + sine[:, None] * local_x + cosine[:, None] * local_y),
                axis=-1,
            )
            footprints = shapely.polygons(corners)
            inside = shapely.covers(map_polygon, footprints)
            collision = (
                shapely.intersects(footprints, obstacle_union)
                if obstacle_union is not None else np.zeros(len(footprints), dtype=np.bool_)
            )
            valid[sample, candidate] = bool(np.all(inside) and not np.any(collision))
    return torch.from_numpy(valid).to(output["poses"].device)


def load_planner_checkpoint(
    path: str | Path,
    device: str | torch.device | None = None,
    *,
    use_ema: bool = True,
) -> tuple[HopePathPlanner, HopePathConfig, dict[str, object]]:
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = config_from_dict(checkpoint["config"])
    model = HopePathPlanner(config).to(device)
    model.load_state_dict(checkpoint["model"])
    if use_ema and "ema" in checkpoint:
        ema = ExponentialMovingAverage(model, config.training.ema_decay)
        ema.load_state_dict(checkpoint["ema"])
        ema.copy_to(model)
    model.eval()
    return model, config, checkpoint


@torch.no_grad()
def generate_candidate_output(
    model: HopePathPlanner,
    batch: Mapping[str, torch.Tensor],
    *,
    candidate_count: int | None = None,
    mirror_tta: bool = False,
    goal_tta_offsets: Sequence[GoalTTAOffset] = (),
) -> dict[str, torch.Tensor]:
    """Generate native, reflected and optional goal-perturbed candidates."""

    def infer_one(source: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        result = model.infer(source, candidate_count=candidate_count)
        if mirror_tta:
            mirrored_batch = mirror_inference_batch(source)
            mirrored_output = unmirror_planner_output(
                model.infer(mirrored_batch, candidate_count=candidate_count)
            )
            result = concatenate_candidate_outputs(result, mirrored_output)
        return result

    output = infer_one(batch)
    seen: set[GoalTTAOffset] = set()
    for raw_offset in goal_tta_offsets:
        offset = tuple(map(float, raw_offset))
        if len(offset) != 3:
            raise ValueError("goal TTA offsets must contain longitudinal, lateral and yaw")
        if offset == (0.0, 0.0, 0.0) or offset in seen:
            continue
        seen.add(offset)
        augmented = infer_one(perturb_goal_inference_batch(batch, model.config, offset))
        output = concatenate_candidate_outputs(output, augmented)
    return output


@torch.no_grad()
def select_candidate_output(
    output: dict[str, torch.Tensor],
    batch: Mapping[str, torch.Tensor],
    config: HopePathConfig,
    *,
    exact_collision: bool = True,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Apply exact collision and goal-envelope selection to candidates."""

    exact = exact_candidate_collision_free(output, batch, config) if exact_collision else None
    geometry = evaluate_inference_batch(
        output, batch, config, collision_free_override=exact
    )
    output["selected"] = geometry["selected"]
    return output, geometry


@torch.no_grad()
def infer_and_select(
    model: HopePathPlanner,
    batch: Mapping[str, torch.Tensor],
    config: HopePathConfig,
    *,
    exact_collision: bool = True,
    candidate_count: int | None = None,
    mirror_tta: bool = False,
    goal_tta_offsets: Sequence[GoalTTAOffset] = (),
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    output = generate_candidate_output(
        model, batch, candidate_count=candidate_count, mirror_tta=mirror_tta,
        goal_tta_offsets=goal_tta_offsets,
    )
    return select_candidate_output(
        output, batch, config, exact_collision=exact_collision
    )


@torch.no_grad()
def infer_ensemble_and_select(
    models: Sequence[HopePathPlanner],
    batch: Mapping[str, torch.Tensor],
    config: HopePathConfig,
    *,
    exact_collision: bool = True,
    candidate_count: int | None = None,
    mirror_tta: bool = False,
    goal_tta_offsets: Sequence[GoalTTAOffset] = (),
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Union candidates from multiple compatible checkpoints, then select."""

    if not models:
        raise ValueError("ensemble requires at least one model")
    outputs = [
        generate_candidate_output(
            model, batch, candidate_count=candidate_count, mirror_tta=mirror_tta,
            goal_tta_offsets=goal_tta_offsets,
        )
        for model in models
    ]
    output = outputs[0]
    for additional in outputs[1:]:
        output = concatenate_candidate_outputs(output, additional)
    return select_candidate_output(
        output, batch, config, exact_collision=exact_collision
    )
