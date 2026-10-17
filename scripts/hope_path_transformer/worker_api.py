"""Python 3.11 worker API for live CVT-to-path-Transformer inference."""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw

from hybrid_astar_8ch_bev import VehicleConfig as LiveVehicleConfig
from rl_ogm.collision import ExactRasterFootprintCollisionChecker

from .geometry import (
    clip_obstacle_polygons_to_raster_extent,
    distance_transform_edt,
    local_to_pixel,
    rasterize_scene_base,
    rasterize_pose_mask,
    swept_path_poses,
    select_route_nearest_polygons,
)
from .inference import (
    concatenate_candidate_outputs,
    generate_candidate_output,
    load_planner_checkpoint,
)
from .metrics import evaluate_inference_batch
from .model import (
    anchor_terminal_pose_trace as anchor_terminal_pose_trace_torch,
    controls_from_pose_trace,
    terminal_pose_error,
)
from path_refiner_rl.core import (
    PathRefinerSpec,
    build_policy_observation,
    refine_paths,
    score_paths,
    select_base_path,
)
from path_refiner_rl.model import PathRefinerPolicy


ROOT = Path(__file__).resolve().parents[2]


def _wrap_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def _resolve_profile_path(value: str | Path, profile_path: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    root_candidate = ROOT / path
    if root_candidate.is_file():
        return root_candidate
    return profile_path.parent / path


def _profile_entries(profile: Mapping[str, object]) -> list[Mapping[str, object]]:
    """Return validated checkpoint entries for a single model or ensemble."""

    primary = profile.get("primary")
    if not isinstance(primary, Mapping):
        raise ValueError("supervised path profile requires a primary checkpoint")
    additional = list(profile.get("additional") or [])
    if any(not isinstance(entry, Mapping) for entry in additional):
        raise ValueError("supervised path profile has an invalid additional checkpoint")
    entries = [primary, *additional]
    if bool(profile.get("single_model_only", False)) and len(entries) != 1:
        raise ValueError(
            "single-model supervised path profile must contain exactly one checkpoint"
        )
    return entries


def _resolve_checkpoint_entry(
    entry: Mapping[str, object], profile_path: Path
) -> tuple[Path, str]:
    """Resolve a preferred checkpoint and its ordered fallbacks."""

    configured = entry.get("checkpoint")
    if not configured:
        raise ValueError("checkpoint entry is missing 'checkpoint'")
    candidates: list[tuple[str, object]] = [("checkpoint", configured)]
    fallback = entry.get("fallback_checkpoint")
    if fallback:
        candidates.append(("fallback_checkpoint", fallback))
    for value in list(entry.get("fallback_checkpoints") or []):
        candidates.append(("fallback_checkpoints", value))

    attempted = []
    for source, value in candidates:
        path = _resolve_profile_path(str(value), profile_path).resolve()
        attempted.append(str(path))
        if path.is_file():
            return path, source
    raise FileNotFoundError(
        "no configured supervised path checkpoint exists: {}".format(
            ", ".join(attempted)
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _points_source_to_local(points: np.ndarray, start: Sequence[float]) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    dx = values[:, 0] - float(start[0])
    dy = values[:, 1] - float(start[1])
    cosine, sine = math.cos(float(start[2])), math.sin(float(start[2]))
    return np.column_stack((cosine * dx + sine * dy, -sine * dx + cosine * dy))


def _pose_source_to_local(pose: Sequence[float], start: Sequence[float]) -> tuple[float, float, float]:
    xy = _points_source_to_local(np.asarray(pose[:2], dtype=np.float64)[None], start)[0]
    return float(xy[0]), float(xy[1]), _wrap_pi(float(pose[2]) - float(start[2]))


def _poses_local_to_source(poses: np.ndarray, start: Sequence[float]) -> np.ndarray:
    values = np.asarray(poses, dtype=np.float64).reshape((-1, 3)).copy()
    cosine, sine = math.cos(float(start[2])), math.sin(float(start[2]))
    x = float(start[0]) + cosine * values[:, 0] - sine * values[:, 1]
    y = float(start[1]) + sine * values[:, 0] + cosine * values[:, 1]
    yaw = np.asarray([_wrap_pi(value + float(start[2])) for value in values[:, 2]])
    return np.column_stack((x, y, yaw))


def _mask_array(value: object) -> np.ndarray:
    mask = np.asarray(value, dtype=np.uint8)
    if mask.ndim == 3:
        mask = mask[0]
    if mask.ndim != 2:
        raise ValueError("live occupancy must be a 2-D mask")
    return np.ascontiguousarray(mask > 0, dtype=np.uint8)


def _mask_polygons_local(
    mask: np.ndarray,
    bev_config: Mapping[str, object],
    start: Sequence[float],
) -> list[np.ndarray]:
    """Encode each 8-connected live obstacle as a PCA-oriented rectangle."""

    height, width = mask.shape
    x_min = float(bev_config.get("x_min", -10.0))
    x_max = float(bev_config.get("x_max", 10.0))
    y_min = float(bev_config.get("y_min", -10.0))
    y_max = float(bev_config.get("y_max", 10.0))
    x_resolution = (x_max - x_min) / float(height)
    y_resolution = (y_max - y_min) / float(width)
    occupied = np.asarray(mask, dtype=bool)
    visited = np.zeros_like(occupied)
    polygons: list[np.ndarray] = []
    for start_row, start_col in np.argwhere(occupied):
        start_row, start_col = int(start_row), int(start_col)
        if visited[start_row, start_col]:
            continue
        stack = [(start_row, start_col)]
        visited[start_row, start_col] = True
        cells = []
        while stack:
            row, col = stack.pop()
            cells.append((row, col))
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = row + dr, col + dc
                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and occupied[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))
        if not cells:
            continue
        cells_array = np.asarray(cells, dtype=np.float64)
        source = np.column_stack(
            (
                x_max - (cells_array[:, 0] + 0.5) * x_resolution,
                y_min + (cells_array[:, 1] + 0.5) * y_resolution,
            )
        )
        center = source.mean(axis=0)
        centered = source - center
        if len(source) >= 2 and float(np.max(np.linalg.norm(centered, axis=1))) > 1.0e-6:
            _, eigenvectors = np.linalg.eigh(centered.T @ centered)
            axes = eigenvectors[:, ::-1]
        else:
            axes = np.eye(2, dtype=np.float64)
        projected = centered @ axes
        margin = 0.5 * math.hypot(x_resolution, y_resolution)
        low = projected.min(axis=0) - margin
        high = projected.max(axis=0) + margin
        corners_projected = np.asarray(
            ((low[0], low[1]), (high[0], low[1]),
             (high[0], high[1]), (low[0], high[1])),
            dtype=np.float64,
        )
        rectangle_source = corners_projected @ axes.T + center
        polygons.append(
            _points_source_to_local(rectangle_source, start).astype(np.float32)
        )
    return polygons


def _warp_live_mask_to_model(
    mask: np.ndarray,
    bev_config: Mapping[str, object],
    start: Sequence[float],
    data,
) -> np.ndarray:
    """Nearest-cell resample of the CVT mask into the model's local raster."""

    height, width = mask.shape
    x_min = float(bev_config.get("x_min", -10.0))
    x_max = float(bev_config.get("x_max", 10.0))
    y_min = float(bev_config.get("y_min", -10.0))
    y_max = float(bev_config.get("y_max", 10.0))
    size = int(data.raster_size)
    half = 0.5 * float(data.extent_m)
    local_x = np.linspace(
        float(data.raster_center_x_m) - half,
        float(data.raster_center_x_m) + half,
        size,
    )
    local_y = np.linspace(
        float(data.raster_center_y_m) + half,
        float(data.raster_center_y_m) - half,
        size,
    )
    grid_x, grid_y = np.meshgrid(local_x, local_y)
    cosine, sine = math.cos(float(start[2])), math.sin(float(start[2]))
    source_x = float(start[0]) + cosine * grid_x - sine * grid_y
    source_y = float(start[1]) + sine * grid_x + cosine * grid_y
    rows = np.floor((x_max - source_x) / (x_max - x_min) * height).astype(np.int64)
    cols = np.floor((source_y - y_min) / (y_max - y_min) * width).astype(np.int64)
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    output = np.ones((size, size), dtype=np.uint8)
    output[valid] = mask[rows[valid], cols[valid]]
    return output


def _map_occupancy(map_polygon: np.ndarray, data) -> np.ndarray:
    image = Image.new("L", (int(data.raster_size), int(data.raster_size)), color=1)
    pixels = [
        tuple(point) for point in local_to_pixel(map_polygon, data)
    ]
    if len(pixels) >= 3:
        ImageDraw.Draw(image).polygon(pixels, fill=0)
    return np.asarray(image, dtype=np.uint8)


def _distance_to_route(polygon: np.ndarray, goal_xy: Sequence[float]) -> float:
    point = np.asarray(polygon, dtype=np.float64).mean(axis=0)
    goal = np.asarray(goal_xy, dtype=np.float64)
    denominator = float(np.dot(goal, goal))
    ratio = 0.0 if denominator <= 1.0e-9 else float(np.clip(np.dot(point, goal) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - ratio * goal))


def _vector_polygon(polygon: np.ndarray, max_vertices: int) -> np.ndarray:
    points = np.asarray(polygon, dtype=np.float32).reshape((-1, 2))
    if len(points) <= max_vertices:
        return points
    indices = np.linspace(0, len(points), max_vertices, endpoint=False).astype(np.int64)
    return points[indices]


def _pack_polygons(
    polygons: Sequence[np.ndarray], max_obstacles: int, max_vertices: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = min(len(polygons), int(max_obstacles))
    values = np.zeros((int(max_obstacles), int(max_vertices), 2), dtype=np.float32)
    vertex_mask = np.zeros((int(max_obstacles), int(max_vertices)), dtype=np.bool_)
    polygon_mask = np.zeros(int(max_obstacles), dtype=np.bool_)
    for index, polygon in enumerate(polygons[:count]):
        points = _vector_polygon(polygon, int(max_vertices))
        size = min(len(points), int(max_vertices))
        if size < 3:
            continue
        values[index, :size] = points[:size]
        vertex_mask[index, :size] = True
        polygon_mask[index] = True
    return values, vertex_mask, polygon_mask


def _pack_collision_polygons(
    polygons: Sequence[np.ndarray], max_vertices: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not polygons:
        return (
            np.zeros((1, max_vertices, 2), dtype=np.float32),
            np.zeros((1, max_vertices), dtype=np.bool_),
            np.zeros(1, dtype=np.bool_),
        )
    values = np.zeros((len(polygons), max_vertices, 2), dtype=np.float32)
    vertex_mask = np.zeros((len(polygons), max_vertices), dtype=np.bool_)
    polygon_mask = np.zeros(len(polygons), dtype=np.bool_)
    for index, polygon in enumerate(polygons):
        points = _vector_polygon(polygon, max_vertices)
        size = min(len(points), max_vertices)
        if size < 3:
            continue
        values[index, :size] = points[:size]
        vertex_mask[index, :size] = True
        polygon_mask[index] = True
    return values, vertex_mask, polygon_mask


def _live_vehicle_config(value: Mapping[str, object]) -> LiveVehicleConfig:
    vehicle = LiveVehicleConfig()
    for key, item in dict(value or {}).items():
        if hasattr(vehicle, key):
            setattr(vehicle, key, item)
    return vehicle


def _minimum_polygon_boundary_clearance(
    points: np.ndarray, polygon: np.ndarray
) -> float:
    """Return the shortest point-to-edge distance for points inside a polygon."""

    values = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    boundary = np.asarray(polygon, dtype=np.float64).reshape((-1, 2))
    if len(values) == 0 or len(boundary) < 3:
        return float("inf")
    minimum = float("inf")
    for index in range(len(boundary)):
        begin = boundary[index]
        end = boundary[(index + 1) % len(boundary)]
        edge = end - begin
        denominator = float(np.dot(edge, edge))
        if denominator <= 1.0e-12:
            distance = np.linalg.norm(values - begin[None], axis=1)
        else:
            ratio = np.clip(
                ((values - begin[None]) @ edge) / denominator, 0.0, 1.0
            )
            closest = begin[None] + ratio[:, None] * edge[None]
            distance = np.linalg.norm(values - closest, axis=1)
        minimum = min(minimum, float(distance.min(initial=float("inf"))))
    return minimum


def _live_candidate_geometry(
    output: Mapping[str, torch.Tensor],
    safety_occupancy: np.ndarray,
    start: Sequence[float],
    boundary_source: Sequence[Sequence[float]],
    vehicle: LiveVehicleConfig,
    target_polygon_source: Sequence[Sequence[float]] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return exact live validity, clearance and terminal-slot containment."""

    checker = ExactRasterFootprintCollisionChecker(safety_occupancy, vehicle)
    poses = output["poses"].detach().float().cpu().numpy()
    lengths = output["lengths"].detach().cpu().numpy()
    valid = np.zeros(lengths.shape, dtype=np.bool_)
    clearance = np.full(lengths.shape, -np.inf, dtype=np.float32)
    final_inside = np.ones(lengths.shape, dtype=np.bool_)
    boundary = np.asarray(boundary_source, dtype=np.float64).reshape((-1, 2))
    target_polygon = np.asarray(
        target_polygon_source if target_polygon_source is not None else [],
        dtype=np.float64,
    ).reshape((-1, 2))
    for sample in range(len(poses)):
        for candidate in range(poses.shape[1]):
            length = int(lengths[sample, candidate])
            swept_local = swept_path_poses(
                poses[sample, candidate, :length], max_spacing_m=0.1
            )
            swept_source = _poses_local_to_source(swept_local, start)
            collision_free, minimum_clearance = (
                _vectorized_exact_raster_path_clearance(
                    checker, swept_source
                )
            )
            inside = bool(collision_free)
            if inside and len(boundary) >= 3:
                corners = _vehicle_corners_for_poses(swept_source, vehicle)
                inside = bool(
                    np.all(
                        _points_in_polygon(
                            corners.reshape((-1, 2)), boundary, tolerance=1.0e-6
                        )
                    )
                )
                if inside:
                    minimum_clearance = min(
                        minimum_clearance,
                        _minimum_polygon_boundary_clearance(
                            corners.reshape((-1, 2)), boundary
                        ),
                    )
            if len(target_polygon) >= 3:
                endpoint_corners = _vehicle_corners_for_poses(
                    swept_source[-1:], vehicle
                ).reshape((-1, 2))
                final_inside[sample, candidate] = bool(
                    np.all(
                        _points_in_polygon(
                            endpoint_corners, target_polygon, tolerance=0.18
                        )
                    )
                )
            valid[sample, candidate] = bool(collision_free and inside)
            clearance[sample, candidate] = float(minimum_clearance)
    device = output["poses"].device
    return (
        torch.from_numpy(valid).to(device),
        torch.from_numpy(clearance).to(device),
        torch.from_numpy(final_inside).to(device),
    )


def _live_candidate_collision_free(
    output: Mapping[str, torch.Tensor],
    safety_occupancy: np.ndarray,
    start: Sequence[float],
    boundary_source: Sequence[Sequence[float]],
    vehicle: LiveVehicleConfig,
) -> torch.Tensor:
    """Compatibility wrapper returning exact live collision validity only."""

    valid, _, _ = _live_candidate_geometry(
        output, safety_occupancy, start, boundary_source, vehicle
    )
    return valid


def _vectorized_exact_raster_path_clear(
    checker: ExactRasterFootprintCollisionChecker, poses: np.ndarray
) -> bool:
    """Vectorized equivalent of ExactRasterFootprintCollisionChecker."""

    valid, _ = _vectorized_exact_raster_path_clearance(checker, poses)
    return valid


def _vectorized_exact_raster_path_clearance(
    checker: ExactRasterFootprintCollisionChecker, poses: np.ndarray
) -> tuple[bool, float]:
    """Validate a swept path and return its minimum physical separation."""

    rows = np.asarray(poses, dtype=np.float64).reshape((-1, 3))
    if len(rows) == 0:
        return False, -float("inf")
    vehicle = checker.vehicle
    half_length = 0.5 * float(vehicle.length) + float(checker.footprint_margin)
    half_width = 0.5 * float(vehicle.width) + float(checker.footprint_margin)
    cosine, sine = np.cos(rows[:, 2]), np.sin(rows[:, 2])
    center_x = (
        rows[:, 0]
        + float(vehicle.rear_axle_to_center) * cosine
        - float(vehicle.rear_axle_lateral_offset) * sine
    )
    center_y = (
        rows[:, 1]
        + float(vehicle.rear_axle_to_center) * sine
        + float(vehicle.rear_axle_lateral_offset) * cosine
    )
    extent_x = half_length * np.abs(cosine) + half_width * np.abs(sine)
    extent_y = half_length * np.abs(sine) + half_width * np.abs(cosine)
    bev_boundary_clearance = np.minimum.reduce(
        (
            center_x - extent_x + 10.0,
            10.0 - center_x - extent_x,
            center_y - extent_y + 10.0,
            10.0 - center_y - extent_y,
        )
    )
    minimum_separation = float(
        bev_boundary_clearance.min(initial=float("inf"))
    )
    if minimum_separation < 0.0:
        return False, minimum_separation
    if len(checker.obstacle_x) == 0:
        return True, minimum_separation
    cell_half = float(checker.cell_half)
    for begin in range(0, len(rows), 64):
        end = min(len(rows), begin + 64)
        dx = checker.obstacle_x[None, :] - center_x[begin:end, None]
        dy = checker.obstacle_y[None, :] - center_y[begin:end, None]
        c = cosine[begin:end, None]
        s = sine[begin:end, None]
        local_x = dx * c + dy * s
        local_y = -dx * s + dy * c
        projection = cell_half * (np.abs(c) + np.abs(s))
        pair_separation = np.max(
            np.stack(
                (
                    np.abs(local_x) - (half_length + projection),
                    np.abs(local_y) - (half_width + projection),
                    np.abs(dx) - (cell_half + extent_x[begin:end, None]),
                    np.abs(dy) - (cell_half + extent_y[begin:end, None]),
                ),
                axis=0,
            ),
            axis=0,
        )
        batch_minimum = float(
            pair_separation.min(initial=float("inf"))
        )
        minimum_separation = min(minimum_separation, batch_minimum)
        if batch_minimum <= 1.0e-9:
            return False, minimum_separation
    return True, minimum_separation


def _vehicle_corners_for_poses(
    poses: np.ndarray, vehicle: LiveVehicleConfig
) -> np.ndarray:
    rows = np.asarray(poses, dtype=np.float64).reshape((-1, 3))
    half_length, half_width = 0.5 * float(vehicle.length), 0.5 * float(vehicle.width)
    local = np.asarray(
        (
            (float(vehicle.rear_axle_to_center) + half_length,
             float(vehicle.rear_axle_lateral_offset) - half_width),
            (float(vehicle.rear_axle_to_center) + half_length,
             float(vehicle.rear_axle_lateral_offset) + half_width),
            (float(vehicle.rear_axle_to_center) - half_length,
             float(vehicle.rear_axle_lateral_offset) + half_width),
            (float(vehicle.rear_axle_to_center) - half_length,
             float(vehicle.rear_axle_lateral_offset) - half_width),
        ),
        dtype=np.float64,
    )
    cosine, sine = np.cos(rows[:, 2]), np.sin(rows[:, 2])
    x = rows[:, None, 0] + cosine[:, None] * local[None, :, 0] - sine[:, None] * local[None, :, 1]
    y = rows[:, None, 1] + sine[:, None] * local[None, :, 0] + cosine[:, None] * local[None, :, 1]
    return np.stack((x, y), axis=-1)


def _points_in_polygon(
    points: np.ndarray, polygon: np.ndarray, tolerance: float = 0.0
) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    polygon = np.asarray(polygon, dtype=np.float64).reshape((-1, 2))
    inside = np.zeros(len(points), dtype=np.bool_)
    near_edge = np.zeros(len(points), dtype=np.bool_)
    x, y = points[:, 0], points[:, 1]
    for index in range(len(polygon)):
        x0, y0 = polygon[index]
        x1, y1 = polygon[(index + 1) % len(polygon)]
        dx, dy = float(x1 - x0), float(y1 - y0)
        denominator = dx * dx + dy * dy
        if tolerance > 0.0 and denominator > 1.0e-12:
            ratio = np.clip(((x - x0) * dx + (y - y0) * dy) / denominator, 0.0, 1.0)
            near_edge |= np.hypot(x - (x0 + ratio * dx), y - (y0 + ratio * dy)) <= tolerance
        crossing_edge = (y0 > y) != (y1 > y)
        crossing_x = dx * (y - y0) / (dy if abs(dy) > 1.0e-12 else 1.0e-12) + x0
        inside ^= crossing_edge & (x < crossing_x)
    return inside | near_edge


def _candidate_control_metrics(
    output: Mapping[str, torch.Tensor], wheelbase_m: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure path length, gear switches and steering variation per candidate."""

    poses = output["poses"].detach().float().cpu().numpy()
    curvature = output["curvature"].detach().float().cpu().numpy()
    gear = output["gear"].detach().cpu().numpy()
    lengths = output["lengths"].detach().cpu().numpy()
    path_length = np.zeros(lengths.shape, dtype=np.float32)
    gear_switches = np.zeros(lengths.shape, dtype=np.float32)
    steering_variation = np.zeros(lengths.shape, dtype=np.float32)
    for sample in range(len(poses)):
        for candidate in range(poses.shape[1]):
            length = max(2, int(lengths[sample, candidate]))
            rows = poses[sample, candidate, :length]
            path_length[sample, candidate] = float(
                np.linalg.norm(np.diff(rows[:, :2], axis=0), axis=1).sum()
            )
            gears = gear[sample, candidate, :length]
            gear_switches[sample, candidate] = float(
                np.count_nonzero(gears[1:] != gears[:-1])
            )
            steering = np.arctan(
                curvature[sample, candidate, :length] * float(wheelbase_m)
            )
            steering_variation[sample, candidate] = float(
                np.abs(np.diff(steering)).sum()
            )
    device = output["poses"].device
    return (
        torch.from_numpy(path_length).to(device),
        torch.from_numpy(gear_switches).to(device),
        torch.from_numpy(steering_variation).to(device),
    )


def _select_runtime_candidate(
    *,
    fallback_selected: torch.Tensor,
    exact_success: torch.Tensor,
    collision_free: torch.Tensor,
    final_inside: torch.Tensor,
    live_clearance_m: torch.Tensor,
    goal_position_error_m: torch.Tensor,
    goal_yaw_error_deg: torch.Tensor,
    path_length_m: torch.Tensor,
    gear_switches: torch.Tensor,
    steering_variation_rad: torch.Tensor,
    minimum_clearance_m: float,
    goal_position_tolerance_m: float,
    goal_yaw_tolerance_deg: float,
    weights: Mapping[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Select a controller-friendly candidate from the robust feasible set."""

    robust_success = (
        exact_success.to(torch.bool)
        & collision_free.to(torch.bool)
        & final_inside.to(torch.bool)
        & (live_clearance_m >= float(minimum_clearance_m))
    )
    normalized_goal_error = torch.maximum(
        goal_position_error_m / max(1.0e-6, float(goal_position_tolerance_m)),
        goal_yaw_error_deg / max(1.0e-6, float(goal_yaw_tolerance_deg)),
    )
    clearance_reward = (
        (live_clearance_m - float(minimum_clearance_m)) / 0.75
    ).clamp(0.0, 1.0)
    cost = (
        float(weights.get("goal_error", 1.0)) * normalized_goal_error
        + float(weights.get("steering_variation", 0.08))
        * steering_variation_rad
        + float(weights.get("gear_switches", 0.35)) * gear_switches
        + float(weights.get("path_length", 0.02)) * path_length_m
        - float(weights.get("clearance", 1.25)) * clearance_reward
    )
    clearance_safe = (
        collision_free.to(torch.bool)
        & (live_clearance_m >= float(minimum_clearance_m))
    )
    feasible_cost = cost.masked_fill(~robust_success, float("inf"))
    selected_robust = feasible_cost.argmin(dim=1)
    any_robust = robust_success.any(dim=1)
    # If no candidate reaches the strict parking-success envelope, still
    # return the best physically executable candidate.  The live adapter will
    # follow it while keeping the strict result available as a diagnostic.
    safe_cost = cost.masked_fill(~clearance_safe, float("inf"))
    selected_safe = safe_cost.argmin(dim=1)
    any_safe = clearance_safe.any(dim=1)
    selected = torch.where(
        any_robust,
        selected_robust,
        torch.where(any_safe, selected_safe, fallback_selected),
    )
    return selected, robust_success, cost


class SupervisedPathWorkerAPI:
    """Load a profiled single model or ensemble and plan from a live CVT BEV."""

    def __init__(self, profile_path: str | Path, device: str = "auto"):
        self.profile_path = Path(profile_path).resolve()
        if not self.profile_path.is_file():
            raise FileNotFoundError(self.profile_path)
        self.profile = json.loads(self.profile_path.read_text(encoding="utf-8-sig"))
        if self.profile.get("format") != "hope-path-transformer-inference-profile-v1":
            raise ValueError("unsupported supervised path inference profile format")
        self.device = torch.device(
            "cuda" if str(device) == "auto" and torch.cuda.is_available() else
            "cpu" if str(device) == "auto" else str(device)
        )
        entries = _profile_entries(self.profile)
        self.models = []
        self.checkpoints = []
        self.config = None
        for entry in entries:
            path, checkpoint_source = _resolve_checkpoint_entry(
                entry, self.profile_path
            )
            use_ema = str(entry.get("weights", "ema")).lower() != "raw"
            model, config, checkpoint = load_planner_checkpoint(
                path, self.device, use_ema=use_ema
            )
            if self.config is not None and (
                config.vehicle != self.config.vehicle
                or config.data != self.config.data
                or config.model != self.config.model
            ):
                raise RuntimeError("ensemble checkpoints have incompatible model/data geometry")
            self.config = config
            self.models.append(model)
            self.checkpoints.append(
                {
                    "path": str(path),
                    "epoch": checkpoint.get("epoch"),
                    "weights": "ema" if use_ema else "raw",
                    "sha256": _sha256(path),
                    "configured_checkpoint": str(entry.get("checkpoint")),
                    "checkpoint_source": checkpoint_source,
                    "fallback_used": checkpoint_source != "checkpoint",
                }
            )
        if not self.models or self.config is None:
            raise RuntimeError("supervised path profile contains no checkpoints")
        runtime_terminal_anchor = dict(
            self.profile.get("runtime_terminal_anchor") or {}
        )
        self.runtime_terminal_anchor_enabled = bool(
            runtime_terminal_anchor.get("enabled", False)
        )
        self.runtime_terminal_anchor_coarse_steps = int(
            runtime_terminal_anchor.get(
                "coarse_steps",
                self.config.model.terminal_anchor_steps,
            )
        )
        if self.runtime_terminal_anchor_enabled:
            if self.runtime_terminal_anchor_coarse_steps < 2:
                raise ValueError(
                    "runtime terminal anchor coarse_steps must be at least two"
                )
            # The checkpoint architecture is unchanged. Only geometry
            # post-processing and its validation contract are enabled.
            self.config = replace(
                self.config,
                model=replace(
                    self.config.model,
                    hard_goal_terminal_pose=True,
                    terminal_anchor_steps=(
                        self.runtime_terminal_anchor_coarse_steps
                    ),
                ),
            )
        self.mirror_tta = bool(self.profile.get("mirror_tta", True))
        self.goal_tta_offsets = []
        seen_goal_offsets = set()
        for raw_offset in self.profile.get("goal_tta_offsets", []):
            if not isinstance(raw_offset, (list, tuple)) or len(raw_offset) != 3:
                raise ValueError(
                    "goal_tta_offsets entries must be [longitudinal_m, lateral_m, yaw_deg]"
                )
            offset = tuple(float(value) for value in raw_offset)
            if offset != (0.0, 0.0, 0.0) and offset not in seen_goal_offsets:
                seen_goal_offsets.add(offset)
                self.goal_tta_offsets.append(offset)
        profile_candidates = int(self.profile.get("candidates_per_model", 0))
        augmentation_count = (2 if self.mirror_tta else 1) * (
            1 + len(self.goal_tta_offsets)
        )
        # The profile records candidates after all TTA. HopePathPlanner.infer
        # expects only the native learned-query count.
        if profile_candidates > 0:
            if profile_candidates % augmentation_count:
                raise ValueError(
                    "candidates_per_model is not divisible by the configured TTA count"
                )
            profile_candidates //= augmentation_count
        self.candidate_count = profile_candidates or None
        actual_total = (
            len(self.models)
            * augmentation_count
            * int(self.candidate_count or self.config.model.candidate_count)
        )
        declared_total = int(self.profile.get("total_candidates", actual_total))
        if declared_total != actual_total:
            raise ValueError(
                "profile total_candidates={} but runtime would generate {}".format(
                    declared_total, actual_total
                )
            )
        runtime_selection = dict(self.profile.get("runtime_selection") or {})
        self.minimum_live_clearance_m = max(
            0.0, float(runtime_selection.get("minimum_clearance_m", 0.10))
        )
        self.selection_weights = {
            "goal_error": float(runtime_selection.get("goal_error_weight", 1.0)),
            "steering_variation": float(
                runtime_selection.get("steering_variation_weight", 0.08)
            ),
            "gear_switches": float(
                runtime_selection.get("gear_switch_weight", 0.35)
            ),
            "path_length": float(runtime_selection.get("path_length_weight", 0.02)),
            "clearance": float(runtime_selection.get("clearance_weight", 1.25)),
        }
        self.path_refiner_policy = None
        self.path_refiner_spec = None
        self.path_refiner_checkpoint = None
        self.path_refiner_candidate_count = 0
        refiner_entry = self.profile.get("path_refiner")
        if isinstance(refiner_entry, Mapping) and bool(
            refiner_entry.get("enabled", True)
        ):
            refiner_path, refiner_source = _resolve_checkpoint_entry(
                refiner_entry, self.profile_path
            )
            refiner_checkpoint = torch.load(
                refiner_path, map_location=self.device, weights_only=False
            )
            self.path_refiner_spec = PathRefinerSpec(**refiner_checkpoint["spec"])
            self.path_refiner_spec.validate()
            refiner_hidden_dim = int(
                refiner_checkpoint.get("arguments", {}).get("hidden_dim", 256)
            )
            self.path_refiner_policy = PathRefinerPolicy(
                raster_channels=self.path_refiner_spec.raster_channels,
                vector_dim=self.path_refiner_spec.vector_dim,
                action_knots=self.path_refiner_spec.action_dim,
                hidden_dim=refiner_hidden_dim,
                topology_count=self.path_refiner_spec.topology_count,
            ).to(self.device)
            use_refiner_ema = str(
                refiner_entry.get("weights", "ema")
            ).lower() != "raw"
            if (
                use_refiner_ema
                and "ema" in refiner_checkpoint
                and "shadow" in refiner_checkpoint["ema"]
            ):
                refiner_state = self.path_refiner_policy.state_dict()
                refiner_state.update(refiner_checkpoint["ema"]["shadow"])
                self.path_refiner_policy.load_state_dict(refiner_state)
            else:
                self.path_refiner_policy.load_state_dict(
                    refiner_checkpoint["policy"]
                )
            self.path_refiner_policy.requires_grad_(False)
            self.path_refiner_policy.eval()
            self.path_refiner_candidate_count = max(
                1, int(refiner_entry.get("candidate_count", 1))
            )
            self.path_refiner_checkpoint = {
                "path": str(refiner_path),
                "weights": "ema" if use_refiner_ema else "raw",
                "sha256": _sha256(refiner_path),
                "format": refiner_checkpoint.get("format"),
                "update": refiner_checkpoint.get("update"),
                "episodes": refiner_checkpoint.get("episodes"),
                "checkpoint_source": refiner_source,
                "fallback_used": refiner_source != "checkpoint",
            }
        self.warmup_latency_ms = self._warmup()

    @torch.no_grad()
    def _append_path_refiner_candidate(
        self,
        output: Mapping[str, torch.Tensor],
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], int | None]:
        """Append one deterministic correction while retaining every base candidate."""

        if self.path_refiner_policy is None or self.path_refiner_spec is None:
            return dict(output), None
        base_candidate_count = int(output["poses"].shape[1])
        base, _geometry = select_base_path(output, batch, self.config)
        zero = torch.zeros(
            (len(batch["bev"]), 1, int(self.path_refiner_spec.action_dim)),
            device=self.device,
        )
        anchored = refine_paths(
            base, batch["goal"], zero, self.config, self.path_refiner_spec
        )
        anchored_score = score_paths(
            anchored, batch, self.config, self.path_refiner_spec, zero
        )
        observation = build_policy_observation(
            batch,
            anchored,
            anchored_score,
            self.config,
            self.path_refiner_spec,
        )
        action = self.path_refiner_policy.fixed_candidate_group(
            observation,
            self.path_refiner_candidate_count,
        )
        refined = refine_paths(
            base,
            batch["goal"],
            action,
            self.config,
            self.path_refiner_spec,
        )
        return concatenate_candidate_outputs(output, refined), base_candidate_count

    @torch.no_grad()
    def _warmup(self) -> float:
        """Pay CUDA kernel/allocator setup once during worker startup."""

        if self.device.type != "cuda":
            return 0.0
        started = time.perf_counter()
        data = self.config.data
        half = 0.5 * float(data.extent_m)
        center_x = float(data.raster_center_x_m)
        center_y = float(data.raster_center_y_m)
        batch, _, _, _, _, _ = self._build_batches(
            {
                "safety_occupancy": np.zeros((200, 200), dtype=np.uint8),
                "start_rear_pose": (0.0, 0.0, 0.0),
                "goal_rear_pose": (1.0, 0.0, 0.0),
                "map_polygon": (
                    (center_x - half, center_y - half),
                    (center_x + half, center_y - half),
                    (center_x + half, center_y + half),
                    (center_x - half, center_y + half),
                ),
                "bev_config": {
                    "x_min": -10.0,
                    "x_max": 10.0,
                    "y_min": -10.0,
                    "y_max": 10.0,
                },
            }
        )
        for model in self.models:
            warm_output = generate_candidate_output(
                model,
                batch,
                candidate_count=self.candidate_count,
                mirror_tta=self.mirror_tta,
                goal_tta_offsets=self.goal_tta_offsets,
            )
            self._append_path_refiner_candidate(warm_output, batch)
        torch.cuda.synchronize(self.device)
        return 1000.0 * (time.perf_counter() - started)

    def status(self) -> dict[str, object]:
        base_total_candidates = (
            len(self.models)
            * (2 if self.mirror_tta else 1)
            * (1 + len(self.goal_tta_offsets))
            * int(self.candidate_count or self.config.model.candidate_count)
        )
        appended_refiner_candidates = int(self.path_refiner_candidate_count)
        return {
            "ok": True,
            "available": True,
            "planner_algorithm": "supervised_path",
            "profile": str(self.profile_path),
            "profile_name": self.profile.get("name"),
            "device": str(self.device),
            "model_count": len(self.models),
            "inference_mode": (
                "single_model" if len(self.models) == 1 else "ensemble"
            ),
            "single_model_only": bool(
                self.profile.get("single_model_only", False)
            ),
            "mirror_tta": self.mirror_tta,
            "goal_tta_offsets": [list(offset) for offset in self.goal_tta_offsets],
            "candidates_per_model": (2 if self.mirror_tta else 1)
            * (1 + len(self.goal_tta_offsets))
            * int(self.candidate_count or self.config.model.candidate_count),
            "base_total_candidates": base_total_candidates,
            "total_candidates": (
                base_total_candidates + appended_refiner_candidates
            ),
            "path_refiner": {
                "available": self.path_refiner_policy is not None,
                "appended_candidates": appended_refiner_candidates,
                "checkpoint": self.path_refiner_checkpoint,
            },
            "runtime_selection": {
                "require_exact_goal": True,
                "require_live_collision_free": True,
                "require_final_vehicle_inside_slot": True,
                "minimum_clearance_m": self.minimum_live_clearance_m,
                "weights": dict(self.selection_weights),
            },
            "runtime_terminal_anchor": {
                "enabled": bool(self.runtime_terminal_anchor_enabled),
                "coarse_steps": int(
                    self.runtime_terminal_anchor_coarse_steps
                ),
                "dense_steps": int(
                    self.runtime_terminal_anchor_coarse_steps
                    * self.config.data.dense_substeps
                ),
            },
            "checkpoints": list(self.checkpoints),
            "warmup_latency_ms": float(self.warmup_latency_ms),
        }

    def _build_batches(self, message: Mapping[str, object]):
        config = self.config
        start = tuple(map(float, message["start_rear_pose"][:3]))
        goal_local = _pose_source_to_local(message["goal_rear_pose"], start)
        goal = np.asarray(
            (goal_local[0], goal_local[1], math.sin(goal_local[2]), math.cos(goal_local[2])),
            dtype=np.float32,
        )
        start_direction = 1 if int(message.get("start_direction", 1)) >= 0 else -1
        mask = _mask_array(message["safety_occupancy"])
        model_mask = _mask_array(message.get("model_occupancy", mask))
        bev_config = dict(message.get("bev_config") or {})
        model_polygon_source = message.get("model_obstacle_polygons")
        canonical_input = bool(model_polygon_source)
        if canonical_input:
            polygons = [
                _points_source_to_local(
                    np.asarray(polygon, dtype=np.float64), start
                ).astype(np.float32)
                for polygon in model_polygon_source
                if len(polygon) >= 3
            ]
            polygons = clip_obstacle_polygons_to_raster_extent(
                polygons, config.data
            )
            vector_polygons = select_route_nearest_polygons(
                polygons, goal_local[:2], config.data.max_obstacles
            )
        else:
            polygons = _mask_polygons_local(model_mask, bev_config, start)
            polygons = clip_obstacle_polygons_to_raster_extent(
                polygons, config.data
            )
            vector_polygons = select_route_nearest_polygons(
                polygons, goal_local[:2], config.data.max_obstacles
            )

        boundary_source = message.get("model_map_polygon") or message.get("map_polygon")
        if boundary_source is None or len(boundary_source) < 3:
            boundary_source = (
                (float(bev_config.get("x_min", -10.0)), float(bev_config.get("y_min", -10.0))),
                (float(bev_config.get("x_max", 10.0)), float(bev_config.get("y_min", -10.0))),
                (float(bev_config.get("x_max", 10.0)), float(bev_config.get("y_max", 10.0))),
                (float(bev_config.get("x_min", -10.0)), float(bev_config.get("y_max", 10.0))),
            )
        map_polygon = _points_source_to_local(
            np.asarray(boundary_source, dtype=np.float64), start
        ).astype(np.float32)
        if canonical_input:
            occupancy, sdf = rasterize_scene_base(
                map_polygon, polygons, config.data
            )
        else:
            map_occupancy = _map_occupancy(map_polygon, config.data)
            live_occupancy = _warp_live_mask_to_model(
                model_mask, bev_config, start, config.data
            )
            occupancy = np.maximum(map_occupancy, live_occupancy).astype(np.uint8)
            free = occupancy == 0
            sdf = (
                distance_transform_edt(free)
                - distance_transform_edt(~free)
            ) * float(config.data.resolution_m)
        start_mask = rasterize_pose_mask((0.0, 0.0, 0.0), config.data, config.vehicle)
        goal_mask = rasterize_pose_mask(goal_local, config.data, config.vehicle)
        bev = np.stack(
            (
                occupancy.astype(np.float32),
                1.0 - occupancy.astype(np.float32),
                np.clip(sdf / float(config.data.sdf_clip_m), -1.0, 1.0),
                start_mask.astype(np.float32) / 255.0,
                goal_mask.astype(np.float32) / 255.0,
            ),
            axis=0,
        )
        vector_values, vector_vertices, vector_polygon_mask = _pack_polygons(
            vector_polygons, config.data.max_obstacles, config.data.max_vertices
        )
        def tensor(value, dtype=None):
            return torch.as_tensor(value, dtype=dtype, device=self.device).unsqueeze(0)

        batch = {
            "bev": tensor(bev, torch.float32),
            "goal": tensor(goal, torch.float32),
            "start_direction": tensor(start_direction, torch.int64),
            "obstacle_vertices": tensor(vector_values, torch.float32),
            "obstacle_vertex_mask": tensor(vector_vertices, torch.bool),
            "obstacle_mask": tensor(vector_polygon_mask, torch.bool),
            "map_polygon": tensor(map_polygon, torch.float32),
        }
        return (
            batch,
            start,
            goal_local,
            len(polygons),
            mask,
            np.asarray(boundary_source, dtype=np.float64),
        )

    @torch.no_grad()
    def plan(self, message: Mapping[str, object]) -> dict[str, object]:
        started = time.perf_counter()
        (
            batch,
            start,
            goal_local,
            polygon_count,
            safety_occupancy,
            boundary_source,
        ) = self._build_batches(message)
        half = 0.5 * float(self.config.data.extent_m)
        center_x = float(self.config.data.raster_center_x_m)
        center_y = float(self.config.data.raster_center_y_m)
        if not (
            center_x - half <= goal_local[0] <= center_x + half
            and center_y - half <= goal_local[1] <= center_y + half
        ):
            return {
                "ok": True,
                "success": False,
                "message": "selected goal is outside the learned 20m x 20m BEV",
                "states": [],
                "worker_latency_ms": 1000.0 * (time.perf_counter() - started),
            }
        outputs = []
        for model in self.models:
            outputs.append(
                generate_candidate_output(
                    model,
                    batch,
                    candidate_count=self.candidate_count,
                    mirror_tta=self.mirror_tta,
                    goal_tta_offsets=self.goal_tta_offsets,
                )
            )
        output = outputs[0]
        for additional in outputs[1:]:
            output = concatenate_candidate_outputs(output, additional)
        # Geometry selection uses large negative sentinels (for infeasible
        # candidates) that are intentionally outside float16's range.
        output = {
            key: value.float()
            if torch.is_tensor(value) and value.is_floating_point()
            else value
            for key, value in output.items()
        }
        output, refiner_candidate_index = self._append_path_refiner_candidate(
            output, batch
        )
        if self.runtime_terminal_anchor_enabled:
            anchor_position_error, anchor_yaw_error = terminal_pose_error(
                output["poses"], batch["goal"], output["lengths"]
            )
            anchored_poses = anchor_terminal_pose_trace_torch(
                output["poses"],
                batch["goal"],
                output["lengths"],
                self.runtime_terminal_anchor_coarse_steps
                * self.config.data.dense_substeps,
            )
            direction = output["gear"].to(anchored_poses.dtype).mul(2.0).sub(1.0)
            anchored_controls = controls_from_pose_trace(
                anchored_poses,
                direction,
                output["lengths"],
                self.config.vehicle.wheelbase_m,
            )
            output["unanchored_runtime_poses"] = output["poses"]
            output["poses"] = anchored_poses
            output.update(anchored_controls)
            output["terminal_anchor_position_correction_m"] = (
                anchor_position_error
            )
            output["terminal_anchor_yaw_correction_rad"] = anchor_yaw_error
        live_vehicle_values = dict(message.get("vehicle_config") or {})
        live_vehicle_config = _live_vehicle_config(live_vehicle_values)
        execution_occupancy = _mask_array(
            message.get("execution_occupancy", safety_occupancy)
        )
        collision_free, live_clearance, final_inside = _live_candidate_geometry(
            output,
            execution_occupancy,
            start,
            boundary_source,
            live_vehicle_config,
            message.get("target_polygon"),
        )
        geometry = evaluate_inference_batch(
            output, batch, self.config, collision_free_override=collision_free
        )
        wheelbase = float(
            live_vehicle_values.get(
                "wheel_base", self.config.vehicle.wheelbase_m
            )
        )
        path_length, gear_switches, steering_variation = (
            _candidate_control_metrics(output, wheelbase)
        )
        selected_tensor, robust_success, runtime_cost = (
            _select_runtime_candidate(
                fallback_selected=geometry["selected"],
                exact_success=geometry["candidate_success"],
                collision_free=collision_free,
                final_inside=final_inside,
                live_clearance_m=live_clearance,
                goal_position_error_m=geometry[
                    "candidate_goal_position_error_m"
                ],
                goal_yaw_error_deg=geometry["candidate_goal_yaw_error_deg"],
                path_length_m=path_length,
                gear_switches=gear_switches,
                steering_variation_rad=steering_variation,
                minimum_clearance_m=self.minimum_live_clearance_m,
                goal_position_tolerance_m=(
                    self.config.data.goal_position_tolerance_m
                ),
                goal_yaw_tolerance_deg=self.config.data.goal_yaw_tolerance_deg,
                weights=self.selection_weights,
            )
        )
        selected = int(selected_tensor[0].item())
        length = int(output["lengths"][0, selected].item())
        poses_local = output["poses"][0, selected, :length].float().cpu().numpy()
        poses_source = _poses_local_to_source(poses_local, start)
        curvature = output["curvature"][0, selected, :length].float().cpu().numpy()
        gear = output["gear"][0, selected, :length].cpu().numpy()
        steer = np.arctan(curvature * wheelbase)
        states = np.column_stack((poses_source, np.where(gear > 0, 1.0, -1.0), steer))
        success = bool(robust_success[0, selected].item())
        selected_position_error = float(
            geometry["candidate_goal_position_error_m"][0, selected].item()
        )
        selected_yaw_error = float(
            geometry["candidate_goal_yaw_error_deg"][0, selected].item()
        )
        selected_clearance = float(live_clearance[0, selected].item())
        return {
            "ok": True,
            "success": success,
            "message": (
                "supervised path candidate satisfies robust live execution constraints"
                if success
                else "no supervised candidate satisfies robust live execution constraints"
            ),
            "states": states.tolist(),
            "metadata": {
                "selected_candidate": selected,
                "candidate_count": int(output["poses"].shape[1]),
                "base_candidate_count": int(
                    output["poses"].shape[1]
                    - (
                        self.path_refiner_candidate_count
                        if refiner_candidate_index is not None
                        else 0
                    )
                ),
                "path_refiner_candidate_index": refiner_candidate_index,
                "path_refiner_candidate_count": int(
                    self.path_refiner_candidate_count
                ),
                "selected_from_path_refiner": bool(
                    refiner_candidate_index is not None
                    and selected >= int(refiner_candidate_index)
                ),
                "path_refiner_checkpoint": self.path_refiner_checkpoint,
                "candidate_success_count": int(robust_success[0].sum().item()),
                "exact_goal_candidate_count": int(
                    geometry["candidate_geometry_success"][0].sum().item()
                ),
                "clearance_qualified_candidate_count": int(
                    (
                        collision_free[0]
                        & (live_clearance[0] >= self.minimum_live_clearance_m)
                    ).sum().item()
                ),
                "final_inside_candidate_count": int(final_inside[0].sum().item()),
                "collision_free_candidate_count": int(collision_free[0].sum().item()),
                "no_collision_free_candidate": bool(
                    geometry["no_collision_free_candidate"][0].item()
                ),
                "goal_position_error_m": selected_position_error,
                "goal_yaw_error_deg": selected_yaw_error,
                "terminal_anchor_position_correction_m": float(
                    geometry["candidate_anchor_position_correction_m"][
                        0, selected
                    ].item()
                ),
                "terminal_anchor_yaw_correction_deg": float(
                    geometry["candidate_anchor_yaw_correction_deg"][
                        0, selected
                    ].item()
                ),
                "terminal_anchor_correction_valid": bool(
                    geometry["candidate_anchor_correction_valid"][
                        0, selected
                    ].item()
                ),
                "minimum_required_clearance_m": self.minimum_live_clearance_m,
                "selected_live_clearance_m": selected_clearance,
                "execution_collision_mask": (
                    "physical"
                    if "execution_occupancy" in message
                    else "safety_compatibility_fallback"
                ),
                "selected_exact_goal_success": bool(
                    geometry["candidate_geometry_success"][0, selected].item()
                ),
                "selected_final_vehicle_inside_slot": bool(
                    final_inside[0, selected].item()
                ),
                "selected_path_length_m": float(
                    path_length[0, selected].item()
                ),
                "selected_gear_switches": int(
                    gear_switches[0, selected].item()
                ),
                "selected_steering_total_variation_deg": math.degrees(
                    float(steering_variation[0, selected].item())
                ),
                "selected_runtime_cost": float(
                    runtime_cost[0, selected].item()
                ),
                "selection_policy": (
                    "robust_exact_goal_clearance_controller_friendly_v1"
                ),
                "occupancy_polygon_count": int(polygon_count),
                "vector_polygon_count": int(min(polygon_count, self.config.data.max_obstacles)),
                "primitive_count": int(length),
                "profile_name": self.profile.get("name"),
                "model_count": len(self.models),
                "inference_mode": (
                    "single_model" if len(self.models) == 1 else "ensemble"
                ),
                "mirror_tta": self.mirror_tta,
                "checkpoint": dict(self.checkpoints[0]),
                "model_input_mode": str(
                    message.get("model_input_mode") or "live_mask_warp"
                ),
                "model_source_occupancy_cells": int(
                    np.count_nonzero(
                        _mask_array(
                            message.get(
                                "model_occupancy",
                                message["safety_occupancy"],
                            )
                        )
                    )
                ),
                "model_lidar_included": bool(
                    message.get("model_lidar_included", False)
                ),
                "model_obstacle_source": str(
                    message.get("model_obstacle_source") or "occupancy_mask"
                ),
                "model_target_source": str(
                    message.get("model_target_source") or "goal_rear_pose"
                ),
                "model_boundary_source": str(
                    message.get("model_boundary_source") or "provided_map"
                ),
            },
            "worker_latency_ms": 1000.0 * (time.perf_counter() - started),
        }
