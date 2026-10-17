"""Metric geometry, rasterization, and primitive conversion utilities."""

from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
try:
    from scipy.ndimage import distance_transform_edt
except ImportError:  # Live CVT worker keeps a deliberately small environment.
    from hybrid_astar_8ch_bev import euclidean_distance_transform

    def distance_transform_edt(value):
        mask = np.asarray(value, dtype=bool)
        # scipy.ndimage returns distance to the nearest zero for non-zero
        # pixels. The project's native EDT takes the complementary convention:
        # distance to True obstacle cells.
        return euclidean_distance_transform(~mask, 1.0)

from .config import DataConfig, VehicleConfig


# Chord-length midpoint primitives need a tiny geometric-curvature allowance:
# theta / chord is up to ~0.09% above theta / arc for the 0.2 m collector step.
CURVATURE_DISCRETIZATION_MARGIN = 1.002


def wrap_angle(value: np.ndarray | float) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    return (array + math.pi) % (2.0 * math.pi) - math.pi


def anchor_terminal_pose_trace(
    poses: np.ndarray,
    goal_pose: Sequence[float],
    bridge_steps: int,
) -> np.ndarray:
    """Apply a smooth hard boundary condition to one finite pose trace.

    The prefix is unchanged.  A quintic smoothstep distributes the terminal
    position/yaw residual over at most ``bridge_steps`` final intervals, has
    zero correction slope at both splice and goal, and writes the last rear-
    axle pose exactly.  This avoids the untrackable jump caused by replacing
    only the final array row.
    """

    source = np.asarray(poses, dtype=np.float64)
    if source.ndim != 2 or source.shape[1] < 3 or len(source) < 2:
        raise ValueError("poses must have shape [N>=2, 3+]")
    if int(bridge_steps) < 2:
        raise ValueError("bridge_steps must be at least two")
    goal = np.asarray(goal_pose, dtype=np.float64).reshape(-1)
    if len(goal) < 3 or not np.isfinite(goal[:3]).all():
        raise ValueError("goal_pose must contain finite x, y and yaw")

    result = source.copy()
    endpoint = len(result) - 1
    bridge_begin = max(1, endpoint - int(bridge_steps))
    denominator = max(1, endpoint - bridge_begin)
    progress = np.clip(
        (np.arange(len(result), dtype=np.float64) - bridge_begin)
        / float(denominator),
        0.0,
        1.0,
    )
    weight = progress**3 * (10.0 - 15.0 * progress + 6.0 * progress**2)
    xy_error = goal[:2] - result[endpoint, :2]
    yaw_error = float(wrap_angle(goal[2] - result[endpoint, 2]))
    result[:, :2] += weight[:, None] * xy_error[None]
    result[:, 2] = wrap_angle(result[:, 2] + weight * yaw_error)
    result[endpoint, :3] = goal[:3]
    return result.astype(np.float32)


def pose_trace_controls(
    poses: np.ndarray,
    gear: np.ndarray,
    wheelbase_m: float,
) -> dict[str, np.ndarray]:
    """Recompute executable control labels after terminal anchoring.

    Curvature is derived from the anchored yaw change and signed travelled
    distance.  The terminal motion and steering are explicitly zero because
    the final pose is a stopped boundary condition, not another primitive.
    """

    rows = np.asarray(poses, dtype=np.float64)
    direction_class = np.asarray(gear).reshape(-1)
    if rows.ndim != 2 or rows.shape[1] < 3 or len(rows) < 2:
        raise ValueError("poses must have shape [N>=2, 3+]")
    if len(direction_class) != len(rows):
        raise ValueError("gear and pose lengths must match")
    if float(wheelbase_m) <= 0.0:
        raise ValueError("wheelbase_m must be positive")

    delta_xy = np.diff(rows[:, :2], axis=0)
    step_edges = np.linalg.norm(delta_xy, axis=1)
    yaw_delta = wrap_angle(np.diff(rows[:, 2]))
    direction = np.where(direction_class[:-1] > 0, 1.0, -1.0)
    signed_step = direction * step_edges
    curvature_edges = np.divide(
        yaw_delta,
        signed_step,
        out=np.zeros_like(yaw_delta),
        where=step_edges > 1.0e-8,
    )
    curvature = np.concatenate((curvature_edges, np.zeros(1, dtype=np.float64)))
    step = np.concatenate((step_edges, np.zeros(1, dtype=np.float64)))
    steering = np.arctan(float(wheelbase_m) * curvature)
    steering[-1] = 0.0
    return {
        "poses": rows[:, :3].astype(np.float32),
        "curvature": curvature.astype(np.float32),
        "steering": steering.astype(np.float32),
        "step": step.astype(np.float32),
        "gear": direction_class.astype(np.int64),
    }


def points_world_to_local(points: np.ndarray, start_pose: Sequence[float]) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    x, y, yaw = map(float, start_pose[:3])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    delta = points - np.array((x, y), dtype=np.float64)
    return np.stack(
        (cosine * delta[:, 0] + sine * delta[:, 1],
         -sine * delta[:, 0] + cosine * delta[:, 1]),
        axis=1,
    ).astype(np.float32)


def states_world_to_local(states: np.ndarray, start_pose: Sequence[float]) -> np.ndarray:
    result = np.asarray(states, dtype=np.float32).copy()
    result[:, :2] = points_world_to_local(result[:, :2], start_pose)
    result[:, 2] = wrap_angle(result[:, 2].astype(np.float64) - float(start_pose[2]))
    return result


def pose_world_to_local(pose: Sequence[float], start_pose: Sequence[float]) -> np.ndarray:
    xy = points_world_to_local(np.asarray(pose[:2])[None, :], start_pose)[0]
    return np.asarray(
        (xy[0], xy[1], float(wrap_angle(float(pose[2]) - float(start_pose[2])))),
        dtype=np.float32,
    )


def _filled_directions(speed: np.ndarray) -> np.ndarray:
    direction = np.sign(np.asarray(speed, dtype=np.float64)).astype(np.int8)
    if not np.any(direction):
        return np.ones_like(direction)
    last = np.int8(0)
    for index in range(len(direction)):
        if direction[index]:
            last = direction[index]
        elif last:
            direction[index] = last
    last = np.int8(0)
    for index in range(len(direction) - 1, -1, -1):
        if direction[index]:
            last = direction[index]
        elif last:
            direction[index] = last
    return direction


def count_gear_switches(speed: np.ndarray) -> int:
    direction = _filled_directions(speed)
    return int(np.count_nonzero(direction[1:] != direction[:-1]))


def resample_motion_primitives(
    local_states: np.ndarray,
    step_m: float,
    wheelbase_m: float,
) -> dict[str, np.ndarray]:
    """Resample each constant-gear segment without smoothing across switches."""

    states = np.asarray(local_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 5:
        raise ValueError("path states must have shape [N>=2, 5]")
    direction = _filled_directions(states[:, 3])
    cuts = np.flatnonzero(direction[1:] != direction[:-1]) + 1
    bounds = np.concatenate(([0], cuts, [len(states)]))
    segments: list[np.ndarray] = []
    gears: list[np.ndarray] = []

    for segment_index, (begin, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        segment = states[begin:end]
        if segment_index and len(segment):
            # The previous pose is the zero-speed gear-change anchor.
            segment = np.concatenate((states[begin - 1:begin], segment), axis=0)
        if len(segment) < 2:
            continue
        delta = np.hypot(np.diff(segment[:, 0]), np.diff(segment[:, 1]))
        keep = np.concatenate(([True], delta > 1.0e-6))
        segment = segment[keep]
        if len(segment) < 2:
            continue
        distance = np.concatenate(
            ([0.0], np.cumsum(np.hypot(np.diff(segment[:, 0]), np.diff(segment[:, 1]))))
        )
        total = float(distance[-1])
        if total <= 1.0e-6:
            continue
        samples = np.arange(0.0, total, float(step_m), dtype=np.float64)
        if not len(samples) or total - samples[-1] > 1.0e-5:
            samples = np.append(samples, total)
        yaw = np.unwrap(segment[:, 2])
        values = np.stack(
            [np.interp(samples, distance, segment[:, column]) for column in (0, 1)],
            axis=1,
        )
        sampled_yaw = np.interp(samples, distance, yaw)
        sampled_steer = np.interp(samples, distance, segment[:, 4])
        sampled = np.column_stack((values, wrap_angle(sampled_yaw), sampled_steer))
        if segments and np.linalg.norm(sampled[0, :2] - segments[-1][-1, :2]) < 1.0e-5:
            sampled = sampled[1:]
            samples = samples[1:]
        if not len(sampled):
            continue
        gear = int(direction[min(begin, len(direction) - 1)])
        segments.append(sampled)
        gears.append(np.full(len(sampled), gear, dtype=np.int8))

    if not segments:
        raise ValueError("path contains no segment with positive travel distance")
    merged = np.concatenate(segments, axis=0)
    merged_gear = np.concatenate(gears, axis=0)
    # The exact collector start is the rear-axle origin.  Preserve it as token 0.
    if np.linalg.norm(merged[0, :2]) > 1.0e-5 or abs(float(merged[0, 2])) > 1.0e-5:
        merged = np.concatenate((np.zeros((1, 4), dtype=np.float64), merged), axis=0)
        merged_gear = np.concatenate((merged_gear[:1], merged_gear), axis=0)
    poses = merged[:, :3]
    delta_xy = np.diff(poses[:, :2], axis=0)
    yaw_delta = wrap_angle(np.diff(poses[:, 2]))
    midpoint_yaw = poses[:-1, 2] + 0.5 * yaw_delta
    forward_projection = (
        delta_xy[:, 0] * np.cos(midpoint_yaw)
        + delta_xy[:, 1] * np.sin(midpoint_yaw)
    )
    step = np.linalg.norm(delta_xy, axis=1)
    fallback_direction = merged_gear[1:].astype(np.float64)
    motion_direction = np.where(
        np.abs(forward_projection) > 1.0e-6,
        np.where(forward_projection >= 0.0, 1.0, -1.0),
        fallback_direction,
    )
    curvature_motion = np.divide(
        yaw_delta,
        motion_direction * step,
        out=np.zeros_like(yaw_delta),
        where=step > 1.0e-8,
    )
    curvature = np.zeros(len(poses), dtype=np.float64)
    step_magnitude = np.zeros(len(poses), dtype=np.float64)
    gear_class = np.zeros(len(poses), dtype=np.int64)
    curvature[:-1] = curvature_motion
    step_magnitude[:-1] = step
    gear_class[:-1] = motion_direction > 0.0
    if len(poses) > 1:
        curvature[-1] = curvature[-2]
        gear_class[-1] = gear_class[-2]
    kinematic_steering = np.arctan(float(wheelbase_m) * curvature)
    return {
        "poses": poses.astype(np.float32),
        "curvature": curvature.astype(np.float32),
        # The optimizer's saved steering column is retained for audits, but
        # it is not kinematically consistent with the saved pose trace.  The
        # supervised control target must agree with pose-derived curvature.
        "steering": kinematic_steering.astype(np.float32),
        "source_steering": merged[:, 3].astype(np.float32),
        "gear": gear_class,
        "step": step_magnitude.astype(np.float32),
    }


def resample_aligned_dense_primitives(
    local_states: np.ndarray,
    coarse_step_m: float,
    wheelbase_m: float,
    substeps: int,
) -> dict[str, np.ndarray]:
    """Sample original GT at subdivisions aligned to every coarse interval.

    Unlike a separate fixed-grid resampling, this guarantees
    ``dense_length == (coarse_length - 1) * substeps + 1`` even at gear cusps
    and short residual segments.  Each dense value is interpolated directly
    from the original ~0.045 m planner-3 trace, not from the 0.2 m poses.
    """

    if int(substeps) < 1:
        raise ValueError("substeps must be positive")
    if int(substeps) == 1:
        return resample_motion_primitives(local_states, coarse_step_m, wheelbase_m)
    states = np.asarray(local_states, dtype=np.float64)
    if states.ndim != 2 or states.shape[0] < 2 or states.shape[1] < 5:
        raise ValueError("path states must have shape [N>=2, 5]")
    direction = _filled_directions(states[:, 3])
    cuts = np.flatnonzero(direction[1:] != direction[:-1]) + 1
    bounds = np.concatenate(([0], cuts, [len(states)]))
    segments: list[np.ndarray] = []
    gears: list[np.ndarray] = []

    for segment_index, (begin, end) in enumerate(zip(bounds[:-1], bounds[1:])):
        segment = states[begin:end]
        if segment_index and len(segment):
            segment = np.concatenate((states[begin - 1:begin], segment), axis=0)
        if len(segment) < 2:
            continue
        delta = np.hypot(np.diff(segment[:, 0]), np.diff(segment[:, 1]))
        segment = segment[np.concatenate(([True], delta > 1.0e-6))]
        if len(segment) < 2:
            continue
        distance = np.concatenate(
            ([0.0], np.cumsum(np.hypot(np.diff(segment[:, 0]), np.diff(segment[:, 1]))))
        )
        total = float(distance[-1])
        if total <= 1.0e-6:
            continue
        coarse = np.arange(0.0, total, float(coarse_step_m), dtype=np.float64)
        if not len(coarse) or total - coarse[-1] > 1.0e-5:
            coarse = np.append(coarse, total)
        fractions = np.arange(int(substeps), dtype=np.float64) / float(substeps)
        dense = (
            coarse[:-1, None]
            + (coarse[1:] - coarse[:-1])[:, None] * fractions[None]
        ).reshape(-1)
        dense = np.append(dense, coarse[-1])
        yaw = np.unwrap(segment[:, 2])
        values = np.stack(
            [np.interp(dense, distance, segment[:, column]) for column in (0, 1)],
            axis=1,
        )
        sampled = np.column_stack(
            (
                values,
                wrap_angle(np.interp(dense, distance, yaw)),
                np.interp(dense, distance, segment[:, 4]),
            )
        )
        if segments and np.linalg.norm(sampled[0, :2] - segments[-1][-1, :2]) < 1.0e-5:
            sampled = sampled[1:]
        if not len(sampled):
            continue
        gear = int(direction[min(begin, len(direction) - 1)])
        segments.append(sampled)
        gears.append(np.full(len(sampled), gear, dtype=np.int8))

    if not segments:
        raise ValueError("path contains no segment with positive travel distance")
    merged = np.concatenate(segments, axis=0)
    merged_gear = np.concatenate(gears, axis=0)
    if np.linalg.norm(merged[0, :2]) > 1.0e-5 or abs(float(merged[0, 2])) > 1.0e-5:
        merged = np.concatenate((np.zeros((1, 4), dtype=np.float64), merged), axis=0)
        merged_gear = np.concatenate((merged_gear[:1], merged_gear), axis=0)
    poses = merged[:, :3]
    delta_xy = np.diff(poses[:, :2], axis=0)
    yaw_delta = wrap_angle(np.diff(poses[:, 2]))
    midpoint_yaw = poses[:-1, 2] + 0.5 * yaw_delta
    forward_projection = (
        delta_xy[:, 0] * np.cos(midpoint_yaw)
        + delta_xy[:, 1] * np.sin(midpoint_yaw)
    )
    step = np.linalg.norm(delta_xy, axis=1)
    fallback_direction = merged_gear[1:].astype(np.float64)
    motion_direction = np.where(
        np.abs(forward_projection) > 1.0e-6,
        np.where(forward_projection >= 0.0, 1.0, -1.0),
        fallback_direction,
    )
    curvature_motion = np.divide(
        yaw_delta,
        motion_direction * step,
        out=np.zeros_like(yaw_delta),
        where=step > 1.0e-8,
    )
    curvature = np.zeros(len(poses), dtype=np.float64)
    step_magnitude = np.zeros(len(poses), dtype=np.float64)
    gear_class = np.zeros(len(poses), dtype=np.int64)
    curvature[:-1] = curvature_motion
    step_magnitude[:-1] = step
    gear_class[:-1] = motion_direction > 0.0
    if len(poses) > 1:
        curvature[-1] = curvature[-2]
        gear_class[-1] = gear_class[-2]
    kinematic_steering = np.arctan(float(wheelbase_m) * curvature)
    return {
        "poses": poses.astype(np.float32),
        "curvature": curvature.astype(np.float32),
        "steering": kinematic_steering.astype(np.float32),
        "source_steering": merged[:, 3].astype(np.float32),
        "gear": gear_class,
        "step": step_magnitude.astype(np.float32),
    }


def densify_executable_primitives(
    primitives: Mapping[str, np.ndarray],
    wheelbase_m: float,
    substeps: int,
) -> dict[str, np.ndarray]:
    """Densify an already executable coarse rollout like the Torch model."""

    substeps = int(substeps)
    if substeps < 1:
        raise ValueError("substeps must be positive")
    curvature = np.asarray(primitives["curvature"], dtype=np.float64)
    step = np.asarray(primitives["step"], dtype=np.float64)
    gear = np.asarray(primitives["gear"], dtype=np.int64)
    if curvature.ndim != 1 or step.shape != curvature.shape or gear.shape != curvature.shape:
        raise ValueError("primitive controls must be aligned one-dimensional arrays")
    if len(curvature) < 2:
        raise ValueError("at least two primitive nodes are required")
    steering = np.arctan(float(wheelbase_m) * curvature)
    fraction = np.arange(substeps, dtype=np.float64) / float(substeps)
    dense_steering = (
        steering[:-1, None]
        + (steering[1:] - steering[:-1])[:, None] * fraction[None]
    ).reshape(-1)
    dense_steering = np.append(dense_steering, steering[-1])
    dense_curvature = np.tan(dense_steering) / float(wheelbase_m)
    dense_gear = np.append(
        np.repeat(gear[:-1], substeps), gear[-1]
    ).astype(np.int64)
    dense_step = np.append(
        np.repeat(step[:-1] / float(substeps), substeps), 0.0
    )
    direction = np.where(dense_gear[:-1] > 0, 1.0, -1.0)
    signed_step = dense_step[:-1] * direction
    yaw_delta = signed_step * dense_curvature[:-1]
    yaw_after = np.cumsum(yaw_delta)
    yaw_before = np.concatenate(([0.0], yaw_after[:-1]))
    midpoint = yaw_before + 0.5 * yaw_delta
    poses = np.column_stack((
        np.concatenate(([0.0], np.cumsum(signed_step * np.cos(midpoint)))),
        np.concatenate(([0.0], np.cumsum(signed_step * np.sin(midpoint)))),
        np.concatenate(([0.0], wrap_angle(yaw_after))),
    ))
    return {
        "poses": poses.astype(np.float32),
        "curvature": dense_curvature.astype(np.float32),
        "steering": dense_steering.astype(np.float32),
        "gear": dense_gear,
        "step": dense_step.astype(np.float32),
    }


def local_to_pixel(points: np.ndarray, data: DataConfig) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape((-1, 2))
    half = 0.5 * float(data.extent_m)
    scale = (float(data.raster_size) - 1.0) / float(data.extent_m)
    return np.stack(
        (
            (points[:, 0] - float(data.raster_center_x_m) + half) * scale,
            (half - (points[:, 1] - float(data.raster_center_y_m))) * scale,
        ),
        axis=1,
    )


def clip_polygon_to_raster_extent(
    polygon: np.ndarray,
    data: DataConfig,
    *,
    minimum_area_m2: float = 1.0e-6,
) -> np.ndarray:
    """Clip one metric polygon to the exact BEV window.

    The rasterizer naturally clips drawing at the image boundary.  Vector
    inputs must use the same visible geometry; otherwise an obstacle can be
    absent from the raster branch while still appearing in the vector branch.
    Sutherland-Hodgman clipping keeps the implementation dependency-free and
    handles the convex parked-vehicle polygons used by both collectors.
    """

    points = np.asarray(polygon, dtype=np.float64).reshape((-1, 2))
    if len(points) < 3 or not np.isfinite(points).all():
        return np.empty((0, 2), dtype=np.float32)
    half = 0.5 * float(data.extent_m)
    bounds = (
        (0, float(data.raster_center_x_m) - half, True),
        (0, float(data.raster_center_x_m) + half, False),
        (1, float(data.raster_center_y_m) - half, True),
        (1, float(data.raster_center_y_m) + half, False),
    )

    def inside(point, axis, boundary, keep_greater):
        return bool(
            point[axis] >= boundary - 1.0e-9
            if keep_greater
            else point[axis] <= boundary + 1.0e-9
        )

    output = points
    for axis, boundary, keep_greater in bounds:
        if len(output) == 0:
            break
        source = output
        clipped = []
        previous = source[-1]
        previous_inside = inside(previous, axis, boundary, keep_greater)
        for current in source:
            current_inside = inside(current, axis, boundary, keep_greater)
            if current_inside != previous_inside:
                denominator = float(current[axis] - previous[axis])
                if abs(denominator) > 1.0e-12:
                    ratio = (boundary - float(previous[axis])) / denominator
                    intersection = previous + ratio * (current - previous)
                    intersection[axis] = boundary
                    clipped.append(intersection)
            if current_inside:
                clipped.append(current)
            previous = current
            previous_inside = current_inside
        output = np.asarray(clipped, dtype=np.float64).reshape((-1, 2))
    if len(output) < 3:
        return np.empty((0, 2), dtype=np.float32)
    area = 0.5 * abs(
        float(
            np.dot(output[:, 0], np.roll(output[:, 1], -1))
            - np.dot(output[:, 1], np.roll(output[:, 0], -1))
        )
    )
    if area < float(minimum_area_m2):
        return np.empty((0, 2), dtype=np.float32)
    return output.astype(np.float32)


def clip_obstacle_polygons_to_raster_extent(
    polygons: Iterable[np.ndarray], data: DataConfig
) -> list[np.ndarray]:
    """Return only obstacle geometry that is visible in the BEV window."""

    visible = []
    for polygon in polygons:
        clipped = clip_polygon_to_raster_extent(polygon, data)
        if len(clipped) >= 3:
            visible.append(clipped)
    return visible


def polygon_route_distance(
    polygon: np.ndarray, goal_xy: Sequence[float]
) -> float:
    """Distance from a polygon centre to the causal start-to-goal segment."""

    point = np.asarray(polygon, dtype=np.float64).reshape((-1, 2)).mean(axis=0)
    goal = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    denominator = float(np.dot(goal, goal))
    ratio = (
        0.0
        if denominator <= 1.0e-9
        else float(np.clip(np.dot(point, goal) / denominator, 0.0, 1.0))
    )
    return float(np.linalg.norm(point - ratio * goal))


def select_route_nearest_polygons(
    polygons: Sequence[np.ndarray],
    goal_xy: Sequence[float],
    maximum: int,
) -> list[np.ndarray]:
    """Select polygons without consulting the future/GT path."""

    scored = [
        (polygon_route_distance(polygon, goal_xy), index, polygon)
        for index, polygon in enumerate(polygons)
    ]
    return [
        item[2]
        for item in sorted(scored, key=lambda item: (item[0], item[1]))[
            : max(0, int(maximum))
        ]
    ]


def rasterize_scene_base(
    map_polygon_local: np.ndarray,
    obstacle_polygons_local: Iterable[np.ndarray],
    data: DataConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return binary occupancy and signed distance in metres."""

    size = int(data.raster_size)
    image = Image.new("L", (size, size), color=1)
    draw = ImageDraw.Draw(image)
    map_pixels = [tuple(value) for value in local_to_pixel(map_polygon_local, data)]
    draw.polygon(map_pixels, fill=0)
    for polygon in obstacle_polygons_local:
        pixels = [tuple(value) for value in local_to_pixel(polygon, data)]
        if len(pixels) >= 3:
            draw.polygon(pixels, fill=1)
    occupancy = np.asarray(image, dtype=np.uint8)
    free = occupancy == 0
    sdf = (
        distance_transform_edt(free) - distance_transform_edt(~free)
    ) * float(data.resolution_m)
    return occupancy, sdf.astype(np.float32)


def quantize_sdf(sdf_m: np.ndarray, clip_m: float) -> np.ndarray:
    normalized = np.clip(np.asarray(sdf_m) / float(clip_m), -1.0, 1.0)
    return np.rint((normalized + 1.0) * 127.5).astype(np.uint8)


def decode_sdf(value: np.ndarray) -> np.ndarray:
    return np.asarray(value, dtype=np.float32) / 127.5 - 1.0


def vehicle_polygon(pose: Sequence[float], vehicle: VehicleConfig) -> np.ndarray:
    x, y, yaw = map(float, pose[:3])
    corners = np.asarray(
        [
            (-vehicle.rear_overhang_m, -0.5 * vehicle.width_m),
            (vehicle.wheelbase_m + vehicle.front_overhang_m, -0.5 * vehicle.width_m),
            (vehicle.wheelbase_m + vehicle.front_overhang_m, 0.5 * vehicle.width_m),
            (-vehicle.rear_overhang_m, 0.5 * vehicle.width_m),
        ],
        dtype=np.float64,
    )
    cosine, sine = math.cos(yaw), math.sin(yaw)
    rotation = np.asarray(((cosine, -sine), (sine, cosine)))
    return (corners @ rotation.T + np.asarray((x, y))).astype(np.float32)


def swept_path_poses(path: np.ndarray, *, max_spacing_m: float = 0.1) -> np.ndarray:
    """Interpolate a pose path at no more than the requested center spacing."""

    if max_spacing_m <= 0.0:
        raise ValueError("max_spacing_m must be positive")
    poses = np.asarray(path, dtype=np.float64)
    if len(poses) == 0:
        return poses.reshape((0, 3))
    if len(poses) > 1:
        begin, end = poses[:-1], poses[1:]
        distance = np.linalg.norm(end[:, :2] - begin[:, :2], axis=1)
        # The epsilon prevents a float32 0.2 m step from becoming three pieces.
        subdivisions = np.maximum(
            1, np.ceil(distance / max_spacing_m - 1.0e-6).astype(np.int64)
        )
        yaw_delta = wrap_angle(end[:, 2] - begin[:, 2])
        pieces = [poses[:1]]
        # Ordering is irrelevant for a set-intersection check.  Vectorizing by
        # subdivision index avoids a Python loop over every path edge.
        for step_index in range(1, int(np.max(subdivisions)) + 1):
            valid = subdivisions >= step_index
            fraction = step_index / subdivisions[valid].astype(np.float64)
            interpolated = np.empty((int(np.count_nonzero(valid)), 3), dtype=np.float64)
            interpolated[:, :2] = (
                begin[valid, :2]
                + fraction[:, None] * (end[valid, :2] - begin[valid, :2])
            )
            interpolated[:, 2] = begin[valid, 2] + fraction * yaw_delta[valid]
            pieces.append(interpolated)
        swept_array = np.concatenate(pieces, axis=0)
    else:
        swept_array = poses
    return swept_array


def exact_path_collision_reason(
    path: np.ndarray,
    obstacle_polygons: Sequence[np.ndarray],
    map_polygon: np.ndarray,
    vehicle: VehicleConfig,
    *,
    max_spacing_m: float = 0.1,
) -> str | None:
    """Return an exact swept-polygon rejection reason, or ``None``.

    Poses are interpolated at no more than ``max_spacing_m`` center travel so
    the check also catches collisions introduced between resampled GT nodes.
    """

    import shapely
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    swept_array = swept_path_poses(path, max_spacing_m=max_spacing_m)
    if len(swept_array) == 0:
        return "invalid_path_array"
    footprint_local = np.asarray((
        (-vehicle.rear_overhang_m, -0.5 * vehicle.width_m),
        (vehicle.wheelbase_m + vehicle.front_overhang_m, -0.5 * vehicle.width_m),
        (vehicle.wheelbase_m + vehicle.front_overhang_m, 0.5 * vehicle.width_m),
        (-vehicle.rear_overhang_m, 0.5 * vehicle.width_m),
    ), dtype=np.float64)
    cosine, sine = np.cos(swept_array[:, 2]), np.sin(swept_array[:, 2])
    corners = np.stack((
        swept_array[:, None, 0] + cosine[:, None] * footprint_local[:, 0]
        - sine[:, None] * footprint_local[:, 1],
        swept_array[:, None, 1] + sine[:, None] * footprint_local[:, 0]
        + cosine[:, None] * footprint_local[:, 1],
    ), axis=-1)
    footprints = shapely.polygons(corners)
    boundary = Polygon(map_polygon)
    if not bool(np.all(shapely.covers(boundary, footprints))):
        return "resampled_out_of_bounds"
    obstacles = [Polygon(polygon) for polygon in obstacle_polygons if len(polygon) >= 3]
    if obstacles:
        obstacle_union = unary_union(obstacles)
        if bool(np.any(shapely.intersects(footprints, obstacle_union))):
            return "resampled_collision"
    return None


def rasterize_pose_mask(
    pose: Sequence[float], data: DataConfig, vehicle: VehicleConfig
) -> np.ndarray:
    image = Image.new("L", (data.raster_size, data.raster_size), color=0)
    pixels = [tuple(value) for value in local_to_pixel(vehicle_polygon(pose, vehicle), data)]
    ImageDraw.Draw(image).polygon(pixels, fill=255)
    return np.asarray(image, dtype=np.uint8)


def map_bounds_polygon(bounds: Sequence[float]) -> np.ndarray:
    xmin, xmax, ymin, ymax = map(float, bounds)
    return np.asarray(((xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)))


def raster_extent_polygon(data: DataConfig) -> np.ndarray:
    """Return the local-coordinate polygon represented by the BEV raster."""

    half = 0.5 * float(data.extent_m)
    center_x = float(data.raster_center_x_m)
    center_y = float(data.raster_center_y_m)
    return np.asarray(
        (
            (center_x - half, center_y - half),
            (center_x + half, center_y - half),
            (center_x + half, center_y + half),
            (center_x - half, center_y + half),
        ),
        dtype=np.float64,
    )
