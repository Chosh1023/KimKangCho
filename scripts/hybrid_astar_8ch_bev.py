#!/usr/bin/env python3
"""Plan and visualize collision-free parking paths on the packed 8-channel BEV.

This is an offline, dependency-light Hybrid A* implementation for the cache at
``C:/CARLA_DATA_CVT_RECORDINGS_20260705_124057``.  It includes:

* exact bicycle-model motion primitives with forward and reverse gears;
* a non-holonomic SE(2) search state including direction and steering bin;
* a holonomic obstacle-aware Dijkstra heuristic;
* complete Reeds-Shepp analytic expansion to a terminal alignment pose;
* a constrained zero-steer reverse docking phase into the exact parking pose;
* conservative full-vehicle collision checking on the 0.1 m BEV raster;
* steering/gear/proximity costs, dense final validation and speed profiling;
* PNG, JSON and CSV output directly from the bit-packed 8-channel cache.

Coordinate convention: x+ is ego-forward, y+ is ego-right, yaw=0 points
forward.  The search reference point is the rear axle; rendered paths are the
vehicle bounding-box centre.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import reeds_shepp_path as rs


CHANNELS = (
    "ego_vehicle_mask",
    "ego_vehicle_outline",
    "bev_vehicle_seg",
    "bev_vehicle_poly",
    "parking_seg",
    "free_parking_poly",
    "target_goal_seg",
    "target_goal_poly",
)
DEFAULT_CACHE_DIR = Path(
    r"C:\CARLA_DATA_CVT_RECORDINGS_20260705_124057\policy_cache"
    r"\targetbev8_image20_widebev_act_deterministic_v3"
)
DEFAULT_OUTPUT_DIR = Path("reports/hybrid_astar_8ch_bev")
BEV_X_MIN = -10.0
BEV_X_MAX = 10.0
BEV_Y_MIN = -10.0
BEV_Y_MAX = 10.0
BEV_RESOLUTION = 0.1
BEV_SIZE = 200


def wrap_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def angle_error(a: float, b: float) -> float:
    return abs(wrap_pi(float(a) - float(b)))


def read_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def write_json(path: Path, value: dict) -> None:
    def safe(item):
        if isinstance(item, dict):
            return {str(key): safe(entry) for key, entry in item.items()}
        if isinstance(item, (list, tuple)):
            return [safe(entry) for entry in item]
        if isinstance(item, np.generic):
            return safe(item.item())
        if isinstance(item, float) and not math.isfinite(item):
            return None
        return item

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(safe(value), handle, ensure_ascii=False, indent=2, allow_nan=False)


def native_path(value: str) -> Path:
    value = str(value)
    if os.name == "nt" and value.startswith("/mnt/") and len(value) > 7:
        return Path(value[5].upper() + ":/" + value[7:])
    return Path(value)


def point_in_polygon(point: Sequence[float], polygon: Sequence[Sequence[float]], tolerance: float = 0.0) -> bool:
    x, y = float(point[0]), float(point[1])
    pts = np.asarray(polygon, dtype=np.float64).reshape((-1, 2))
    if len(pts) < 3:
        return False
    inside = False
    j = len(pts) - 1
    for i in range(len(pts)):
        xi, yi = pts[i]
        xj, yj = pts[j]
        dx = xj - xi
        dy = yj - yi
        denom = dx * dx + dy * dy
        if denom > 0.0:
            t = max(0.0, min(1.0, ((x - xi) * dx + (y - yi) * dy) / denom))
            if math.hypot(x - (xi + t * dx), y - (yi + t * dy)) <= tolerance:
                return True
        if (yi > y) != (yj > y):
            denominator = yj - yi
            if abs(denominator) < 1e-12:
                denominator = 1e-12 if denominator >= 0.0 else -1e-12
            crossing = (xj - xi) * (y - yi) / denominator + xi
            if x < crossing:
                inside = not inside
        j = i
    return inside


def point_to_polygon_edge_distance(point: Sequence[float], polygon: Sequence[Sequence[float]]) -> float:
    """Return the shortest Euclidean distance from a point to a polygon edge."""
    x, y = float(point[0]), float(point[1])
    pts = np.asarray(polygon, dtype=np.float64).reshape((-1, 2))
    if len(pts) < 2:
        return 0.0
    best = float("inf")
    for index in range(len(pts)):
        x0, y0 = pts[index]
        x1, y1 = pts[(index + 1) % len(pts)]
        dx = float(x1 - x0)
        dy = float(y1 - y0)
        denom = dx * dx + dy * dy
        if denom <= 1e-12:
            distance = math.hypot(x - float(x0), y - float(y0))
        else:
            t = max(0.0, min(1.0, ((x - float(x0)) * dx + (y - float(y0)) * dy) / denom))
            distance = math.hypot(x - (float(x0) + t * dx), y - (float(y0) + t * dy))
        best = min(best, distance)
    return float(best)


def rectangle_polygon(center: Sequence[float], yaw: float, length: float, width: float) -> List[List[float]]:
    cx, cy = map(float, center[:2])
    half_l = 0.5 * float(length)
    half_w = 0.5 * float(width)
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    output = []
    for lx, ly in ((half_l, -half_w), (half_l, half_w), (-half_l, half_w), (-half_l, -half_w)):
        output.append([cx + c * lx - s * ly, cy + s * lx + c * ly])
    return output


def polygon_long_axis_yaw(polygon: Sequence[Sequence[float]]) -> float:
    points = np.asarray(polygon, dtype=np.float64).reshape((-1, 2))
    if len(points) < 2:
        return 0.0
    best_length = -1.0
    best_yaw = 0.0
    for index in range(len(points)):
        delta = points[(index + 1) % len(points)] - points[index]
        length = float(np.hypot(delta[0], delta[1]))
        if length > best_length:
            best_length = length
            best_yaw = math.atan2(float(delta[1]), float(delta[0]))
    return wrap_pi(best_yaw)


def directed_slot_yaw(polygon: Sequence[Sequence[float]], reference_yaw: float) -> float:
    axis = polygon_long_axis_yaw(polygon)
    opposite = wrap_pi(axis + math.pi)
    return axis if angle_error(axis, reference_yaw) <= angle_error(opposite, reference_yaw) else opposite


def binary_dilate(mask: np.ndarray, radius_pixels: int) -> np.ndarray:
    source = np.asarray(mask, dtype=bool)
    radius = max(0, int(radius_pixels))
    if radius == 0:
        return source.copy()
    h, w = source.shape
    output = np.zeros_like(source)
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if dr * dr + dc * dc > radius * radius:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = src_r0 + dr
            dst_r1 = src_r1 + dr
            dst_c0 = src_c0 + dc
            dst_c1 = src_c1 + dc
            output[dst_r0:dst_r1, dst_c0:dst_c1] |= source[src_r0:src_r1, src_c0:src_c1]
    return output


def _edt_1d(values: np.ndarray) -> np.ndarray:
    """Felzenszwalb-Huttenlocher exact squared 1-D Euclidean transform."""

    f = np.asarray(values, dtype=np.float64)
    n = len(f)
    v = np.zeros(n, dtype=np.int32)
    z = np.empty(n + 1, dtype=np.float64)
    result = np.empty(n, dtype=np.float64)
    k = 0
    v[0] = 0
    z[0] = -np.inf
    z[1] = np.inf
    for q in range(1, n):
        q2 = float(q * q)
        while True:
            vk = int(v[k])
            denominator = 2.0 * float(q - vk)
            separation = ((f[q] + q2) - (f[vk] + float(vk * vk))) / denominator
            if separation > z[k] or k == 0:
                break
            k -= 1
        if separation <= z[k] and k == 0:
            separation = z[k] + 1e-12
        k += 1
        v[k] = q
        z[k] = separation
        z[k + 1] = np.inf
    k = 0
    for q in range(n):
        while z[k + 1] < q:
            k += 1
        delta = q - int(v[k])
        result[q] = float(delta * delta) + f[int(v[k])]
    return result


def euclidean_distance_transform(obstacle_mask: np.ndarray, resolution: float) -> np.ndarray:
    obstacle = np.asarray(obstacle_mask, dtype=bool)
    h, w = obstacle.shape
    if not obstacle.any():
        return np.full((h, w), 1e3, dtype=np.float32)
    large = float((h + w) * (h + w) + 1)
    work = np.where(obstacle, 0.0, large).astype(np.float64)
    vertical = np.empty_like(work)
    for col in range(w):
        vertical[:, col] = _edt_1d(work[:, col])
    squared = np.empty_like(vertical)
    for row in range(h):
        squared[row, :] = _edt_1d(vertical[row, :])
    return (np.sqrt(squared) * float(resolution)).astype(np.float32)


@dataclass
class VehicleConfig:
    blueprint: str = "vehicle.tesla.model3"
    map_name: str = "Town05"
    semantic_tag: int = 14
    generation: int = 1
    base_type: str = "car"
    special_type: str = "electric"
    wheel_count: int = 4
    default_role_name: str = "autopilot"
    default_color_rgb: Tuple[int, int, int] = (17, 37, 103)
    length: float = 4.7917795
    width: float = 2.16345
    height: float = 1.487660
    bbox_center_x: float = 0.02921997
    bbox_center_y: float = 0.000005
    bbox_center_z: float = 0.735861
    wheel_base: float = 3.004647
    rear_axle_to_center: float = 1.41563697
    rear_axle_lateral_offset: float = 0.000003
    front_axle_x: float = 1.618230
    rear_axle_x: float = -1.386417
    front_track: float = 1.666828
    rear_track: float = 1.666828
    front_wheel_max_steer_deg: float = 70.0
    planning_steer_utilization: float = 0.98
    steering_curve: Tuple[Tuple[float, float], ...] = (
        (0.0, 1.0),
        (20.0, 0.9),
        (60.0, 0.8),
        (120.0, 0.7),
    )
    forward_speed_limit_kmh: float = 12.0
    reverse_speed_limit_kmh: float = 10.0
    simulation_hz: float = 20.0
    fixed_delta_seconds: float = 0.05
    mass_kg: float = 1845.0
    drag_coefficient: float = 0.15
    center_of_mass_x: float = 0.45
    center_of_mass_y: float = 0.0
    center_of_mass_z: float = -0.30
    max_rpm: float = 15000.0
    moment_of_inertia_kgm2: float = 1.0
    damping_rate_full_throttle: float = 0.15
    damping_rate_zero_throttle_clutch_engaged: float = 2.0
    damping_rate_zero_throttle_clutch_disengaged: float = 0.35
    automatic_gearbox: bool = True
    gear_switch_time_s: float = 0.0
    clutch_strength_kgm2ps: float = 10.0
    final_ratio: float = 9.0
    forward_gear_ratio: float = 1.0
    torque_curve_rpm_nm: Tuple[Tuple[float, float], ...] = (
        (-9.31094, 743.0),
        (9000.0, 743.0),
        (9500.0, 660.0),
        (10400.0, 605.0),
        (12500.0, 528.0),
        (15000.0, 502.0),
    )
    sweep_wheel_collision: bool = False
    front_left_wheel_xyz_m: Tuple[float, float, float] = (1.618225, -0.833416, 0.302955)
    front_right_wheel_xyz_m: Tuple[float, float, float] = (1.618235, 0.833412, 0.302955)
    rear_left_wheel_xyz_m: Tuple[float, float, float] = (-1.386417, -0.833412, 0.302960)
    rear_right_wheel_xyz_m: Tuple[float, float, float] = (-1.386417, 0.833416, 0.302956)
    wheel_radius_m: float = 0.37
    tire_friction: float = 3.5
    wheel_damping_rate: float = 0.25
    max_brake_torque_nm: float = 700.0
    rear_handbrake_torque_nm: float = 1400.0
    longitudinal_stiffness: float = 3000.0
    lateral_stiffness_max_load: float = 3.0
    lateral_stiffness_value: float = 20.0
    safety_margin: float = 0.10
    collision_circles: int = 7
    raster_cell_margin_scale: float = 1.0

    @property
    def max_steer(self) -> float:
        """Maximum planning bicycle angle from the inner Ackermann wheel."""

        inner = math.radians(float(self.front_wheel_max_steer_deg) * float(self.planning_steer_utilization))
        inner_radius = float(self.wheel_base) / max(1e-9, math.tan(inner))
        center_radius = inner_radius + 0.5 * float(self.front_track)
        return math.atan(float(self.wheel_base) / max(1e-9, center_radius))

    @property
    def max_steer_deg(self) -> float:
        return math.degrees(self.max_steer)

    @property
    def max_curvature(self) -> float:
        return math.tan(self.max_steer) / float(self.wheel_base)

    @property
    def min_turning_radius(self) -> float:
        return 1.0 / max(1e-9, self.max_curvature)

    def ackermann_wheel_angles(self, bicycle_steer: float) -> Tuple[float, float]:
        """Return signed inner/outer front-wheel angles for a bicycle angle."""

        steer = float(bicycle_steer)
        if abs(steer) < 1e-10:
            return 0.0, 0.0
        radius = float(self.wheel_base) / max(1e-9, math.tan(abs(steer)))
        inner_radius = max(1e-6, radius - 0.5 * float(self.front_track))
        outer_radius = radius + 0.5 * float(self.front_track)
        sign = 1.0 if steer >= 0.0 else -1.0
        return (
            sign * math.atan(float(self.wheel_base) / inner_radius),
            sign * math.atan(float(self.wheel_base) / outer_radius),
        )

    def steering_scale_at_speed(self, speed_kmh: float) -> float:
        speed = max(0.0, float(speed_kmh))
        curve = sorted((float(x), float(y)) for x, y in self.steering_curve)
        if speed <= curve[0][0]:
            return curve[0][1]
        for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
            if speed <= x1:
                ratio = (speed - x0) / max(1e-9, x1 - x0)
                return y0 + ratio * (y1 - y0)
        return curve[-1][1]

    def max_speed_for_bicycle_steer_kmh(self, bicycle_steer: float, direction: int) -> float:
        inner, _ = self.ackermann_wheel_angles(bicycle_steer)
        required_ratio = abs(inner) / max(1e-9, math.radians(float(self.front_wheel_max_steer_deg)))
        software_limit = self.forward_speed_limit_kmh if int(direction) >= 0 else self.reverse_speed_limit_kmh
        if required_ratio <= self.steering_scale_at_speed(software_limit) + 1e-12:
            return float(software_limit)
        curve = sorted((float(x), float(y)) for x, y in self.steering_curve)
        if required_ratio > curve[0][1] + 1e-12:
            return 0.0
        for (x0, y0), (x1, y1) in zip(curve, curve[1:]):
            if y0 + 1e-12 >= required_ratio >= y1 - 1e-12:
                denominator = y1 - y0
                if abs(denominator) < 1e-12:
                    continue
                ratio = (required_ratio - y0) / denominator
                speed = x0 + ratio * (x1 - x0)
                return max(0.0, min(float(software_limit), float(speed)))
        return float(software_limit)

    def required_carla_steer_command(self, bicycle_steer: float, speed_kmh: float) -> float:
        inner, _ = self.ackermann_wheel_angles(bicycle_steer)
        available = math.radians(float(self.front_wheel_max_steer_deg)) * self.steering_scale_at_speed(speed_kmh)
        return inner / max(1e-9, available)

    def carla_steer_command(self, bicycle_steer: float, speed_kmh: float) -> float:
        required = self.required_carla_steer_command(bicycle_steer, speed_kmh)
        return max(-1.0, min(1.0, required))

    def derived_specs(self) -> dict:
        return {
            "ackermann_planning_bicycle_max_steer_deg": self.max_steer_deg,
            "planning_inner_wheel_max_steer_deg": float(self.front_wheel_max_steer_deg) * float(self.planning_steer_utilization),
            "minimum_rear_axle_turning_radius_m": self.min_turning_radius,
            "rear_axle_local_x_m": float(self.bbox_center_x) - float(self.rear_axle_to_center),
            "rear_axle_local_y_m": float(self.bbox_center_y) - float(self.rear_axle_lateral_offset),
        }

    def rear_to_center(self, x: float, y: float, yaw: float) -> Tuple[float, float, float]:
        offset_x = float(self.rear_axle_to_center)
        offset_y = float(self.rear_axle_lateral_offset)
        c = math.cos(yaw)
        s = math.sin(yaw)
        return x + c * offset_x - s * offset_y, y + s * offset_x + c * offset_y, wrap_pi(yaw)

    def center_to_rear(self, x: float, y: float, yaw: float) -> Tuple[float, float, float]:
        offset_x = float(self.rear_axle_to_center)
        offset_y = float(self.rear_axle_lateral_offset)
        c = math.cos(yaw)
        s = math.sin(yaw)
        return x - c * offset_x + s * offset_y, y - s * offset_x - c * offset_y, wrap_pi(yaw)

    def corners_from_rear(self, x: float, y: float, yaw: float, margin: float = 0.0) -> List[List[float]]:
        cx, cy, _ = self.rear_to_center(float(x), float(y), float(yaw))
        return rectangle_polygon(
            (cx, cy),
            yaw,
            float(self.length) + 2.0 * float(margin),
            float(self.width) + 2.0 * float(margin),
        )

    def covering_circles(self) -> Tuple[np.ndarray, float]:
        count = max(2, int(self.collision_circles))
        half_l = 0.5 * float(self.length)
        half_w = 0.5 * float(self.width)
        back = float(self.rear_axle_to_center) - half_l
        front = float(self.rear_axle_to_center) + half_l
        segment = (front - back) / float(count)
        centers_x = back + (np.arange(count, dtype=np.float64) + 0.5) * segment
        centers_y = np.full(count, float(self.rear_axle_lateral_offset), dtype=np.float64)
        centers = np.column_stack((centers_x, centers_y))
        radius = math.hypot(0.5 * segment, half_w) + float(self.safety_margin)
        return centers, radius


@dataclass
class PlannerConfig:
    xy_resolution: float = 0.20
    yaw_resolution_deg: float = 4.0
    primitive_length: float = 0.45
    integration_step: float = 0.05
    steering_samples: int = 9
    max_steer_index_delta: int = 1
    gear_switch_steer_bins: int = 1
    reverse_multiplier: float = 1.20
    gear_switch_cost: float = 3.5
    steer_cost: float = 0.18
    steer_change_cost: float = 1.10
    proximity_distance: float = 1.00
    proximity_cost: float = 0.75
    heuristic_weight: float = 1.05
    heuristic_downsample: int = 2
    heuristic_inflation: float = 0.35
    nonholonomic_heuristic_radius_m: float = 12.0
    analytic_distance: float = 3.20
    relaxed_analytic_distance: float = 6.0
    analytic_interval: int = 6
    analytic_min_expansions: int = 24
    analytic_immediate_ratio: float = 1.45
    analytic_refine_expansions: int = 900
    relaxed_refine_expansions: int = 0
    analytic_steer_transition_cost: float = 2.50
    analytic_short_segment_m: float = 0.75
    analytic_short_segment_cost: float = 1.50
    analytic_max_steer_jump_bins: float = 2.0
    analytic_steer_ramp_length_m: float = 0.65
    analytic_connector_position_tolerance_m: float = 0.08
    analytic_connector_yaw_tolerance_deg: float = 1.5
    analytic_connector_opt_iterations: int = 18
    max_time_s: float = 12.0
    max_expansions: int = 80000
    goal_xy_tolerance: float = 0.15
    goal_yaw_tolerance_deg: float = 3.0
    terminal_dock_enabled: bool = True
    terminal_dock_distances_m: Tuple[float, ...] = (2.40, 2.00, 1.60, 1.20)
    terminal_dock_min_length_m: float = 1.15
    terminal_dock_yaw_tolerance_deg: float = 1.0
    terminal_dock_lateral_tolerance_m: float = 0.05
    terminal_dock_steer_tolerance_deg: float = 0.10
    # Offline quality searches may deliberately spend extra time across
    # geometrically distinct staging distances.  Latency-bounded live planner
    # profiles override this to 1.0 so the advertised wall-clock budget stays
    # a hard cap.
    terminal_candidate_total_time_multiplier: float = 1.60
    terminal_first_candidate_fraction: float = 0.38
    terminal_candidate_min_time_s: float = 0.60
    terminal_prefer_shortest_first: bool = False
    # Feasibility-first profiles normally return the first reachable staging
    # pose.  For a multi-cusp first result, however, the remaining staging
    # budget can often discover a much faster one-switch parking topology.
    # Keep the first route as a guaranteed fallback while comparing those
    # remaining terminal corridors.
    terminal_continue_after_gear_switches: int = 0

    @property
    def yaw_resolution(self) -> float:
        return math.radians(float(self.yaw_resolution_deg))


@dataclass
class BEVFrame:
    cache_index: int
    session_index: int
    session_id: str
    side: str
    frame: int
    session_path: Path
    maps: np.ndarray
    goal_center: Tuple[float, float]
    goal_yaw: float
    recorded_goal_yaw: float
    goal_length: float
    goal_width: float
    goal_polygon: List[List[float]]
    target_valid: bool
    target_iou: float
    start_reverse: bool
    speed_kmh: float

    @property
    def obstacle_mask(self) -> np.ndarray:
        obstacle = np.asarray((self.maps[2] > 0) | (self.maps[3] > 0), dtype=bool)
        ego_clear = binary_dilate(self.maps[0] > 0, 3)
        obstacle[ego_clear] = False
        return obstacle


class PackedBEVCache:
    def __init__(self, cache_dir: Path):
        self.cache_dir = Path(cache_dir)
        self.metadata = read_json(self.cache_dir / "metadata.json")
        if not self.metadata.get("complete"):
            raise RuntimeError("8-channel cache is incomplete: {}".format(self.cache_dir))
        channels = tuple(self.metadata.get("target_bev_channels") or ())
        if channels != CHANNELS:
            raise RuntimeError("unexpected channel order: {}".format(channels))
        self.base_dir = Path(self.metadata["base_cache_dir"])
        self.base_metadata = read_json(self.base_dir / "metadata.json")
        self.sessions = read_json(self.base_dir / "sessions.json")
        self.packed = np.load(str(self.cache_dir / "target_bev_packed.npy"), mmap_mode="r")
        self.session_indices = np.load(str(self.base_dir / "session_index.npy"), mmap_mode="r")
        self.frames = np.load(str(self.base_dir / "frame.npy"), mmap_mode="r")
        self.goal_vectors = np.load(str(self.base_dir / "goal_vector.npy"), mmap_mode="r")
        expected = (int(self.metadata["total_frames"]), 8, 200, 25)
        if self.packed.shape != expected or self.packed.dtype != np.uint8:
            raise RuntimeError("invalid packed raster: {} {} expected {}".format(self.packed.shape, self.packed.dtype, expected))
        for name, array in (
            ("session_index", self.session_indices),
            ("frame", self.frames),
            ("goal_vector", self.goal_vectors),
        ):
            if len(array) != len(self.packed):
                raise RuntimeError("{} is not aligned with packed raster".format(name))

    def __len__(self) -> int:
        return len(self.packed)

    @property
    def vehicle_config(self) -> VehicleConfig:
        spec = self.base_metadata.get("vehicle_spec") or {}
        length = float(spec.get("length", 4.7917795))
        return VehicleConfig(
            length=length,
            width=float(spec.get("width", 2.16345)),
            bbox_center_x=float(spec.get("center_x", 0.02921997)),
            # The cache stores only the collision box.  All wheel/physics
            # defaults below are from the attached 2026-07-12 CARLA
            # vehicle.tesla.model3 runtime specification.
            bbox_center_y=0.000005,
            wheel_base=3.004647,
            rear_axle_to_center=float(spec.get("center_x", 0.02921997)) - (-1.386417),
            rear_axle_lateral_offset=0.000003,
            front_axle_x=1.618230,
            rear_axle_x=-1.386417,
            front_track=1.666828,
            rear_track=1.666828,
            front_wheel_max_steer_deg=70.0,
        )

    def find_index(self, session_id: str, frame: int) -> int:
        matches = [item for item in self.sessions if str(item["session_id"]) == str(session_id)]
        if not matches:
            raise KeyError("session is not present in the packed cache: {}".format(session_id))
        session = matches[0]
        start = int(session["offset"])
        end = start + int(session["count"])
        local = np.nonzero(np.asarray(self.frames[start:end]) == int(frame))[0]
        if len(local) == 0:
            raise KeyError("frame {} is not present in session {}".format(frame, session_id))
        return start + int(local[0])

    def load(self, index: int) -> BEVFrame:
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        maps = np.unpackbits(np.asarray(self.packed[index]), axis=-1, count=BEV_SIZE, bitorder="big")
        if maps.shape != (8, BEV_SIZE, BEV_SIZE):
            raise RuntimeError("bad decoded map shape: {}".format(maps.shape))
        sid = int(self.session_indices[index])
        frame = int(self.frames[index])
        session = self.sessions[sid]
        session_path = native_path(session["path"])
        goal_vector = np.asarray(self.goal_vectors[index], dtype=np.float64)
        center = (float(goal_vector[0] * 20.0), float(goal_vector[1] * 20.0))
        yaw = math.atan2(float(goal_vector[2]), float(goal_vector[3]))
        recorded_yaw = yaw
        length = float(goal_vector[4] * 8.0)
        width = float(goal_vector[5] * 4.0)
        goal_polygon = rectangle_polygon(center, yaw, length, width)
        target_valid = True
        target_iou = float("nan")
        goal_path = session_path / "cvt_bev" / "target_goal" / ("{:06d}.json".format(frame))
        if goal_path.exists():
            raw_goal = read_json(goal_path)
            raw_center = raw_goal.get("center_ego")
            if raw_center is not None and len(raw_center) >= 2:
                center = (float(raw_center[0]), float(raw_center[1]))
            if raw_goal.get("yaw_ego_deg") is not None:
                yaw = math.radians(float(raw_goal["yaw_ego_deg"]))
                recorded_yaw = yaw
            length = float(raw_goal.get("length", length))
            width = float(raw_goal.get("width", width))
            if len(raw_goal.get("polygon_ego") or []) >= 3:
                goal_polygon = [[float(x), float(y)] for x, y in raw_goal["polygon_ego"]]
            else:
                goal_polygon = rectangle_polygon(center, yaw, length, width)
            target_valid = bool(raw_goal.get("target_candidate_valid", True))
            target_iou = float(raw_goal.get("target_candidate_iou", float("nan")))
        # The saved yaw follows the demonstrated final vehicle and can be
        # visibly skewed inside a narrow slot.  Use the slot's long axis for a
        # geometrically valid parked pose, resolving its 180-degree ambiguity
        # with the recorded yaw.
        yaw = directed_slot_yaw(goal_polygon, recorded_yaw)
        state_path = session_path / "state" / ("{:06d}.json".format(frame))
        start_reverse = False
        speed_kmh = 0.0
        if state_path.exists():
            state = read_json(state_path)
            start_reverse = bool((state.get("control") or {}).get("reverse", False))
            speed_kmh = float(state.get("speed_kmh", 0.0))
        return BEVFrame(
            cache_index=index,
            session_index=sid,
            session_id=str(session["session_id"]),
            side=str(session["side"]),
            frame=frame,
            session_path=session_path,
            maps=maps.astype(np.uint8, copy=False),
            goal_center=center,
            goal_yaw=wrap_pi(yaw),
            recorded_goal_yaw=wrap_pi(recorded_yaw),
            goal_length=length,
            goal_width=width,
            goal_polygon=goal_polygon,
            target_valid=target_valid,
            target_iou=target_iou,
            start_reverse=start_reverse,
            speed_kmh=speed_kmh,
        )


class CollisionChecker:
    """Conservative rectangle-vs-raster collision checking using covering circles."""

    def __init__(self, obstacle_mask: np.ndarray, vehicle: VehicleConfig):
        self.obstacle = np.asarray(obstacle_mask, dtype=bool)
        if self.obstacle.shape != (BEV_SIZE, BEV_SIZE):
            raise ValueError("obstacle mask must be 200x200")
        self.vehicle = vehicle
        self.distance = euclidean_distance_transform(self.obstacle, BEV_RESOLUTION)
        self.circle_offsets, self.circle_radius = vehicle.covering_circles()
        self.cell_half_diagonal = (
            math.sqrt(2.0)
            * 0.5
            * BEV_RESOLUTION
            * max(0.0, float(vehicle.raster_cell_margin_scale))
        )
        obstacle_rows, obstacle_cols = np.nonzero(self.obstacle)
        self.obstacle_x = BEV_X_MAX - (obstacle_rows.astype(np.float64) + 0.5) * BEV_RESOLUTION
        self.obstacle_y = BEV_Y_MIN + (obstacle_cols.astype(np.float64) + 0.5) * BEV_RESOLUTION

    @staticmethod
    def _nearest_cells(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        row_float = (BEV_X_MAX - x) / BEV_RESOLUTION - 0.5
        col_float = (y - BEV_Y_MIN) / BEV_RESOLUTION - 0.5
        rows = np.rint(row_float).astype(np.int32)
        cols = np.rint(col_float).astype(np.int32)
        valid = (rows >= 0) & (rows < BEV_SIZE) & (cols >= 0) & (cols < BEV_SIZE)
        rows = np.clip(rows, 0, BEV_SIZE - 1)
        cols = np.clip(cols, 0, BEV_SIZE - 1)
        cell_x = BEV_X_MAX - (rows.astype(np.float64) + 0.5) * BEV_RESOLUTION
        cell_y = BEV_Y_MIN + (cols.astype(np.float64) + 0.5) * BEV_RESOLUTION
        offset = np.hypot(x - cell_x, y - cell_y)
        return rows, cols, np.where(valid, offset, np.inf)

    def check_poses(
        self,
        xs: Sequence[float],
        ys: Sequence[float],
        yaws: Sequence[float],
    ) -> Tuple[bool, float]:
        x = np.asarray(xs, dtype=np.float64).reshape((-1,))
        y = np.asarray(ys, dtype=np.float64).reshape((-1,))
        yaw = np.asarray(yaws, dtype=np.float64).reshape((-1,))
        if not (len(x) == len(y) == len(yaw)) or len(x) == 0:
            return False, -float("inf")
        c = np.cos(yaw)
        s = np.sin(yaw)

        half_l = 0.5 * self.vehicle.length + self.vehicle.safety_margin
        half_w = 0.5 * self.vehicle.width + self.vehicle.safety_margin
        offset_x = float(self.vehicle.rear_axle_to_center)
        offset_y = float(self.vehicle.rear_axle_lateral_offset)
        center_x = x + offset_x * c - offset_y * s
        center_y = y + offset_x * s + offset_y * c
        for lx, ly in ((half_l, half_w), (half_l, -half_w), (-half_l, half_w), (-half_l, -half_w)):
            corner_x = center_x + lx * c - ly * s
            corner_y = center_y + lx * s + ly * c
            if np.any(corner_x < BEV_X_MIN) or np.any(corner_x > BEV_X_MAX):
                return False, -float("inf")
            if np.any(corner_y < BEV_Y_MIN) or np.any(corner_y > BEV_Y_MAX):
                return False, -float("inf")

        pose_clearance = np.full(len(x), np.inf, dtype=np.float64)
        for local_x, local_y in self.circle_offsets:
            circle_x = x + float(local_x) * c - float(local_y) * s
            circle_y = y + float(local_x) * s + float(local_y) * c
            rows, cols, sample_offset = self._nearest_cells(circle_x, circle_y)
            if not np.isfinite(sample_offset).all():
                return False, -float("inf")
            lower_distance = (
                self.distance[rows, cols].astype(np.float64)
                - sample_offset
                - self.cell_half_diagonal
            )
            clearance = lower_distance - float(self.circle_radius)
            pose_clearance = np.minimum(pose_clearance, clearance)

        # Covering circles are a fast conservative broad phase.  Near a
        # rectangle corner they can overlap a cell that the actual vehicle
        # does not.  Resolve only those candidates with an exact oriented-box
        # test against obstacle pixel cells, expanded by half a pixel diagonal.
        candidate_indices = np.nonzero(pose_clearance <= 0.0)[0]
        expanded_l = 0.5 * self.vehicle.length + self.vehicle.safety_margin + self.cell_half_diagonal
        expanded_w = 0.5 * self.vehicle.width + self.vehicle.safety_margin + self.cell_half_diagonal
        for index in candidate_indices:
            if len(self.obstacle_x) == 0:
                pose_clearance[index] = 1e3
                continue
            dx = self.obstacle_x - center_x[index]
            dy = self.obstacle_y - center_y[index]
            local_x = dx * c[index] + dy * s[index]
            local_y = -dx * s[index] + dy * c[index]
            outside_x = np.abs(local_x) - expanded_l
            outside_y = np.abs(local_y) - expanded_w
            if np.any((outside_x <= 0.0) & (outside_y <= 0.0)):
                return False, float(min(pose_clearance.min(initial=0.0), 0.0))
            exact_clearance = np.hypot(np.maximum(outside_x, 0.0), np.maximum(outside_y, 0.0))
            pose_clearance[index] = float(exact_clearance.min(initial=float("inf")))
        return True, float(pose_clearance.min(initial=float("inf")))

    def pose_ok(self, pose: Sequence[float]) -> Tuple[bool, float]:
        return self.check_poses([pose[0]], [pose[1]], [pose[2]])


class HolonomicHeuristic:
    def __init__(
        self,
        obstacle_distance: np.ndarray,
        goal_xy: Sequence[float],
        downsample: int = 2,
        inflation: float = 0.35,
    ):
        self.downsample = max(1, int(downsample))
        self.resolution = BEV_RESOLUTION * self.downsample
        blocked_full = np.asarray(obstacle_distance <= float(inflation), dtype=bool)
        size = BEV_SIZE // self.downsample
        cropped = blocked_full[: size * self.downsample, : size * self.downsample]
        self.blocked = cropped.reshape(size, self.downsample, size, self.downsample).any(axis=(1, 3))
        self.cost = np.full((size, size), np.inf, dtype=np.float32)
        gr, gc = self._cell(float(goal_xy[0]), float(goal_xy[1]))
        self.blocked[gr, gc] = False
        self._dijkstra(gr, gc)

    def _cell(self, x: float, y: float) -> Tuple[int, int]:
        row = int(math.floor((BEV_X_MAX - float(x)) / self.resolution))
        col = int(math.floor((float(y) - BEV_Y_MIN) / self.resolution))
        row = max(0, min(self.cost.shape[0] - 1, row))
        col = max(0, min(self.cost.shape[1] - 1, col))
        return row, col

    def _dijkstra(self, goal_row: int, goal_col: int) -> None:
        self.cost[goal_row, goal_col] = 0.0
        heap: List[Tuple[float, int, int]] = [(0.0, goal_row, goal_col)]
        motions = (
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2.0)),
            (-1, 1, math.sqrt(2.0)),
            (1, -1, math.sqrt(2.0)),
            (1, 1, math.sqrt(2.0)),
        )
        h, w = self.cost.shape
        while heap:
            current, row, col = heapq.heappop(heap)
            if current > float(self.cost[row, col]) + 1e-6:
                continue
            for dr, dc, scale in motions:
                nr, nc = row + dr, col + dc
                if nr < 0 or nr >= h or nc < 0 or nc >= w or self.blocked[nr, nc]:
                    continue
                value = current + self.resolution * scale
                if value + 1e-6 < float(self.cost[nr, nc]):
                    self.cost[nr, nc] = value
                    heapq.heappush(heap, (value, nr, nc))

    def value(self, x: float, y: float, fallback_goal: Sequence[float]) -> float:
        row, col = self._cell(x, y)
        value = float(self.cost[row, col])
        if not math.isfinite(value):
            return 1.35 * math.hypot(float(x) - float(fallback_goal[0]), float(y) - float(fallback_goal[1]))
        return value


@dataclass
class SearchNode:
    x: float
    y: float
    yaw: float
    direction: int
    steer: float
    steer_index: int
    cost: float
    parent: Optional[int]
    segment: np.ndarray


@dataclass
class PlanResult:
    success: bool
    message: str
    states: np.ndarray
    expansions: int
    elapsed_s: float
    analytic_expansions: int
    min_clearance_m: float = float("nan")
    validation: dict = field(default_factory=dict)
    terminal_dock: dict = field(default_factory=dict)


class HybridAStarPlanner:
    def __init__(
        self,
        vehicle: VehicleConfig,
        config: PlannerConfig,
        obstacle_mask: np.ndarray,
        guide_states: Optional[np.ndarray] = None,
        guide_weight: float = 0.0,
        guide_free_radius_m: float = 0.75,
        terminal_obstacle_mask: Optional[np.ndarray] = None,
    ):
        self.vehicle = vehicle
        self.config = config
        self.collision = CollisionChecker(obstacle_mask, vehicle)
        self.terminal_collision = (
            self.collision
            if terminal_obstacle_mask is None
            else CollisionChecker(terminal_obstacle_mask, vehicle)
        )
        self.guide_weight = max(0.0, float(guide_weight))
        self.guide_free_radius_m = max(0.0, float(guide_free_radius_m))
        self.guide_distance: Optional[np.ndarray] = None
        if self.guide_weight > 0.0 and guide_states is not None:
            guide = np.asarray(guide_states, dtype=np.float64)
            if guide.ndim == 2 and len(guide) and guide.shape[1] >= 2:
                guide_mask = np.zeros((BEV_SIZE, BEV_SIZE), dtype=bool)
                rows = np.rint((BEV_X_MAX - guide[:, 0]) / BEV_RESOLUTION - 0.5).astype(np.int32)
                cols = np.rint((guide[:, 1] - BEV_Y_MIN) / BEV_RESOLUTION - 0.5).astype(np.int32)
                valid = (rows >= 0) & (rows < BEV_SIZE) & (cols >= 0) & (cols < BEV_SIZE)
                guide_mask[rows[valid], cols[valid]] = True
                if guide_mask.any():
                    self.guide_distance = euclidean_distance_transform(guide_mask, BEV_RESOLUTION)
        samples = max(3, int(config.steering_samples))
        if samples % 2 == 0:
            samples += 1
        self.steers = np.linspace(-vehicle.max_steer, vehicle.max_steer, samples, dtype=np.float64)
        self._yaw_bins = max(1, int(round(2.0 * math.pi / config.yaw_resolution)))
        self._nonholonomic_cache: Dict[Tuple[int, int, int], float] = {}

    def _guide_penalty(self, x: float, y: float) -> float:
        """Softly keep a quality search in the topology of a feasible route."""
        if self.guide_distance is None or self.guide_weight <= 0.0:
            return 0.0
        row = int(round((BEV_X_MAX - float(x)) / BEV_RESOLUTION - 0.5))
        col = int(round((float(y) - BEV_Y_MIN) / BEV_RESOLUTION - 0.5))
        if row < 0 or row >= BEV_SIZE or col < 0 or col >= BEV_SIZE:
            return float(self.guide_weight * 100.0)
        outside = max(0.0, float(self.guide_distance[row, col]) - self.guide_free_radius_m)
        return float(self.guide_weight * outside * outside)

    def _state_key(self, node: SearchNode) -> Tuple[int, int, int, int, int]:
        ix = int(round((node.x - BEV_X_MIN) / self.config.xy_resolution))
        iy = int(round((node.y - BEV_Y_MIN) / self.config.xy_resolution))
        iyaw = int(round((wrap_pi(node.yaw) + math.pi) / self.config.yaw_resolution)) % self._yaw_bins
        return ix, iy, iyaw, 1 if node.direction > 0 else -1, int(node.steer_index)

    def _successor_steer_indices(self, node: SearchNode, direction: int) -> List[int]:
        """Return steering-continuous lattice controls for one expansion.

        In the same gear the steering bin can move only to an adjacent value.
        At a D/R cusp the controller stops and releases steering to centre, so
        the next segment starts from the centre bin or one adjacent bin.
        """
        center = len(self.steers) // 2
        if int(direction) == int(node.direction):
            anchor = int(node.steer_index)
            delta = max(0, int(self.config.max_steer_index_delta))
        else:
            anchor = center
            delta = max(0, int(self.config.gear_switch_steer_bins))
        begin = max(0, anchor - delta)
        end = min(len(self.steers) - 1, anchor + delta)
        return sorted(range(begin, end + 1), key=lambda index: (abs(index - anchor), abs(index - center)))

    def _simulate(self, node: SearchNode, direction: int, steer: float, steer_index: int) -> Optional[SearchNode]:
        cfg = self.config
        steps = max(1, int(math.ceil(cfg.primitive_length / cfg.integration_step)))
        ds = cfg.primitive_length / float(steps)
        start_steer = float(node.steer) if int(direction) == int(node.direction) else 0.0
        x, y, yaw = node.x, node.y, node.yaw
        rows = []
        for step in range(steps):
            ratio_mid = (float(step) + 0.5) / float(steps)
            ratio_end = (float(step) + 1.0) / float(steps)
            steer_mid = start_steer + ratio_mid * (float(steer) - start_steer)
            steer_end = start_steer + ratio_end * (float(steer) - start_steer)
            curvature = math.tan(steer_mid) / float(self.vehicle.wheel_base)
            signed = float(direction) * ds
            if abs(curvature) < 1e-10:
                x += signed * math.cos(yaw)
                y += signed * math.sin(yaw)
            else:
                next_yaw = yaw + signed * curvature
                x += (math.sin(next_yaw) - math.sin(yaw)) / curvature
                y += (math.cos(yaw) - math.cos(next_yaw)) / curvature
                yaw = wrap_pi(next_yaw)
            rows.append((x, y, yaw, int(direction), float(steer_end)))
        segment = np.asarray(rows, dtype=np.float64)
        clear, min_clearance = self.collision.check_poses(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
        )
        if not clear:
            return None
        cost = cfg.primitive_length * (1.0 if direction > 0 else cfg.reverse_multiplier)
        if int(direction) != int(node.direction):
            cost += cfg.gear_switch_cost
        cost += cfg.primitive_length * cfg.steer_cost * abs(float(steer)) / max(1e-9, self.vehicle.max_steer)
        cost += cfg.steer_change_cost * abs(float(steer) - start_steer) / max(1e-9, self.vehicle.max_steer)
        if min_clearance < cfg.proximity_distance:
            ratio = max(0.0, (cfg.proximity_distance - min_clearance) / max(1e-9, cfg.proximity_distance))
            cost += cfg.proximity_cost * cfg.primitive_length * ratio * ratio
        return SearchNode(
            x=float(segment[-1, 0]),
            y=float(segment[-1, 1]),
            yaw=float(segment[-1, 2]),
            direction=int(direction),
            steer=float(steer),
            steer_index=int(steer_index),
            cost=float(node.cost + cost),
            parent=None,
            segment=segment,
        )

    def _heuristic(self, node: SearchNode, goal: Sequence[float], holonomic: HolonomicHeuristic) -> float:
        position = holonomic.value(node.x, node.y, goal)
        heading = self.vehicle.min_turning_radius * angle_error(node.yaw, goal[2])
        nonholonomic = float(heading)
        if math.hypot(float(node.x) - float(goal[0]), float(node.y) - float(goal[1])) <= float(
            self.config.nonholonomic_heuristic_radius_m
        ):
            state_key = self._state_key(node)
            cache_key = (state_key[0], state_key[1], state_key[2])
            cached = self._nonholonomic_cache.get(cache_key)
            if cached is None:
                candidates = rs.generate_paths(
                    (node.x, node.y, node.yaw),
                    goal,
                    self.vehicle.max_curvature,
                    max_length_m=50.0,
                )
                cached = min((float(path.total_length) for path in candidates), default=float(heading))
                self._nonholonomic_cache[cache_key] = float(cached)
            nonholonomic = max(nonholonomic, float(cached))
        return max(float(position), float(nonholonomic))

    def _analytic_candidates(self, node: SearchNode, goal: Sequence[float]) -> Iterable[Tuple[float, rs.ReedsSheppPath]]:
        paths = rs.calc_paths(
            (node.x, node.y, node.yaw),
            goal,
            self.vehicle.max_curvature,
            step_size=self.config.integration_step,
            max_length_m=50.0,
            max_steer=self.vehicle.max_steer,
        )
        ranked = []
        for path in paths:
            value = rs.path_cost(
                path,
                reverse_multiplier=self.config.reverse_multiplier,
                gear_switch_cost=self.config.gear_switch_cost,
                steer_cost=self.config.steer_cost,
                steer_change_cost=self.config.steer_change_cost,
            )
            first_direction = path.directions[1] if len(path.directions) > 1 else node.direction
            if int(first_direction) != int(node.direction):
                value += self.config.gear_switch_cost
                entry_steer = 0.0
            else:
                entry_steer = float(node.steer)
            controls = [
                0.0 if ctype == rs.STRAIGHT else self.vehicle.max_steer if ctype == rs.LEFT else -self.vehicle.max_steer
                for ctype in path.ctypes
            ]
            if controls:
                value += self.config.analytic_steer_transition_cost * abs(controls[0] - entry_steer) / max(
                    1e-9, self.vehicle.max_steer
                )
            for index in range(1, len(controls)):
                same_gear = float(path.lengths[index - 1]) * float(path.lengths[index]) > 0.0
                if same_gear:
                    change_ratio = abs(float(controls[index]) - float(controls[index - 1])) / max(
                        1e-9, self.vehicle.max_steer
                    )
                    value += self.config.analytic_steer_transition_cost * change_ratio
                    if min(abs(float(path.lengths[index - 1])), abs(float(path.lengths[index]))) < float(
                        self.config.analytic_short_segment_m
                    ):
                        value += self.config.analytic_short_segment_cost * change_ratio
            ranked.append((float(value), path))
        ranked.sort(key=lambda item: item[0])
        return ranked

    def _try_analytic(
        self,
        node: SearchNode,
        goal: Sequence[float],
        enforce_steering_continuity: bool = True,
    ) -> Tuple[Optional[np.ndarray], float]:
        for cost, path in self._analytic_candidates(node, goal):
            if len(path.x) < 2:
                continue
            steer_step = float(abs(self.steers[1] - self.steers[0])) if len(self.steers) > 1 else self.vehicle.max_steer
            allowed_jump = float(self.config.analytic_max_steer_jump_bins) * steer_step + 1e-9
            directions = np.asarray(path.directions, dtype=np.int8)
            steers = np.asarray(path.steers, dtype=np.float64)
            entry_direction = int(directions[1]) if len(directions) > 1 else int(node.direction)
            entry_steer = 0.0 if entry_direction != int(node.direction) else float(node.steer)
            if (
                enforce_steering_continuity
                and len(steers) > 1
                and entry_direction == int(node.direction)
                and abs(float(steers[1]) - entry_steer) > allowed_jump
            ):
                continue
            if enforce_steering_continuity and len(steers) > 1:
                same_gear = directions[1:] == directions[:-1]
                if np.any(np.abs(np.diff(steers))[same_gear] > allowed_jump):
                    continue
            clear, min_clearance = self.collision.check_poses(path.x, path.y, path.yaw)
            if not clear:
                continue
            states = np.column_stack(
                (
                    np.asarray(path.x, dtype=np.float64),
                    np.asarray(path.y, dtype=np.float64),
                    np.asarray(path.yaw, dtype=np.float64),
                    np.asarray(path.directions, dtype=np.float64),
                    np.asarray(path.steers, dtype=np.float64),
                )
            )
            return states[1:], float(cost)
        return None, float("inf")

    @staticmethod
    def _reconstruct(nodes: List[SearchNode], node_id: int, analytic: Optional[np.ndarray] = None) -> np.ndarray:
        chunks = []
        current: Optional[int] = int(node_id)
        while current is not None:
            node = nodes[current]
            chunks.append(node.segment)
            current = node.parent
        chunks.reverse()
        path = np.concatenate(chunks, axis=0)
        if analytic is not None and len(analytic):
            path = np.concatenate((path, analytic), axis=0)
        if len(path) <= 1:
            return path
        keep = np.ones(len(path), dtype=bool)
        delta = np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))
        keep[1:] = (delta > 1e-6) | (np.abs(np.diff(path[:, 2])) > 1e-7)
        return path[keep]

    def plan(self, start: Sequence[float], goal: Sequence[float], start_direction: int = 1) -> PlanResult:
        started = time.perf_counter()
        start = (float(start[0]), float(start[1]), wrap_pi(float(start[2])))
        goal = (float(goal[0]), float(goal[1]), wrap_pi(float(goal[2])))
        start_ok, _ = self.collision.pose_ok(start)
        if not start_ok:
            return PlanResult(False, "start pose is in collision", np.empty((0, 5)), 0, time.perf_counter() - started, 0)
        goal_ok, _ = self.collision.pose_ok(goal)
        if not goal_ok:
            return PlanResult(False, "goal pose is in collision", np.empty((0, 5)), 0, time.perf_counter() - started, 0)

        holonomic = HolonomicHeuristic(
            self.collision.distance,
            goal,
            downsample=self.config.heuristic_downsample,
            inflation=self.config.heuristic_inflation,
        )
        start_direction = 1 if int(start_direction) >= 0 else -1
        center_steer = len(self.steers) // 2
        start_node = SearchNode(
            x=start[0],
            y=start[1],
            yaw=start[2],
            direction=start_direction,
            steer=0.0,
            steer_index=center_steer,
            cost=0.0,
            parent=None,
            segment=np.asarray([[start[0], start[1], start[2], start_direction, 0.0]], dtype=np.float64),
        )
        nodes = [start_node]
        best_cost: Dict[Tuple[int, int, int, int, int], float] = {self._state_key(start_node): 0.0}
        heap: List[Tuple[float, int, int]] = []
        counter = 0
        h0 = self._heuristic(start_node, goal, holonomic)
        heapq.heappush(
            heap,
            (
                self.config.heuristic_weight * h0 + self._guide_penalty(start_node.x, start_node.y),
                counter,
                0,
            ),
        )
        expansions = 0
        analytic_expansions = 0
        closest_id = 0
        closest_h = h0
        start_lower_bound = max(
            math.hypot(start[0] - goal[0], start[1] - goal[1]),
            self.vehicle.min_turning_radius * angle_error(start[2], goal[2]),
            0.1,
        )
        incumbent_states: Optional[np.ndarray] = None
        incumbent_score = float("inf")
        incumbent_clearance = float("nan")
        relaxed_states: Optional[np.ndarray] = None
        relaxed_score = float("inf")
        relaxed_clearance = float("nan")

        while heap:
            elapsed = time.perf_counter() - started
            if elapsed >= self.config.max_time_s:
                break
            if expansions >= self.config.max_expansions:
                break
            _, _, node_id = heapq.heappop(heap)
            node = nodes[node_id]
            key = self._state_key(node)
            if node.cost > best_cost.get(key, float("inf")) + 1e-7:
                continue
            expansions += 1
            if incumbent_states is not None and expansions >= max(1, self.config.analytic_refine_expansions):
                return PlanResult(
                    True,
                    "refined analytic Reeds-Shepp connection",
                    incumbent_states,
                    expansions,
                    time.perf_counter() - started,
                    analytic_expansions,
                    min_clearance_m=incumbent_clearance,
                )
            if (
                relaxed_states is not None
                and int(self.config.relaxed_refine_expansions) > 0
                and expansions >= int(self.config.relaxed_refine_expansions)
            ):
                return PlanResult(
                    True,
                    "feasibility-first relaxed Reeds-Shepp connection",
                    relaxed_states,
                    expansions,
                    time.perf_counter() - started,
                    analytic_expansions,
                    min_clearance_m=relaxed_clearance,
                )
            heuristic = self._heuristic(node, goal, holonomic)
            if heuristic < closest_h:
                closest_h = heuristic
                closest_id = node_id

            distance_to_goal = math.hypot(node.x - goal[0], node.y - goal[1])
            should_shoot = (
                distance_to_goal <= max(
                    float(self.config.analytic_distance),
                    float(self.config.relaxed_analytic_distance),
                )
                and expansions >= max(1, int(self.config.analytic_min_expansions))
                and expansions % max(1, self.config.analytic_interval) == 0
            )
            if should_shoot:
                analytic_expansions += 1
                analytic, analytic_cost = (None, float("inf"))
                if distance_to_goal <= float(self.config.analytic_distance):
                    analytic, analytic_cost = self._try_analytic(node, goal)
                if analytic is None:
                    relaxed, relaxed_cost = self._try_analytic(
                        node,
                        goal,
                        enforce_steering_continuity=False,
                    )
                    if relaxed is not None:
                        candidate = self._reconstruct(nodes, node_id, relaxed)
                        clear, min_clearance = self.collision.check_poses(
                            candidate[:, 0], candidate[:, 1], candidate[:, 2]
                        )
                        candidate_score = float(node.cost + relaxed_cost + 8.0)
                        if clear and candidate_score < relaxed_score:
                            relaxed_states = candidate
                            relaxed_score = candidate_score
                            relaxed_clearance = min_clearance
                if analytic is not None:
                    states = self._reconstruct(nodes, node_id, analytic)
                    clear, min_clearance = self.collision.check_poses(states[:, 0], states[:, 1], states[:, 2])
                    if clear:
                        candidate_score = float(node.cost + analytic_cost)
                        if candidate_score < incumbent_score:
                            incumbent_states = states
                            incumbent_score = candidate_score
                            incumbent_clearance = min_clearance
                        candidate_length = float(np.hypot(np.diff(states[:, 0]), np.diff(states[:, 1])).sum())
                        if candidate_length <= self.config.analytic_immediate_ratio * start_lower_bound:
                            return PlanResult(
                                True,
                                "analytic Reeds-Shepp connection",
                                states,
                                expansions,
                                time.perf_counter() - started,
                                analytic_expansions,
                                min_clearance_m=min_clearance,
                            )

            xy_ok = distance_to_goal <= self.config.goal_xy_tolerance
            yaw_ok = angle_error(node.yaw, goal[2]) <= math.radians(self.config.goal_yaw_tolerance_deg)
            if xy_ok and yaw_ok:
                states = self._reconstruct(nodes, node_id)
                clear, min_clearance = self.collision.check_poses(states[:, 0], states[:, 1], states[:, 2])
                if clear:
                    return PlanResult(
                        True,
                        "goal tolerance reached",
                        states,
                        expansions,
                        time.perf_counter() - started,
                        analytic_expansions,
                        min_clearance_m=min_clearance,
                    )

            directions = (node.direction, -node.direction)
            for direction in directions:
                for steer_index in self._successor_steer_indices(node, int(direction)):
                    steer = self.steers[steer_index]
                    child = self._simulate(node, int(direction), float(steer), int(steer_index))
                    if child is None:
                        continue
                    child.parent = node_id
                    child_key = self._state_key(child)
                    if child.cost + 1e-8 >= best_cost.get(child_key, float("inf")):
                        continue
                    best_cost[child_key] = child.cost
                    nodes.append(child)
                    child_id = len(nodes) - 1
                    counter += 1
                    h = self._heuristic(child, goal, holonomic)
                    priority = (
                        child.cost
                        + self.config.heuristic_weight * h
                        + self._guide_penalty(child.x, child.y)
                    )
                    heapq.heappush(heap, (priority, counter, child_id))

        if incumbent_states is not None:
            return PlanResult(
                True,
                "best analytic Reeds-Shepp connection before search limit",
                incumbent_states,
                expansions,
                time.perf_counter() - started,
                analytic_expansions,
                min_clearance_m=incumbent_clearance,
            )
        if relaxed_states is not None:
            return PlanResult(
                True,
                "best near-goal relaxed Reeds-Shepp connector before search limit",
                relaxed_states,
                expansions,
                time.perf_counter() - started,
                analytic_expansions,
                min_clearance_m=relaxed_clearance,
            )
        partial = self._reconstruct(nodes, closest_id)
        reason = "time limit" if time.perf_counter() - started >= self.config.max_time_s else "search exhausted"
        return PlanResult(
            False,
            "{}; closest heuristic {:.2f} m".format(reason, closest_h),
            partial,
            expansions,
            time.perf_counter() - started,
            analytic_expansions,
        )

    def _reverse_dock_candidate(
        self,
        goal: Sequence[float],
        distance_m: float,
    ) -> Tuple[Tuple[float, float, float], np.ndarray, float]:
        """Build an exact zero-curvature reverse segment from an aligned approach pose."""
        gx, gy, gyaw = float(goal[0]), float(goal[1]), wrap_pi(float(goal[2]))
        distance_m = float(distance_m)
        approach = (
            gx + distance_m * math.cos(gyaw),
            gy + distance_m * math.sin(gyaw),
            gyaw,
        )
        steps = max(1, int(math.ceil(distance_m / self.config.integration_step)))
        rows = []
        for index in range(steps + 1):
            progress = distance_m * float(index) / float(steps)
            rows.append(
                (
                    approach[0] - progress * math.cos(gyaw),
                    approach[1] - progress * math.sin(gyaw),
                    gyaw,
                    -1,
                    0.0,
                )
            )
        segment = np.asarray(rows, dtype=np.float64)
        segment[-1, :3] = (gx, gy, gyaw)
        clear, min_clearance = self.terminal_collision.check_poses(
            segment[:, 0],
            segment[:, 1],
            segment[:, 2],
        )
        if not clear:
            return approach, np.empty((0, 5), dtype=np.float64), float(min_clearance)
        return approach, segment, float(min_clearance)

    def plan_with_terminal_dock(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        start_direction: int = 1,
    ) -> PlanResult:
        """Plan to an aligned staging pose, then insert with a straight reverse motion.

        The exact goal is deliberately not given to the free-form search.  This
        prevents a short one-shot Reeds-Shepp solution from satisfying only the
        goal centre while arriving with steering or heading still changing.
        """
        if not self.config.terminal_dock_enabled:
            return self.plan(start, goal, start_direction=start_direction)

        started = time.perf_counter()
        candidates = []
        for distance_m in self.config.terminal_dock_distances_m:
            distance_m = float(distance_m)
            if distance_m + 1e-9 < float(self.config.terminal_dock_min_length_m):
                continue
            approach, dock_segment, dock_clearance = self._reverse_dock_candidate(goal, distance_m)
            candidates.append(
                {
                    "distance_m": distance_m,
                    "approach_pose": list(approach),
                    "dock_collision_free": bool(len(dock_segment)),
                    "dock_min_clearance_m": dock_clearance,
                    "segment": dock_segment,
                }
            )

        feasible = [item for item in candidates if item["dock_collision_free"]]
        feasible.sort(
            key=lambda item: (
                (
                    float(item["distance_m"])
                    if self.config.terminal_prefer_shortest_first
                    else -float(item["distance_m"])
                ),
                -float(item["dock_min_clearance_m"]),
            )
        )
        # Longest docking is preferred for alignment quality, but the shortest
        # valid staging pose is the most likely escape when an aisle obstacle
        # blocks the 2.4 m pose.  Alternate the two ends so it is tested second
        # rather than after several nearly identical long candidates.
        staging_order = []
        left = 0
        right = len(feasible) - 1
        while left <= right:
            staging_order.append(feasible[left])
            left += 1
            if left <= right:
                staging_order.append(feasible[right])
                right -= 1
        feasible = staging_order
        total_expansions = 0
        total_analytic = 0
        best_partial = np.empty((0, 5), dtype=np.float64)
        failure_messages = []
        candidate_attempts = []
        successful_candidates = []
        last_approach_validation = {}
        original_plan_time_s = max(0.05, float(self.config.max_time_s))
        total_multiplier = max(
            1.0,
            float(self.config.terminal_candidate_total_time_multiplier),
        )
        staging_total_time_s = (
            original_plan_time_s
            if len(feasible) <= 1
            else original_plan_time_s * total_multiplier
        )
        staging_deadline = started + staging_total_time_s

        for candidate_index, candidate in enumerate(feasible):
            remaining_time_s = staging_deadline - time.perf_counter()
            if remaining_time_s <= 0.05 and total_expansions > 0:
                break
            candidates_left = max(1, len(feasible) - candidate_index)
            reserve_per_candidate_s = min(
                float(self.config.terminal_candidate_min_time_s),
                max(0.10, remaining_time_s / float(candidates_left)),
            )
            if candidate_index == 0 and candidates_left > 1:
                desired_time_s = max(
                    reserve_per_candidate_s,
                    float(self.config.terminal_first_candidate_fraction)
                    * staging_total_time_s,
                )
                reserved_time_s = reserve_per_candidate_s * float(
                    candidates_left - 1
                )
                candidate_time_s = min(
                    desired_time_s,
                    max(reserve_per_candidate_s, remaining_time_s - reserved_time_s),
                )
            else:
                candidate_time_s = max(
                    reserve_per_candidate_s,
                    remaining_time_s / float(candidates_left),
                )
            candidate_time_s = max(
                0.10,
                min(original_plan_time_s, candidate_time_s),
            )
            approach = candidate["approach_pose"]
            self.config.max_time_s = float(candidate_time_s)
            try:
                approach_result = self.plan(
                    start,
                    approach,
                    start_direction=start_direction,
                )
            finally:
                self.config.max_time_s = original_plan_time_s
            last_approach_validation = dict(
                getattr(approach_result, "validation", {}) or {}
            )
            candidate_attempts.append(
                {
                    "distance_m": float(candidate["distance_m"]),
                    "time_budget_s": float(candidate_time_s),
                    "success": bool(approach_result.success),
                    "message": str(approach_result.message),
                }
            )
            total_expansions += int(approach_result.expansions)
            total_analytic += int(approach_result.analytic_expansions)
            if len(approach_result.states) > len(best_partial):
                best_partial = approach_result.states
            if not approach_result.success:
                failure_messages.append(
                    "{:.2f}m staging pose: {}".format(float(candidate["distance_m"]), approach_result.message)
                )
                continue

            dock_segment = np.asarray(candidate["segment"], dtype=np.float64)
            states = np.concatenate((approach_result.states, dock_segment[1:]), axis=0)
            approach_clear, approach_clearance = self.collision.check_poses(
                approach_result.states[:, 0],
                approach_result.states[:, 1],
                approach_result.states[:, 2],
            )
            dock_clear, dock_clearance = self.terminal_collision.check_poses(
                dock_segment[:, 0],
                dock_segment[:, 1],
                dock_segment[:, 2],
            )
            if not approach_clear or not dock_clear:
                failure_messages.append(
                    "combined approach or strict docking path is in collision"
                )
                continue
            min_clearance = min(
                float(approach_clearance),
                float(dock_clearance),
            )
            terminal_start_index = max(0, len(approach_result.states) - 1)
            terminal_info = {
                "enabled": True,
                "mode": "aligned_zero_steer_reverse",
                "direction": -1,
                "distance_m": float(candidate["distance_m"]),
                "approach_pose_rear_axle": list(map(float, approach)),
                "terminal_start_index": int(terminal_start_index),
                "candidate_count": len(candidates),
                "collision_free_candidate_count": len(feasible),
                "candidate_attempts": list(candidate_attempts),
                "staging_total_time_budget_s": float(
                    staging_total_time_s
                ),
                "dock_min_clearance_m": float(candidate["dock_min_clearance_m"]),
                "approach_collision_profile": (
                    "terminal_strict"
                    if self.terminal_collision is self.collision
                    else "separate_physical_approach"
                ),
                "terminal_collision_profile": "strict",
                "approach_planner_message": str(approach_result.message),
                "approach_planner_algorithm": str(
                    (
                        getattr(approach_result, "validation", {})
                        or {}
                    ).get("planner_algorithm", "hybrid_astar")
                ),
                "relaxed_analytic_connector": "relaxed Reeds-Shepp" in str(approach_result.message),
            }
            approach_algorithm = terminal_info[
                "approach_planner_algorithm"
            ]
            approach_label = {
                "hjba_star": "HJBA*",
                "la3t_star": "LA3T*",
            }.get(approach_algorithm, "Hybrid A*")
            candidate_result = PlanResult(
                True,
                "{} alignment plus constrained straight reverse docking ({})".format(
                    approach_label,
                    approach_result.message,
                ),
                states,
                total_expansions,
                time.perf_counter() - started,
                total_analytic,
                min_clearance_m=float(min_clearance),
                validation=dict(
                    getattr(approach_result, "validation", {}) or {}
                ),
                terminal_dock=terminal_info,
            )
            quality = path_quality_metrics(states, self.vehicle)
            continue_threshold = max(
                0,
                int(self.config.terminal_continue_after_gear_switches),
            )
            should_compare_remaining = bool(
                continue_threshold > 0
                and int(quality["gear_switches"]) >= continue_threshold
                and candidate_index + 1 < len(feasible)
                and staging_deadline - time.perf_counter() > 0.05
            )
            if not should_compare_remaining and not successful_candidates:
                candidate_result.terminal_dock[
                    "candidate_attempts"
                ] = list(candidate_attempts)
                candidate_result.terminal_dock[
                    "selection_reason"
                ] = "first executable terminal topology"
                return candidate_result
            successful_candidates.append(
                (
                    (
                        int(quality["gear_switches"]),
                        float(quality["path_length_m"]),
                        float(quality["steering_total_variation_deg"]),
                    ),
                    candidate_result,
                )
            )
            if not should_compare_remaining:
                _, selected_result = min(
                    successful_candidates,
                    key=lambda item: item[0],
                )
                selected_result.terminal_dock[
                    "candidate_attempts"
                ] = list(candidate_attempts)
                selected_result.terminal_dock[
                    "selection_reason"
                ] = "lowest-gear executable terminal topology"
                selected_result.terminal_dock[
                    "compared_successful_candidate_count"
                ] = len(successful_candidates)
                selected_result.elapsed_s = float(
                    time.perf_counter() - started
                )
                return selected_result

        if successful_candidates:
            _, selected_result = min(
                successful_candidates,
                key=lambda item: item[0],
            )
            selected_result.terminal_dock[
                "candidate_attempts"
            ] = list(candidate_attempts)
            selected_result.terminal_dock[
                "selection_reason"
            ] = "lowest-gear executable terminal topology"
            selected_result.terminal_dock[
                "compared_successful_candidate_count"
            ] = len(successful_candidates)
            selected_result.elapsed_s = float(
                time.perf_counter() - started
            )
            return selected_result

        reason = "no collision-free terminal reverse corridor" if not feasible else "no reachable terminal staging pose"
        if failure_messages:
            reason += "; " + " | ".join(failure_messages[:4])
        return PlanResult(
            False,
            reason,
            best_partial,
            total_expansions,
            time.perf_counter() - started,
            total_analytic,
            validation=last_approach_validation,
            terminal_dock={
                "enabled": True,
                "mode": "aligned_zero_steer_reverse",
                "candidate_count": len(candidates),
                "collision_free_candidate_count": len(feasible),
                "candidate_attempts": list(candidate_attempts),
                "staging_total_time_budget_s": float(
                    staging_total_time_s
                ),
            },
        )


def path_metrics(states: np.ndarray) -> dict:
    states = np.asarray(states, dtype=np.float64)
    if len(states) == 0:
        return {
            "path_length_m": 0.0,
            "forward_length_m": 0.0,
            "reverse_length_m": 0.0,
            "gear_switches": 0,
            "max_steer_deg": 0.0,
        }
    ds = np.zeros(len(states), dtype=np.float64)
    if len(states) > 1:
        ds[1:] = np.hypot(np.diff(states[:, 0]), np.diff(states[:, 1]))
    directions = np.where(states[:, 3] >= 0.0, 1, -1)
    return {
        "path_length_m": float(ds.sum()),
        "forward_length_m": float(ds[directions > 0].sum()),
        "reverse_length_m": float(ds[directions < 0].sum()),
        "gear_switches": int(np.count_nonzero(directions[1:] != directions[:-1])),
        "max_steer_deg": float(np.degrees(np.max(np.abs(states[:, 4]), initial=0.0))),
    }


def path_quality_metrics(states: np.ndarray, vehicle: Optional[VehicleConfig] = None) -> dict:
    """Measure path quality independently of the downstream controller."""
    rows = np.asarray(states, dtype=np.float64)
    if rows.ndim != 2 or len(rows) == 0 or rows.shape[1] < 5:
        return {
            "path_length_m": 0.0,
            "gear_switches": 0,
            "steering_total_variation_deg": 0.0,
            "max_same_gear_steer_jump_deg": 0.0,
            "steering_effort_m": 0.0,
            "direction_changes_per_10m": 0.0,
        }
    ds = np.zeros(len(rows), dtype=np.float64)
    if len(rows) > 1:
        ds[1:] = np.hypot(np.diff(rows[:, 0]), np.diff(rows[:, 1]))
    directions = np.where(rows[:, 3] >= 0.0, 1, -1)
    gear_switches = int(np.count_nonzero(directions[1:] != directions[:-1]))
    steer_delta = np.abs(np.diff(rows[:, 4])) if len(rows) > 1 else np.empty(0, dtype=np.float64)
    same_gear = directions[1:] == directions[:-1]
    moving_delta = steer_delta[same_gear]
    max_steer = float(vehicle.max_steer) if vehicle is not None else max(
        1e-9, float(np.max(np.abs(rows[:, 4]), initial=0.0))
    )
    length = float(ds.sum())
    return {
        "path_length_m": length,
        "gear_switches": gear_switches,
        "steering_total_variation_deg": float(math.degrees(float(moving_delta.sum()))),
        "max_same_gear_steer_jump_deg": float(math.degrees(float(moving_delta.max(initial=0.0)))),
        "steering_effort_m": float(np.sum(ds * np.abs(rows[:, 4]) / max(1e-9, max_steer))),
        "direction_changes_per_10m": float(10.0 * gear_switches / max(1e-6, length)),
    }


def validate_plan(
    result: PlanResult,
    frame: BEVFrame,
    vehicle: VehicleConfig,
    planner_config: PlannerConfig,
    collision: CollisionChecker,
) -> dict:
    states = np.asarray(result.states, dtype=np.float64)
    validation = {
        "valid": False,
        "planner_success": bool(result.success),
        "target_candidate_valid": bool(frame.target_valid),
        "collision_free": False,
        "vehicle_inside_target": False,
        "target_free_overlap_ratio": 0.0,
        "target_obstacle_overlap_pixels": 0,
        "terminal_position_error_m": float("inf"),
        "terminal_yaw_error_deg": float("inf"),
        "terminal_dock_valid": False,
        "terminal_reverse_straight_length_m": 0.0,
        "terminal_max_yaw_error_deg": float("inf"),
        "terminal_max_lateral_error_m": float("inf"),
        "terminal_max_abs_steer_deg": float("inf"),
        "terminal_final_direction": 0,
        "final_slot_min_edge_clearance_m": 0.0,
        "min_clearance_m": float("nan"),
        "max_kinematic_residual_rad": float("inf"),
        "max_sample_spacing_m": float("inf"),
        "failure_reasons": [],
    }
    reasons: List[str] = []
    if not result.success:
        reasons.append("planner did not reach the goal")
    if not frame.target_valid:
        reasons.append("target_candidate_valid is false")
    target_mask = frame.maps[6] > 0
    free_mask = frame.maps[5] > 0
    target_pixels = int(target_mask.sum())
    free_overlap = int(np.logical_and(target_mask, free_mask).sum())
    free_ratio = free_overlap / float(max(1, target_pixels))
    target_obstacle_overlap = int(np.logical_and(target_mask, frame.obstacle_mask).sum())
    validation["target_free_overlap_ratio"] = float(free_ratio)
    validation["target_obstacle_overlap_pixels"] = target_obstacle_overlap
    if target_pixels == 0:
        reasons.append("target_goal_seg is empty")
    elif free_ratio < 0.50:
        reasons.append("target goal is not supported by free_parking_poly")
    if len(states) == 0:
        reasons.append("empty trajectory")
        validation["failure_reasons"] = reasons
        return validation

    collision_free, min_clearance = collision.check_poses(states[:, 0], states[:, 1], states[:, 2])
    validation["collision_free"] = bool(collision_free)
    validation["min_clearance_m"] = float(min_clearance)
    if not collision_free:
        reasons.append("dense vehicle footprint collides with the BEV obstacle mask or boundary")

    end = states[-1]
    center_x, center_y, center_yaw = vehicle.rear_to_center(float(end[0]), float(end[1]), float(end[2]))
    position_error = math.hypot(center_x - frame.goal_center[0], center_y - frame.goal_center[1])
    yaw_error = angle_error(center_yaw, frame.goal_yaw)
    validation["terminal_position_error_m"] = float(position_error)
    validation["terminal_yaw_error_deg"] = float(math.degrees(yaw_error))
    if position_error > max(0.25, planner_config.goal_xy_tolerance + 0.05):
        reasons.append("terminal position error exceeds tolerance")
    if yaw_error > math.radians(max(5.0, planner_config.goal_yaw_tolerance_deg + 1.0)):
        reasons.append("terminal yaw error exceeds tolerance")

    final_corners = vehicle.corners_from_rear(float(end[0]), float(end[1]), float(end[2]), margin=0.0)
    inside = all(point_in_polygon(corner, frame.goal_polygon, tolerance=0.06) for corner in final_corners)
    goal_rear = vehicle.center_to_rear(frame.goal_center[0], frame.goal_center[1], frame.goal_yaw)
    goal_vehicle_corners = vehicle.corners_from_rear(*goal_rear, margin=0.0)
    validation["vehicle_inside_target"] = bool(inside)
    validation["final_vehicle_corners"] = final_corners
    validation["goal_vehicle_corners"] = goal_vehicle_corners
    validation["final_slot_min_edge_clearance_m"] = float(
        min((point_to_polygon_edge_distance(corner, frame.goal_polygon) for corner in final_corners), default=0.0)
    )
    if not inside:
        reasons.append("full final vehicle footprint is not inside the target slot")

    if planner_config.terminal_dock_enabled:
        dock = result.terminal_dock or {}
        terminal_start = int(dock.get("terminal_start_index", -1))
        dock_mode_ok = dock.get("mode") == "aligned_zero_steer_reverse"
        segment_ok = 0 <= terminal_start < len(states) - 1
        if dock_mode_ok and segment_ok:
            terminal_states = states[terminal_start + 1 :]
            terminal_spacing = np.hypot(
                np.diff(states[terminal_start:, 0]),
                np.diff(states[terminal_start:, 1]),
            )
            terminal_length = float(terminal_spacing.sum())
            terminal_yaw_error = float(
                np.degrees(
                    max((angle_error(float(yaw), frame.goal_yaw) for yaw in terminal_states[:, 2]), default=float("inf"))
                )
            )
            normal_x = -math.sin(frame.goal_yaw)
            normal_y = math.cos(frame.goal_yaw)
            lateral = np.abs(
                (terminal_states[:, 0] - goal_rear[0]) * normal_x
                + (terminal_states[:, 1] - goal_rear[1]) * normal_y
            )
            terminal_lateral_error = float(lateral.max(initial=0.0))
            terminal_steer = float(np.degrees(np.abs(terminal_states[:, 4]).max(initial=0.0)))
            terminal_directions_ok = bool(np.all(terminal_states[:, 3] < 0.0))
            validation["terminal_reverse_straight_length_m"] = terminal_length
            validation["terminal_max_yaw_error_deg"] = terminal_yaw_error
            validation["terminal_max_lateral_error_m"] = terminal_lateral_error
            validation["terminal_max_abs_steer_deg"] = terminal_steer
            validation["terminal_final_direction"] = -1 if end[3] < 0.0 else 1
            dock_valid = (
                terminal_directions_ok
                and terminal_length + 1e-6 >= planner_config.terminal_dock_min_length_m
                and terminal_yaw_error <= planner_config.terminal_dock_yaw_tolerance_deg + 1e-6
                and terminal_lateral_error <= planner_config.terminal_dock_lateral_tolerance_m + 1e-6
                and terminal_steer <= planner_config.terminal_dock_steer_tolerance_deg + 1e-6
            )
            validation["terminal_dock_valid"] = bool(dock_valid)
        if not validation["terminal_dock_valid"]:
            reasons.append("terminal motion is not a sufficiently long aligned zero-steer reverse docking segment")

    if len(states) > 1:
        spacing = np.hypot(np.diff(states[:, 0]), np.diff(states[:, 1]))
        yaw_delta = np.asarray([wrap_pi(value) for value in np.diff(states[:, 2])], dtype=np.float64)
        signed_distance = spacing * np.where(states[1:, 3] >= 0.0, 1.0, -1.0)
        predicted = signed_distance * np.tan(states[1:, 4]) / float(vehicle.wheel_base)
        residual = np.abs(np.asarray([wrap_pi(value) for value in yaw_delta - predicted]))
        max_residual = float(residual.max(initial=0.0))
        max_spacing = float(spacing.max(initial=0.0))
    else:
        max_residual = 0.0
        max_spacing = 0.0
    validation["max_kinematic_residual_rad"] = max_residual
    validation["max_sample_spacing_m"] = max_spacing
    if max_residual > 2e-3:
        reasons.append("trajectory does not reproduce the bicycle-model integration")
    if max_spacing > planner_config.integration_step * 1.25 + 1e-6:
        reasons.append("trajectory sampling is too sparse for continuous collision validation")
    if np.max(np.abs(states[:, 4]), initial=0.0) > vehicle.max_steer + 1e-8:
        reasons.append("steering exceeds vehicle limit")

    validation["failure_reasons"] = reasons
    validation["valid"] = not reasons
    return validation


def build_trajectory_records(states: np.ndarray, vehicle: VehicleConfig) -> List[dict]:
    states = np.asarray(states, dtype=np.float64)
    count = len(states)
    if count == 0:
        return []
    spacing = np.zeros(count, dtype=np.float64)
    if count > 1:
        spacing[1:] = np.hypot(np.diff(states[:, 0]), np.diff(states[:, 1]))
    directions = np.where(states[:, 3] >= 0.0, 1, -1)
    curvature = np.tan(states[:, 4]) / float(vehicle.wheel_base)
    software_speed_limit_kmh = np.where(
        directions > 0,
        float(vehicle.forward_speed_limit_kmh),
        float(vehicle.reverse_speed_limit_kmh),
    )
    curvature_speed_limit = software_speed_limit_kmh / 3.6 / (1.0 + 2.5 * np.abs(curvature))
    steering_speed_limit = np.asarray(
        [
            vehicle.max_speed_for_bicycle_steer_kmh(float(steer), int(direction)) / 3.6
            for steer, direction in zip(states[:, 4], directions)
        ],
        dtype=np.float64,
    )
    speed_limit = np.minimum(curvature_speed_limit, steering_speed_limit)
    speed = speed_limit.copy()

    boundaries = [0]
    boundaries.extend((np.nonzero(directions[1:] != directions[:-1])[0] + 1).tolist())
    boundaries.append(count)
    acceleration = 1.0
    deceleration = 1.3
    for begin, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= begin:
            continue
        speed[begin] = 0.0
        speed[end - 1] = 0.0
        for index in range(begin + 1, end):
            reachable = math.sqrt(max(0.0, speed[index - 1] ** 2 + 2.0 * acceleration * spacing[index]))
            speed[index] = min(speed[index], reachable)
        for index in range(end - 2, begin - 1, -1):
            reachable = math.sqrt(max(0.0, speed[index + 1] ** 2 + 2.0 * deceleration * spacing[index + 1]))
            speed[index] = min(speed[index], reachable)

    records = []
    elapsed = 0.0
    distance = 0.0
    for index, state in enumerate(states):
        if index > 0:
            distance += float(spacing[index])
            average_speed = 0.5 * (float(speed[index - 1]) + float(speed[index]))
            elapsed += float(spacing[index]) / max(0.20, average_speed)
        center_x, center_y, _ = vehicle.rear_to_center(float(state[0]), float(state[1]), float(state[2]))
        speed_kmh = abs(float(speed[index])) * 3.6
        steering_scale = vehicle.steering_scale_at_speed(speed_kmh)
        inner_steer, outer_steer = vehicle.ackermann_wheel_angles(float(state[4]))
        carla_steer = vehicle.carla_steer_command(float(state[4]), speed_kmh)
        required_carla_steer = vehicle.required_carla_steer_command(float(state[4]), speed_kmh)
        records.append(
            {
                "index": int(index),
                "time_s": float(elapsed),
                "distance_m": float(distance),
                "rear_x_m": float(state[0]),
                "rear_y_m": float(state[1]),
                "center_x_m": float(center_x),
                "center_y_m": float(center_y),
                "yaw_rad": float(wrap_pi(state[2])),
                "yaw_deg": float(math.degrees(wrap_pi(state[2]))),
                "direction": int(directions[index]),
                "gear": "F" if directions[index] > 0 else "R",
                "speed_mps": float(directions[index] * speed[index]),
                "speed_kmh": float(directions[index] * speed_kmh),
                "software_speed_limit_kmh": float(software_speed_limit_kmh[index]),
                "steer_rad": float(state[4]),
                "steer_deg": float(math.degrees(state[4])),
                "bicycle_steer_deg": float(math.degrees(state[4])),
                "inner_front_wheel_steer_deg": float(math.degrees(inner_steer)),
                "outer_front_wheel_steer_deg": float(math.degrees(outer_steer)),
                "steering_scale": float(steering_scale),
                "carla_steer_command": float(carla_steer),
                "required_carla_steer_command": float(required_carla_steer),
                "steer_normalized": float(carla_steer),
                "curvature_1pm": float(curvature[index]),
                "simulation_tick_20hz": int(round(elapsed * float(vehicle.simulation_hz))),
            }
        )
    return records


def save_trajectory_csv(path: Path, records: Sequence[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def bev_to_canvas(x: float, y: float, origin: Tuple[int, int], size: int) -> Tuple[int, int]:
    u = origin[0] + int(round((float(y) - BEV_Y_MIN) / (BEV_Y_MAX - BEV_Y_MIN) * (size - 1)))
    v = origin[1] + int(round((BEV_X_MAX - float(x)) / (BEV_X_MAX - BEV_X_MIN) * (size - 1)))
    return u, v


CHANNEL_COLORS = (
    (245, 196, 55),
    (255, 225, 105),
    (224, 68, 68),
    (255, 115, 70),
    (58, 116, 72),
    (63, 190, 94),
    (38, 175, 205),
    (95, 225, 245),
)


def fused_bev_image(frame: BEVFrame) -> Image.Image:
    maps = frame.maps > 0
    rgb = np.full((BEV_SIZE, BEV_SIZE, 3), (22, 25, 29), dtype=np.uint8)
    rgb[maps[4]] = (38, 70, 46)
    rgb[maps[5]] = (45, 125, 65)
    obstacle = frame.obstacle_mask
    rgb[obstacle] = (165, 53, 53)
    rgb[maps[3] & obstacle] = (215, 72, 55)
    rgb[maps[6]] = (30, 125, 150)
    rgb[maps[7]] = (78, 220, 240)
    rgb[maps[0]] = (225, 175, 40)
    rgb[maps[1]] = (255, 225, 90)
    return Image.fromarray(rgb, mode="RGB")


def draw_dashed_polygon(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Tuple[int, int]],
    fill: Tuple[int, int, int],
    width: int = 2,
    dash: float = 10.0,
) -> None:
    closed = list(points) + [points[0]]
    for begin, end in zip(closed, closed[1:]):
        dx = float(end[0] - begin[0])
        dy = float(end[1] - begin[1])
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            continue
        cursor = 0.0
        while cursor < length:
            stop = min(length, cursor + dash)
            p0 = (begin[0] + dx * cursor / length, begin[1] + dy * cursor / length)
            p1 = (begin[0] + dx * stop / length, begin[1] + dy * stop / length)
            draw.line((p0, p1), fill=fill, width=width)
            cursor += 2.0 * dash


def render_plan_image(
    frame: BEVFrame,
    result: PlanResult,
    validation: dict,
    vehicle: VehicleConfig,
    planner_config: PlannerConfig,
    output_path: Path,
) -> None:
    canvas_w, canvas_h = 1510, 1040
    map_origin = (24, 104)
    map_size = 900
    canvas = Image.new("RGB", (canvas_w, canvas_h), (17, 19, 22))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    large_bev = fused_bev_image(frame).resize((map_size, map_size), Image.Resampling.NEAREST)
    canvas.paste(large_bev, map_origin)

    for meter in range(-10, 11):
        color = (78, 82, 88) if meter % 5 == 0 else (48, 52, 57)
        width = 2 if meter % 5 == 0 else 1
        u0, _ = bev_to_canvas(0.0, meter, map_origin, map_size)
        _, v0 = bev_to_canvas(meter, 0.0, map_origin, map_size)
        draw.line((u0, map_origin[1], u0, map_origin[1] + map_size - 1), fill=color, width=width)
        draw.line((map_origin[0], v0, map_origin[0] + map_size - 1, v0), fill=color, width=width)

    target_pixels = [bev_to_canvas(x, y, map_origin, map_size) for x, y in frame.goal_polygon]
    draw.line(target_pixels + [target_pixels[0]], fill=(110, 242, 255), width=4)
    goal_center = bev_to_canvas(frame.goal_center[0], frame.goal_center[1], map_origin, map_size)
    goal_arrow = bev_to_canvas(
        frame.goal_center[0] + 1.2 * math.cos(frame.goal_yaw),
        frame.goal_center[1] + 1.2 * math.sin(frame.goal_yaw),
        map_origin,
        map_size,
    )
    draw.line((goal_center, goal_arrow), fill=(110, 242, 255), width=5)

    states = np.asarray(result.states, dtype=np.float64)
    center_points: List[Tuple[float, float, float, int]] = []
    for state in states:
        cx, cy, cyaw = vehicle.rear_to_center(float(state[0]), float(state[1]), float(state[2]))
        center_points.append((cx, cy, cyaw, 1 if state[3] >= 0.0 else -1))
    if len(center_points) > 1:
        for previous, current in zip(center_points, center_points[1:]):
            color = (75, 170, 255) if current[3] > 0 else (255, 165, 55)
            draw.line(
                (bev_to_canvas(previous[0], previous[1], map_origin, map_size), bev_to_canvas(current[0], current[1], map_origin, map_size)),
                fill=color,
                width=7,
            )

        accumulated = 0.0
        last_footprint = -10.0
        for index, point in enumerate(center_points):
            if index > 0:
                accumulated += math.hypot(point[0] - center_points[index - 1][0], point[1] - center_points[index - 1][1])
            is_cusp = index > 0 and point[3] != center_points[index - 1][3]
            should_draw = index in (0, len(center_points) - 1) or is_cusp or accumulated - last_footprint >= 1.25
            if not should_draw:
                continue
            last_footprint = accumulated
            corners = rectangle_polygon((point[0], point[1]), point[2], vehicle.length, vehicle.width)
            pixels = [bev_to_canvas(x, y, map_origin, map_size) for x, y in corners]
            color = (110, 195, 255) if point[3] > 0 else (255, 195, 90)
            draw.line(pixels + [pixels[0]], fill=color, width=2)
            if is_cusp:
                px = bev_to_canvas(point[0], point[1], map_origin, map_size)
                draw.ellipse((px[0] - 7, px[1] - 7, px[0] + 7, px[1] + 7), fill=(240, 80, 225), outline=(255, 255, 255), width=2)

    if center_points:
        start_pixel = bev_to_canvas(center_points[0][0], center_points[0][1], map_origin, map_size)
        end_pixel = bev_to_canvas(center_points[-1][0], center_points[-1][1], map_origin, map_size)
        draw.ellipse((start_pixel[0] - 8, start_pixel[1] - 8, start_pixel[0] + 8, start_pixel[1] + 8), fill=(85, 245, 105))
        draw.ellipse((end_pixel[0] - 8, end_pixel[1] - 8, end_pixel[0] + 8, end_pixel[1] + 8), fill=(255, 245, 105))

    goal_rear = vehicle.center_to_rear(frame.goal_center[0], frame.goal_center[1], frame.goal_yaw)
    goal_vehicle = vehicle.corners_from_rear(*goal_rear)
    goal_vehicle_pixels = [bev_to_canvas(x, y, map_origin, map_size) for x, y in goal_vehicle]
    draw_dashed_polygon(draw, goal_vehicle_pixels, fill=(250, 250, 250), width=3, dash=11.0)

    dock = result.terminal_dock or {}
    if dock.get("approach_pose_rear_axle"):
        approach_rear = dock["approach_pose_rear_axle"]
        approach_center = vehicle.rear_to_center(*approach_rear)
        approach_pixel = bev_to_canvas(approach_center[0], approach_center[1], map_origin, map_size)
        draw.ellipse(
            (approach_pixel[0] - 9, approach_pixel[1] - 9, approach_pixel[0] + 9, approach_pixel[1] + 9),
            fill=(215, 80, 235),
            outline=(255, 255, 255),
            width=2,
        )

    valid = bool(validation.get("valid"))
    status = "VALID" if valid else "INVALID / PARTIAL"
    status_color = (90, 245, 115) if valid else (255, 105, 90)
    draw.text((24, 18), "8-channel BEV Hybrid A* parking trajectory", fill=(245, 248, 250), font=font)
    draw.text(
        (24, 42),
        "{} | session={} side={} frame={} cache_index={}".format(status, frame.session_id, frame.side, frame.frame, frame.cache_index),
        fill=status_color,
        font=font,
    )
    metrics = path_metrics(states)
    draw.text(
        (24, 64),
        "goal=({:.2f},{:.2f},{:.1f}deg)  length={:.2f}m  F={:.2f}m R={:.2f}m  switches={}  clearance={:.3f}m".format(
            frame.goal_center[0],
            frame.goal_center[1],
            math.degrees(frame.goal_yaw),
            metrics["path_length_m"],
            metrics["forward_length_m"],
            metrics["reverse_length_m"],
            metrics["gear_switches"],
            float(validation.get("min_clearance_m", float("nan"))),
        ),
        fill=(215, 222, 230),
        font=font,
    )
    draw.text(
        (24, 84),
        "expansions={} analytic={} runtime={:.3f}s endpoint_error={:.3f}m/{:.3f}deg dock={:.2f}m".format(
            result.expansions,
            result.analytic_expansions,
            result.elapsed_s,
            float(validation.get("terminal_position_error_m", float("nan"))),
            float(validation.get("terminal_yaw_error_deg", float("nan"))),
            float(validation.get("terminal_reverse_straight_length_m", 0.0)),
        ),
        fill=(190, 200, 210),
        font=font,
    )

    right_x = 950
    thumb = 200
    cell_w = 270
    cell_h = 225
    for channel_index, name in enumerate(CHANNELS):
        row = channel_index // 2
        col = channel_index % 2
        x0 = right_x + col * cell_w
        y0 = 104 + row * cell_h
        color = CHANNEL_COLORS[channel_index]
        image = np.full((BEV_SIZE, BEV_SIZE, 3), (25, 28, 32), dtype=np.uint8)
        image[frame.maps[channel_index] > 0] = color
        panel = Image.fromarray(image, mode="RGB").resize((thumb, thumb), Image.Resampling.NEAREST)
        canvas.paste(panel, (x0, y0))
        draw.rectangle((x0, y0, x0 + thumb, y0 + thumb), outline=(100, 105, 112), width=1)
        draw.text((x0, y0 + thumb + 5), "{}: {}".format(channel_index, name), fill=(220, 225, 230), font=font)

    draw.text((950, 24), "Path: blue=forward, orange=reverse, magenta=staging/gear switch", fill=(215, 222, 230), font=font)
    draw.text((950, 46), "BEV: cyan=slot, dashed white=exact goal vehicle, yellow=ego", fill=(215, 222, 230), font=font)
    draw.text(
        (950, 68),
        "Tesla Model 3: WB={:.3f}m track={:.3f}m wheel_max={:.1f}deg bicycle_max={:.1f}deg Rmin={:.2f}m".format(
            vehicle.wheel_base,
            vehicle.front_track,
            vehicle.front_wheel_max_steer_deg,
            vehicle.max_steer_deg,
            vehicle.min_turning_radius,
        ),
        fill=(215, 222, 230),
        font=font,
    )
    if validation.get("failure_reasons"):
        draw.text((950, 86), "Reasons: " + "; ".join(validation["failure_reasons"]), fill=(255, 120, 105), font=font)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def make_contact_sheet(image_paths: Sequence[Path], output_path: Path) -> None:
    images = []
    for path in image_paths:
        with Image.open(path) as source:
            image = source.convert("RGB")
            image.thumbnail((720, 500), Image.Resampling.LANCZOS)
            images.append(image.copy())
    if not images:
        return
    columns = 2 if len(images) > 1 else 1
    rows = int(math.ceil(len(images) / float(columns)))
    cell_w, cell_h = 740, 520
    sheet = Image.new("RGB", (columns * cell_w, 46 + rows * cell_h), (18, 20, 23))
    draw = ImageDraw.Draw(sheet)
    draw.text((18, 16), "Hybrid A* on packed 8-channel CVT BEV", fill=(245, 245, 245), font=ImageFont.load_default())
    for index, image in enumerate(images):
        x = (index % columns) * cell_w + 10
        y = 46 + (index // columns) * cell_h + 10
        sheet.paste(image, (x, y))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def parse_indices(args: argparse.Namespace, cache: PackedBEVCache) -> List[int]:
    selected: List[int] = []
    for value in args.index or []:
        selected.append(int(value))
    if args.indices:
        selected.extend(int(item.strip()) for item in str(args.indices).split(",") if item.strip())
    if args.session_id:
        selected.append(cache.find_index(args.session_id, args.frame))
    if args.sample_count > 0:
        session_positions = np.linspace(0, len(cache.sessions) - 1, int(args.sample_count)).round().astype(int)
        for sid in session_positions:
            session = cache.sessions[int(sid)]
            local_frame = min(20, int(session["count"]) - 1)
            selected.append(int(session["offset"]) + local_frame)
    if not selected:
        # Fixed representative L/R frames verified against this cache.  Fall
        # back to quantiles if a different cache is supplied.
        representatives = [55370, 136480, 56045, 46876]
        selected = [value for value in representatives if value < len(cache)]
        if not selected:
            selected = np.linspace(0, len(cache) - 1, min(4, len(cache))).round().astype(int).tolist()
    output = []
    seen = set()
    for value in selected:
        if value in seen:
            continue
        if not 0 <= value < len(cache):
            raise IndexError("cache index out of range: {}".format(value))
        seen.add(value)
        output.append(value)
    return output


def run_frame(
    frame: BEVFrame,
    vehicle: VehicleConfig,
    planner_config: PlannerConfig,
    output_root: Path,
    start_direction_mode: str,
) -> dict:
    start_center = (float(vehicle.bbox_center_x), 0.0, 0.0)
    start = vehicle.center_to_rear(*start_center)
    goal = vehicle.center_to_rear(frame.goal_center[0], frame.goal_center[1], frame.goal_yaw)
    if start_direction_mode == "forward":
        start_direction = 1
    elif start_direction_mode == "reverse":
        start_direction = -1
    else:
        start_direction = -1 if frame.start_reverse else 1

    planner = HybridAStarPlanner(vehicle, planner_config, frame.obstacle_mask)
    if frame.target_valid:
        result = planner.plan_with_terminal_dock(start, goal, start_direction=start_direction)
    else:
        result = PlanResult(False, "invalid target candidate", np.empty((0, 5)), 0, 0.0, 0)
    validation = validate_plan(result, frame, vehicle, planner_config, planner.collision)
    records = build_trajectory_records(result.states, vehicle)
    max_required_command = max((abs(float(item["required_carla_steer_command"])) for item in records), default=0.0)
    max_inner_wheel_steer = max((abs(float(item["inner_front_wheel_steer_deg"])) for item in records), default=0.0)
    validation["max_required_carla_steer_command"] = float(max_required_command)
    validation["max_inner_front_wheel_steer_deg"] = float(max_inner_wheel_steer)
    validation["front_wheel_max_steer_deg"] = float(vehicle.front_wheel_max_steer_deg)
    if max_required_command > 1.0 + 1e-6:
        validation["failure_reasons"].append("speed-dependent CARLA steering command exceeds [-1,1]")
        validation["valid"] = False
    if max_inner_wheel_steer > vehicle.front_wheel_max_steer_deg + 1e-6:
        validation["failure_reasons"].append("Ackermann inner wheel exceeds CARLA max_steer_angle")
        validation["valid"] = False
    result.validation = validation
    metrics = path_metrics(result.states)

    name = "{}_{}_f{:06d}_i{:06d}".format(frame.side, frame.session_id, frame.frame, frame.cache_index)
    output_dir = Path(output_root) / name
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "trajectory.csv"
    image_path = output_dir / "bev_hybrid_astar.png"
    json_path = output_dir / "plan.json"
    save_trajectory_csv(csv_path, records)
    render_plan_image(frame, result, validation, vehicle, planner_config, image_path)

    payload = {
        "schema_version": 1,
        "source": {
            "cache_index": frame.cache_index,
            "session_index": frame.session_index,
            "session_id": frame.session_id,
            "side": frame.side,
            "frame": frame.frame,
            "session_path": str(frame.session_path),
            "channel_order": list(CHANNELS),
            "coordinate": {
                "frame": "ego",
                "x_axis": "forward",
                "y_axis": "right",
                "yaw_zero": "forward",
                "bounds_m": [BEV_X_MIN, BEV_X_MAX, BEV_Y_MIN, BEV_Y_MAX],
                "resolution_m": BEV_RESOLUTION,
            },
        },
        "goal": {
            "center_xy_m": list(frame.goal_center),
            "yaw_rad": frame.goal_yaw,
            "yaw_deg": math.degrees(frame.goal_yaw),
            "yaw_source": "parking_slot_long_axis_directed_by_recorded_final_yaw",
            "recorded_final_yaw_rad": frame.recorded_goal_yaw,
            "recorded_final_yaw_deg": math.degrees(frame.recorded_goal_yaw),
            "length_m": frame.goal_length,
            "width_m": frame.goal_width,
            "polygon_xy_m": frame.goal_polygon,
            "candidate_valid": frame.target_valid,
            "candidate_iou": frame.target_iou,
        },
        "start": {
            "center_pose": list(start_center),
            "rear_axle_pose": list(start),
            "initial_direction": start_direction,
            "recorded_speed_kmh": frame.speed_kmh,
        },
        "vehicle": {**asdict(vehicle), "derived": vehicle.derived_specs()},
        "planner": asdict(planner_config),
        "result": {
            "search_success": result.success,
            "trackable": bool(validation["valid"]),
            "message": result.message,
            "expansions": result.expansions,
            "analytic_expansions": result.analytic_expansions,
            "elapsed_s": result.elapsed_s,
            **metrics,
            "terminal_dock": result.terminal_dock,
        },
        "validation": validation,
        "artifacts": {
            "trajectory_csv": str(csv_path.resolve()),
            "visualization_png": str(image_path.resolve()),
        },
        "trajectory": records,
        "limitations": [
            "The packed 8-channel raster represents vehicles but has no wall, curb, post, pedestrian or unknown-space channel.",
            "Vehicle geometry and steering limits come from the documented 2026-07-12 CARLA Tesla Model 3 runtime specification, not from the BEV cache.",
        ],
    }
    write_json(json_path, payload)
    return {
        "cache_index": frame.cache_index,
        "session_id": frame.session_id,
        "side": frame.side,
        "frame": frame.frame,
        "valid": bool(validation["valid"]),
        "message": result.message,
        "elapsed_s": result.elapsed_s,
        "expansions": result.expansions,
        "path_length_m": metrics["path_length_m"],
        "gear_switches": metrics["gear_switches"],
        "min_clearance_m": validation["min_clearance_m"],
        "terminal_position_error_m": validation["terminal_position_error_m"],
        "terminal_yaw_error_deg": validation["terminal_yaw_error_deg"],
        "terminal_dock_valid": validation["terminal_dock_valid"],
        "terminal_reverse_straight_length_m": validation["terminal_reverse_straight_length_m"],
        "final_slot_min_edge_clearance_m": validation["final_slot_min_edge_clearance_m"],
        "image": str(image_path.resolve()),
        "plan": str(json_path.resolve()),
        "trajectory": str(csv_path.resolve()),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--index", type=int, action="append", default=[])
    parser.add_argument("--indices", default="", help="comma-separated packed-cache indices")
    parser.add_argument("--session-id", default="")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=0, help="sample evenly across cached sessions")
    parser.add_argument("--max-time-s", type=float, default=12.0)
    parser.add_argument("--max-expansions", type=int, default=80000)
    parser.add_argument("--wheel-base", type=float, default=None)
    parser.add_argument("--rear-axle-to-center", type=float, default=None)
    parser.add_argument("--rear-axle-lateral-offset", type=float, default=None)
    parser.add_argument("--front-track", type=float, default=None)
    parser.add_argument("--rear-track", type=float, default=None)
    parser.add_argument(
        "--front-wheel-max-steer-deg",
        "--max-steer-deg",
        dest="front_wheel_max_steer_deg",
        type=float,
        default=None,
    )
    parser.add_argument("--planning-steer-utilization", type=float, default=None)
    parser.add_argument("--safety-margin", type=float, default=0.10)
    parser.add_argument("--start-direction", choices=("auto", "forward", "reverse"), default="auto")
    parser.add_argument("--no-contact-sheet", action="store_true")
    parser.add_argument("--strict", action="store_true", help="return a nonzero code if any trajectory is invalid")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    cache = PackedBEVCache(args.cache_dir)
    indices = parse_indices(args, cache)
    vehicle = cache.vehicle_config
    if args.wheel_base is not None:
        vehicle.wheel_base = float(args.wheel_base)
    if args.rear_axle_to_center is not None:
        vehicle.rear_axle_to_center = float(args.rear_axle_to_center)
    if args.rear_axle_lateral_offset is not None:
        vehicle.rear_axle_lateral_offset = float(args.rear_axle_lateral_offset)
    if args.front_track is not None:
        vehicle.front_track = float(args.front_track)
    if args.rear_track is not None:
        vehicle.rear_track = float(args.rear_track)
    if args.front_wheel_max_steer_deg is not None:
        vehicle.front_wheel_max_steer_deg = float(args.front_wheel_max_steer_deg)
    if args.planning_steer_utilization is not None:
        vehicle.planning_steer_utilization = float(args.planning_steer_utilization)
    vehicle.rear_axle_x = float(vehicle.bbox_center_x) - float(vehicle.rear_axle_to_center)
    vehicle.front_axle_x = float(vehicle.rear_axle_x) + float(vehicle.wheel_base)
    vehicle.safety_margin = float(args.safety_margin)
    planner_config = PlannerConfig(max_time_s=float(args.max_time_s), max_expansions=int(args.max_expansions))
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for index in indices:
        frame = cache.load(index)
        row = run_frame(frame, vehicle, planner_config, args.out_dir, args.start_direction)
        summary_rows.append(row)
        print(
            "index={} session={} side={} frame={} valid={} runtime={:.3f}s length={:.2f}m clearance={:.3f}m image={}".format(
                row["cache_index"],
                row["session_id"],
                row["side"],
                row["frame"],
                row["valid"],
                row["elapsed_s"],
                row["path_length_m"],
                row["min_clearance_m"],
                row["image"],
            ),
            flush=True,
        )

    valid_count = sum(bool(row["valid"]) for row in summary_rows)
    summary = {
        "cache_dir": str(Path(args.cache_dir).resolve()),
        "cached_sessions": int(cache.metadata["total_sessions"]),
        "cached_frames": int(cache.metadata["total_frames"]),
        "raw_scenarios_note": "735 raw directories; R/20260710_205302 is excluded from the valid packed cache",
        "requested": len(summary_rows),
        "valid": valid_count,
        "invalid": len(summary_rows) - valid_count,
        "vehicle": {**asdict(vehicle), "derived": vehicle.derived_specs()},
        "planner": asdict(planner_config),
        "samples": summary_rows,
    }
    summary_path = args.out_dir / "summary.json"
    write_json(summary_path, summary)
    if not args.no_contact_sheet:
        make_contact_sheet([Path(row["image"]) for row in summary_rows], args.out_dir / "contact_sheet.png")
    print("summary={} valid={}/{}".format(summary_path.resolve(), valid_count, len(summary_rows)), flush=True)
    if args.strict and valid_count != len(summary_rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
