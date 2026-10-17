"""Raw-shard preparation and processed-shard PyTorch dataset."""

from __future__ import annotations

import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from .config import HopePathConfig
from .geometry import (
    anchor_terminal_pose_trace,
    clip_obstacle_polygons_to_raster_extent,
    decode_sdf,
    map_bounds_polygon,
    points_world_to_local,
    pose_trace_controls,
    pose_world_to_local,
    quantize_sdf,
    rasterize_pose_mask,
    rasterize_scene_base,
    resample_aligned_dense_primitives,
    resample_motion_primitives,
    select_route_nearest_polygons,
    states_world_to_local,
    wrap_angle,
)


PROCESSED_SCHEMA = "hope-path-processed-shard-v2"
BEV_CHANNELS = ("occupancy", "free", "signed_distance", "start", "goal")


def target_topology_modes(
    poses: np.ndarray,
    gears: np.ndarray,
    masks: np.ndarray,
    goals: np.ndarray,
) -> np.ndarray:
    """Classify GT paths into eight stable proposal-query families.

    The bits encode initial gear, simple/multi-switch motion, and the signed
    side of the largest deviation from the direct start-goal corridor.  This
    turns anonymous Best-of-K queries into repeatable route specialists.
    """

    count = len(poses)
    result = np.zeros(count, dtype=np.int64)
    for index in range(count):
        length = max(1, int(np.count_nonzero(masks[index])))
        active_gear = gears[index, :length]
        initial_forward = int(int(active_gear[0]) == 1)
        switches = int(np.count_nonzero(active_gear[1:] != active_gear[:-1]))
        multi_switch = int(switches >= 2)
        active_pose = poses[index, :length]
        goal = goals[index]
        cross = goal[0] * active_pose[:, 1] - goal[1] * active_pose[:, 0]
        detour_index = int(np.argmax(np.abs(cross)))
        left_detour = int(float(cross[detour_index]) >= 0.0)
        result[index] = initial_forward * 4 + multi_switch * 2 + left_detour
    return result


def split_for_shard(index: int) -> str:
    if index < 16:
        return "train"
    if index < 18:
        return "validation"
    return "test"


def discover_complete_raw_shards(root: str | Path) -> list[dict[str, object]]:
    """Only expose atomically complete collector shards.

    The active collector creates the NPZ at the very end and then writes
    ``summary.json``.  Requiring both files and validating successful_paths
    prevents preprocessing from touching an active shard.
    """

    root = Path(root)
    discovered: list[dict[str, object]] = []
    for directory in sorted(root.glob("shard_*")):
        if not directory.is_dir():
            continue
        try:
            shard_index = int(directory.name.rsplit("_", 1)[-1])
        except ValueError:
            continue
        summary_path = directory / "summary.json"
        if not summary_path.is_file():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8-sig"))
            successes = int(summary["successful_paths"])
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        data_path = directory / "hope_2d_paths_{}.npz".format(successes)
        if not data_path.is_file() or successes <= 0:
            continue
        discovered.append(
            {
                "index": shard_index,
                "name": directory.name,
                "successes": successes,
                "data": data_path,
                "summary": summary_path,
                "split": split_for_shard(shard_index),
            }
        )
    return discovered


def _raw_obstacles(raw: Mapping[str, np.ndarray], sample: int) -> list[np.ndarray]:
    sample_offsets = raw["obstacle_sample_offsets"]
    polygon_offsets = raw["obstacle_polygon_offsets"]
    vertices = raw["obstacle_vertices"]
    first_polygon = int(sample_offsets[sample])
    last_polygon = int(sample_offsets[sample + 1])
    return [
        np.asarray(
            vertices[int(polygon_offsets[index]):int(polygon_offsets[index + 1])],
            dtype=np.float32,
        )
        for index in range(first_polygon, last_polygon)
    ]


def _pack_polygons(
    polygons: Sequence[np.ndarray], goal: Sequence[float], config: HopePathConfig
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = config.data
    polygons = select_route_nearest_polygons(
        polygons, goal[:2], data.max_obstacles
    )
    values = np.zeros((data.max_obstacles, data.max_vertices, 2), dtype=np.float32)
    vertex_mask = np.zeros((data.max_obstacles, data.max_vertices), dtype=np.bool_)
    polygon_mask = np.zeros(data.max_obstacles, dtype=np.bool_)
    for polygon_index, polygon in enumerate(polygons[: data.max_obstacles]):
        count = min(len(polygon), data.max_vertices)
        if count < 2:
            continue
        values[polygon_index, :count] = polygon[:count]
        vertex_mask[polygon_index, :count] = True
        polygon_mask[polygon_index] = True
    return values, vertex_mask, polygon_mask


def _path_length(states: np.ndarray) -> float:
    if len(states) < 2:
        return 0.0
    return float(np.hypot(np.diff(states[:, 0]), np.diff(states[:, 1])).sum())


def _quality_weight(path_length: float, gear_switches: int) -> float:
    penalty = 0.025 * max(0, gear_switches - 3) + 0.015 * max(0.0, path_length - 20.0)
    return float(max(0.20, math.exp(-penalty)))


def _first_motion_direction(states: np.ndarray) -> int:
    """Return the first non-zero GT gear using the common state layout."""

    if states.ndim != 2 or states.shape[1] < 4:
        raise ValueError("path states do not contain a direction column")
    distance = np.hypot(np.diff(states[:, 0]), np.diff(states[:, 1]))
    moving = np.flatnonzero(distance > 1.0e-6)
    if len(moving):
        value = float(states[int(moving[0]) + 1, 3])
        if abs(value) > 1.0e-6:
            return 1 if value > 0.0 else -1
    for value in states[1:, 3]:
        if abs(float(value)) > 1.0e-6:
            return 1 if float(value) > 0.0 else -1
    raise ValueError("path has no non-zero motion direction")


def _atomic_savez(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def prepare_raw_shard(
    raw_path: str | Path,
    output_path: str | Path,
    config: HopePathConfig,
    *,
    max_samples: int | None = None,
) -> dict[str, object]:
    """Convert one immutable collector shard into a fixed training contract."""

    config.validate()
    raw_path, output_path = Path(raw_path), Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    accepted: dict[str, list[np.ndarray | int | float]] = {
        "source_index": [], "level_id": [], "quality_weight": [],
        "goal": [], "bev_base": [], "obstacle_vertices": [],
        "obstacle_vertex_mask": [], "obstacle_mask": [], "map_polygon": [], "target_pose": [],
        "target_curvature": [], "target_step": [], "target_gear": [], "target_mask": [],
        "target_length": [], "path_length_m": [], "gear_switches": [],
        "initial_direction": [],
    }
    if config.data.dense_substeps > 1:
        accepted.update({
            "dense_target_pose": [],
            "dense_target_steering": [],
            "dense_source_steering": [],
            "dense_target_gear": [],
            "dense_target_mask": [],
            "dense_target_length": [],
        })
    rejected: list[tuple[int, str]] = []
    rejection_counts: Counter[str] = Counter()

    with np.load(raw_path, allow_pickle=False) as raw:
        path_offsets = raw["path_offsets"]
        sample_count = min(len(path_offsets) - 1, max_samples or len(path_offsets) - 1)
        for sample in range(sample_count):
            reason = ""
            begin, end = int(path_offsets[sample]), int(path_offsets[sample + 1])
            states_world = np.asarray(raw["path_states"][begin:end], dtype=np.float32)
            start = np.asarray(raw["start_poses"][sample], dtype=np.float32)
            goal_world = np.asarray(raw["goal_poses"][sample], dtype=np.float32)
            if len(states_world) < 2 or not np.isfinite(states_world).all():
                reason = "invalid_path_array"
            local_states = states_world_to_local(states_world, start) if not reason else states_world
            goal = pose_world_to_local(goal_world, start)
            path_length = _path_length(local_states)
            gear_switches = int(raw["gear_switches"][sample])
            goal_position_error = float(np.linalg.norm(local_states[-1, :2] - goal[:2])) if not reason else math.inf
            goal_yaw_error = abs(float(wrap_angle(local_states[-1, 2] - goal[2]))) if not reason else math.inf
            if not reason and path_length > config.data.primitive_step_m * (config.data.primitive_count - 1):
                reason = "path_too_long"
            if not reason and gear_switches > config.data.max_gear_switches:
                reason = "excessive_gear_switches"
            if not reason and goal_position_error > config.data.goal_position_tolerance_m:
                reason = "terminal_position_error"
            if not reason and math.degrees(goal_yaw_error) > config.data.goal_yaw_tolerance_deg:
                reason = "terminal_yaw_error"
            if not reason and np.max(np.abs(local_states[:, 4])) > config.vehicle.max_steer_rad + 1.0e-3:
                reason = "steering_limit"
            try:
                primitives = resample_motion_primitives(
                    local_states, config.data.primitive_step_m, config.vehicle.wheelbase_m
                ) if not reason else None
            except ValueError:
                primitives = None
                reason = reason or "primitive_conversion"
            if primitives is not None and len(primitives["poses"]) > config.data.primitive_count:
                reason = "primitive_overflow"
            dense = None
            if not reason and config.data.dense_substeps > 1:
                try:
                    dense = resample_aligned_dense_primitives(
                        local_states,
                        config.data.primitive_step_m,
                        config.vehicle.wheelbase_m,
                        config.data.dense_substeps,
                    )
                except ValueError:
                    reason = "dense_primitive_conversion"
                if dense is not None:
                    expected_dense_length = (
                        (len(primitives["poses"]) - 1)
                        * config.data.dense_substeps
                        + 1
                    )
                    if len(dense["poses"]) != expected_dense_length:
                        reason = "dense_primitive_alignment"
                    elif len(dense["poses"]) > config.data.dense_primitive_count:
                        reason = "dense_primitive_overflow"
            if not reason and config.model.hard_goal_terminal_pose:
                try:
                    anchored_pose = anchor_terminal_pose_trace(
                        primitives["poses"],
                        goal,
                        config.model.terminal_anchor_steps,
                    )
                    anchored_controls = pose_trace_controls(
                        anchored_pose,
                        primitives["gear"],
                        config.vehicle.wheelbase_m,
                    )
                    if (
                        np.max(np.abs(anchored_controls["steering"]))
                        > config.vehicle.max_steer_rad + 1.0e-3
                    ):
                        reason = "terminal_anchor_steering_limit"
                    else:
                        primitives.update(anchored_controls)
                        path_length = _path_length(anchored_pose)
                    if dense is not None and not reason:
                        anchored_dense_pose = anchor_terminal_pose_trace(
                            dense["poses"],
                            goal,
                            (
                                config.model.terminal_anchor_steps
                                * config.data.dense_substeps
                            ),
                        )
                        anchored_dense_controls = pose_trace_controls(
                            anchored_dense_pose,
                            dense["gear"],
                            config.vehicle.wheelbase_m,
                        )
                        if (
                            np.max(np.abs(anchored_dense_controls["steering"]))
                            > config.vehicle.max_steer_rad + 1.0e-3
                        ):
                            reason = "terminal_anchor_steering_limit"
                        else:
                            dense.update(anchored_dense_controls)
                except ValueError:
                    reason = "terminal_anchor_conversion"
            if reason:
                rejected.append((sample, reason))
                rejection_counts[reason] += 1
                continue

            obstacles_world = _raw_obstacles(raw, sample)
            obstacles_local = clip_obstacle_polygons_to_raster_extent(
                [points_world_to_local(polygon, start) for polygon in obstacles_world],
                config.data,
            )
            bounds_world = map_bounds_polygon(raw["map_bounds"][sample])
            bounds_local = points_world_to_local(bounds_world, start)
            occupancy, sdf = rasterize_scene_base(bounds_local, obstacles_local, config.data)
            packed, vertex_mask, obstacle_mask = _pack_polygons(
                obstacles_local, goal, config
            )

            length = len(primitives["poses"])
            pose_target = np.zeros((config.data.primitive_count, 3), dtype=np.float32)
            curvature_target = np.zeros(config.data.primitive_count, dtype=np.float32)
            step_target = np.zeros(config.data.primitive_count, dtype=np.float32)
            gear_target = np.zeros(config.data.primitive_count, dtype=np.int64)
            path_mask = np.zeros(config.data.primitive_count, dtype=np.bool_)
            pose_target[:length] = primitives["poses"]
            curvature_target[:length] = primitives["curvature"]
            step_target[:length] = primitives["step"]
            gear_target[:length] = primitives["gear"]
            path_mask[:length] = True
            accepted["source_index"].append(sample)
            accepted["level_id"].append(int(raw["level_ids"][sample]))
            accepted["quality_weight"].append(_quality_weight(path_length, gear_switches))
            accepted["goal"].append(
                np.asarray((goal[0], goal[1], math.sin(goal[2]), math.cos(goal[2])), dtype=np.float32)
            )
            accepted["bev_base"].append(np.stack((occupancy, quantize_sdf(sdf, config.data.sdf_clip_m))))
            accepted["obstacle_vertices"].append(packed)
            accepted["obstacle_vertex_mask"].append(vertex_mask)
            accepted["obstacle_mask"].append(obstacle_mask)
            accepted["map_polygon"].append(bounds_local)
            accepted["target_pose"].append(pose_target)
            accepted["target_curvature"].append(curvature_target)
            accepted["target_step"].append(step_target)
            accepted["target_gear"].append(gear_target)
            accepted["target_mask"].append(path_mask)
            accepted["target_length"].append(length)
            accepted["path_length_m"].append(path_length)
            accepted["gear_switches"].append(gear_switches)
            accepted["initial_direction"].append(
                1 if int(primitives["gear"][0]) > 0 else -1
            )
            if dense is not None:
                dense_length = len(dense["poses"])
                dense_pose = np.zeros(
                    (config.data.dense_primitive_count, 3), dtype=np.float32
                )
                dense_steering = np.zeros(
                    config.data.dense_primitive_count, dtype=np.float32
                )
                dense_gear = np.zeros(
                    config.data.dense_primitive_count, dtype=np.int64
                )
                dense_mask = np.zeros(
                    config.data.dense_primitive_count, dtype=np.bool_
                )
                dense_pose[:dense_length] = dense["poses"]
                dense_steering[:dense_length] = dense["steering"]
                dense_gear[:dense_length] = dense["gear"]
                dense_mask[:dense_length] = True
                accepted["dense_target_pose"].append(dense_pose)
                accepted["dense_target_steering"].append(dense_steering)
                dense_source_steering = np.zeros(
                    config.data.dense_primitive_count, dtype=np.float32
                )
                dense_source_steering[:dense_length] = dense["source_steering"]
                accepted["dense_source_steering"].append(dense_source_steering)
                accepted["dense_target_gear"].append(dense_gear)
                accepted["dense_target_mask"].append(dense_mask)
                accepted["dense_target_length"].append(dense_length)

    if not accepted["source_index"]:
        raise RuntimeError("no accepted paths in {}".format(raw_path))
    arrays = {key: np.asarray(value) for key, value in accepted.items()}
    arrays["rejected_source_index"] = np.asarray([item[0] for item in rejected], dtype=np.int64)
    arrays["rejected_reason"] = np.asarray([item[1] for item in rejected], dtype="U40")
    _atomic_savez(output_path, **arrays)
    return {
        "schema": PROCESSED_SCHEMA,
        "raw": str(raw_path.resolve()),
        "output": str(output_path.resolve()),
        "considered": int(len(accepted["source_index"]) + len(rejected)),
        "accepted": int(len(accepted["source_index"])),
        "rejected": int(len(rejected)),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "bev_channels": list(BEV_CHANNELS),
    }


class ProcessedShardDataset:
    """Map-style dataset for one processed shard.

    A shard is deliberately loaded as a unit.  The trainer shuffles shard order
    and then sample order, avoiding random-access decompression across all 20
    large NPZ files.
    """

    def __init__(self, path: str | Path, config: HopePathConfig, mirror: bool = False):
        self.path = Path(path)
        self.config = config
        self.mirror = bool(mirror)
        with np.load(self.path, allow_pickle=False) as source:
            self.arrays = {key: source[key] for key in source.files if not key.startswith("rejected_")}
        self.topology_modes = target_topology_modes(
            self.arrays["target_pose"],
            self.arrays["target_gear"],
            self.arrays["target_mask"],
            self.arrays["goal"],
        )
        if (
            config.training.balance_initial_direction
            and "initial_direction" not in self.arrays
        ):
            raise RuntimeError(
                "direction-balanced training requires initial_direction in {}".format(
                    self.path
                )
            )
        dense_required = config.data.dense_substeps > 1
        dense_keys = {
            "dense_target_pose", "dense_target_steering", "dense_target_gear",
            "dense_target_mask", "dense_target_length",
        }
        if dense_required and not dense_keys.issubset(self.arrays):
            missing = ", ".join(sorted(dense_keys.difference(self.arrays)))
            raise RuntimeError(
                "dense training data is missing {} in {}".format(missing, self.path)
            )
        self._start_mask = rasterize_pose_mask((0.0, 0.0, 0.0), config.data, config.vehicle)

    def __len__(self) -> int:
        return int(len(self.arrays["source_index"]))

    def __getitem__(self, index: int) -> dict[str, object]:
        import torch
        base = self.arrays["bev_base"][index]
        occupancy = base[0].astype(np.float32)
        sdf = decode_sdf(base[1])
        goal_vector = self.arrays["goal"][index].astype(np.float32).copy()
        goal_yaw = math.atan2(float(goal_vector[2]), float(goal_vector[3]))
        goal_mask = rasterize_pose_mask(
            (goal_vector[0], goal_vector[1], goal_yaw), self.config.data, self.config.vehicle
        ).astype(np.float32) / 255.0
        bev = np.stack(
            (occupancy, 1.0 - occupancy, sdf,
             self._start_mask.astype(np.float32) / 255.0, goal_mask), axis=0
        )
        obstacle_vertices = self.arrays["obstacle_vertices"][index].astype(np.float32).copy()
        map_polygon = self.arrays["map_polygon"][index].astype(np.float32).copy()
        target_pose = self.arrays["target_pose"][index].astype(np.float32).copy()
        curvature = self.arrays["target_curvature"][index].astype(np.float32).copy()
        dense_pose = None
        dense_steering = None
        if "dense_target_pose" in self.arrays:
            dense_pose = self.arrays["dense_target_pose"][index].astype(np.float32).copy()
            dense_steering = self.arrays["dense_target_steering"][index].astype(np.float32).copy()
        mirrored = self.mirror and np.random.random() < 0.5
        if mirrored:
            # local_to_pixel maps lateral y to the image row (H) axis.  A
            # y -> -y reflection must therefore flip H, not the column/x axis.
            bev = bev[:, ::-1, :].copy()
            goal_vector[1] *= -1.0
            goal_vector[2] *= -1.0
            obstacle_vertices[..., 1] *= -1.0
            map_polygon[..., 1] *= -1.0
            target_pose[:, 1] *= -1.0
            target_pose[:, 2] *= -1.0
            curvature *= -1.0
            if dense_pose is not None and dense_steering is not None:
                dense_pose[:, 1] *= -1.0
                dense_pose[:, 2] *= -1.0
                dense_steering *= -1.0
        result = {
            "bev": torch.from_numpy(bev),
            "goal": torch.from_numpy(goal_vector),
            "obstacle_vertices": torch.from_numpy(obstacle_vertices),
            "obstacle_vertex_mask": torch.from_numpy(
                self.arrays["obstacle_vertex_mask"][index].astype(np.bool_)
            ),
            "obstacle_mask": torch.from_numpy(self.arrays["obstacle_mask"][index].astype(np.bool_)),
            "map_polygon": torch.from_numpy(map_polygon),
            "target_pose": torch.from_numpy(target_pose),
            "target_curvature": torch.from_numpy(curvature),
            "target_step": torch.from_numpy(self.arrays["target_step"][index].astype(np.float32)),
            "target_gear": torch.from_numpy(self.arrays["target_gear"][index].astype(np.int64)),
            "target_mask": torch.from_numpy(self.arrays["target_mask"][index].astype(np.bool_)),
            "target_length": torch.tensor(int(self.arrays["target_length"][index]), dtype=torch.long),
            "quality_weight": torch.tensor(float(self.arrays["quality_weight"][index])),
            "level_id": torch.tensor(int(self.arrays["level_id"][index]), dtype=torch.long),
            "source_index": torch.tensor(int(self.arrays["source_index"][index]), dtype=torch.long),
            # Reflection exchanges the left/right topology bit while keeping
            # initial gear and switch complexity unchanged.
            "target_topology_mode": torch.tensor(
                int(self.topology_modes[index]) ^ int(mirrored), dtype=torch.long
            ),
        }
        if "initial_direction" in self.arrays:
            result["start_direction"] = torch.tensor(
                float(self.arrays["initial_direction"][index]),
                dtype=torch.float32,
            )
        if dense_pose is not None and dense_steering is not None:
            result.update({
                "dense_target_pose": torch.from_numpy(dense_pose),
                "dense_target_steering": torch.from_numpy(dense_steering),
                "dense_target_gear": torch.from_numpy(
                    self.arrays["dense_target_gear"][index].astype(np.int64)
                ),
                "dense_target_mask": torch.from_numpy(
                    self.arrays["dense_target_mask"][index].astype(np.bool_)
                ),
                "dense_target_length": torch.tensor(
                    int(self.arrays["dense_target_length"][index]), dtype=torch.long
                ),
            })
        return result


def processed_manifest_entries(root: str | Path, split: str | None = None) -> list[dict[str, object]]:
    root = Path(root)
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("processed manifest not found: {}".format(manifest_path))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("schema") != PROCESSED_SCHEMA:
        raise RuntimeError("processed manifest is not schema {}".format(PROCESSED_SCHEMA))
    audit_path = root / "dataset_summary.json"
    if not audit_path.is_file():
        raise RuntimeError("dataset has not passed preflight validation: {}".format(audit_path))
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    totals = manifest.get("totals", {})
    expected = (
        int(totals.get("accepted", -1)),
        int(totals.get("rejected", -1)),
        len(manifest.get("shards", [])),
    )
    observed = (
        int(audit.get("accepted", -2)),
        int(audit.get("rejected", -2)),
        int(audit.get("shards", -2)),
    )
    if (
        not audit.get("valid", False)
        or audit.get("processed_schema") != PROCESSED_SCHEMA
        or observed != expected
    ):
        raise RuntimeError("dataset audit is missing, stale, or invalid; run validation again")
    exact_filter = manifest.get("exact_geometry_filter", {})
    if exact_filter.get("schema") != "hope-path-exact-geometry-filter-v1":
        raise RuntimeError("dataset has not passed the exact geometry filter")
    entries = list(manifest.get("shards", []))
    for entry in entries:
        candidate = Path(str(entry.get("data", "")))
        if not candidate.is_file() and not (root / candidate.name).is_file():
            raise FileNotFoundError("processed shard not found: {}".format(candidate))
    if split is not None:
        entries = [entry for entry in entries if entry.get("split") == split]
    return entries
