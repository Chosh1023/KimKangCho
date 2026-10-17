#!/usr/bin/env python
"""Materialize K=8 supervised-model paths for the real CARLA follower test.

The trained model runs in the Python/PyTorch environment, while CARLA 0.9.15
uses its Python 3.7 environment.  This script is the deterministic bridge: it
selects processed test scenes, generates four native plus four mirrored
candidates, applies the exact offline geometry selector, and writes the chosen
path in the same actor-local format as the collected Planner-3 GT.

The resulting directory is intentionally compatible with
``validate_collected_planner3_gt_carla.py``.  That validator then replays the
model path through the production HybridPathFollower and CARLA PhysX without
ever loading the saved GT path.
"""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from hope_path_transformer.dataset import (  # noqa: E402
    ProcessedShardDataset,
    processed_manifest_entries,
)
from hope_path_transformer.engine import move_to_device  # noqa: E402
from hope_path_transformer.inference import (  # noqa: E402
    generate_candidate_output,
    load_planner_checkpoint,
)
from hope_path_transformer.metrics import evaluate_inference_batch  # noqa: E402
from hope_path_transformer.worker_api import (  # noqa: E402
    LiveVehicleConfig,
    _candidate_control_metrics,
    _live_candidate_geometry,
    _select_runtime_candidate,
)


SCHEMA = "hope-path-k8-carla-eval-materialization-v1"


def wrap_pi(value: float) -> float:
    return (float(value) + math.pi) % (2.0 * math.pi) - math.pi


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def repair_sample_dir(raw_value: str) -> Path:
    """Repair old absolute paths whose Korean Desktop component was mangled."""
    candidate = Path(str(raw_value))
    if candidate.is_dir():
        return candidate.resolve()
    normalized = str(raw_value).replace("/", "\\")
    marker = "\\run-logs\\"
    position = normalized.lower().find(marker)
    if position >= 0:
        repaired = ROOT / Path(normalized[position + len(marker):].replace("\\", "/"))
        repaired = ROOT / "run-logs" / repaired.relative_to(ROOT)
        if repaired.is_dir():
            return repaired.resolve()
    raise FileNotFoundError("raw sample directory not found: {}".format(raw_value))


def load_dedup_rows(dataset_root: Path) -> dict[str, dict[str, Any]]:
    path = dataset_root / "dedup_kept.jsonl"
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["uid"])] = row
    return rows


def resolve_shard(dataset_root: Path, entry: dict[str, Any]) -> Path:
    candidate = Path(str(entry["data"]))
    if candidate.is_file():
        return candidate.resolve()
    local = dataset_root / candidate.name
    if local.is_file():
        return local.resolve()
    raise FileNotFoundError("processed shard not found: {}".format(candidate))


def family_from_uid(uid: str) -> str:
    # The two sources were appended in this order during preprocessing:
    # s00 = right-angle forward-start, s01 = general fixed-scene samples.
    return "right_angle_forward" if str(uid).startswith("s00-") else "general"


def enumerate_test_samples(
    dataset_root: Path,
    entries: list[dict[str, Any]],
    dedup: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for entry in entries:
        shard = resolve_shard(dataset_root, entry)
        with np.load(shard, allow_pickle=False) as source:
            uids = source["source_uid"]
            slots = source["target_slot_id"]
            modes = source["source_mode"]
            for index in range(len(uids)):
                uid = str(uids[index])
                if uid not in dedup:
                    raise RuntimeError("missing dedup provenance for {}".format(uid))
                records.append(
                    {
                        "uid": uid,
                        "shard": shard,
                        "index": int(index),
                        "target_slot_id": str(slots[index]),
                        "source_mode": str(modes[index]),
                        "family": family_from_uid(uid),
                        "dedup": dedup[uid],
                    }
                )
    return records


def choose_balanced(
    records: list[dict[str, Any]], count: int, seed: int
) -> list[dict[str, Any]]:
    """Balance source family first, then target slot, without replacement."""
    rng = random.Random(int(seed))
    families: dict[str, dict[str, deque[dict[str, Any]]]] = defaultdict(dict)
    for family in sorted({str(item["family"]) for item in records}):
        by_slot: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for item in records:
            if item["family"] == family:
                by_slot[str(item["target_slot_id"])].append(item)
        for slot, values in by_slot.items():
            rng.shuffle(values)
            families[family][slot] = deque(values)

    family_order = [name for name in ("right_angle_forward", "general") if name in families]
    family_order.extend(sorted(set(families) - set(family_order)))
    slot_cycles: dict[str, deque[str]] = {}
    for family in family_order:
        slots = sorted(families[family])
        rng.shuffle(slots)
        slot_cycles[family] = deque(slots)

    chosen: list[dict[str, Any]] = []
    active = deque(family_order)
    while active and len(chosen) < int(count):
        family = active.popleft()
        cycle = slot_cycles[family]
        selected = None
        for _ in range(len(cycle)):
            slot = cycle.popleft()
            queue = families[family][slot]
            if queue:
                selected = queue.popleft()
                if queue:
                    cycle.append(slot)
                break
        if selected is not None:
            chosen.append(selected)
        if cycle:
            active.append(family)
    if len(chosen) < int(count):
        raise RuntimeError("requested {}, only selected {}".format(count, len(chosen)))
    for ordinal, item in enumerate(chosen):
        item["evaluation_ordinal"] = int(ordinal)
    return chosen


def poses_local_to_actor(
    poses: np.ndarray, start_state: np.ndarray
) -> np.ndarray:
    start = np.asarray(start_state, dtype=np.float64).reshape(-1)
    yaw = float(start[2])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    values = np.asarray(poses, dtype=np.float64)
    output = np.empty_like(values)
    output[:, 0] = start[0] + cosine * values[:, 0] - sine * values[:, 1]
    output[:, 1] = start[1] + sine * values[:, 0] + cosine * values[:, 1]
    output[:, 2] = np.asarray([wrap_pi(yaw + value) for value in values[:, 2]])
    return output


def metadata_fields(sample_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    metadata = load_json(sample_dir / "metadata.json")
    scene = dict(metadata.get("scene") or {})
    collection = dict(metadata.get("collection") or {})
    origin = scene.get("ego_start_world_pose") or collection.get("ego_start_world_pose")
    if origin is None or len(origin) < 3:
        raise RuntimeError("sample metadata has no ego_start_world_pose: {}".format(sample_dir))
    return scene, collection


def select_like_live_inference(
    model_output: dict[str, torch.Tensor],
    batch: dict[str, object],
    items: list[dict[str, Any]],
    config: Any,
) -> tuple[torch.Tensor, list[dict[str, Any]], list[dict[str, torch.Tensor]]]:
    """Apply the deployed live-geometry/controller-friendly selector per scene."""
    selected_rows = []
    runtime_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, torch.Tensor]] = []
    weights = {
        "goal_error": 1.0,
        "steering_variation": 0.08,
        "gear_switches": 0.35,
        "path_length": 0.02,
        "clearance": 1.25,
    }
    vehicle = LiveVehicleConfig()
    for index, item in enumerate(items):
        sample_output = {
            key: value[index : index + 1] if isinstance(value, torch.Tensor) else value
            for key, value in model_output.items()
        }
        sample_batch = {
            key: value[index : index + 1] if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        }
        raw_sample = repair_sample_dir(str(item["dedup"]["sample_dir"]))
        with np.load(raw_sample / "planner_input.npz", allow_pickle=False) as source:
            safety = np.asarray(source["obstacle_safety"], dtype=np.uint8)
            start = np.asarray(source["start_state"], dtype=np.float64)
            boundary = np.asarray(source["parking_lot_polygon_xy"], dtype=np.float64)
            target_polygon = np.asarray(source["target_slot_polygon_xy"], dtype=np.float64)
        collision_free, live_clearance, final_inside = _live_candidate_geometry(
            sample_output,
            safety,
            start,
            boundary,
            vehicle,
            target_polygon,
        )
        geometry = evaluate_inference_batch(
            sample_output,
            sample_batch,
            config,
            collision_free_override=collision_free,
        )
        path_length, gear_switches, steering_variation = _candidate_control_metrics(
            sample_output, config.vehicle.wheelbase_m
        )
        selected, robust_success, cost = _select_runtime_candidate(
            fallback_selected=geometry["selected"],
            exact_success=geometry["candidate_success"],
            collision_free=collision_free,
            final_inside=final_inside,
            live_clearance_m=live_clearance,
            goal_position_error_m=geometry["candidate_goal_position_error_m"],
            goal_yaw_error_deg=geometry["candidate_goal_yaw_error_deg"],
            path_length_m=path_length,
            gear_switches=gear_switches,
            steering_variation_rad=steering_variation,
            minimum_clearance_m=0.25,
            goal_position_tolerance_m=config.data.goal_position_tolerance_m,
            goal_yaw_tolerance_deg=config.data.goal_yaw_tolerance_deg,
            weights=weights,
        )
        chosen = int(selected[0].item())
        sample_output["selected"] = selected
        selected_rows.append(selected)
        geometry_rows.append(geometry)
        runtime_rows.append(
            {
                "robust_success": bool(robust_success[0, chosen].item()),
                "robust_candidate_count": int(robust_success[0].sum().item()),
                "final_inside": bool(final_inside[0, chosen].item()),
                "final_inside_candidate_count": int(final_inside[0].sum().item()),
                "live_clearance_m": float(live_clearance[0, chosen].item()),
                "path_length_m": float(path_length[0, chosen].item()),
                "gear_switches": int(gear_switches[0, chosen].item()),
                "steering_variation_rad": float(
                    steering_variation[0, chosen].item()
                ),
                "selection_cost": float(cost[0, chosen].item()),
            }
        )
    return torch.cat(selected_rows, dim=0), runtime_rows, geometry_rows


def materialize_one(
    output_root: Path,
    item: dict[str, Any],
    model_output: dict[str, torch.Tensor],
    geometry: dict[str, torch.Tensor],
    batch_index: int,
    wheelbase_m: float,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = int(model_output["selected"][batch_index].item())
    length = int(model_output["lengths"][batch_index, selected].item())
    length = max(2, min(length, int(model_output["poses"].shape[2])))
    poses = model_output["poses"][batch_index, selected, :length].float().cpu().numpy()
    curvature = (
        model_output["curvature"][batch_index, selected, :length].float().cpu().numpy()
    )
    gear = model_output["gear"][batch_index, selected, :length].cpu().numpy()

    raw_sample = repair_sample_dir(str(item["dedup"]["sample_dir"]))
    with np.load(raw_sample / "planner_input.npz", allow_pickle=False) as source:
        start_state = np.asarray(source["start_state"], dtype=np.float64)
        target_polygon = np.asarray(source["target_slot_polygon_xy"], dtype=np.float64)
    actor_poses = poses_local_to_actor(poses, start_state)
    steer = np.arctan(curvature.astype(np.float64) * float(wheelbase_m))
    direction = np.where(gear > 0, 1.0, -1.0)
    path = np.column_stack((actor_poses, direction, steer)).astype(np.float64)
    if not np.all(np.isfinite(path)):
        raise RuntimeError("non-finite predicted path for {}".format(item["uid"]))
    if np.linalg.norm(path[0, :2] - start_state[:2]) > 1.0e-4:
        raise RuntimeError("predicted path origin mismatch for {}".format(item["uid"]))

    scene, collection = metadata_fields(raw_sample)
    origin = scene.get("ego_start_world_pose") or collection.get("ego_start_world_pose")
    ordinal = int(item["evaluation_ordinal"])
    relative = Path("samples") / "sample_{:05d}".format(ordinal)
    destination = output_root / relative
    destination.mkdir(parents=True, exist_ok=False)
    np.save(destination / "path.npy", path)
    np.savez_compressed(
        destination / "planner_input.npz",
        target_slot_polygon_xy=target_polygon.astype(np.float64),
        start_state=start_state,
    )

    selected_success = bool(
        geometry["candidate_success"][batch_index, selected].item()
    )
    candidate_success_count = int(geometry["candidate_success"][batch_index].sum().item())
    collision_free_count = int(
        geometry["candidate_collision_free"][batch_index].sum().item()
    )
    record = {
        "schema_version": SCHEMA,
        "accepted": True,
        "sample_relpath": relative.as_posix(),
        "ordinal": ordinal,
        "target_slot_id": str(item["target_slot_id"]),
        "start_position_sector": int(scene.get("start_position_sector", collection.get("start_position_sector", -1))),
        "start_yaw_sector": int(scene.get("start_yaw_sector", collection.get("start_yaw_sector", -1))),
        "ego_start_world_pose": [float(value) for value in origin[:3]],
        "planner_algorithm": "supervised_single_model_k8_exact_selector",
        "model_candidate_count": int(model_output["poses"].shape[1]),
        "selected_candidate": selected,
        "offline_selected_success": selected_success,
        "offline_candidate_success_count": candidate_success_count,
        "offline_collision_free_candidate_count": collision_free_count,
        "offline_no_collision_free_candidate": bool(
            geometry["no_collision_free_candidate"][batch_index].item()
        ),
        "offline_goal_position_error_m": float(
            geometry["candidate_goal_position_error_m"][batch_index, selected].item()
        ),
        "offline_goal_yaw_error_deg": float(
            geometry["candidate_goal_yaw_error_deg"][batch_index, selected].item()
        ),
        "offline_minimum_clearance_m": float(
            geometry["candidate_minimum_clearance_m"][batch_index, selected].item()
        ),
        "predicted_path_nodes": int(len(path)),
        "source_uid": str(item["uid"]),
        "source_family": str(item["family"]),
        "source_mode": str(item["source_mode"]),
        "source_sample_dir": str(raw_sample),
        "source_gt_path_used": False,
    }
    if runtime is not None:
        record.update(
            {
                "runtime_robust_success": bool(runtime["robust_success"]),
                "runtime_robust_candidate_count": int(runtime["robust_candidate_count"]),
                "runtime_final_vehicle_inside_slot": bool(runtime["final_inside"]),
                "runtime_final_inside_candidate_count": int(
                    runtime["final_inside_candidate_count"]
                ),
                "runtime_live_clearance_m": float(runtime["live_clearance_m"]),
                "runtime_path_length_m": float(runtime["path_length_m"]),
                "runtime_gear_switches": int(runtime["gear_switches"]),
                "runtime_steering_variation_rad": float(
                    runtime["steering_variation_rad"]
                ),
                "runtime_selection_cost": float(runtime["selection_cost"]),
            }
        )
    (destination / "metadata.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--selection-mode", choices=("live-runtime", "offline"), default="live-runtime"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.count < 1 or args.batch_size < 1:
        raise ValueError("count and batch-size must be positive")

    dataset_root = Path(args.dataset).expanduser().resolve()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    output_root = Path(args.output).expanduser().resolve()
    if output_root.exists():
        if not args.force:
            raise RuntimeError("output already exists; pass --force: {}".format(output_root))
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    entries = processed_manifest_entries(dataset_root, split="test")
    dedup = load_dedup_rows(dataset_root)
    available = enumerate_test_samples(dataset_root, entries, dedup)
    selected = choose_balanced(available, min(args.count, len(available)), args.seed)
    by_shard: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for item in selected:
        by_shard[Path(item["shard"])].append(item)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, config, checkpoint_payload = load_planner_checkpoint(
        checkpoint, device=device, use_ema=True
    )
    records: list[dict[str, Any]] = []
    print(
        "[K8 MATERIALIZE] available={} selected={} families={} device={}".format(
            len(available),
            len(selected),
            {name: sum(item["family"] == name for item in selected) for name in sorted({str(item["family"]) for item in selected})},
            device,
        ),
        flush=True,
    )
    completed = 0
    for shard in sorted(by_shard, key=str):
        dataset = ProcessedShardDataset(shard, config, mirror=False)
        items = sorted(by_shard[shard], key=lambda value: int(value["index"]))
        for offset in range(0, len(items), int(args.batch_size)):
            chunk = items[offset : offset + int(args.batch_size)]
            batch = default_collate([dataset[int(item["index"])] for item in chunk])
            batch = move_to_device(batch, device)
            model_output = generate_candidate_output(
                model,
                batch,
                candidate_count=4,
                mirror_tta=True,
            )
            if int(model_output["poses"].shape[1]) != 8:
                raise RuntimeError(
                    "K=8 contract violated: generated {} candidates".format(
                        int(model_output["poses"].shape[1])
                    )
                )
            runtime_rows: list[dict[str, Any] | None]
            if args.selection_mode == "live-runtime":
                selected_tensor, live_rows, geometry_rows = select_like_live_inference(
                    model_output, batch, chunk, config
                )
                model_output["selected"] = selected_tensor
                runtime_rows = list(live_rows)
            else:
                geometry_batch = evaluate_inference_batch(model_output, batch, config)
                model_output["selected"] = geometry_batch["selected"]
                geometry_rows = [
                    {
                        key: value[index : index + 1]
                        for key, value in geometry_batch.items()
                    }
                    for index in range(len(chunk))
                ]
                runtime_rows = [None] * len(chunk)
            for batch_index, item in enumerate(chunk):
                sample_output = {
                    key: value[batch_index : batch_index + 1]
                    if isinstance(value, torch.Tensor)
                    else value
                    for key, value in model_output.items()
                }
                records.append(
                    materialize_one(
                        output_root,
                        item,
                        sample_output,
                        geometry_rows[batch_index],
                        0,
                        config.vehicle.wheelbase_m,
                        runtime_rows[batch_index],
                    )
                )
                completed += 1
            print(
                "[K8 MATERIALIZE] {}/{} latest_shard={}".format(
                    completed, len(selected), shard.name
                ),
                flush=True,
            )

    records.sort(key=lambda value: int(value["ordinal"]))
    with (output_root / "manifest.jsonl").open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    offline_passed = sum(bool(record["offline_selected_success"]) for record in records)
    summary = {
        "schema_version": SCHEMA,
        "dataset": str(dataset_root),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "device": str(device),
        "candidate_count_native": 4,
        "mirror_tta": True,
        "candidate_count_total": 8,
        "selector": str(args.selection_mode),
        "seed": int(args.seed),
        "available_test_samples": int(len(available)),
        "materialized": int(len(records)),
        "offline_passed": int(offline_passed),
        "offline_failed": int(len(records) - offline_passed),
        "offline_success_rate": float(offline_passed / max(1, len(records))),
        "family_counts": {
            name: int(sum(record["source_family"] == name for record in records))
            for name in sorted({str(record["source_family"]) for record in records})
        },
        "source_gt_path_used": False,
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[K8 MATERIALIZE DONE] {}".format(json.dumps(summary, ensure_ascii=False)), flush=True)


if __name__ == "__main__":
    main()
