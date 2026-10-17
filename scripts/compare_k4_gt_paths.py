"""Paired saved-scene K4/Planner-3 path audit; never uses GT to select a candidate.

Uses the deployed strict-anchor profile, saved processed inputs, and the live
physical-raster geometry/selection functions. No CARLA server is required.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hope_path_transformer.dataset import ProcessedShardDataset, processed_manifest_entries
from hope_path_transformer.engine import move_to_device
from hope_path_transformer.geometry import states_world_to_local
from hope_path_transformer.inference import generate_candidate_output, load_planner_checkpoint
from hope_path_transformer.metrics import evaluate_inference_batch
from hope_path_transformer.model import anchor_terminal_pose_trace, controls_from_pose_trace, terminal_pose_error
from hope_path_transformer.worker_api import LiveVehicleConfig, _live_candidate_geometry, _select_runtime_candidate, _candidate_control_metrics
import hope_path_transformer.worker_api as runtime_geometry
from materialize_supervised_k8_carla_eval_paths import load_dedup_rows, repair_sample_dir, resolve_shard, sha256_file


INPUT_KEYS = ("bev", "goal", "start_direction", "obstacle_vertices", "obstacle_vertex_mask", "obstacle_mask", "map_polygon")
ORIGINAL_RASTER_CHECK = runtime_geometry._vectorized_exact_raster_path_clearance


def pruned_raster_check(checker, poses):
    """Exact broad-phase pruning for the unchanged production SAT checker.

    Cells outside each 64-pose bounding box expanded by an established upper
    bound cannot lower the minimum separating-axis distance. Preserve the
    production checker's global BEV check and first-colliding-block behavior.
    """
    rows = np.asarray(poses, dtype=np.float64).reshape(-1, 3)
    empty = copy.copy(checker)
    empty.obstacle_x = np.empty(0); empty.obstacle_y = np.empty(0)
    valid, minimum = ORIGINAL_RASTER_CHECK(empty, rows)
    if not valid or not len(checker.obstacle_x):
        return valid, minimum
    vehicle = checker.vehicle
    hl = .5 * vehicle.length + checker.footprint_margin
    hw = .5 * vehicle.width + checker.footprint_margin
    c, s = np.cos(rows[:, 2]), np.sin(rows[:, 2])
    cx = rows[:, 0] + vehicle.rear_axle_to_center * c - vehicle.rear_axle_lateral_offset * s
    cy = rows[:, 1] + vehicle.rear_axle_to_center * s + vehicle.rear_axle_lateral_offset * c
    ex = hl * np.abs(c) + hw * np.abs(s)
    ey = hl * np.abs(s) + hw * np.abs(c)
    for begin in range(0, len(rows), 64):
        section = slice(begin, begin + 64)
        margin = max(minimum, 1e-9) + checker.cell_half + 1e-10
        keep = ((checker.obstacle_x >= np.min(cx[section] - ex[section]) - margin)
            & (checker.obstacle_x <= np.max(cx[section] + ex[section]) + margin)
            & (checker.obstacle_y >= np.min(cy[section] - ey[section]) - margin)
            & (checker.obstacle_y <= np.max(cy[section] + ey[section]) + margin))
        if not np.any(keep):
            continue
        reduced = copy.copy(checker)
        reduced.obstacle_x = checker.obstacle_x[keep]
        reduced.obstacle_y = checker.obstacle_y[keep]
        valid, separation = ORIGINAL_RASTER_CHECK(reduced, rows[section])
        minimum = min(minimum, separation)
        if not valid:
            return False, minimum
    return True, minimum


def enable_pruning():
    runtime_geometry._vectorized_exact_raster_path_clearance = pruned_raster_check


def path_metrics(path: np.ndarray, goal: np.ndarray, initial_direction: float = 1) -> dict:
    """N x 5 rear-axle poses, +/-1 gear, steering in radians; only valid rows."""
    p = np.asarray(path, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 5 or len(p) < 2 or not np.isfinite(p).all():
        raise ValueError("path must contain at least two finite N x 5 states")
    ds = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    length = float(ds.sum())
    dyaw = np.arctan2(np.sin(np.diff(p[:, 2])), np.cos(np.diff(p[:, 2])))
    dsteer = np.abs(np.diff(p[:, 4]))
    gears = p[:, 3] > 0
    # Keep the deployed row-transition count; also expose moving-edge count
    # so zero-length terminal gear changes are detectable.
    moving_gear = gears[:-1][ds > 1e-6]
    midpoint_yaw = p[:-1, 2] + .5 * dyaw
    longitudinal = np.sum(np.diff(p[:, :2], axis=0) * np.column_stack((np.cos(midpoint_yaw), np.sin(midpoint_yaw))), axis=1)
    geometric_gear = longitudinal[ds > 1e-6] > 0
    geometric_switches = int(np.count_nonzero(geometric_gear[1:] != geometric_gear[:-1]))
    yaw_error = float(np.arctan2(np.sin(p[-1, 2] - goal[2]), np.cos(p[-1, 2] - goal[2])))
    return {
        "length_m": length,
        "gear_switches": int(np.count_nonzero(gears[1:] != gears[:-1])),
        "moving_gear_switches": int(np.count_nonzero(moving_gear[1:] != moving_gear[:-1])),
        "motion_gear_switches": geometric_switches,
        "gear_switches_including_initial": geometric_switches + int(len(geometric_gear) > 0 and geometric_gear[0] != (initial_direction > 0)),
        "reverse_length_m": float(ds[~gears[:-1]].sum()),
        "heading_variation_deg": float(np.rad2deg(np.abs(dyaw).sum())),
        "steering_variation_deg": float(np.rad2deg(dsteer.sum())),
        "steering_variation_deg_per_m": float(np.rad2deg(dsteer.sum()) / max(length, 1e-9)),
        "max_abs_steer_deg": float(np.rad2deg(np.abs(p[:, 4]).max())),
        "goal_position_error_m": float(np.linalg.norm(p[-1, :2] - goal[:2])),
        "goal_yaw_error_deg": abs(math.degrees(yaw_error)),
    }


def anchor_output(output: dict, batch: dict, config, coarse_steps: int) -> dict:
    """Same post-forward terminal bridge as SupervisedPathWorkerAPI.plan."""
    output = {k: v.float() if torch.is_tensor(v) and v.is_floating_point() else v for k, v in output.items()}
    position_error, yaw_error = terminal_pose_error(output["poses"], batch["goal"], output["lengths"])
    poses = anchor_terminal_pose_trace(output["poses"], batch["goal"], output["lengths"], coarse_steps * config.data.dense_substeps)
    controls = controls_from_pose_trace(poses, output["gear"].to(poses.dtype) * 2 - 1, output["lengths"], config.vehicle.wheelbase_m)
    output["poses"] = poses
    output.update(controls)
    output["terminal_anchor_position_correction_m"] = position_error
    output["terminal_anchor_yaw_correction_rad"] = yaw_error
    return output


def inspect_scene(task: dict) -> dict:
    """CPU geometry on all four candidates plus both references in one scene."""
    raw = repair_sample_dir(task["provenance"]["sample_dir"])
    with np.load(raw / "planner_input.npz", allow_pickle=False) as z:
        start = z["start_state"].astype(np.float64)
        boundary = z["parking_lot_polygon_xy"].astype(np.float64)
        slot = z["target_slot_polygon_xy"].astype(np.float64)
        mask_key = "obstacle_physical" if "obstacle_physical" in z else "obstacle_safety"
        occupancy = z[mask_key].astype(np.uint8)
    raw_gt = states_world_to_local(np.load(raw / "path.npy"), start).astype(np.float64)
    processed = task["processed"]
    paths = task["paths"] + [raw_gt, processed]
    if np.linalg.norm(raw_gt[0, :3]) > 1e-4:
        raise ValueError(f"GT start does not match scene: {raw}")
    if any(np.linalg.norm(p[0, :3]) > 1e-4 for p in paths):
        raise ValueError(f"candidate start does not match GT: {raw}")
    lengths = np.asarray([[len(p) for p in paths]], dtype=np.int64)
    poses = np.zeros((1, len(paths), int(lengths.max()), 3), dtype=np.float32)
    for j, path in enumerate(paths):
        poses[0, j, :len(path)] = path[:, :3]
    valid, clearance, inside = _live_candidate_geometry(
        {"poses": torch.from_numpy(poses), "lengths": torch.from_numpy(lengths)},
        occupancy, start, boundary, LiveVehicleConfig(), slot,
    )
    valid, clearance, inside = [x.numpy()[0] for x in (valid, clearance, inside)]
    metrics = []
    for j, p in enumerate(paths):
        m = path_metrics(p, task["goal"], start[3])
        m.update(collision_free=bool(valid[j]), clearance_m=float(clearance[j]), final_inside=bool(inside[j]))
        m["common_feasible"] = bool(valid[j] and inside[j] and clearance[j] >= task["minimum_clearance"]
            and m["goal_position_error_m"] <= task["position_tolerance"]
            and m["goal_yaw_error_deg"] <= task["yaw_tolerance"])
        metrics.append(m)
    return {"metrics": metrics, "raw_sample_dir": str(raw), "mask": mask_key}


def describe(values) -> dict:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    if not len(a):
        return {"n": 0}
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "p95": float(np.percentile(a, 95)), "min": float(a.min()), "max": float(a.max())}


def paired_summary(rows: list[dict], reference: str) -> dict:
    if not rows:
        return {"n": 0}
    result = {"n": len(rows)}
    for metric in ("length_m", "gear_switches", "moving_gear_switches", "motion_gear_switches", "gear_switches_including_initial", "reverse_length_m", "heading_variation_deg", "steering_variation_deg", "steering_variation_deg_per_m"):
        gt = np.asarray([r[reference][metric] for r in rows], dtype=float)
        pred = np.asarray([r["k4"][metric] for r in rows], dtype=float)
        delta = pred - gt
        item = {"gt": describe(gt), "k4": describe(pred), "delta": describe(delta),
                "worse_fraction": float((delta > 1e-6).mean()), "equal_fraction": float((np.abs(delta) <= 1e-6).mean()), "better_fraction": float((delta < -1e-6).mean())}
        if np.all(gt > 1e-9):
            item["paired_relative_percent"] = describe(100 * delta / gt)
        if gt.sum() > 0:
            item["aggregate_relative_percent"] = float(100 * delta.sum() / gt.sum())
        result[metric] = item
    # Resample scenes (not nodes). Descriptive paired uncertainty only.
    delta = np.asarray([r["k4"]["length_m"] - r[reference]["length_m"] for r in rows])
    relative = np.asarray([100 * (r["k4"]["length_m"] / r[reference]["length_m"] - 1) for r in rows])
    rng = np.random.default_rng(42)
    boot_d, boot_r = [], []
    for _ in range(1000):
        idx = rng.integers(0, len(rows), len(rows))
        boot_d.append(delta[idx].mean()); boot_r.append(relative[idx].mean())
    result["length_mean_delta_bootstrap95_m"] = np.percentile(boot_d, [2.5, 97.5]).tolist()
    result["length_mean_relative_bootstrap95_percent"] = np.percentile(boot_r, [2.5, 97.5]).tolist()
    return result


def summarize(rows: list[dict], manifest: dict) -> dict:
    pairs = [r for r in rows if r["k4"]["strict_success"] and r["gt_raw"]["common_feasible"]]
    processed_pairs = [r for r in rows if r["k4"]["strict_success"] and r["gt_processed"]["common_feasible"]]
    valid_predictions = [r for r in rows if r["k4"]["strict_success"]]
    result = dict(manifest, samples=len(rows),
        counts={
            "k4_strict_success": len(valid_predictions),
            "k4_collision_free": sum(r["k4"]["collision_free"] for r in rows),
            "k4_common_feasible": sum(r["k4"]["common_feasible"] for r in rows),
            "gt_raw_common_feasible": sum(r["gt_raw"]["common_feasible"] for r in rows),
            "gt_processed_common_feasible": sum(r["gt_processed"]["common_feasible"] for r in rows),
            "paired_raw_common_feasible_and_k4_strict": len(pairs),
        },
        paired_raw=paired_summary(pairs, "gt_raw"),
        paired_processed=paired_summary(processed_pairs, "gt_processed"),
        k4_strict_success_vs_all_saved_raw_references=paired_summary(valid_predictions, "gt_raw"),
        all_rows_descriptive_including_failures=paired_summary(rows, "gt_raw"),
        by_source={}, by_gt_switches={},
    )
    for key, value in (("source_group", "by_source"), ("gt_switch_group", "by_gt_switches")):
        for label in sorted({r[key] for r in rows}):
            group = [r for r in rows if r[key] == label]
            paired = [r for r in pairs if r[key] == label]
            result[value][label] = {"total": len(group), "k4_success": sum(r["k4"]["strict_success"] for r in group), "paired": paired_summary(paired, "gt_raw")}
    result["selection_gap_length_m"] = describe([r["selection_gap_length_m"] for r in valid_predictions])
    return result


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="config/hope_path_transformer_town05_town04_hardcase_v2_feasibility_k4_strict_anchor_inference.json")
    parser.add_argument("--dataset", default="data/town05_town04_physx_combined_dense_v4_hardcase_v2")
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--geometry-workers", type=int, default=4)
    parser.add_argument("--processes", action="store_true")
    parser.add_argument("--prune-raster", action="store_true")
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()
    if args.prune_raster:
        enable_pruning()
    torch.set_num_threads(2)
    torch.manual_seed(42)
    output_root = Path(args.output).resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    profile_path = (ROOT / args.profile).resolve()
    profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    if profile.get("mirror_tta") or profile.get("goal_tta_offsets") or profile.get("additional") or profile["total_candidates"] != 4:
        raise ValueError("this audit requires native single-model K4")
    checkpoint_path = (ROOT / profile["primary"]["checkpoint"]).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, model_config, checkpoint = load_planner_checkpoint(checkpoint_path, device, use_ema=True)
    epoch = checkpoint.get("epoch")
    del checkpoint
    config = replace(model_config, model=replace(model_config.model, hard_goal_terminal_pose=True,
        terminal_anchor_steps=profile["runtime_terminal_anchor"]["coarse_steps"]))
    if not profile["runtime_terminal_anchor"]["enabled"] or model_config.model.hard_goal_terminal_pose:
        raise ValueError("unexpected anchor contract")
    selection = profile["runtime_selection"]
    minimum_clearance = selection["minimum_clearance_m"]
    weights = {"goal_error": selection["goal_error_weight"], "steering_variation": selection["steering_variation_weight"],
        "gear_switches": selection["gear_switch_weight"], "path_length": selection["path_length_weight"], "clearance": selection["clearance_weight"]}
    dataset_root = (ROOT / args.dataset).resolve()
    provenance = load_dedup_rows(dataset_root)
    entries = processed_manifest_entries(dataset_root, args.split)
    manifest = {"schema": "k4-gt-paired-path-audit-v1", "split": args.split, "max_samples": args.max_samples,
        "dataset": str(dataset_root), "checkpoint": str(checkpoint_path), "checkpoint_epoch": epoch,
        "checkpoint_sha256": sha256_file(checkpoint_path), "profile": profile,
        "dataset_manifest_sha256": sha256_file(dataset_root / "manifest.json"),
        "script_sha256": sha256_file(Path(__file__)), "torch_version": torch.__version__, "device": str(device),
        "inference_input": "saved processed scene inputs only; no target path tensors passed to model",
        "execution_geometry": "saved physical 200x200 raster, production swept footprint and slot checks",
        "gt": "saved Planner-3 li_iterative_ocp path.npy; no globally optimal certificate",
        "scope": "offline planned paths; not camera perception, CARLA tracking or actual driven distance",
        "exact_raster_broad_phase_pruning": args.prune_raster,
        "bootstrap_unit": "scene; does not account for shared parking-layout correlations"}
    write_json(output_root / "run_manifest.json", manifest)
    rows = []
    timings = {"network": 0.0, "geometry": 0.0, "selection": 0.0}
    started = time.perf_counter()
    executor = ProcessPoolExecutor if args.processes else ThreadPoolExecutor
    with executor(max_workers=args.geometry_workers, initializer=enable_pruning if args.prune_raster else None) as pool, (output_root / "per_scene.jsonl").open("w", encoding="utf-8") as stream:
        for entry in entries:
            shard_path = resolve_shard(dataset_root, entry)
            dataset = ProcessedShardDataset(shard_path, config, mirror=False)
            selected_paths, selected_lengths, selected_uids = [], [], []
            for begin in range(0, len(dataset), args.batch_size):
                indices = list(range(begin, min(begin + args.batch_size, len(dataset))))
                if args.max_samples is not None:
                    indices = indices[:max(0, args.max_samples - len(rows))]
                if not indices:
                    break
                full_batch = default_collate([dataset[i] for i in indices])
                batch = move_to_device({k: full_batch[k] for k in INPUT_KEYS}, device)
                tick = time.perf_counter()
                with torch.inference_mode():
                    prediction = generate_candidate_output(model, batch, candidate_count=4, mirror_tta=False)
                    prediction = anchor_output(prediction, batch, config, config.model.terminal_anchor_steps)
                # Free intermediate decoder outputs before CPU validation.
                prediction = {k: prediction[k] for k in ("poses", "lengths", "gear", "curvature", "steering", "scores", "terminal_anchor_position_correction_m", "terminal_anchor_yaw_correction_rad")}
                arrays = {k: v.detach().cpu().numpy() for k, v in prediction.items()}
                timings["network"] += time.perf_counter() - tick
                tasks = []
                for j, idx in enumerate(indices):
                    uid = str(dataset.arrays["source_uid"][idx])
                    pose = arrays["poses"][j]
                    paths = [np.column_stack((pose[c, :n], 2 * arrays["gear"][j, c, :n] - 1, arrays["steering"][j, c, :n])) for c, n in enumerate(arrays["lengths"][j])]
                    n = int(dataset.arrays["dense_target_length"][idx])
                    processed = np.column_stack((dataset.arrays["dense_target_pose"][idx, :n],
                        2 * dataset.arrays["dense_target_gear"][idx, :n] - 1, dataset.arrays["dense_target_steering"][idx, :n]))
                    goal = dataset.arrays["goal"][idx]
                    tasks.append({"paths": paths, "processed": processed, "provenance": provenance[uid],
                        "goal": np.array([goal[0], goal[1], math.atan2(goal[2], goal[3])]),
                        "minimum_clearance": minimum_clearance, "position_tolerance": config.data.goal_position_tolerance_m,
                        "yaw_tolerance": config.data.goal_yaw_tolerance_deg})
                tick = time.perf_counter()
                scenes = list(pool.map(inspect_scene, tasks))
                timings["geometry"] += time.perf_counter() - tick
                def tensor_metric(name, dtype=torch.float32):
                    return torch.tensor([[m[name] for m in s["metrics"][:4]] for s in scenes], dtype=dtype, device=device)
                collision_free = tensor_metric("collision_free", torch.bool)
                tick = time.perf_counter()
                with torch.inference_mode():
                    geometry = evaluate_inference_batch(prediction, batch, config, collision_free_override=collision_free)
                    runtime_length, runtime_switches, runtime_steering = _candidate_control_metrics(prediction, config.vehicle.wheelbase_m)
                    selected, success, costs = _select_runtime_candidate(fallback_selected=geometry["selected"],
                        exact_success=geometry["candidate_success"], collision_free=collision_free,
                        final_inside=tensor_metric("final_inside", torch.bool), live_clearance_m=tensor_metric("clearance_m"),
                        goal_position_error_m=geometry["candidate_goal_position_error_m"], goal_yaw_error_deg=geometry["candidate_goal_yaw_error_deg"],
                        path_length_m=runtime_length, gear_switches=runtime_switches,
                        steering_variation_rad=runtime_steering,
                        minimum_clearance_m=minimum_clearance, goal_position_tolerance_m=config.data.goal_position_tolerance_m,
                        goal_yaw_tolerance_deg=config.data.goal_yaw_tolerance_deg, weights=weights)
                selected = selected.cpu().numpy(); success = success.cpu().numpy(); costs = costs.cpu().numpy()
                geom = {k: v.cpu().numpy() for k, v in geometry.items()}
                timings["selection"] += time.perf_counter() - tick
                for j, idx in enumerate(indices):
                    uid = str(dataset.arrays["source_uid"][idx]); chosen = int(selected[j])
                    scene = scenes[j]; candidate_metrics = scene["metrics"][:4]
                    for c, m in enumerate(candidate_metrics):
                        m.update(strict_success=bool(success[j, c]), trackable=bool(geom["candidate_trackable"][j, c]),
                            anchor_correction_valid=bool(geom["candidate_anchor_correction_valid"][j, c]),
                            anchor_position_correction_m=float(arrays["terminal_anchor_position_correction_m"][j, c]),
                            anchor_yaw_correction_deg=float(np.rad2deg(arrays["terminal_anchor_yaw_correction_rad"][j, c])),
                            selection_cost=float(costs[j, c]))
                    feasible_lengths = [m["length_m"] for m in candidate_metrics if m["strict_success"]]
                    gt = scene["metrics"][4]
                    row = {"uid": uid, "shard": shard_path.name, "shard_index": idx, "raw_sample_dir": scene["raw_sample_dir"],
                        "source_group": uid[:3], "slot_id": str(dataset.arrays["target_slot_id"][idx]),
                        "start_pose_mode": str(dataset.arrays["start_pose_mode"][idx]), "start_direction": int(dataset.arrays["initial_direction"][idx]),
                        "gt_switch_group": str(gt["gear_switches"]), "collision_mask": scene["mask"], "selected_candidate": chosen,
                        "k4": candidate_metrics[chosen], "gt_raw": gt, "gt_processed": scene["metrics"][5], "candidates": candidate_metrics,
                        "selection_gap_length_m": candidate_metrics[chosen]["length_m"] - min(feasible_lengths) if feasible_lengths else None}
                    rows.append(row)
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                    path = tasks[j]["paths"][chosen]
                    padded = np.zeros((config.data.dense_primitive_count, 5), dtype=np.float32)
                    padded[:len(path)] = path
                    selected_paths.append(padded); selected_lengths.append(len(path)); selected_uids.append(uid)
                stream.flush()
                elapsed = time.perf_counter() - started
                if begin == 0 or (begin // args.batch_size) % 10 == 0:
                    print(f"[COMPARE] n={len(rows)} elapsed={elapsed:.1f}s scenes/s={len(rows)/elapsed:.2f} strict={sum(r['k4']['strict_success'] for r in rows)} stages={timings}", flush=True)
            np.savez_compressed(output_root / (shard_path.stem + "_selected.npz"),
                paths=np.asarray(selected_paths), lengths=np.asarray(selected_lengths), uids=np.asarray(selected_uids))
            if args.max_samples is not None and len(rows) >= args.max_samples:
                break
    manifest["elapsed_seconds"] = time.perf_counter() - started
    manifest["timings_seconds"] = timings
    summary = summarize(rows, manifest)
    write_json(output_root / "summary.json", summary)
    print(json.dumps({"samples": len(rows), "counts": summary["counts"], "elapsed_seconds": manifest["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
