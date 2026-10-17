"""Estimate T + w integral(a^2 + v^2 delta_dot^2) for saved, fixed paths.

This is a common retiming audit, NOT reconstruction of original GT OCP controls
and NOT a full vehicle-dynamics/closed-loop or globally optimal route benchmark.
All original moving steering knots are preserved. Squared speed is piecewise
linear in arc length; each interval has constant acceleration. Integrals are
analytic. Stops are imposed at geometric direction changes and zero-length
steering intervals. No path/model/production configuration is modified.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import ctypes
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "analysis_artifacts/k4_gt_comparison_test_20260916"
STRUCTURE = ROOT / "analysis_artifacts/k4_structure_analysis_20260916"
PARAMS = dict(weight=.5, max_speed=2.5, max_accel=2., max_steer_rate=.6,
              max_steer=.980994, zero_distance_m=1.e-6, steer_tolerance_rad=1.e-5)


@lru_cache(None)
def casadi():
    import casadi as ca
    if os.name == "nt":
        package_dir = str(Path(ca.__file__).parent)
        os.environ["PATH"] = package_dir + os.pathsep + os.environ.get("PATH", "")
        for name in ("libipopt-3.dll", "libcasadi_nlpsol_ipopt.dll"):
            filename = Path(package_dir) / name
            if filename.is_file():
                ctypes.WinDLL(str(filename))
    return ca


@lru_cache(maxsize=48)
def solver_for_size(n):
    ca = casadi()
    inside = ca.SX.sym("b", n-2)
    p = ca.SX.sym("p", 2*(n-1)+1)
    h, slope, w = p[:n-1], p[n-1:2*(n-1)], p[-1]
    b = ca.vertcat(0., inside, 0.)
    q = ca.vertcat(0., ca.sqrt(inside), 0.)
    q0, q1 = q[:-1], q[1:]
    dt = 2*h/(q0+q1)
    accel = (b[1:]-b[:-1])/(2*h)
    fourth_mean = (q0**4+q0**3*q1+q0**2*q1**2+q0*q1**3+q1**4)/5
    duration = ca.sum1(dt)
    accel_cost = ca.dot(accel**2, dt)
    steer_cost = ca.dot(slope**2 * fourth_mean, dt)
    objective = duration + w*(accel_cost+steer_cost)
    nlp = dict(x=inside, p=p, f=objective, g=b[1:]-b[:-1])
    solver = ca.nlpsol("timing", "ipopt", nlp, {
        "print_time": False, "ipopt.print_level": 0, "ipopt.sb": "yes",
        "ipopt.max_iter": 350, "ipopt.tol": 1.e-8,
        "ipopt.acceptable_tol": 1.e-6, "ipopt.bound_relax_factor": 0.,
        "ipopt.mu_strategy": "adaptive",
    })
    evaluate = ca.Function("evaluate", [inside, p], [duration, accel_cost, steer_cost])
    return solver, evaluate


def subdivide(h, steer, refinement=1):
    """Preserve original steering knots; add nodes, never smooth or coarsen."""
    total = float(np.sum(h))
    original_s = np.r_[0., np.cumsum(h)]
    chunks = [np.linspace(original_s[i], original_s[i+1],
                          max(refinement, int(math.ceil(d/.05))*refinement)+1)[:-1]
              for i, d in enumerate(h)]
    sample = np.r_[np.concatenate(chunks), total]
    # Bucket dimensions to reuse symbolic NLPs; split widest gaps only.
    target = max(32, int(math.ceil(len(sample)/32))*32)
    while len(sample) < target:
        gaps = np.diff(sample)
        count = min(target-len(sample), len(gaps))
        indices = np.argsort(gaps)[-count:]
        sample = np.sort(np.r_[sample, .5*(sample[indices]+sample[indices+1])])
    return np.diff(sample), np.interp(sample, original_s, steer)


def optimize_run(h, steer, weight=.5, refinement=1, start_scale=.6, keep_trace=False):
    h, steer = subdivide(h, steer, refinement)
    slope = np.diff(steer)/h
    cap = np.full(len(steer), PARAMS["max_speed"]**2)
    rate_cap = (PARAMS["max_steer_rate"]/np.maximum(np.abs(slope), 1.e-12))**2
    cap[:-1] = np.minimum(cap[:-1], rate_cap)
    cap[1:] = np.minimum(cap[1:], rate_cap)
    cap[[0, -1]] = 0.
    feasible = cap.copy()
    accel_limit = 2*PARAMS["max_accel"]*h
    for i in range(1, len(cap)):
        feasible[i] = min(feasible[i], feasible[i-1]+accel_limit[i-1])
    for i in range(len(cap)-2, -1, -1):
        feasible[i] = min(feasible[i], feasible[i+1]+accel_limit[i])
    lower = np.minimum(1.e-12, cap[1:-1]*1.e-6)
    x0 = np.maximum(lower*2, feasible[1:-1]*start_scale)
    params = np.r_[h, slope, weight]
    solver, evaluate = solver_for_size(len(steer))
    sol = solver(x0=x0, p=params, lbx=lower, ubx=cap[1:-1],
                 lbg=-accel_limit, ubg=accel_limit)
    stats = solver.stats()
    inside = np.asarray(sol["x"]).ravel()
    b = np.r_[0., inside, 0.]
    speed = np.sqrt(np.maximum(b, 0.))
    accel = np.diff(b)/(2*h)
    dt = 2*h/(speed[:-1]+speed[1:])
    rate = np.abs(slope)*np.maximum(speed[:-1], speed[1:])
    duration, ac, sc = [float(v) for v in evaluate(inside, params)]
    violation = max(0., float(np.max(np.abs(accel)))-PARAMS["max_accel"],
                    float(np.max(rate))-PARAMS["max_steer_rate"],
                    float(np.max(speed))-PARAMS["max_speed"])
    valid = bool(stats["success"] and violation <= 1.e-5 and np.isfinite(duration+ac+sc))
    result = dict(success=valid, solver_status=stats["return_status"],
                  iterations=stats["iter_count"], duration_s=duration,
                  acceleration_integral=ac, steering_integral=sc,
                  comfort_integral=ac+sc, objective=duration+weight*(ac+sc),
                  constraint_violation=violation, max_speed=float(np.max(speed)),
                  max_accel=float(np.max(np.abs(accel))), max_steer_rate=float(np.max(rate)),
                  nodes=len(steer))
    if keep_trace:
        result["trace"] = dict(h=h.tolist(), steer=steer.tolist(), speed=speed.tolist(), dt=dt.tolist())
    return result


def retime(path, weight=.5, refinement=1, start_scale=.6, keep_trace=False):
    p = np.asarray(path, dtype=np.float64)
    max_steer = float(np.max(np.abs(p[:, 4])))
    if max_steer > PARAMS["max_steer"]+PARAMS["steer_tolerance_rad"]:
        return dict(success=False, reason="steering_limit_exceeded", max_steer_rad=max_steer)
    ds = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
    dyaw = np.arctan2(np.sin(np.diff(p[:, 2])), np.cos(np.diff(p[:, 2])))
    heading = p[:-1, 2]+.5*dyaw
    dxy = np.diff(p[:, :2], axis=0)
    longitudinal = np.sum(dxy*np.column_stack((np.cos(heading), np.sin(heading))), axis=1)
    lateral = np.sum(dxy*np.column_stack((-np.sin(heading), np.cos(heading))), axis=1)
    gear = np.where(longitudinal >= 0, 1, -1)
    moving = ds > PARAMS["zero_distance_m"]
    moving_gears = gear[moving]
    gear_changes = int(np.count_nonzero(np.diff(moving_gears)))
    stationary_steer_time = (abs(p[0, 4])+abs(p[-1, 4])+
                             float(np.abs(np.diff(p[:, 4]))[~moving].sum()))/PARAMS["max_steer_rate"]
    runs = []
    begin = None
    for i in range(len(ds)):
        if moving[i]:
            if begin is None:
                begin = i
            elif gear[i] != gear[i-1]:
                runs.append((begin, i)); begin = i
        elif begin is not None:
            runs.append((begin, i)); begin = None
    if begin is not None:
        runs.append((begin, len(ds)))
    results = [optimize_run(ds[a:b], p[a:b+1, 4], weight, refinement, start_scale, keep_trace)
               for a, b in runs]
    if not results:
        return dict(success=False, reason="no_moving_run")
    duration = sum(r["duration_s"] for r in results)+stationary_steer_time
    ac = sum(r["acceleration_integral"] for r in results)
    sc = sum(r["steering_integral"] for r in results)
    # These diagnostics expose why fixed geometry + stored steering is only a
    # surrogate: XY/yaw/steering do not necessarily satisfy a bicycle exactly.
    predicted_dyaw = gear*ds*np.tan(.5*(p[:-1, 4]+p[1:, 4]))/3.004647
    result = dict(success=all(r["success"] for r in results),
        reason="ok" if all(r["success"] for r in results) else "solver_failure",
        duration_s=duration, acceleration_integral=ac, steering_integral=sc,
        comfort_integral=ac+sc, objective=duration+weight*(ac+sc),
        moving_duration_s=duration-stationary_steer_time,
        stationary_steer_time_s=stationary_steer_time, length_m=float(ds.sum()),
        neglected_distance_m=float(ds[~moving].sum()), gear_changes=gear_changes,
        run_count=len(runs), first_moving_gear=int(moving_gears[0]),
        last_moving_gear=int(moving_gears[-1]), max_steer_rad=max_steer,
        max_speed=max(r["max_speed"] for r in results),
        max_accel=max(r["max_accel"] for r in results),
        max_steer_rate=max(r["max_steer_rate"] for r in results),
        constraint_violation=max(r["constraint_violation"] for r in results),
        iterations=sum(r["iterations"] for r in results),
        statuses=[r["solver_status"] for r in results],
        lateral_displacement_residual_m=float(np.abs(lateral[moving]).sum()),
        bicycle_yaw_residual_deg=float(np.rad2deg(np.abs(dyaw[moving]-predicted_dyaw[moving]).sum())))
    if keep_trace:
        result["runs"] = results
    return result


@lru_cache(maxsize=1)
def selected_shard(name):
    with np.load(AUDIT / (Path(name).stem+"_selected.npz")) as z:
        return {k: z[k] for k in z.files}


def load_paths(row):
    saved = selected_shard(row["shard"])
    idx = row["shard_index"]
    assert str(saved["uids"][idx]) == row["uid"]
    k4 = saved["paths"][idx, :int(saved["lengths"][idx])].copy()
    gt = np.load(Path(row["raw_sample_dir"]) / "path.npy")
    return gt, k4


def work(job):
    row, weight, refinement = job
    gt, k4 = load_paths(row)
    result = {k: row[k] for k in ("uid", "source_group", "start_direction")}
    result["required_maneuver"] = row["required_maneuver"]
    result["true_ocp_convergence"] = row["true_ocp_convergence"]
    for name, path in (("gt", gt), ("k4", k4)):
        try:
            result[name] = retime(path, weight, refinement)
        except Exception as exc:
            result[name] = dict(success=False, reason="exception", message=str(exc))
    return result


def summaries(rows):
    paired = [r for r in rows if r["gt"]["success"] and r["k4"]["success"]]
    same_direction = [r for r in paired if r["gt"]["first_moving_gear"] == r["k4"]["first_moving_gear"] == r["start_direction"]]
    cohorts = dict(all_valid=paired, same_initial_direction=same_direction)
    for label in sorted(set(r["required_maneuver"] for r in paired)):
        cohorts["maneuver_"+label] = [r for r in paired if r["required_maneuver"] == label]
    output = {}
    for name, sample in cohorts.items():
        group = {"n": len(sample), "metrics": {}}
        for metric in ("length_m", "duration_s", "acceleration_integral", "steering_integral", "comfort_integral", "objective", "gear_changes", "stationary_steer_time_s", "lateral_displacement_residual_m", "bicycle_yaw_residual_deg"):
            a = np.array([r["gt"][metric] for r in sample]); b = np.array([r["k4"][metric] for r in sample])
            if not len(a):
                continue
            group["metrics"][metric] = dict(gt_mean=float(a.mean()), k4_mean=float(b.mean()),
                change_of_means_pct=float(100*(b.mean()/a.mean()-1)) if a.mean() else None,
                k4_lower_pct=float(100*np.mean(b < a)),
                delta_median=float(np.median(b-a)),
                relative_median_pct=float(np.median(100*(b-a)/np.maximum(a, 1.e-12))),
                relative_p10_pct=float(np.percentile(100*(b-a)/np.maximum(a, 1.e-12), 10)),
                relative_p90_pct=float(np.percentile(100*(b-a)/np.maximum(a, 1.e-12), 90)))
        if sample:
            group["shorter_but_higher_J"] = int(sum(r["k4"]["length_m"] < r["gt"]["length_m"] and r["k4"]["objective"] > r["gt"]["objective"] for r in sample))
            group["max_constraint_violation"] = max(r[k]["constraint_violation"] for r in sample for k in ("gt", "k4"))
        output[name] = group
    from collections import Counter
    return dict(attempted=len(rows), cohorts=output,
        gt_status=dict(Counter(r["gt"]["reason"] for r in rows)),
        k4_status=dict(Counter(r["k4"]["reason"] for r in rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--weight", type=float, default=.5)
    parser.add_argument("--refinement", type=int, default=1)
    args = parser.parse_args()
    out = (ROOT / args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = [json.loads(line) for line in (AUDIT / "per_scene.jsonl").read_text(encoding="utf-8").splitlines()]
    provenance = {r["uid"]: r for r in map(json.loads, (STRUCTURE/"gt_maneuver_rows.jsonl").read_text(encoding="utf-8").splitlines())}
    rows = [r for r in rows if r["gt_raw"]["common_feasible"] and r["k4"]["strict_success"]]
    for row in rows:
        row.update({k: provenance[row["uid"]][k] for k in ("required_maneuver", "true_ocp_convergence")})
    if args.limit:
        chosen = np.random.default_rng(20260916).choice(len(rows), min(args.limit, len(rows)), replace=False)
        rows = [rows[i] for i in sorted(chosen)]
    manifest = dict(schema="fixed-path-common-retiming-v1", parameters=PARAMS,
                    args=vars(args), paired_available=9227, scenes=len(rows),
                    source_audit=str(AUDIT), source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    start = time.perf_counter(); results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, (out/"per_scene.jsonl").open("w", encoding="utf-8") as stream:
        for result in pool.map(work, ((r, args.weight, args.refinement) for r in rows), chunksize=4):
            results.append(result); stream.write(json.dumps(result)+"\n"); stream.flush()
            if len(results) % 25 == 0 or len(results) == len(rows):
                print(f"[RETIME] {len(results)}/{len(rows)} elapsed={time.perf_counter()-start:.1f}s", flush=True)
    summary = summaries(results); summary["elapsed_s"] = time.perf_counter()-start
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
