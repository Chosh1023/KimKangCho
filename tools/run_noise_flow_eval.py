"""Experiment 2 (noise Flow, no warm start) on experiments/eval_uids.json.

Phases (all outputs are new files in experiments/results/; nothing existing is
overwritten):

  main       N=16 candidates, Euler 16 steps, top-K=3 on the 500 scenes
             -> exp2_noise_flow_{repr}.jsonl
  sweep      N in {1,4,8,16,32} x steps in {4,8,16} x top-K in {1,3,5}
             -> exp2_noise_flow_{repr}_sweep.jsonl (resumable per combination)
  baselines  steering smoothness of methods 2-4 from the existing artifacts
             -> exp2_baseline_smoothness.jsonl
  summarize  tables + time/J figure -> exp2_noise_flow_{repr}_summary.json/.png

Timing: single process, CUDA-synchronized, from input encoding through the
final selection, one untimed warm-up call per configuration, candidate
validation in 8 worker processes (the HA*+Flow measurement setting).
HA* + IPOPT time = clean single-process HA* plan time + clean single-process
IPOPT time (timing_hastar.jsonl + timing_ipopt.jsonl, reused).
Steering smoothness (same definition for every method, dwell included):
mean |steering rate| = total steering variation / duration; violation =
max interval steering rate > max_steer_rate.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import argparse
import json
import math
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'tools'))

import numpy as np  # noqa: E402

from run_experiments import (DATASET, SCENES, CACHE, TARGETS, RESULTS, eval_uids, gt_table,  # noqa: E402
                             read_jsonl)

SWEEP_N = (1, 4, 8, 16, 32)
SWEEP_STEPS = (4, 8, 16)
SWEEP_K = (1, 3, 5)
LIMIT_TOL = 1e-8


def smoothness(trace, max_steer_rate=None, limit=None):
    """Mean |steering rate| over the time trace (dwell included) and the limit check."""
    tr = np.asarray(trace, dtype=float)
    duration = float(tr[-1, 0])
    variation = float(np.abs(np.diff(tr[:, 5])).sum())
    out = dict(mean_abs_steer_rate=variation/duration if duration > 0 else None, duration_s=duration,
               steering_variation_rad=variation)
    if max_steer_rate is not None:
        out['max_steer_rate'] = float(max_steer_rate)
        out['steer_rate_violation'] = bool(max_steer_rate > limit+LIMIT_TOL)
    return out


def environment(device):
    import torch
    cpu = platform.processor()
    try:
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    except OSError:
        pass
    return dict(gpu=torch.cuda.get_device_name(0) if device == 'cuda' else None, cpu=cpu, cpu_threads=os.cpu_count(),
                torch=torch.__version__, python=platform.python_version(), os=platform.platform(),
                process='single process; CUDA-synchronized wall clock; one untimed warm-up per configuration',
                validation='candidate validation in 8 worker processes')


def open_solver(args):
    from k4_flow_solver.noise_flow_solver import NoiseFlowSolver
    from k4_flow_solver.experiment_eval import GOAL_POSITION_M, GOAL_YAW_DEG
    solver = NoiseFlowSolver(args.checkpoint, device=args.device, validation_workers=args.validation_workers)
    solver.set_goal_tolerance(GOAL_POSITION_M, math.radians(GOAL_YAW_DEG))
    return solver


def scene_row(uid, report, trajectory, gt, cfg, repr_name, keep_candidates):
    from k4_flow_solver.experiment_eval import timing_record
    rows = report.get('candidate_rows', [])
    selected = report['selected_index']
    row = dict(uid=uid, method=f'noise_flow_{repr_name}', success=report['status'] == 'valid_trajectory',
               J=report['selected_J'], gt_structure=gt['structure'], ranked_structures=report['ranked_structures'],
               selected_structure=report['selected_structure'],
               structure_match=(report['selected_structure'] == gt['structure']) if selected is not None else None,
               gt_structure_in_topk=gt['structure'] in report['ranked_structures'],
               candidates=report['candidates'], top_k=report['top_k'], steps=report['steps'],
               verified_candidates=report['verified_candidates'], failure_counts=report['failure_counts'],
               scene_failure=report['scene_failure'], timings=report['timings'],
               candidate_steer_rate_violations=sum(1 for r in rows if np.isfinite(r['max_steer_rate'])
                                                   and r['max_steer_rate'] > cfg.max_steer_rate+LIMIT_TOL),
               time=timing_record(warm_start_s=None, optimizer_s=report['timings']['total_s'], device=report['device'],
                                  warmup_excluded=report['warmup_excluded']))
    if trajectory is not None:
        row['smoothness'] = smoothness(trajectory, rows[selected]['max_steer_rate'], cfg.max_steer_rate)
    if keep_candidates:
        row['candidate_rows'] = [dict(structure=r['structure'], structure_rank=r['structure_rank'], J=r['J'],
                                      verified=r['verified_feasible'], failure=r['failure'],
                                      max_steer_rate=r['max_steer_rate']) for r in rows]
    return row


def run_configuration(solver, uids, scenes, gt, *, candidates, top_k, steps, seed, repr_name, keep_candidates, label):
    solver.warmup(scenes[uids[0]], candidates=candidates, top_k=top_k, steps=steps, seed=seed)
    out = []
    for i, uid in enumerate(uids):
        trajectory, report = solver.plan(scenes[uid], candidates=candidates, top_k=top_k, steps=steps, seed=seed)
        out.append(scene_row(uid, report, trajectory, gt[uid], solver.cfg, repr_name, keep_candidates))
        if (i+1) % 100 == 0:
            print(json.dumps(dict(configuration=label, done=i+1, success=sum(r['success'] for r in out),
                                  mean_total_s=float(np.mean([r['timings']['total_s'] for r in out])))), flush=True)
    return out


def load_inputs(limit):
    from k4_flow_solver.prepare import load_scene
    uids = eval_uids()[:limit] if limit else eval_uids()
    return uids, {u: load_scene(SCENES/u) for u in uids}, gt_table(uids)


def phase_main(args):
    target = args.out_dir/f'exp2_noise_flow_{args.repr}.jsonl'
    if target.exists():
        raise FileExistsError(f'{target} exists; not overwriting')
    uids, scenes, gt = load_inputs(args.limit)
    solver = open_solver(args)
    started = time.perf_counter()
    rows = run_configuration(solver, uids, scenes, gt, candidates=16, top_k=3, steps=16, seed=args.seed,
                             repr_name=args.repr, keep_candidates=True, label='main N16 S16 K3')
    solver.close()
    meta = dict(record='meta', checkpoint=str(Path(args.checkpoint).resolve()), epoch=solver.epoch,
                representation=solver.representation, candidates=16, top_k=3, steps=16, seed=args.seed,
                environment=environment(args.device), wall_s=time.perf_counter()-started)
    with target.open('w', encoding='utf-8') as f:
        f.write(json.dumps(meta)+'\n')
        for r in rows:
            f.write(json.dumps(r, default=float)+'\n')
    print(json.dumps(dict(phase='main', scenes=len(rows), success=sum(r['success'] for r in rows), file=str(target))))


def power_state():
    """Battery/AC state (Windows) recorded with every timed configuration."""
    import subprocess
    out = subprocess.run(['powershell.exe', '-NoProfile', '-Command',
                          '$b=Get-CimInstance Win32_Battery; $s=Get-CimInstance -Namespace root/wmi -ClassName BatteryStatus; '
                          '"$($b.BatteryStatus) $($b.EstimatedChargeRemaining) $($s.PowerOnline)"'],
                         capture_output=True, text=True).stdout.split()
    return dict(battery_status=int(out[0]), charge=int(out[1]), power_online=out[2] == 'True') if len(out) == 3 else dict(raw=out)


def phase_config(args):
    """One (candidates, steps, top-K) configuration on the 500 scenes, with candidate rows."""
    n, steps, k = args.candidates, args.steps, args.top_k
    target = args.out_dir/f'exp2_noise_flow_{args.repr}_N{n}_S{steps}_K{k}.jsonl'
    if target.exists():
        raise FileExistsError(f'{target} exists; not overwriting')
    uids, scenes, gt = load_inputs(args.limit)
    solver = open_solver(args)
    before = power_state()
    started = time.perf_counter()
    rows = run_configuration(solver, uids, scenes, gt, candidates=n, top_k=k, steps=steps, seed=args.seed,
                             repr_name=args.repr, keep_candidates=True, label=f'N{n} S{steps} K{k}')
    wall = time.perf_counter()-started
    solver.close()
    meta = dict(record='meta', checkpoint=str(Path(args.checkpoint).resolve()), epoch=solver.epoch,
                representation=solver.representation, candidates=n, top_k=k, steps=steps, seed=args.seed,
                environment=environment(args.device), wall_s=wall, power_before=before, power_after=power_state())
    with target.open('w', encoding='utf-8') as f:
        f.write(json.dumps(meta)+'\n')
        for r in rows:
            f.write(json.dumps(r, default=float)+'\n')
    print(json.dumps(dict(phase='config', N=n, steps=steps, K=k, scenes=len(rows), success=sum(r['success'] for r in rows),
                          file=str(target))))


def phase_sweep(args):
    uids, scenes, gt = load_inputs(args.limit)
    target = args.out_dir/f'exp2_noise_flow_{args.repr}_sweep.jsonl'
    done = set()
    if target.exists():
        counts = {}
        for r in read_jsonl(target):
            if r.get('record') == 'scene':
                key = (r['candidates'], r['steps'], r['top_k_requested'])
                counts[key] = counts.get(key, 0)+1
        done = {k for k, v in counts.items() if v == len(uids)}
    solver = open_solver(args)
    with target.open('a', encoding='utf-8') as f:
        if not done:
            f.write(json.dumps(dict(record='meta', checkpoint=str(Path(args.checkpoint).resolve()), epoch=solver.epoch,
                                    representation=solver.representation, seed=args.seed, scenes=len(uids),
                                    environment=environment(args.device)))+'\n')
        for n in SWEEP_N:
            for steps in SWEEP_STEPS:
                for k in SWEEP_K:
                    if (n, steps, k) in done:
                        continue
                    started = time.perf_counter()
                    rows = run_configuration(solver, uids, scenes, gt, candidates=n, top_k=k, steps=steps, seed=args.seed,
                                             repr_name=args.repr, keep_candidates=False, label=f'N{n} S{steps} K{k}')
                    for r in rows:
                        f.write(json.dumps(dict(record='scene', top_k_requested=k, **r), default=float)+'\n')
                    f.flush()
                    print(json.dumps(dict(phase='sweep', N=n, steps=steps, K=k, success=sum(r['success'] for r in rows),
                                          wall_s=time.perf_counter()-started)), flush=True)
    solver.close()


def baseline_scene(task):
    uid, gt = task
    from k4_flow_solver.config import SolverConfig
    from k4_flow_solver.numerical import decode, from_teacher, cost, time_trace
    from k4_flow_solver.warmstart import load_warm_start, controls_from_hybrid_astar
    manifest = json.loads((DATASET/'manifest.json').read_text(encoding='utf-8'))
    cfg = SolverConfig(**manifest['config'])
    controls = {'3_ocp_gt': decode(gt['latent'], gt['structure'], cfg)}
    result_file = TARGETS/'samples'/uid[:3]/uid/'result.json'
    if result_file.is_file() and json.loads(result_file.read_text(encoding='utf-8')).get('success'):
        controls['2_hybrid_astar_ipopt'] = from_teacher(result_file.parent/'trajectory.npz', cfg)[0]
    warm = CACHE/(uid+'.npz')
    if warm.is_file():
        path, meta = load_warm_start(warm)
        try:
            controls['4_hybrid_astar'] = controls_from_hybrid_astar(path, meta['vehicle']['wheel_base'], cfg)[0]
        except ValueError:
            pass
    out = []
    for method, c in controls.items():
        out.append(dict(uid=uid, method=method, structure=int(gt['structure']) if method == '3_ocp_gt' else None,
                        gears=c.gears.tolist(),
                        smoothness=smoothness(time_trace(c, cfg), cost(c, cfg)['max_steer_rate'], cfg.max_steer_rate)))
    return out


def phase_baselines(args):
    uids = eval_uids()
    gt = gt_table(uids)
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for block in pool.map(baseline_scene, [(u, gt[u]) for u in uids]) for r in block]
    target = args.out_dir/'exp2_baseline_smoothness.jsonl'
    if target.exists():
        raise FileExistsError(f'{target} exists; not overwriting')
    with target.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, default=float)+'\n')
    print(json.dumps(dict(phase='baselines', rows=len(rows))))


def full_stats(values):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    if not len(a):
        return dict(n=0)
    return dict(n=int(len(a)), min=float(a.min()), median=float(np.median(a)), max=float(a.max()), mean=float(a.mean()))


def phase_summarize(args):
    from k4_flow_solver.config import SolverConfig
    uids = eval_uids()
    offline = {}
    for r in read_jsonl(RESULTS/'offline_methods.jsonl'):
        offline.setdefault(r['method'], {})[r['uid']] = r
    gt = offline['3_ocp_gt']
    main_rows = read_jsonl(args.out_dir/f'exp2_noise_flow_{args.repr}.jsonl')
    meta = main_rows[0]
    flow = {r['uid']: r for r in main_rows[1:]}
    ha_time = {r['uid']: r['plan_s'] for r in read_jsonl(RESULTS/'timing_hastar.jsonl')}
    ipopt_time = {r['uid']: r['solve_s'] for r in read_jsonl(RESULTS/'timing_ipopt.jsonl')}
    smooth = {}
    for r in read_jsonl(args.out_dir/'exp2_baseline_smoothness.jsonl'):
        smooth.setdefault(r['method'], {})[r['uid']] = r['smoothness']
    name = f'noise_flow_{args.repr}'
    methods = {name: flow, '2_hybrid_astar_ipopt': offline['2_hybrid_astar_ipopt'], '3_ocp_gt': gt,
               '4_hybrid_astar': offline['4_hybrid_astar']}
    smooth[name] = {u: r['smoothness'] for u, r in flow.items() if r.get('smoothness')}
    ok = {m: [u for u in uids if res.get(u, {}).get('success')] for m, res in methods.items()}
    common = [u for u in uids if all(u in set(v) for v in ok.values())]

    def times(m, subset):
        if m == name:
            return [flow[u]['timings']['total_s'] for u in subset]
        if m == '2_hybrid_astar_ipopt':
            return [ha_time[u]+(ipopt_time.get(u) or 0.0) for u in subset if u in ha_time]
        if m == '4_hybrid_astar':
            return [ha_time[u] for u in subset if u in ha_time]
        return []

    table = {}
    for m, res in methods.items():
        s = ok[m]
        entry = dict(success=len(s), scenes=len(uids), success_rate=len(s)/len(uids),
                     J_method_success=full_stats([res[u]['J'] for u in s]),
                     J_over_gt_method_success=full_stats([res[u]['J']/gt[u]['J'] for u in s if gt[u]['success']]),
                     J_common=full_stats([res[u]['J'] for u in common]),
                     J_over_gt_common=full_stats([res[u]['J']/gt[u]['J'] for u in common]),
                     time_s_success=full_stats(times(m, s)), time_s_all=full_stats(times(m, uids)),
                     mean_abs_steer_rate_success=full_stats([smooth[m][u]['mean_abs_steer_rate'] for u in s if u in smooth.get(m, {})]),
                     mean_abs_steer_rate_common=full_stats([smooth[m][u]['mean_abs_steer_rate'] for u in common if u in smooth.get(m, {})]),
                     steer_rate_violation_rate_success=(float(np.mean([smooth[m][u]['steer_rate_violation'] for u in s
                                                                       if u in smooth.get(m, {})])) if s else None))
        table[m] = entry
    rows = list(flow.values())
    selected = [r for r in rows if r['success']]
    failures = {}
    for r in rows:
        if not r['success']:
            failures[r['scene_failure']] = failures.get(r['scene_failure'], 0)+1
    candidate_failures = {}
    for r in rows:
        for k, v in r['failure_counts'].items():
            candidate_failures[k] = candidate_failures.get(k, 0)+v
    total_candidates = sum(r['candidates'] for r in rows)
    table[name].update(
        structure_selected_equals_gt=float(np.mean([r['structure_match'] for r in selected])) if selected else None,
        structure_gt_in_topk=float(np.mean([r['gt_structure_in_topk'] for r in rows])),
        structure_top1_equals_gt=float(np.mean([r['ranked_structures'][0] == r['gt_structure'] for r in rows])),
        scene_failure_classes=failures, candidate_failure_classes=candidate_failures,
        candidate_success_rate=sum(r['verified_candidates'] for r in rows)/total_candidates,
        candidate_steer_rate_violation_rate=sum(r['candidate_steer_rate_violations'] for r in rows)/total_candidates,
        timing_breakdown={k: full_stats([r['timings'][k] for r in rows])
                          for k in ('input_encoding_s', 'generation_s', 'postprocess_validation_s', 'total_s')})
    for m in ('2_hybrid_astar_ipopt', '4_hybrid_astar'):
        reasons = {}
        for u in uids:
            r = methods[m].get(u, {})
            if not r.get('success'):
                reason = r.get('reason') or ('collision' if r.get('eval', {}).get('collision_free') is False else
                                             'goal_not_reached' if r.get('eval', {}).get('terminal_ok') is False else
                                             'limit_violation' if r.get('eval', {}).get('controls_ok') is False else 'other')
                reasons[reason] = reasons.get(reason, 0)+1
        table[m]['scene_failure_classes'] = reasons
    reference = dict(success_rate=table['2_hybrid_astar_ipopt']['success_rate'],
                     mean_time_s_all=table['2_hybrid_astar_ipopt']['time_s_all']['mean'],
                     mean_J_over_gt=table['2_hybrid_astar_ipopt']['J_over_gt_method_success']['mean'])
    sweep = []
    sweep_file = args.out_dir/f'exp2_noise_flow_{args.repr}_sweep.jsonl'
    if sweep_file.is_file():
        groups = {}
        for r in read_jsonl(sweep_file):
            if r.get('record') == 'scene':
                groups.setdefault((r['candidates'], r['steps'], r['top_k_requested']), []).append(r)
        for (n, steps, k), rs in sorted(groups.items()):
            s = [r for r in rs if r['success']]
            entry = dict(N=n, steps=steps, K=k, scenes=len(rs), success_rate=len(s)/len(rs),
                         mean_J=float(np.mean([r['J'] for r in s])) if s else None,
                         mean_J_over_gt=float(np.mean([r['J']/gt[r['uid']]['J'] for r in s])) if s else None,
                         mean_time_s_all=float(np.mean([r['timings']['total_s'] for r in rs])),
                         mean_time_s_success=float(np.mean([r['timings']['total_s'] for r in s])) if s else None)
            entry['better_than_hastar_ipopt'] = bool(entry['mean_J_over_gt'] is not None
                                                     and entry['mean_time_s_all'] < reference['mean_time_s_all']
                                                     and entry['mean_J_over_gt'] < reference['mean_J_over_gt'])
            entry['success_3pp_below_hastar_ipopt'] = bool(entry['success_rate'] < reference['success_rate']-.03)
            sweep.append(entry)
    cfg = SolverConfig(**json.loads((DATASET/'manifest.json').read_text(encoding='utf-8'))['config'])
    summary = dict(scenes=len(uids), common_success_scenes=len(common), methods=table, sweep=sweep,
                   hastar_ipopt_reference=reference, noise_flow_run=meta,
                   definitions=dict(
                       J='T + 0.5*integral(a^2 + v^2*steer_rate^2) dt, common evaluator (numerical.validate), goal 0.1 m / 5 deg',
                       J_over_gt='per-scene J / OCP GT J (GT = experiment 3, same evaluator)',
                       common='scenes where the noise Flow, HA*+IPOPT, OCP GT and HA* alone all succeed',
                       time_noise_flow='input encoding -> final selection, CUDA-synchronized, warm-up excluded, 8 validation workers',
                       time_hastar_ipopt='clean single-process HA* plan time + clean single-process IPOPT solve time (reused)',
                       time_s_all='all 500 scenes (failed attempts included; HA* failures have HA* time only)',
                       mean_abs_steer_rate='total steering variation / duration over the time trace (dwell included)',
                       steer_rate_violation=f'max interval steering rate > {cfg.max_steer_rate} rad/s',
                       sweep_plot='x = mean time over all 500 scenes, y = mean J/GT over each method\'s successful scenes'))
    out = args.out_dir/f'exp2_noise_flow_{args.repr}_summary.json'
    out.write_bytes(json.dumps(summary, indent=1, default=float).encode('utf-8'))
    print(json.dumps(dict(phase='summarize', file=str(out), common=len(common))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('phase', choices=('main', 'config', 'sweep', 'baselines', 'summarize'))
    p.add_argument('--checkpoint', default='run-logs/noise_flow_gt_bspline64_v1/best.pt')
    p.add_argument('--repr', default='bspline64')
    p.add_argument('--device', default='cuda')
    p.add_argument('--validation-workers', type=int, default=8)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=20260917)
    p.add_argument('--limit', type=int, help='first N evaluation scenes (timing trials)')
    p.add_argument('--candidates', type=int, default=16, help='config phase: candidates N')
    p.add_argument('--steps', type=int, default=16, help='config phase: Euler steps')
    p.add_argument('--top-k', type=int, default=3, help='config phase: structure top-K')
    p.add_argument('--out-dir', type=Path, default=RESULTS, help='where the new exp2 files go (trial runs: elsewhere)')
    args = p.parse_args()
    if args.limit and args.out_dir.resolve() == RESULTS.resolve():
        raise ValueError('--limit requires --out-dir outside experiments/results so trial runs never write result files')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dict(main=phase_main, config=phase_config, sweep=phase_sweep, baselines=phase_baselines, summarize=phase_summarize)[args.phase](args)


if __name__ == '__main__':
    main()
