"""Experiments 1-4 on experiments/eval_uids.json with the common evaluator.

  1  Hybrid A* + Flow solver (mode A: HA* structure fixed; mode B: predicted)
  2  Hybrid A* + IPOPT (lab fast multiphase OCP, HA* gear order fixed, offline setting)
  3  OCP GT (original training GT; reference solution, no runtime measured)
  4  Hybrid A* alone (retimed HA* controls)

Every method is scored by experiment_eval (numerical.validate, goal tolerance
0.1 m / 5 deg).  Phases (run in this order; all write experiments/results/):

  offline      J/feasibility of methods 2-4 from existing artifacts (parallel)
  flow         method 1 with a checkpoint, single process, CUDA-synchronized timing
  time-hastar  re-plan HA* single process (clean timing; checks the cached path is reproduced)
  time-ipopt   re-solve OCP single process on a subset (clean timing)
  flow-ipopt   HA* -> Flow -> IPOPT in one process (superseded: OCP timing shared the process with the GPU model)
  flow-save / ipopt-from-flow
               the same pipeline measured in separate processes (Flow outputs saved, then OCP alone)
  summarize    tables in experiments/results/summary.json
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
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'tools'))

import numpy as np  # noqa: E402

DATASET = ROOT/'data/k4_numeric_flow_full_v1_20260917'
SCENES = ROOT/'data/flow_scenes'
CACHE = ROOT/'data/warmstart_cache/hybrid_astar_e3000_v1'
TARGETS = ROOT/'data/ocp_hybrid_astar/all_v1'
RESULTS = ROOT/'experiments/results'


def eval_uids(path=ROOT/'experiments/eval_uids.json'):
    return json.loads(Path(path).read_text(encoding='utf-8'))['uids']


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, default=float)+'\n')


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text(encoding='utf-8').splitlines() if l.strip()]


def gt_table(uids):
    manifest = json.loads((DATASET/'manifest.json').read_text(encoding='utf-8'))
    wanted, table = set(uids), {}
    for shard in manifest['shards']:
        if shard['split'] != 'validation':
            continue
        with np.load(DATASET/shard['file'], allow_pickle=False) as z:
            for i, u in enumerate(z['uids']):
                if str(u) in wanted:
                    table[str(u)] = dict(latent=z['latent'][i], structure=int(z['structure'][i]), target_J=float(z['target_J'][i]))
    return table


def offline_scene(task):
    """Methods 2-4 for one uid from cached artifacts."""
    uid, gt = task
    from k4_flow_solver.config import SolverConfig
    from k4_flow_solver.numerical import decode, from_teacher
    from k4_flow_solver.prepare import load_scene
    from k4_flow_solver.experiment_eval import evaluate_controls, evaluate_path, timing_record
    from k4_flow_solver.warmstart import load_warm_start
    manifest = json.loads((DATASET/'manifest.json').read_text(encoding='utf-8'))
    cfg = SolverConfig(**manifest['config'])
    scene = load_scene(SCENES/uid)
    out = []
    # 3: OCP GT (reference; its collection time is not part of the release)
    gt_controls = decode(gt['latent'], gt['structure'], cfg)
    r3 = evaluate_controls(gt_controls, scene, cfg=cfg)
    out.append(dict(uid=uid, method='3_ocp_gt', success=r3['verified_feasible'], J=r3['J'], gt_teacher_J=gt['target_J'],
                    eval=r3, time=timing_record(warm_start_s=None, optimizer_s=None, device='n/a', warmup_excluded=False)))
    warm = CACHE/(uid+'.npz')
    if not warm.is_file():
        for m in ('2_hybrid_astar_ipopt', '4_hybrid_astar'):
            out.append(dict(uid=uid, method=m, success=False, J=None, reason='hybrid_astar_failed',
                            time=timing_record(warm_start_s=None, optimizer_s=None, device='cpu', warmup_excluded=False)))
        return out
    path, meta = load_warm_start(warm)
    ha_s = float(meta['plan_time_s'])
    r4 = evaluate_path(path, scene, cfg=cfg, wheelbase_m=meta['vehicle']['wheel_base'])
    out.append(dict(uid=uid, method='4_hybrid_astar', success=r4['verified_feasible'], J=r4['J'], eval=r4,
                    time=timing_record(warm_start_s=ha_s, optimizer_s=0.0, device='cpu', warmup_excluded=False),
                    time_scope='cached HA* time measured with 12 parallel planners'))
    result_file = TARGETS/'samples'/uid[:3]/uid/'result.json'
    record = json.loads(result_file.read_text(encoding='utf-8')) if result_file.is_file() else {}
    if record.get('success'):
        controls, _ = from_teacher(result_file.parent/'trajectory.npz', cfg)
        r2 = evaluate_controls(controls, scene, cfg=cfg)
        out.append(dict(uid=uid, method='2_hybrid_astar_ipopt', success=r2['verified_feasible'], J=r2['J'], eval=r2,
                        ocp_objective=record['best']['objective'], stages=len(record['stages']),
                        time=timing_record(warm_start_s=ha_s, optimizer_s=record.get('solve_s'), device='cpu', warmup_excluded=False),
                        time_scope='cached HA* (12 parallel) + OCP (14 parallel workers)'))
    else:
        out.append(dict(uid=uid, method='2_hybrid_astar_ipopt', success=False, J=None,
                        reason=record.get('failure_reason', 'no_result'),
                        time=timing_record(warm_start_s=ha_s, optimizer_s=record.get('solve_s'), device='cpu', warmup_excluded=False)))
    return out


def phase_offline(args):
    uids = eval_uids()
    gt = gt_table(uids)
    with ProcessPoolExecutor(args.workers) as pool:
        rows = [r for block in pool.map(offline_scene, [(u, gt[u]) for u in uids]) for r in block]
    write_jsonl(RESULTS/'offline_methods.jsonl', rows)
    print(json.dumps(dict(phase='offline', rows=len(rows))))


def phase_flow(args):
    import torch
    from k4_flow_solver.prepare import load_scene
    from k4_flow_solver.warmstart import load_warm_start
    from k4_flow_solver.warmstart_solver import WarmStartFlowSolver
    from k4_flow_solver.experiment_eval import timing_record, GOAL_POSITION_M, GOAL_YAW_DEG
    uids = eval_uids()
    solver = WarmStartFlowSolver(args.checkpoint, device=args.device, validation_workers=args.validation_workers)
    solver.set_goal_tolerance(GOAL_POSITION_M, math.radians(GOAL_YAW_DEG))
    first = next(u for u in uids if (CACHE/(u+'.npz')).is_file())
    solver.warmup(load_scene(SCENES/first), load_warm_start(CACHE/(first+'.npz'))[0])
    ha_clean = {r['uid']: r['plan_s'] for r in read_jsonl(RESULTS/'timing_hastar.jsonl')} if (RESULTS/'timing_hastar.jsonl').is_file() else {}
    for mode in args.modes:
        rows = []
        for i, uid in enumerate(uids):
            warm = CACHE/(uid+'.npz')
            if not warm.is_file():
                rows.append(dict(uid=uid, method=f'1_flow_{mode}', success=False, J=None, reason='hybrid_astar_failed'))
                continue
            path, meta = load_warm_start(warm)
            _, report = solver.plan(load_scene(SCENES/uid), path, top_k=2, samples_per_structure=8, steps=16,
                                    seed=args.seed, include_warm_start=False, structure_mode=mode)
            report.pop('candidates')
            rows.append(dict(uid=uid, method=f'1_flow_{mode}', success=report['status'] == 'valid_trajectory',
                             J=report['selected_J'], report=report,
                             time=timing_record(warm_start_s=ha_clean.get(uid, meta['plan_time_s']),
                                                optimizer_s=report['timings']['total_s'], device=args.device,
                                                warmup_excluded=report['warmup_excluded'])))
            if (i+1) % 50 == 0:
                print(json.dumps(dict(mode=mode, done=i+1, valid=sum(r['success'] for r in rows))), flush=True)
        write_jsonl(RESULTS/f'exp1_flow_{mode}.jsonl', rows)
    solver.close()
    torch.cuda.empty_cache() if args.device == 'cuda' else None


def phase_time_hastar(args):
    from k4_flow_solver import warmstart as ws
    uids = eval_uids()
    rows = []
    pcfg = ws.planner_config(**ws.AGREED_PLANNER_OVERRIDES)
    for i, uid in enumerate(uids):
        with np.load(SCENES/uid/'planner_input.npz', allow_pickle=False) as z:
            raw = {k: z[k] for k in z.files}
        path, info = ws.plan_hybrid_astar(raw, pcfg, start_direction=1 if float(raw['start_state'][3]) >= 0 else -1)
        cached = CACHE/(uid+'.npz')
        same = bool(path is not None and cached.is_file() and np.array_equal(path, ws.load_warm_start(cached)[0]))
        rows.append(dict(uid=uid, success=path is not None, plan_s=info['plan_time_s'], expansions=info['expansions'],
                         identical_to_cache=same if cached.is_file() else (path is None)))
        if (i+1) % 50 == 0:
            print(json.dumps(dict(phase='time-hastar', done=i+1)), flush=True)
    write_jsonl(RESULTS/'timing_hastar.jsonl', rows)


def phase_time_ipopt(args):
    from solve_hybrid_astar_ocp import solve_scene
    from k4_flow_solver.config import SolverConfig
    from k4_flow_solver.numerical import from_teacher
    from k4_flow_solver.prepare import load_scene
    from k4_flow_solver.experiment_eval import evaluate_controls
    manifest = json.loads((DATASET/'manifest.json').read_text(encoding='utf-8'))
    cfg = SolverConfig(**manifest['config'])
    uids = [u for u in eval_uids() if (CACHE/(u+'.npz')).is_file()][:args.subset]
    out_dir = RESULTS/'ipopt_rerun'
    rows = []
    for uid in uids:
        r = solve_scene((uid, str(CACHE), str(SCENES), str(out_dir)))
        row = dict(uid=uid, success=bool(r['success']), solve_s=r.get('solve_s'), stages=len(r.get('stages', [])))
        if r['success']:
            controls, _ = from_teacher(out_dir/'samples'/uid[:3]/uid/'trajectory.npz', cfg)
            ev = evaluate_controls(controls, load_scene(SCENES/uid), cfg=cfg)
            row.update(J=ev['J'], verified=ev['verified_feasible'])
        rows.append(row)
    write_jsonl(RESULTS/'timing_ipopt.jsonl', rows)


def phase_flow_ipopt(args):
    """HA* -> Flow (mode A) -> IPOPT seeded by the Flow trajectory (HA* path if Flow found none)."""
    from solve_hybrid_astar_ocp import solve_scene
    from k4_flow_solver.config import SolverConfig
    from k4_flow_solver.numerical import from_teacher
    from k4_flow_solver.prepare import load_scene
    from k4_flow_solver.warmstart import load_warm_start
    from k4_flow_solver.warmstart_solver import WarmStartFlowSolver
    from k4_flow_solver.experiment_eval import evaluate_controls, timing_record, GOAL_POSITION_M, GOAL_YAW_DEG
    manifest = json.loads((DATASET/'manifest.json').read_text(encoding='utf-8'))
    cfg = SolverConfig(**manifest['config'])
    uids = eval_uids()
    solver = WarmStartFlowSolver(args.checkpoint, device=args.device, validation_workers=args.validation_workers)
    solver.set_goal_tolerance(GOAL_POSITION_M, math.radians(GOAL_YAW_DEG))
    first = next(u for u in uids if (CACHE/(u+'.npz')).is_file())
    solver.warmup(load_scene(SCENES/first), load_warm_start(CACHE/(first+'.npz'))[0])
    ha_clean = {r['uid']: r['plan_s'] for r in read_jsonl(RESULTS/'timing_hastar.jsonl')}
    out_dir = RESULTS/'flow_ipopt'
    rows = []
    for i, uid in enumerate(uids):
        if not (CACHE/(uid+'.npz')).is_file():
            rows.append(dict(uid=uid, method='5_flow_ipopt', success=False, J=None, reason='hybrid_astar_failed'))
            continue
        scene = load_scene(SCENES/uid)
        path, _ = load_warm_start(CACHE/(uid+'.npz'))
        trajectory, report = solver.plan(scene, path, top_k=2, samples_per_structure=8, steps=16, seed=args.seed,
                                         include_warm_start=False, structure_mode='warm_start')
        flow_s = report['timings']['total_s']
        if trajectory is not None:
            warm, source = trajectory[:, [1, 2, 3, 6, 5]], 'flow'      # [x, y, yaw, gear, steer]
        else:
            warm, source = path, 'hybrid_astar_fallback'
        r = solve_scene((uid, str(CACHE), str(SCENES), str(out_dir), warm, source))
        row = dict(uid=uid, method='5_flow_ipopt', warm_start_source=source, flow_valid=trajectory is not None,
                   flow_J=report['selected_J'], ipopt_success=bool(r['success']), solve_s=r.get('solve_s'),
                   stages=len(r.get('stages', [])), flow_s=flow_s, success=False, J=None,
                   reason=r.get('failure_reason'))
        if r['success']:
            controls, _ = from_teacher(out_dir/'samples'/uid[:3]/uid/'trajectory.npz', cfg)
            ev = evaluate_controls(controls, scene, cfg=cfg)
            row.update(success=ev['verified_feasible'], J=ev['J'], eval=ev)
        row['time'] = timing_record(warm_start_s=ha_clean[uid], optimizer_s=flow_s+(r.get('solve_s') or 0.0),
                                    device=args.device, warmup_excluded=report['warmup_excluded'])
        rows.append(row)
        if (i+1) % 50 == 0:
            print(json.dumps(dict(phase='flow-ipopt', done=i+1, success=sum(x['success'] for x in rows))), flush=True)
    write_jsonl(RESULTS/'exp5_flow_ipopt.jsonl', rows)
    solver.close()


def phase_flow_save(args):
    """Stage 1 of the separated HA* -> Flow -> IPOPT measurement: Flow outputs and Flow timing only."""
    from k4_flow_solver.prepare import load_scene
    from k4_flow_solver.warmstart import load_warm_start
    from k4_flow_solver.warmstart_solver import WarmStartFlowSolver
    from k4_flow_solver.experiment_eval import GOAL_POSITION_M, GOAL_YAW_DEG
    uids = eval_uids()
    out_dir = RESULTS/'flow_trajectories'
    out_dir.mkdir(parents=True, exist_ok=True)
    solver = WarmStartFlowSolver(args.checkpoint, device=args.device, validation_workers=args.validation_workers)
    solver.set_goal_tolerance(GOAL_POSITION_M, math.radians(GOAL_YAW_DEG))
    first = next(u for u in uids if (CACHE/(u+'.npz')).is_file())
    solver.warmup(load_scene(SCENES/first), load_warm_start(CACHE/(first+'.npz'))[0])
    rows = []
    for uid in uids:
        if not (CACHE/(uid+'.npz')).is_file():
            rows.append(dict(uid=uid, hybrid_astar=False))
            continue
        path, _ = load_warm_start(CACHE/(uid+'.npz'))
        trajectory, report = solver.plan(load_scene(SCENES/uid), path, top_k=2, samples_per_structure=8, steps=16,
                                         seed=args.seed, include_warm_start=False, structure_mode='warm_start')
        if trajectory is not None:
            np.savez_compressed(out_dir/(uid+'.npz'), path=trajectory[:, [1, 2, 3, 6, 5]], trajectory=trajectory)
        rows.append(dict(uid=uid, hybrid_astar=True, flow_valid=trajectory is not None, flow_J=report['selected_J'],
                         flow_s=report['timings']['total_s'], warmup_excluded=report['warmup_excluded']))
    solver.close()
    write_jsonl(RESULTS/'exp5_flow_outputs.jsonl', rows)
    print(json.dumps(dict(phase='flow-save', valid=sum(bool(r.get('flow_valid')) for r in rows))), flush=True)


def phase_ipopt_from_flow(args):
    """Stage 2: OCP seeded by the saved Flow paths, in a process without the Flow model (same
    conditions as time-ipopt).  Scenes where Flow found no valid candidate use the HA* path."""
    from solve_hybrid_astar_ocp import solve_scene
    from k4_flow_solver.config import SolverConfig
    from k4_flow_solver.numerical import from_teacher
    from k4_flow_solver.prepare import load_scene
    from k4_flow_solver.warmstart import load_warm_start
    from k4_flow_solver.experiment_eval import evaluate_controls, timing_record
    manifest = json.loads((DATASET/'manifest.json').read_text(encoding='utf-8'))
    cfg = SolverConfig(**manifest['config'])
    flows = {r['uid']: r for r in read_jsonl(RESULTS/'exp5_flow_outputs.jsonl')}
    ha_clean = {r['uid']: r['plan_s'] for r in read_jsonl(RESULTS/'timing_hastar.jsonl')}
    out_dir = RESULTS/'flow_ipopt_v2'
    rows = []
    for i, uid in enumerate(eval_uids()):
        f = flows[uid]
        if not f['hybrid_astar']:
            rows.append(dict(uid=uid, method='5_flow_ipopt', success=False, J=None, reason='hybrid_astar_failed'))
            continue
        if f['flow_valid']:
            with np.load(RESULTS/'flow_trajectories'/(uid+'.npz'), allow_pickle=False) as z:
                warm, source = z['path'].copy(), 'flow'
        else:
            warm, source = load_warm_start(CACHE/(uid+'.npz'))[0], 'hybrid_astar_fallback'
        r = solve_scene((uid, str(CACHE), str(SCENES), str(out_dir), warm, source))
        row = dict(uid=uid, method='5_flow_ipopt', warm_start_source=source, flow_valid=f['flow_valid'], flow_J=f['flow_J'],
                   ipopt_success=bool(r['success']), solve_s=r.get('solve_s'), stages=len(r.get('stages', [])),
                   flow_s=f['flow_s'], success=False, J=None, reason=r.get('failure_reason'),
                   measurement='Flow and OCP in separate processes; OCP process without the Flow model')
        if r['success']:
            controls, _ = from_teacher(out_dir/'samples'/uid[:3]/uid/'trajectory.npz', cfg)
            ev = evaluate_controls(controls, load_scene(SCENES/uid), cfg=cfg)
            row.update(success=ev['verified_feasible'], J=ev['J'], eval=ev)
        row['time'] = timing_record(warm_start_s=ha_clean[uid], optimizer_s=f['flow_s']+(r.get('solve_s') or 0.0),
                                    device='cuda+cpu', warmup_excluded=f['warmup_excluded'])
        rows.append(row)
        if (i+1) % 50 == 0:
            print(json.dumps(dict(phase='ipopt-from-flow', done=i+1, success=sum(x['success'] for x in rows))), flush=True)
    write_jsonl(RESULTS/'exp5_flow_ipopt_v2.jsonl', rows)


def stats(values):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    return dict(n=int(len(a)), mean=float(a.mean()), median=float(np.median(a)), p95=float(np.quantile(a, .95))) if len(a) else dict(n=0)


def phase_summarize(args):
    rows = read_jsonl(RESULTS/'offline_methods.jsonl')
    exp5 = 'exp5_flow_ipopt_v2' if (RESULTS/'exp5_flow_ipopt_v2.jsonl').is_file() else 'exp5_flow_ipopt'
    for name in ('exp1_flow_warm_start', 'exp1_flow_predicted', exp5):
        f = RESULTS/f'{name}.jsonl'
        if f.is_file():
            rows += read_jsonl(f)
    by = {}
    for r in rows:
        by.setdefault(r['method'], {})[r['uid']] = r
    uids = eval_uids()
    gt = by['3_ocp_gt']
    ha_clean = {r['uid']: r for r in read_jsonl(RESULTS/'timing_hastar.jsonl')} if (RESULTS/'timing_hastar.jsonl').is_file() else {}
    ipopt_clean = {r['uid']: r for r in read_jsonl(RESULTS/'timing_ipopt.jsonl')} if (RESULTS/'timing_ipopt.jsonl').is_file() else {}
    all_ok = [u for u in uids if all(by[m].get(u, {}).get('success') for m in by)]
    table = {}
    for m in sorted(by):
        res = by[m]
        ok = [u for u in uids if res.get(u, {}).get('success')]
        paired = [u for u in ok if gt[u]['success']]
        entry = dict(scenes=len(uids), success=len(ok), success_rate=len(ok)/len(uids),
                     J=stats([res[u]['J'] for u in ok]),
                     J_over_gt=stats([res[u]['J']/gt[u]['J'] for u in paired]),
                     J_over_gt_common=stats([res[u]['J']/gt[u]['J'] for u in all_ok]),
                     J_common=stats([res[u]['J'] for u in all_ok]))
        if m == '5_flow_ipopt':
            entry['flow_s'] = stats([res[u]['flow_s'] for u in uids if 'flow_s' in res[u]])
            entry['ipopt_s'] = stats([res[u]['solve_s'] for u in uids if res[u].get('solve_s') is not None])
            entry['warm_start_from_flow'] = sum(res[u].get('warm_start_source') == 'flow' for u in uids)
            entry['warm_start_fallback_hybrid_astar'] = sum(res[u].get('warm_start_source') == 'hybrid_astar_fallback' for u in uids)
        if m.startswith('1_flow'):
            entry['flow_s'] = stats([res[u]['time']['optimizer_s'] for u in ok])
            entry['flow_s_all'] = stats([res[u]['time']['optimizer_s'] for u in uids if res[u].get('time')])
        if ha_clean and m != '3_ocp_gt':
            entry['hybrid_astar_clean_s'] = stats([ha_clean[u]['plan_s'] for u in uids if u in ha_clean])
        if m == '2_hybrid_astar_ipopt':
            entry['ipopt_s_parallel_load'] = stats([res[u]['time']['optimizer_s'] for u in ok])
            if ipopt_clean:
                entry['ipopt_clean_subset'] = dict(n=len(ipopt_clean), solve_s=stats([r['solve_s'] for r in ipopt_clean.values()]),
                                                   verified=sum(bool(r.get('verified')) for r in ipopt_clean.values()))
        table[m] = entry
    summary = dict(eval_uids='experiments/eval_uids.json', scenes=len(uids), common_success_scenes=len(all_ok),
                   goal_tolerance=dict(position_m=0.1, yaw_deg=5.0), methods=table,
                   hybrid_astar_reproduced=(sum(r['identical_to_cache'] for r in ha_clean.values()) if ha_clean else None),
                   notes=['J over GT uses the common evaluator on both; GT = original OCP GT (experiment 3)',
                          'common = scenes where every method succeeds',
                          'not an independent test set (validation split)'])
    (RESULTS/'summary.json').write_bytes(json.dumps(summary, indent=1).encode('utf-8'))
    print(json.dumps(summary, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('phase', choices=('offline', 'flow', 'flow-ipopt', 'flow-save', 'ipopt-from-flow', 'time-hastar',
                                     'time-ipopt', 'summarize'))
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--checkpoint', default='run-logs/warmstart_flow_full_s0.02_v1/best.pt')
    p.add_argument('--device', default='cuda')
    p.add_argument('--validation-workers', type=int, default=8)
    p.add_argument('--modes', nargs='+', default=['warm_start', 'predicted'])
    p.add_argument('--seed', type=int, default=20260917)
    p.add_argument('--subset', type=int, default=100)
    args = p.parse_args()
    dict(offline=phase_offline, flow=phase_flow, **{'flow-ipopt': phase_flow_ipopt, 'flow-save': phase_flow_save,
                                                    'ipopt-from-flow': phase_ipopt_from_flow, 'time-hastar': phase_time_hastar,
                                                    'time-ipopt': phase_time_ipopt},
         summarize=phase_summarize)[args.phase](args)


if __name__ == '__main__':
    main()
