"""Export scene/path data for the noise-Flow path comparison page.

Methods: noise Flow (N candidates, top-K, Euler steps; default N64/K3/S16),
Hybrid A* + IPOPT (offline result, the J reported in the tables), OCP GT and
Hybrid A* alone.  Scenes: evaluation scenes where all four succeed, spread
evenly over the noise-Flow J/GT ranking.  Noise-Flow paths are regenerated
with the same checkpoint, seed and settings as the evaluation run and their J
is checked against the recorded result.  Output: experiments/viewer_noise_flow/paths.json

  python tools/export_noise_flow_viewer_data.py --count 30
"""
from pathlib import Path
import argparse
import json
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'tools'))

import numpy as np  # noqa: E402

from export_path_viewer_data import to_scene, thin, pack, rd, R, DATASET, CACHE, SCENES  # noqa: E402

TARGETS = ROOT/'data/ocp_hybrid_astar/all_v1'


def stats(values):
    a = np.asarray(values, float)
    return dict(min=float(a.min()), median=float(np.median(a)), max=float(a.max()), mean=float(a.mean()), n=int(len(a)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=30)
    p.add_argument('--result', default='exp2_noise_flow_bspline64_N64_S16_K3.jsonl')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'run-logs/noise_flow_gt_bspline64_v1/best.pt')
    p.add_argument('--out', type=Path, default=ROOT/'experiments/viewer_noise_flow/paths.json')
    args = p.parse_args()
    from k4_flow_solver.config import SolverConfig
    from k4_flow_solver.numerical import decode, from_teacher, time_trace, cost
    from k4_flow_solver.conditions import occupancy_rectangles
    from k4_flow_solver.warmstart import load_warm_start, controls_from_hybrid_astar
    from k4_flow_solver.noise_flow_solver import NoiseFlowSolver
    from k4_flow_solver.prepare import load_scene
    manifest = json.loads((DATASET/'manifest.json').read_text(encoding='utf-8'))
    cfg = SolverConfig(**manifest['config'])
    uids = json.loads((ROOT/'experiments/eval_uids.json').read_text(encoding='utf-8'))['uids']
    rows = rd(args.result)
    meta, flow = rows[0], {r['uid']: r for r in rows[1:]}
    off = {}
    for r in rd('offline_methods.jsonl'):
        off.setdefault(r['method'], {})[r['uid']] = r
    ip, gt, ha = off['2_hybrid_astar_ipopt'], off['3_ocp_gt'], off['4_hybrid_astar']
    hastar_t = {r['uid']: r['plan_s'] for r in rd('timing_hastar.jsonl')}
    ipopt_t = {r['uid']: r['solve_s'] for r in rd('timing_ipopt.jsonl')}
    common = [u for u in uids if flow[u]['success'] and ip[u]['success'] and gt[u]['success'] and ha[u]['success']]
    ranked = sorted(common, key=lambda u: flow[u]['J']/gt[u]['J'])
    picks = [ranked[i] for i in np.linspace(0, len(ranked)-1, args.count).round().astype(int)]

    # Summary over the evaluation set (same definitions as the reported tables).
    total = {'flow': lambda u: flow[u]['timings']['total_s'], 'ipopt': lambda u: hastar_t[u]+(ipopt_t.get(u) or 0.0),
             'hastar': lambda u: hastar_t[u]}
    J = {'flow': lambda u: flow[u]['J'], 'ipopt': lambda u: ip[u]['J'], 'gt': lambda u: gt[u]['J'], 'hastar': lambda u: ha[u]['J']}
    ok = {'flow': lambda u: flow[u]['success'], 'ipopt': lambda u: ip[u]['success'], 'gt': lambda u: gt[u]['success'],
          'hastar': lambda u: ha[u]['success']}
    summary = {}
    for key in ('flow', 'ipopt', 'gt', 'hastar'):
        success = sum(ok[key](u) for u in uids)
        entry = dict(success=success, success_rate=success/len(uids), J_common=stats([J[key](u) for u in common]),
                     J_over_gt_common=stats([J[key](u)/gt[u]['J'] for u in common]))
        if key in total:
            entry['time_all'] = stats([total[key](u) for u in uids])
        summary[key] = entry
    failures = {}
    for u in uids:
        if not flow[u]['success']:
            failures[flow[u]['scene_failure']] = failures.get(flow[u]['scene_failure'], 0)+1

    gt_rows = {}
    for shard in manifest['shards']:
        if shard['split'] != 'validation':
            continue
        with np.load(DATASET/shard['file'], allow_pickle=False) as z:
            for i, u in enumerate(z['uids']):
                if str(u) in picks:
                    gt_rows[str(u)] = (z['latent'][i], int(z['structure'][i]))
    solver = NoiseFlowSolver(args.checkpoint, device='cuda', validation_workers=8)
    solver.set_goal_tolerance(.1, math.radians(5))
    solver.warmup(load_scene(SCENES/picks[0]), candidates=meta['candidates'], top_k=meta['top_k'], steps=meta['steps'])
    scenes, mismatches = [], []
    for rank, uid in enumerate(picks):
        scene = load_scene(SCENES/uid)
        trajectory, report = solver.plan(scene, candidates=meta['candidates'], top_k=meta['top_k'], steps=meta['steps'],
                                         seed=meta['seed'], keep_candidates=False)
        if report['status'] != 'valid_trajectory' or abs(report['selected_J']-flow[uid]['J']) > 1e-6*flow[uid]['J']:
            mismatches.append(dict(uid=uid, recorded=flow[uid]['J'], regenerated=report['selected_J']))
        if trajectory is None:
            continue
        start = np.asarray(scene['start_state'], float)
        rects = [[round(10-r1*0.1, 2), round(10-r0*0.1, 2), round(-10+c0*0.1, 2), round(-10+c1*0.1, 2)]
                 for r0, r1, c0, c1 in occupancy_rectangles(scene['obstacle_physical'])]
        ha_path, ha_meta = load_warm_start(CACHE/(uid+'.npz'))
        controls = dict(hastar=controls_from_hybrid_astar(ha_path, ha_meta['vehicle']['wheel_base'], cfg)[0],
                        ipopt=from_teacher(TARGETS/'samples'/uid[:3]/uid/'trajectory.npz', cfg)[0],
                        gt=decode(*gt_rows[uid], cfg))
        methods = {}
        for key, c in controls.items():
            methods[key] = dict(path=pack(thin(to_scene(time_trace(c, cfg), start))), J=round(J[key](uid), 3),
                                duration=round(cost(c, cfg)['duration_s'], 2), gear_changes=int(len(c.gears)-1))
        tr = np.asarray(trajectory)
        gears = [int(g) for i, g in enumerate(tr[:, 6]) if i == 0 or g != tr[i-1, 6]]
        methods['flow'] = dict(path=pack(thin(to_scene(tr, start))), J=round(report['selected_J'], 3),
                               duration=round(float(tr[-1, 0]), 2), gear_changes=len(gears)-1,
                               structure_rank=next((c for c in [flow[uid]['ranked_structures'].index(report['selected_structure'])]), None))
        timing = dict(flow=round(flow[uid]['timings']['total_s'], 3), hastar=round(hastar_t[uid], 3),
                      ipopt=round(ipopt_t[uid], 3) if uid in ipopt_t else None)
        scenes.append(dict(uid=uid, rank=rank, flow_over_gt=round(report['selected_J']/gt[uid]['J'], 3),
                           start=[round(float(v), 3) for v in start[:3]],
                           goal=[round(float(v), 3) for v in scene['requested_goal_pose'][:3]],
                           lot=np.round(scene['parking_lot_polygon_xy'], 3).tolist(),
                           slot=np.round(scene['target_slot_polygon_xy'], 3).tolist(),
                           obstacles=rects, methods=methods, timing=timing))
        print(uid, 'NF/GT', scenes[-1]['flow_over_gt'], flush=True)
    solver.close()
    out = dict(schema='path-viewer-v1', variant='noise-flow', scenes=scenes,
               frame='BEV scene frame: x forward, y ego-right; rear-axle poses',
               vehicle=dict(length=4.7917795, width=2.16345, rear_axle_to_center=1.41563697),
               eval_scenes=len(uids), common_success=len(common), summary=summary, flow_failures=failures,
               flow_setting=dict(candidates=meta['candidates'], top_k=meta['top_k'], steps=meta['steps'], epoch=meta['epoch']),
               regeneration_mismatches=mismatches,
               sources=dict(flow=args.result, ipopt='offline_methods.jsonl + data/ocp_hybrid_astar/all_v1 trajectories',
                            timing='timing_hastar.jsonl + timing_ipopt.jsonl (single-process re-measurement)'))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(json.dumps(out, separators=(',', ':')).encode('utf-8'))
    print('wrote', args.out, round(args.out.stat().st_size/1e6, 2), 'MB', 'mismatches', len(mismatches))


if __name__ == '__main__':
    main()
