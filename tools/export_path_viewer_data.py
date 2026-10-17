"""Export scene/path data for the path comparison page (experiments/viewer/paths.json).

Scenes: evaluation scenes where all four methods succeed, spread evenly over
the Flow J/GT ranking.  Paths are expressed in the scene (BEV) frame:
x forward, y ego-right, rear-axle poses.  Every trajectory is re-derived with
numerical.time_trace from the same controls the common evaluator scored.

  python tools/export_path_viewer_data.py --count 30
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))

import numpy as np  # noqa: E402

R = ROOT/'experiments/results'
DATASET = ROOT/'data/k4_numeric_flow_full_v1_20260917'
CACHE = ROOT/'data/warmstart_cache/hybrid_astar_e3000_v1'
SCENES = ROOT/'data/flow_scenes'


def rd(name):
    return [json.loads(l) for l in (R/name).read_text(encoding='utf-8').splitlines() if l.strip()]


def to_scene(trace, start):
    """time_trace rows [t, x, y, yaw, v, steer, gear] (start-local) -> scene frame."""
    c, s = np.cos(start[2]), np.sin(start[2])
    x = start[0]+c*trace[:, 1]-s*trace[:, 2]
    y = start[1]+s*trace[:, 1]+c*trace[:, 2]
    return np.c_[trace[:, 0], x, y, trace[:, 3]+start[2], np.abs(trace[:, 4]), trace[:, 6]]


def thin(rows, limit=260):
    """Keep gear-change rows and an even subset of the rest."""
    if len(rows) <= limit:
        return rows
    keep = set(np.linspace(0, len(rows)-1, limit).round().astype(int).tolist())
    keep |= set((np.flatnonzero(np.diff(rows[:, 5]) != 0)).tolist()) | set((np.flatnonzero(np.diff(rows[:, 5]) != 0)+1).tolist())
    return rows[sorted(keep)]


def pack(rows):
    return dict(t=np.round(rows[:, 0], 3).tolist(), x=np.round(rows[:, 1], 3).tolist(), y=np.round(rows[:, 2], 3).tolist(),
                v=np.round(rows[:, 4], 3).tolist(), gear=rows[:, 5].astype(int).tolist())


def gear_changes(c):
    return int(len(c.gears)-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--count', type=int, default=30)
    p.add_argument('--out', type=Path, default=ROOT/'experiments/viewer/paths.json')
    args = p.parse_args()
    from k4_flow_solver.config import SolverConfig
    from k4_flow_solver.numerical import decode, from_teacher, time_trace, cost
    from k4_flow_solver.conditions import occupancy_rectangles
    from k4_flow_solver.warmstart import load_warm_start, controls_from_hybrid_astar
    manifest = json.loads((DATASET/'manifest.json').read_text(encoding='utf-8'))
    cfg = SolverConfig(**manifest['config'])
    uids = json.loads((ROOT/'experiments/eval_uids.json').read_text(encoding='utf-8'))['uids']
    off = {}
    for r in rd('offline_methods.jsonl'):
        off.setdefault(r['method'], {})[r['uid']] = r
    flow = {r['uid']: r for r in rd('exp1_flow_warm_start.jsonl')}
    ipopt = {r['uid']: r for r in rd('timing_ipopt.jsonl')}
    hastar_t = {r['uid']: r['plan_s'] for r in rd('timing_hastar.jsonl')}
    gt = off['3_ocp_gt']
    common = [u for u in uids if flow.get(u, {}).get('success') and ipopt.get(u, {}).get('verified')
              and gt[u]['success'] and off['4_hybrid_astar'][u]['success']
              and (R/'flow_trajectories'/(u+'.npz')).is_file()]
    ranked = sorted(common, key=lambda u: flow[u]['J']/gt[u]['J'])
    picks = [ranked[i] for i in np.linspace(0, len(ranked)-1, args.count).round().astype(int)]
    gt_rows = {}
    for shard in manifest['shards']:
        if shard['split'] != 'validation':
            continue
        with np.load(DATASET/shard['file'], allow_pickle=False) as z:
            for i, u in enumerate(z['uids']):
                if str(u) in picks:
                    gt_rows[str(u)] = (z['latent'][i], int(z['structure'][i]))
    scenes = []
    for rank, uid in enumerate(picks):
        with np.load(SCENES/uid/'planner_input.npz', allow_pickle=False) as z:
            raw = {k: z[k] for k in ('obstacle_physical', 'start_state', 'requested_goal_pose',
                                     'parking_lot_polygon_xy', 'target_slot_polygon_xy')}
        start = raw['start_state'].astype(float)
        rects = [[round(10-r1*0.1, 2), round(10-r0*0.1, 2), round(-10+c0*0.1, 2), round(-10+c1*0.1, 2)]
                 for r0, r1, c0, c1 in occupancy_rectangles(raw['obstacle_physical'])]
        methods = {}
        ha_path, _ = load_warm_start(CACHE/(uid+'.npz'))
        ha_controls, _ = controls_from_hybrid_astar(ha_path, cfg=cfg)
        controls = dict(hastar=ha_controls,
                        ipopt=from_teacher(R/'ipopt_rerun'/'samples'/uid[:3]/uid/'trajectory.npz', cfg)[0],
                        gt=decode(*gt_rows[uid], cfg))
        for key, c in controls.items():
            methods[key] = dict(path=pack(thin(to_scene(time_trace(c, cfg), start))), J=round(cost(c, cfg)['J'], 3),
                                duration=round(cost(c, cfg)['duration_s'], 2), gear_changes=gear_changes(c))
        with np.load(R/'flow_trajectories'/(uid+'.npz'), allow_pickle=False) as z:
            flow_trace = z['trajectory']
        fl_gears = [int(g) for i, g in enumerate(flow_trace[:, 6]) if i == 0 or g != flow_trace[i-1, 6]]
        methods['flow'] = dict(path=pack(thin(to_scene(flow_trace, start))), J=round(flow[uid]['J'], 3),
                               duration=round(float(flow_trace[-1, 0]), 2), gear_changes=len(fl_gears)-1)
        # Common-evaluator J (what the results tables report) overrides the local cost where available.
        methods['hastar']['J'] = round(off['4_hybrid_astar'][uid]['J'], 3)
        methods['ipopt']['J'] = round(ipopt[uid]['J'], 3)
        methods['gt']['J'] = round(gt[uid]['J'], 3)
        timing = dict(hastar=round(hastar_t[uid], 2), flow=round(flow[uid]['time']['optimizer_s'], 2),
                      ipopt=round(ipopt[uid]['solve_s'], 2))
        scenes.append(dict(uid=uid, rank=rank, flow_over_gt=round(flow[uid]['J']/gt[uid]['J'], 3),
                           start=[round(v, 3) for v in start[:3]], goal=[round(float(v), 3) for v in raw['requested_goal_pose'][:3]],
                           lot=np.round(raw['parking_lot_polygon_xy'], 3).tolist(),
                           slot=np.round(raw['target_slot_polygon_xy'], 3).tolist(),
                           obstacles=rects, methods=methods, timing=timing))
        print(uid, 'Flow/GT', scenes[-1]['flow_over_gt'], flush=True)
    summary = json.loads((R/'summary.json').read_text(encoding='utf-8'))
    out = dict(schema='path-viewer-v1', scenes=scenes, frame='BEV scene frame: x forward, y ego-right; rear-axle poses',
               vehicle=dict(length=4.7917795, width=2.16345, rear_axle_to_center=1.41563697),
               eval_scenes=len(uids), common_success=len(common), summary_source=str(R/'summary.json'),
               summary_methods=list(summary['methods']))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, separators=(',', ':')), encoding='utf-8')
    print('wrote', args.out, round(args.out.stat().st_size/1e6, 2), 'MB')


if __name__ == '__main__':
    main()
