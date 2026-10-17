"""Re-time the OCP(GT) method on the evaluation scenes (single process).

The dataset GT came from scripts/collect_fast_ocp_gt.py: the source sample's
path.npy (scene frame) -> states_world_to_local -> multi-stage corridor OCP
(IPOPT, CPU 10 s per NLP, J < 14 / 1 % / 35 s stops).  The source folders are
not on this machine; each scene's planner_input.npz keeps that route as
``known_feasible_route`` (gear sequences agree with the GT once the GT's
sub-centimetre phases are ignored).  This tool seeds the identical OCP code
(tools/solve_hybrid_astar_ocp.solve_scene, same loop and settings) with that
route and records the wall time, then scores the result with the common
evaluator and compares it with the dataset GT.  The time of producing the
route itself is not included (it is not recorded anywhere).

  python tools/time_gt_ocp.py                 -> experiments/results/timing_gt_ocp.jsonl
  python tools/time_gt_ocp.py --limit 5 --out-dir <scratch>   (trial)
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
os.environ.setdefault('FAST_OCP_CPU', '10')
from pathlib import Path
import argparse
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(ROOT/'tools'))

import numpy as np  # noqa: E402

from run_experiments import CACHE, SCENES, RESULTS, eval_uids, gt_table  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', type=Path, default=RESULTS)
    p.add_argument('--limit', type=int)
    p.add_argument('--hastar', action='store_true', help='condition check: re-time HA* + IPOPT instead (trial only)')
    args = p.parse_args()
    if args.limit and args.out_dir.resolve() == RESULTS.resolve():
        raise ValueError('--limit requires --out-dir outside experiments/results')
    name = 'timing_hastar_ipopt_check.jsonl' if args.hastar else 'timing_gt_ocp.jsonl'
    target = args.out_dir/name
    work = args.out_dir/('hastar_ipopt_check' if args.hastar else 'gt_ocp_rerun')
    if target.exists() or work.exists():
        raise FileExistsError(f'{target} or {work} exists; not overwriting')
    from solve_hybrid_astar_ocp import solve_scene
    from hope_path_transformer.geometry import states_world_to_local
    from k4_flow_solver.config import SolverConfig
    from k4_flow_solver.numerical import from_teacher
    from k4_flow_solver.prepare import load_scene
    from k4_flow_solver.experiment_eval import evaluate_controls
    manifest = json.loads((ROOT/'data/k4_numeric_flow_full_v1_20260917/manifest.json').read_text(encoding='utf-8'))
    cfg = SolverConfig(**manifest['config'])
    uids = eval_uids()[:args.limit] if args.limit else eval_uids()
    gt = gt_table(uids)
    rows = []
    started = time.perf_counter()
    for i, uid in enumerate(uids):
        if args.hastar:
            if not (CACHE/(uid+'.npz')).is_file():
                continue
            task = (uid, str(CACHE), str(SCENES), str(work))
        else:
            with np.load(SCENES/uid/'planner_input.npz', allow_pickle=False) as z:
                warm = states_world_to_local(z['known_feasible_route'], z['start_state']).astype(float)
            task = (uid, str(CACHE), str(SCENES), str(work), warm, 'known_feasible_route')
        r = solve_scene(task)
        row = dict(uid=uid, success=bool(r['success']), solve_s=r.get('solve_s'), stages=len(r.get('stages', [])),
                   gears=r.get('gears'), stop_reason=r.get('stop_reason'), failure_reason=r.get('failure_reason'),
                   gt_J=gt[uid]['target_J'], gt_structure=gt[uid]['structure'])
        if r['success']:
            controls, _ = from_teacher(work/'samples'/uid[:3]/uid/'trajectory.npz', cfg)
            ev = evaluate_controls(controls, load_scene(SCENES/uid), cfg=cfg)
            row.update(J=ev['J'], verified=ev['verified_feasible'], ocp_objective=r['best']['objective'])
        rows.append(row)
        if (i+1) % 50 == 0:
            done = [x for x in rows if x.get('verified')]
            print(json.dumps(dict(done=i+1, verified=len(done), mean_solve_s=float(np.mean([x['solve_s'] for x in rows])),
                                  wall_s=time.perf_counter()-started)), flush=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('w', encoding='utf-8') as f:
        f.write(json.dumps(dict(record='meta', warm_start='known_feasible_route' if not args.hastar else 'hybrid_astar_e3000_v1',
                                code='tools/solve_hybrid_astar_ocp.solve_scene (same loop/settings as collect_fast_ocp_gt.run_scene)',
                                fast_ocp_cpu_s=float(os.environ['FAST_OCP_CPU']), process='single process, sequential',
                                wall_s=time.perf_counter()-started))+'\n')
        for row in rows:
            f.write(json.dumps(row, default=float)+'\n')
    ok = [x for x in rows if x.get('verified')]
    print(json.dumps(dict(file=str(target), scenes=len(rows), verified=len(ok),
                          mean_solve_s=float(np.mean([x['solve_s'] for x in rows])) if rows else None)))


if __name__ == '__main__':
    main()
