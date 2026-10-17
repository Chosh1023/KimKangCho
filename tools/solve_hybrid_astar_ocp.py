"""Solve the lab fast multiphase OCP from Hybrid A* warm starts (HA* gear order fixed).

Same solver settings as ``scripts/collect_fast_ocp_gt.py`` (multi-stage meshes,
IPOPT max_cpu_time FAST_OCP_CPU=10 s per NLP, stop after the first stage that
ends beyond 35 s, J < 14 early stop, < 1 % refinement stop).  Only the warm
start differs: the stage-1 Hybrid A* cache path instead of the sample's
``path.npy``.  The gear order is the one ``split_path`` extracts from the HA*
path, identical to the Flow x0 structure.

Uses: (1) cost of new x1 targets with the HA* structure, (2) experiment 2
(HA* + IPOPT) in its offline setting.  Source files are never modified.

  python tools/solve_hybrid_astar_ocp.py --out data/ocp_hybrid_astar/val199_v1 --workers 14
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('FAST_OCP_CPU', '10')
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import math
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))

import numpy as np  # noqa: E402

SCHEMA = 'hybrid-astar-fixed-gear-ocp-v1'
TIME_BUDGET_S = 35.0        # collect_fast_ocp_gt.py: break once a stage ends beyond 35 s


def solve_scene(task):
    """collect_fast_ocp_gt.run_scene with the Hybrid A* path as warm start.

    ``task`` = (uid, cache, scenes, output[, warm_path, label]).  With an
    explicit ``warm_path`` (N x 5, cache frame, e.g. a Flow trajectory) that
    path seeds the OCP instead of the cached Hybrid A* path; the training
    target conversion is skipped.
    """
    uid, cache, scenes, output = task[:4]
    warm_path = task[4] if len(task) > 4 else None
    label = task[5] if len(task) > 5 else 'hybrid_astar_e3000_v1'
    import fast_multiphase_ocp as model
    from collect_fast_ocp_gt import dump
    from hope_path_transformer.geometry import states_world_to_local
    from k4_flow_solver import warmstart as ws
    begin = time.perf_counter()
    directory = Path(output)/'samples'/uid[:3]/uid
    result_file = directory/'result.json'
    if result_file.is_file():
        old = json.loads(result_file.read_text(encoding='utf-8'))
        if not old.get('success') or (directory/'trajectory.npz').is_file():
            old['resumed'] = True
            return old
    result = dict(schema=SCHEMA, uid=uid, warm_start=label, success=False, stages=[],
                  global_optimum_certified=False, worker_pid=os.getpid(),
                  fast_ocp_cpu_s=float(os.environ['FAST_OCP_CPU']), time_budget_s=TIME_BUDGET_S)
    try:
        with np.load(Path(scenes)/uid/'planner_input.npz', allow_pickle=False) as z:
            scene = {k: z[k] for k in z.files}
        if warm_path is None:
            path, meta = ws.load_warm_start(Path(cache)/(uid+'.npz'))     # already start-local rear axle
            _, x0_info = ws.controls_from_hybrid_astar(path)
        else:
            path, x0_info = np.asarray(warm_path, dtype=np.float64), None
        requested = np.r_[scene['requested_goal_pose'], -1., 0.][None, :]
        goal = states_world_to_local(requested, scene['start_state'])[0, :3].astype(float)
        raw, gears = model.split_path(path)
        meshes, classic = model.meshes_for(raw, gears)
        goal[2] += 2*np.pi*round((raw[-1][0][-1, 2]-goal[2])/(2*np.pi))
        result.update(gears=gears, gear_changes=len(gears)-1, x0_gears=x0_info['gears'] if x0_info else gears,
                      classic_scope=classic,
                      goal_local=goal.tolist(),
                      scope='original forward/reverse benchmark' if classic else 'fixed HA* gear sequence')
        previous = None
        oldmesh = None
        best_arrays = None
        for stage, mesh in enumerate(meshes):
            prior_best = result.get('best', {}).get('objective')
            start = time.perf_counter()
            seed = model.seed_from_parts(raw, mesh) if previous is None else model.remesh(previous, oldmesh, mesh)
            lengths, parts, _ = model.unpack(seed, mesh)
            corridors = model.ocp.obstacle_corridors(scene, None, None, None, phase_parts=parts, include_far=True)
            problem = model.build(mesh[0], mesh[-1], goal, corridors,
                                  phases=None if classic else list(zip(mesh, gears)))
            solved = model.ocp.solve(problem, seed)
            z = solved.pop('z')
            entry = dict(mesh=[int(n) for n in mesh], solver=solved)
            result['stages'].append(entry)
            if solved['success']:
                previous = z
                oldmesh = mesh
                val, arrays = model.validate(z, mesh, gears, scene, goal, classic)
                entry['validation'] = val
                if val['verified_feasible'] and (not result['success'] or val['objective'] < result['best']['objective']):
                    result['success'] = True
                    result['best'] = val
                    best_arrays = arrays
            entry['total_s'] = time.perf_counter()-start
            if result['success'] and result['best']['objective'] < 14:
                break
            if stage >= 1 and solved['success'] and entry.get('validation', {}).get('verified_feasible') and prior_best is not None:
                if (prior_best-result['best']['objective'])/prior_best < .01:
                    result['stop_reason'] = 'verified refinement improvement below 1 percent'
                    break
            if time.perf_counter()-begin > TIME_BUDGET_S:
                break
        result['solve_s'] = time.perf_counter()-begin
        if result['success']:
            directory.mkdir(parents=True, exist_ok=True)
            best_arrays['start_state'] = scene['start_state']
            temporary = directory/'trajectory.tmp.npz'
            np.savez_compressed(temporary, **best_arrays)
            os.replace(temporary, directory/'trajectory.npz')
            result['trajectory'] = 'trajectory.npz'
            if x0_info is not None:
                result['target_conversion'] = convert_target(directory/'trajectory.npz', Path(scenes)/uid, x0_info['gears'])
        else:
            result['failure_reason'] = 'no continuously verified feasible solution within budget'
    except Exception as exc:
        result.update(failure_reason=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
    result['total_s'] = time.perf_counter()-begin
    dump(result_file, result)
    return result


def convert_target(trajectory, scene_dir, x0_gears):
    """Same teacher -> training-target conversion as k4_flow_solver/prepare.py."""
    from k4_flow_solver.config import SolverConfig
    from k4_flow_solver.numerical import from_teacher, encode, decode, structure_id, repair, validate
    from k4_flow_solver.prepare import load_scene
    manifest = json.loads((ROOT/'data/k4_numeric_flow_full_v1_20260917/manifest.json').read_text(encoding='utf-8'))
    cfg = SolverConfig(**manifest['config'])
    try:
        controls, goal = from_teacher(trajectory, cfg)
        h = structure_id(controls.gears, cfg)
        latent, _ = encode(controls, cfg)
        controls = decode(latent, h, cfg)
        controls, correction = repair(controls, goal, cfg, time_budget_s=1.0)
        checks = validate(controls, goal, load_scene(scene_dir), cfg)
        return dict(ok=bool(checks['verified_feasible']), target_J=checks['J'], structure=int(h),
                    gears=controls.gears.tolist(), x0_structure=int(structure_id(np.asarray(x0_gears), cfg)),
                    same_structure_as_x0=bool(int(h) == int(structure_id(np.asarray(x0_gears), cfg))))
    except Exception as exc:
        return dict(ok=False, error=f'{type(exc).__name__}:{exc}')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cache', type=Path, default=ROOT/'data/warmstart_cache/hybrid_astar_e3000_v1')
    p.add_argument('--uids-file', type=Path, default=ROOT/'data/warmstart_cache/hybrid_astar_val200_e3000_v2/run_config.json')
    p.add_argument('--scenes', type=Path, default=ROOT/'data/flow_scenes')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=14)
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = json.loads(args.uids_file.read_text(encoding='utf-8'))
    uids = [u for u in (data['uids'] if isinstance(data, dict) else data) if (args.cache/(u+'.npz')).is_file()]
    run = dict(schema=SCHEMA, uids=len(uids), workers=args.workers, fast_ocp_cpu_s=float(os.environ['FAST_OCP_CPU']),
               time_budget_s=TIME_BUDGET_S, cache=str(args.cache), argv=sys.argv)
    (args.out/'run_config.json').write_text(json.dumps(dict(run, uid_list=uids), indent=1), encoding='utf-8')
    print(json.dumps(run), flush=True)
    started = time.perf_counter()
    tasks = [(u, str(args.cache), str(args.scenes), str(args.out)) for u in uids]
    with ProcessPoolExecutor(args.workers) as pool, (args.out/'results.jsonl').open('w', encoding='utf-8') as stream:
        for count, future in enumerate(as_completed([pool.submit(solve_scene, t) for t in tasks]), 1):
            r = future.result()
            compact = {k: r.get(k) for k in ('uid', 'success', 'total_s', 'solve_s', 'gears', 'x0_gears', 'gear_changes',
                                              'classic_scope', 'failure_reason', 'error', 'stop_reason', 'target_conversion')}
            compact['objective'] = r.get('best', {}).get('objective')
            compact['stages'] = len(r.get('stages', []))
            stream.write(json.dumps(compact)+'\n')
            stream.flush()
            if count % 10 == 0 or count == len(tasks):
                print(json.dumps(dict(done=count, total=len(tasks), elapsed_s=round(time.perf_counter()-started, 1))), flush=True)
    wall = time.perf_counter()-started
    (args.out/'wall_clock.json').write_text(json.dumps(dict(wall_s=wall, scenes=len(uids), workers=args.workers)), encoding='utf-8')
    print(json.dumps(dict(event='complete', wall_s=round(wall, 1))), flush=True)


if __name__ == '__main__':
    main()
