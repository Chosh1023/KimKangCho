"""Resumable process-parallel GT optimization; never modifies source samples.

Outputs a raw trajectory/control overlay keyed by original UID and split.
It is not a processed training-shard dataset or a new PhysX validation pass.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ.setdefault('FAST_OCP_CPU','10')
import argparse
from collections import Counter,defaultdict,deque
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import math
import numpy as np
import fast_multiphase_ocp as model
from hope_path_transformer.geometry import states_world_to_local

ROOT=Path(__file__).resolve().parents[1]
SCHEMA='k4-fast-ocp-gt-overlay-v1'


def dump(path,value):
    def clean(v):
        if isinstance(v,float) and not math.isfinite(v):return None
        if isinstance(v,dict):return {k:clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        return v
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)


def initialize_worker():
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)


def resolve_source(value):
    p=Path(value)
    if p.is_dir():return p
    normal=value.replace('\\','/');position=normal.lower().find('/run-logs/')
    if position>=0:
        repaired=ROOT/normal[position+1:]
        if repaired.is_dir():return repaired
    raise FileNotFoundError(value)


def run_scene(row,output):
    begin=time.perf_counter();uid=row['uid']
    directory=Path(output)/'samples'/uid[:3]/uid
    result_file=directory/'result.json'
    if result_file.is_file():
        old=json.loads(result_file.read_text(encoding='utf-8'))
        if not old.get('success') or (directory/'trajectory.npz').is_file():
            old['resumed']=True;return old
    result=dict(schema=SCHEMA,uid=uid,split=row['processed_split'],source_sample_dir=row['sample_dir'],
                source_input_sha256=row.get('input_sha256'),source_gt_sha256=row.get('gt_sha256'),
                success=False,stages=[],global_optimum_certified=False,worker_pid=os.getpid())
    try:
        source=resolve_source(row['sample_dir'])
        with np.load(source/'planner_input.npz',allow_pickle=False) as z:scene={k:z[k] for k in z.files}
        path=states_world_to_local(np.load(source/'path.npy',allow_pickle=False),scene['start_state']).astype(float)
        requested=np.r_[scene['requested_goal_pose'],-1.,0.][None,:]
        goal=states_world_to_local(requested,scene['start_state'])[0,:3].astype(float)
        raw,gears=model.split_path(path);meshes,classic=model.meshes_for(raw,gears)
        # Keep yaw continuous across phases, choosing the equivalent requested
        # terminal orientation nearest the source path's final heading.
        goal[2]+=2*np.pi*round((raw[-1][0][-1,2]-goal[2])/(2*np.pi))
        result.update(gears=gears,gear_changes=len(gears)-1,classic_scope=classic,goal_local=goal.tolist(),
                      scope='original forward/reverse benchmark' if classic else 'fixed GT gear sequence, unrestricted maneuver headings')
        previous=None;oldmesh=None;best_arrays=None
        for stage,mesh in enumerate(meshes):
            prior_best=result.get('best',{}).get('objective')
            start=time.perf_counter()
            seed=model.seed_from_parts(raw,mesh) if previous is None else model.remesh(previous,oldmesh,mesh)
            lengths,parts,_=model.unpack(seed,mesh)
            # Include distant obstacle and boundary planes too, so the NLP
            # cannot escape into an object omitted at its initial pose.
            corridors=model.ocp.obstacle_corridors(scene,None,None,None,phase_parts=parts,include_far=True)
            problem=model.build(mesh[0],mesh[-1],goal,corridors,
                                phases=None if classic else list(zip(mesh,gears)))
            solved=model.ocp.solve(problem,seed);z=solved.pop('z')
            entry=dict(mesh=[int(n) for n in mesh],solver=solved)
            result['stages'].append(entry)
            if solved['success']:
                previous=z;oldmesh=mesh
                val,arrays=model.validate(z,mesh,gears,scene,goal,classic)
                entry['validation']=val
                if val['verified_feasible'] and (not result['success'] or val['objective']<result['best']['objective']):
                    result['success']=True;result['best']=val;best_arrays=arrays
            entry['total_s']=time.perf_counter()-start
            if result['success'] and result['best']['objective']<14:break
            if stage>=1 and solved['success'] and entry.get('validation',{}).get('verified_feasible') and prior_best is not None:
                if (prior_best-result['best']['objective'])/prior_best<.01:
                    result['stop_reason']='verified refinement improvement below 1 percent';break
            if time.perf_counter()-begin>35:break
        if result['success']:
            directory.mkdir(parents=True,exist_ok=True)
            arrays=best_arrays
            # Readable physical controls: q >= 0; signed speed = gear*q.
            # Steering-at-rest durations are explicit between gear phases.
            arrays['start_state']=scene['start_state']
            temporary=directory/'trajectory.tmp.npz'
            np.savez_compressed(temporary,**arrays)
            os.replace(temporary,directory/'trajectory.npz')
            result['trajectory']='trajectory.npz'
        else:result['failure_reason']='no continuously verified feasible solution within budget'
    except Exception as exc:
        result.update(failure_reason=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    result['total_s']=time.perf_counter()-begin
    dump(result_file,result)
    return result


def ordered_rows(selection):
    rows=[json.loads(line) for line in selection.read_text(encoding='utf-8').splitlines() if line.strip()]
    assert len({r['uid'] for r in rows})==len(rows)
    rng=np.random.default_rng(20260916);rng.shuffle(rows)
    groups=defaultdict(deque)
    for row in rows:groups[(row['processed_split'],row.get('target_slot_id'),row.get('start_direction'))].append(row)
    result=[]
    while groups:
        for key in list(groups):
            result.append(groups[key].popleft())
            if not groups[key]:del groups[key]
    return result


def source_hashes():
    names=['collect_fast_ocp_gt.py','fast_multiphase_ocp.py','fast_two_phase_model.py','search_k4_two_phase_ocp.py',
           'search_k4_global_ocp.py','certify_k4_small_scene_cost_bounds.py','validate_k4_global_search.py']
    return {name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in names}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'data/k4_fast_ocp_gt_v2_20260916')
    p.add_argument('--selection',type=Path)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--limit',type=int)
    p.add_argument('--pilot-count',type=int,default=300)
    p.add_argument('--pilot-min-success',type=float,default=.75)
    args=p.parse_args();output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    selection=args.selection or output/'source_selection.jsonl'
    rows=ordered_rows(selection)
    if args.limit:rows=rows[:args.limit]
    hashes=source_hashes();hash_file=output/'source_hashes.json'
    if hash_file.exists() and json.loads(hash_file.read_text(encoding='utf-8'))!=hashes:
        raise RuntimeError('source changed: choose a new output directory; refusing mixed-version resume')
    dump(hash_file,hashes)
    snapshot=output/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for name in hashes:shutil.copy2(ROOT/'scripts'/name,snapshot/name)
    lock=output/'RUNNING.lock'
    # Exclusive lock is deliberately not auto-deleted on a hard crash. Check
    # the recorded PID before removing a stale lock and resuming manually.
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    start=time.perf_counter();counts=Counter();split=defaultdict(Counter);gear=defaultdict(Counter);failures=Counter()
    elapsed_sum=0.;completed=0;fresh=0;valid_stream=None;all_stream=None
    state='running_pilot';error=None
    def status():
        elapsed=time.perf_counter()-start
        payload=dict(schema=SCHEMA,status=state,pid=os.getpid(),workers=args.workers,total=len(rows),
            completed=completed,accepted=counts['accepted'],rejected=counts['rejected'],resumed=counts['resumed'],
            elapsed_s=elapsed,fresh_completed=fresh,seconds_per_fresh_result=elapsed/fresh if fresh else None,
            remaining_hours=(len(rows)-completed)*elapsed/fresh/3600 if fresh else None,
            mean_scene_s=elapsed_sum/completed if completed else None,
            splits=dict(split),gear_changes=dict(gear),failures=dict(failures),
            last_update_utc=datetime.now(timezone.utc).isoformat(),error=error,
            output_kind='raw path and speed-control overlay; processed training conversion pending',
            objective='T + 0.5 integral(a^2 + v^2 delta_dot^2) dt',
            minimum_clearance_m=.1,global_optimum_certified=False,original_data_mutated=False)
        dump(output/'progress.json',payload)
    try:
        # Rebuild indexes when resuming, with each persisted result read by its
        # worker; trajectory/result pairs are committed atomically per scene.
        valid_stream=(output/'accepted.jsonl').open('w',encoding='utf-8',buffering=1)
        all_stream=(output/'results.jsonl').open('w',encoding='utf-8',buffering=1)
        def record(r):
            nonlocal completed,fresh,elapsed_sum
            completed+=1;fresh+=not r.get('resumed',False);counts['resumed']+=bool(r.get('resumed'))
            key='accepted' if r['success'] else 'rejected';counts[key]+=1;elapsed_sum+=r['total_s']
            split[r['split']][key]+=1;gear[str(r.get('gear_changes','unknown'))][key]+=1
            compact={k:r[k] for k in ('uid','split','success','total_s','gears','gear_changes','failure_reason','error') if k in r}
            compact['sample_dir']=str(output/'samples'/r['uid'][:3]/r['uid'])
            compact['source_sample_dir']=r['source_sample_dir']
            if r['success']:
                compact['objective']=r['best']['objective'];valid_stream.write(json.dumps(compact,ensure_ascii=False)+'\n')
            else:failures[r.get('failure_reason','unknown')]+=1
            all_stream.write(json.dumps(compact,ensure_ascii=False)+'\n')
            print(json.dumps(dict(completed=completed,total=len(rows),accepted=counts['accepted'],uid=r['uid'],success=r['success'],
                                  J=r.get('best',{}).get('objective'),seconds=round(r['total_s'],3))),flush=True)
            if completed%10==0:status()
        with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize_worker) as pool:
            pilot_end=min(args.pilot_count,len(rows))
            for lo,hi in ((0,pilot_end),(pilot_end,len(rows))):
                if lo==hi:continue
                if lo:
                    rate=counts['accepted']/completed
                    if rate<args.pilot_min_success:
                        state='pilot_needs_review';status();return 2
                    state='running_full';status()
                iterator=iter(rows[lo:hi]);pending={}
                def submit():
                    row=next(iterator,None)
                    if row is not None:pending[pool.submit(run_scene,row,str(output))]=row['uid']
                for _ in range(args.workers*2):submit()
                while pending:
                    done,_=wait(pending,timeout=20,return_when=FIRST_COMPLETED)
                    if not done:status()
                    for future in done:
                        pending.pop(future);record(future.result())
                        if (output/'STOP').exists():
                            state='stop_requested'
                            # Drain just the already submitted jobs.
                        else:submit()
                    if shutil.disk_usage(output).free<10*1024**3:
                        raise RuntimeError('less than 10 GiB disk free; stopping safely')
                if state=='stop_requested':break
        if state!='stop_requested':state='complete'
    except BaseException:
        state='failed';error=traceback.format_exc();raise
    finally:
        status()
        if valid_stream:valid_stream.close()
        if all_stream:all_stream.close()
        lock.unlink(missing_ok=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
