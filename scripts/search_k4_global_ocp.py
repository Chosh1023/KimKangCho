"""Best-effort global search for the three saved forward/reverse scenes.

Hard bicycle dynamics, jointly optimized steering/speed, independent continuous
reconstruction and collision validation. Multi-start IPOPT is explicitly NOT a
global certificate. A separate SCIP script bounds a coarse transcription only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import PchipInterpolator
from scipy.optimize import root
from scipy.integrate import cumulative_simpson, solve_ivp

from compare_k4_gt_ocp_cost import ROOT,casadi
from compare_k4_forward_reverse import OUT as PREVIOUS,polynomials,curve_values,clock_for_curve
from certify_k4_small_scene_cost_bounds import WB,W,VMAX,AMAX,OMAX,DMAX,check_geometry,normalized_clock

GRID_KIND=os.environ.get('K4_OCP_GRID','cosine')
OUT=ROOT/('analysis_artifacts/k4_global_ocp_adaptive_20260916' if GRID_KIND=='cosine' else 'analysis_artifacts/k4_global_ocp_extended_20260916')
COST_C=4/3*(36*W)**.25
EPS=2e-4


def arc_grid(n):
    u=np.linspace(0,1,n+1)
    return .5*(1-np.cos(np.pi*u)) if GRID_KIND=='cosine' else u


def phase_cost(L,delta,q):
    n=len(q)-1;h=L*np.diff(arc_grid(n))
    q0,q1=q[:-1],q[1:]
    dt=2*h/(q0+q1);a=(q1*q1-q0*q0)/(2*h);slope=np.diff(delta)/h
    f4=(q0**4+q0**3*q1+q0*q0*q1*q1+q0*q1**3+q1**4)/5
    ac=float(np.dot(a*a,dt));sc=float(np.dot(slope*slope*f4,dt));T=float(dt.sum())
    return dict(objective=T+W*(ac+sc),duration_s=T,acceleration_integral=ac,steering_integral=sc,
        max_speed=float(q.max()),max_accel=float(abs(a).max()),
        max_steer_rate=float(np.max(abs(slope)*np.maximum(q0,q1))),max_steer=float(abs(delta).max()))


def build_solver(n,goal,mode='rk4',yaw_tolerance=4.999,maxiter=700):
    ca=casadi();d=ca.SX.sym('d');L=ca.SX.sym('L')
    poses=ca.SX.sym('p',3,n+1);delta=ca.SX.sym('delta',n+1);q=ca.SX.sym('q',n+1)
    dwell=ca.SX.sym('dwell',2)
    z=ca.vertcat(d,L,ca.reshape(poses,-1,1),delta,q,dwell)
    g=[];gl=[];gu=[]
    def eq(value):
        g.append(value);gl.extend([0.]*value.numel());gu.extend([0.]*value.numel())
    def box(value,lo,hi):
        g.append(value);gl.extend([lo]*value.numel());gu.extend([hi]*value.numel())
    def rhs(theta,steer):return ca.vertcat(-ca.cos(theta),-ca.sin(theta),-ca.tan(steer)/WB)
    eq(poses[:,0]-ca.vertcat(d,0,0));eq(poses[:2,-1]-np.asarray(goal[:2]))
    tol=math.radians(yaw_tolerance)
    box(poses[2,-1],float(goal[2]-tol),float(goal[2]+tol))
    # The vehicle may steer while stopped at the cusp or goal; count that time.
    objective=COST_C*ca.sqrt(d)+ca.sum1(dwell)
    box((OMAX-EPS)*dwell-ca.vertcat(delta[0],delta[-1]),0.,np.inf)
    box((OMAX-EPS)*dwell+ca.vertcat(delta[0],delta[-1]),0.,np.inf)
    for i in range(n):
        h=L*(arc_grid(n)[i+1]-arc_grid(n)[i])
        p=poses[:,i];mid=.5*(delta[i]+delta[i+1])
        if mode=='midpoint':step=h*rhs(.5*(poses[2,i]+poses[2,i+1]),mid)
        else:
            k1=rhs(p[2],delta[i]);k2=rhs(p[2]+.5*h*k1[2],mid)
            k3=rhs(p[2]+.5*h*k2[2],mid);k4=rhs(p[2]+h*k3[2],delta[i+1])
            step=h/6*(k1+2*k2+2*k3+k4)
        eq(poses[:,i+1]-p-step)
        dt=2*h/(q[i]+q[i+1]);a=(q[i+1]**2-q[i]**2)/(2*h);slope=(delta[i+1]-delta[i])/h
        box(a,-AMAX+EPS,AMAX-EPS)
        box(slope*q[i],-OMAX+EPS,OMAX-EPS);box(slope*q[i+1],-OMAX+EPS,OMAX-EPS)
        f4=(q[i]**4+q[i]**3*q[i+1]+q[i]**2*q[i+1]**2+q[i]*q[i+1]**3+q[i+1]**4)/5
        objective+=dt*(1+W*(a*a+slope*slope*f4))
    solver=ca.nlpsol('reverse_ocp','ipopt',dict(x=z,f=objective,g=ca.vertcat(*g)),{
        'print_time':False,'ipopt.print_level':0,'ipopt.sb':'yes','ipopt.tol':2e-8,
        'ipopt.acceptable_tol':1e-6,'ipopt.max_iter':maxiter,'ipopt.max_cpu_time':90.,
        'ipopt.bound_relax_factor':0.,'ipopt.mu_strategy':'adaptive',
    })
    # Limits exclude paths whose acceleration-only lower bound exceeds 16.5.
    upper_cost=16.5
    dmax=(upper_cost/COST_C-np.sqrt(abs(goal[1])))**2
    lmax=(upper_cost/COST_C-np.sqrt(2.))**2
    lb=np.r_[2.,abs(goal[1]),np.tile([-35.,-35.,-12.],n+1),np.full(n+1,-DMAX+EPS),np.full(n+1,1e-5),0.,0.]
    ub=np.r_[dmax,lmax,np.tile([35.,35.,12.],n+1),np.full(n+1,DMAX-EPS),np.full(n+1,VMAX-EPS),4.,4.]
    dslice=slice(2+3*(n+1),2+4*(n+1));qslice=slice(2+4*(n+1),2+5*(n+1))
    start=qslice.start;lb[[start,start+n]]=0;ub[[start,start+n]]=0
    evaluate=ca.Function('eval_solution',[z],[objective,ca.vertcat(*g)])
    return dict(solver=solver,evaluate=evaluate,lb=lb,ub=ub,gl=np.array(gl),gu=np.array(gu),n=n,mode=mode,goal=np.asarray(goal))


def pack(d,L,poses,delta,q):return np.r_[d,L,poses.reshape(-1),delta,q,abs(delta[0])/(OMAX-EPS),abs(delta[-1])/(OMAX-EPS)]


def unpack(z,n):
    return dict(d=float(z[0]),L=float(z[1]),poses=z[2:2+3*(n+1)].reshape(n+1,3),
        delta=z[2+3*(n+1):2+4*(n+1)],q=z[2+4*(n+1):2+5*(n+1)],dwell=z[-2:])


def seed_from_previous(label,n):
    row=json.loads((PREVIOUS/(label+'_result.json')).read_text(encoding='utf-8'));ref=row['reference']
    poly=polynomials(row['goal'],ref['variables']);r=np.linspace(0,1,10001);v=curve_values(poly,r)
    arc=np.r_[0,cumulative_simpson(v['h'],x=r)];L=float(arc[-1]);rr=PchipInterpolator(arc,r)(L*arc_grid(n))
    v=curve_values(poly,rr);poses=np.c_[v['xy'],v['yaw']]
    q=1.6*np.sqrt(np.maximum(np.sin(np.linspace(0,np.pi,n+1)),0));q[[0,-1]]=0
    delta=v['steer'];delta[[0,-1]]=0
    return pack(ref['forward_length_m'],L,poses,delta,q)


def resample(z,oldn,newn):
    a=unpack(z,oldn);old=arc_grid(oldn);new=arc_grid(newn)
    p=np.column_stack([np.interp(new,old,a['poses'][:,i]) for i in range(3)])
    delta=np.interp(new,old,a['delta']);q=np.sqrt(np.interp(new,old,a['q']**2))
    return pack(a['d'],a['L'],p,delta,q)


def solve(problem,start):
    t=time.perf_counter();start=np.clip(start,problem['lb'],problem['ub'])
    result=problem['solver'](x0=start,lbx=problem['lb'],ubx=problem['ub'],lbg=problem['gl'],ubg=problem['gu'])
    z=np.array(result['x']).ravel();cost,g=problem['evaluate'](z);g=np.array(g).ravel()
    violation=float(max(np.max(problem['gl']-g),np.max(g-problem['gu']),0.))
    stat=problem['solver'].stats()
    return dict(z=z,objective=float(cost),success=bool(stat['success'] and violation<2e-6),
        status=stat['return_status'],iterations=stat['iter_count'],violation=violation,seconds=time.perf_counter()-t)


def exact_reverse(L,delta,trace=False):
    """Piecewise-linear steering, analytic yaw and Gauss quadrature for XY."""
    n=len(delta)-1;h=L*np.diff(arc_grid(n));change=np.diff(delta);base=delta[:-1]
    def yaw_increment(tau):
        t=np.atleast_1d(tau)[None,:];dx=change[:,None]*t
        ratio=np.log1p(-2*np.sin(dx/2)**2-np.tan(base[:,None])*np.sin(dx))
        safe=np.where(abs(change)>1e-10,change,1.)[:,None]
        inc=h[:,None]/(WB*safe)*ratio
        return np.where((abs(change)>1e-10)[:,None],inc,-h[:,None]/WB*np.tan(base[:,None])*t)
    theta=np.r_[0,np.cumsum(yaw_increment([1.])[:,0])]
    nodes,weights=leggauss(24);u=(nodes+1)/2;w=weights/2
    ang=theta[:-1,None]+yaw_increment(u)
    dx=-h*(np.cos(ang)@w);dy=-h*(np.sin(ang)@w)
    x=np.r_[0,np.cumsum(dx)];y=np.r_[0,np.cumsum(dy)]
    result=dict(endpoint=np.array([x[-1],y[-1],theta[-1]]),node_poses=np.c_[x,y,theta])
    if trace:
        m=max(8,int(math.ceil(h.max()/.001)))
        # Integrate each dense subinterval by 4-point Gauss, retaining all knots.
        local=np.linspace(0,1,m+1);gn,gw=leggauss(4)
        subt=(np.arange(m)[:,None]+(gn[None,:]+1)/2)/m
        angles=theta[:-1,None]+yaw_increment(subt.ravel())
        angles=angles.reshape(n,m,4)
        xs=-h[:,None]/m*np.sum(np.cos(angles)*gw[None,None,:]/2,axis=2)
        ys=-h[:,None]/m*np.sum(np.sin(angles)*gw[None,None,:]/2,axis=2)
        tx=x[:-1,None]+np.c_[np.zeros(n),np.cumsum(xs,axis=1)]
        ty=y[:-1,None]+np.c_[np.zeros(n),np.cumsum(ys,axis=1)]
        yaw=theta[:-1,None]+yaw_increment(local)
        steer=base[:,None]+change[:,None]*local[None,:]
        arr=np.stack((tx,ty,yaw,-np.ones_like(tx),steer),axis=-1)
        result['path']=np.vstack((arr[:,:-1,:].reshape(-1,5),arr[-1,-1,:]))
        result['arc']=np.r_[(L*arc_grid(n)[:-1,None]+h[:,None]*local[None,:-1]).ravel(),L]
    return result


def validate(label,z,n,scene,goal):
    a=unpack(z,n);original=a.copy();target_yaw=float(a['poses'][-1,2])
    def residual(p):
        end=exact_reverse(p[0],a['delta']*p[1])['endpoint']
        return [end[1]-goal[1],end[2]-target_yaw]
    correction=root(residual,[a['L'],1.],tol=1e-10)
    L,scale=correction.x;delta=a['delta']*scale;continuous=exact_reverse(L,delta,trace=True)
    d=float(goal[0]-continuous['endpoint'][0]);path=continuous['path'];path[:,0]+=d
    q=a['q'].copy();metrics=phase_cost(L,delta,q)
    slow=max(1.,metrics['max_speed']/VMAX,np.sqrt(metrics['max_accel']/AMAX),metrics['max_steer_rate']/OMAX)*1.00001
    q/=slow;reverse=phase_cost(L,delta,q)
    forward=clock_for_curve(d,lambda f:np.zeros_like(f),quad_n=128,dense=True)
    fpath=np.c_[np.linspace(0,d,int(d/.001)+2),np.zeros((int(d/.001)+2,2)),np.ones(int(d/.001)+2),np.zeros(int(d/.001)+2)]
    # Zero-length intervals represent steering at rest, not extra gear switches.
    finish=path[-1].copy();finish[4]=0.
    full=np.vstack((fpath,path,finish))
    geom=check_geometry(full,scene,float(abs(np.tan(delta)).max()/WB))
    end=continuous['endpoint'].copy();end[0]+=d
    errors=dict(position_m=float(np.linalg.norm(end[:2]-goal[:2])),yaw_rad=float(abs(end[2]-goal[2])))
    feasible=bool(correction.success and abs(L-a['L'])<.05 and abs(scale-1)<.01 and d>=2
        and errors['position_m']<1e-7 and errors['yaw_rad']<=math.radians(5)
        and abs(delta).max()<=DMAX and geom['collision_free'] and geom['final_inside']
        and geom['conservative_continuous_clearance_m']>=.1)
    result={k:forward[k]+reverse[k] for k in ('objective','duration_s','acceleration_integral','steering_integral')}
    dwell=float((abs(delta[0])+abs(delta[-1]))/(OMAX-EPS))
    result['objective']+=dwell;result['duration_s']+=dwell
    result.update(label=label,n=n,grid=GRID_KIND,verified_feasible=feasible,forward=forward,reverse=reverse,
        stationary_steer_time_s=dwell,cusp_steer_rad=float(delta[0]),end_moving_steer_rad=float(delta[-1]),
        length_m=d+L,forward_length_m=d,reverse_length_m=L,gear_changes=1,
        geometry=geom,endpoint_errors=errors,shooting_correction=dict(success=bool(correction.success),
        length_change_m=float(L-a['L']),steering_scale=float(scale),cusp_change_m=float(d-a['d']),speed_slowdown=float(slow)),
        path=full,delta=delta,q=q,d=d,L=L,poses=continuous['node_poses']+np.array([d,0,0]))
    return result


def run_scene(label,starts=32,seed=20260916):
    OUT.mkdir(parents=True,exist_ok=True)
    row=json.loads((PREVIOUS/(label+'_result.json')).read_text(encoding='utf-8'));goal=np.asarray(row['goal'])
    with np.load(Path(row['raw_sample_dir'])/'planner_input.npz') as z:scene={k:z[k] for k in z.files}
    n=40;problem=build_solver(n,goal);base=seed_from_previous(label,n);rng=np.random.default_rng(seed+ord(label))
    candidates=[];logs=[];best=float('inf')
    for i in range(starts):
        start=base.copy() if i==0 else (candidates[i%len(candidates)]['z'].copy() if candidates and i%3 else base.copy())
        if i:
            a=unpack(start,n);u=np.linspace(0,1,n+1)
            # Perturb both the geometry and speed over a broad range.
            amplitude=.15 if i%3==1 else .45
            a['delta']+=sum(rng.normal(0,amplitude)*np.sin(k*np.pi*u) for k in range(1,5))
            a['q']*=rng.uniform(.5,1.5);a['d']+=rng.uniform(-1.8,1.8);a['L']*=rng.uniform(.8,1.25)
            start=pack(a['d'],a['L'],a['poses'],a['delta'],a['q'])
        solved=solve(problem,start);z=solved.pop('z');logs.append(dict(index=i,**solved))
        if solved['success']:
            candidates.append(dict(z=z,**solved));best=min(best,solved['objective'])
        print(label,'start',i,'J',round(solved['objective'],6),'best',round(best,6),solved['status'],flush=True)
        (OUT/(label+'_progress.json')).write_text(json.dumps(dict(starts=logs,best=best),indent=2),encoding='utf-8')
    candidates.sort(key=lambda c:c['objective'])
    np.savez_compressed(OUT/(label+'_coarse_candidates.npz'),z=np.array([c['z'] for c in candidates[:12]]),n=n)
    refined=[]
    for index,c in enumerate(candidates[:4]):
        z=c['z'];oldn=n
        for newn in (80,160,320,640):
            finer=build_solver(newn,goal,maxiter=1000);sol=solve(finer,resample(z,oldn,newn));z=sol.pop('z');oldn=newn
            print(label,'refine',index,newn,sol['objective'],sol['status'],flush=True)
            if not sol['success']:break
            checked=validate(label,z,newn,scene,goal)
            arrays={k:checked.pop(k) for k in ('path','delta','q','poses')}
            checked['solver']=sol
            np.savez_compressed(OUT/(label+f'_candidate{index}_n{newn}.npz'),**arrays,z=z,d=checked['d'],L=checked['L'],n=newn)
            (OUT/(label+f'_candidate{index}_n{newn}.json')).write_text(json.dumps(checked,indent=2),encoding='utf-8')
            refined.append(checked)
        if index>=1 and any(x['n']==640 and x['verified_feasible'] for x in refined):break
    valid=[r for r in refined if r['verified_feasible']]
    result=dict(label=label,uid=row['uid'],goal=goal.tolist(),raw_sample_dir=row['raw_sample_dir'],
        method='hard-dynamics multistart and mesh refinement; not globally certified',
        starts=logs,refined=refined,best=min(valid,key=lambda r:r['objective']) if valid else None)
    (OUT/(label+'_search.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--labels',default='ABC');p.add_argument('--starts',type=int,default=32);p.add_argument('--workers',type=int,default=3);args=p.parse_args()
    if args.workers==1:
        for label in args.labels:run_scene(label,args.starts)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(run_scene,label,args.starts) for label in args.labels]
            for future in as_completed(futures):
                result=future.result();print('FINISHED',result['label'],result['best']['objective'] if result['best'] else None,flush=True)


if __name__=='__main__':main()
