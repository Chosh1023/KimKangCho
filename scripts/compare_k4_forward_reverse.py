"""Analysis-only: straight forward, then smoothly curved reverse parking.

Local optimization supplies a feasible reference, never a global certificate.
Exact-pose Reeds--Shepp length lower bounds are recorded separately. All stored
GT/K4 timing is a retiming surrogate, not the collector's original OCP cost.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import time
import numpy as np
from numpy.polynomial import Polynomial as P
from scipy.integrate import cumulative_simpson, solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from numpy.polynomial.legendre import leggauss

from compare_k4_gt_ocp_cost import ROOT, AUDIT, load_paths, retime
from certify_k4_small_scene_cost_bounds import (
    WB,W,VMAX,AMAX,OMAX,DMAX,A_COEFF,normalized_clock,check_geometry,
)

OUT=ROOT/'analysis_artifacts/k4_forward_reverse_20260916'
UIDS=['s01-o000000030-79bd87db010e','s01-o000000006-fea4f2875948',
      's01-o000000053-3a0643b85139']
R=P([0,1]); BUMP=R**3*(1-R)**3
MATRIX=np.array([[1,1,1],[3,4,5],[6,12,20]],float)
INV=np.linalg.inv(MATRIX)


def read_rows(file):
    return [json.loads(x) for x in file.read_text(encoding='utf-8').splitlines()]


def polynomials(goal,x):
    d,h0,h1,bx0,by0,bx1,by1,theta=x
    p0=np.array([d,0.]); p1=np.asarray(goal[:2])
    v0=np.array([-h0,0.]); v1=-h1*np.array([np.cos(theta),np.sin(theta)])
    aa=INV@np.vstack((p1-p0-v0,v1-v0,np.zeros(2)))
    return [P([p0[k],v0[k],0.,*aa[:,k]])+BUMP*(b0+b1*(2*R-1))
            for k,b0,b1 in [(0,bx0,bx1),(1,by0,by1)]]


def curve_values(polys,r):
    xy=np.stack([p(r) for p in polys],axis=-1)
    d1=np.stack([p.deriv(1)(r) for p in polys],axis=-1)
    d2=np.stack([p.deriv(2)(r) for p in polys],axis=-1)
    d3=np.stack([p.deriv(3)(r) for p in polys],axis=-1)
    h=np.linalg.norm(d1,axis=-1)
    cross=d1[...,0]*d2[...,1]-d1[...,1]*d2[...,0]
    cross_r=d1[...,0]*d3[...,1]-d1[...,1]*d3[...,0]
    # Reverse motion: bicycle curvature is minus the curve's oriented curvature.
    k=-cross/h**3
    k_r=-cross_r/h**3+3*cross*np.sum(d1*d2,axis=-1)/h**5
    steer=np.arctan(WB*k)
    steer_s=WB*k_r/(1+(WB*k)**2)/h
    yaw=np.arctan2(-d1[...,1],-d1[...,0])
    return dict(xy=xy,h=h,curvature=k,steer=steer,steer_s=steer_s,yaw=yaw)


def clock_for_curve(length,steer_s_at_fraction,quad_n=80,dense=False):
    nodes,weights=leggauss(quad_n); u=(nodes+1)/2; weights=weights/2
    f,_,frac=normalized_clock(u)
    sc=length**4*float(np.dot(weights,steer_s_at_fraction(frac)**2*f**4))
    ac=length*length*A_COEFF
    u=np.unique(np.r_[np.linspace(0,1,4001 if dense else 601),
                     np.geomspace(1.e-10,.01,80),1-np.geomspace(1.e-10,.01,80)])
    f,fp,frac=normalized_clock(u)
    max_f=float(f.max()); max_fp=float(abs(fp).max())
    rate_factor=float(np.max(abs(steer_s_at_fraction(frac))*f))
    duration=max((3*W*(ac+sc))**.25,length*max_f/VMAX,
                 np.sqrt(length*max_fp/AMAX),length*rate_factor/OMAX)
    return dict(objective=duration+W*(ac+sc)/duration**3,duration_s=duration,
        acceleration_integral=ac/duration**3,steering_integral=sc/duration**3,
        max_speed=length*max_f/duration,max_accel=length*max_fp/duration**2,
        max_steer_rate=length*rate_factor/duration)


def evaluate(goal,x,resolution=401,quad_n=64,trace=False):
    polys=polynomials(goal,x)
    r=np.linspace(0,1,resolution); values=curve_values(polys,r)
    arc=np.r_[0.,cumulative_simpson(values['h'],x=r)]; length=float(arc[-1])
    r_of_s=PchipInterpolator(arc,r)
    def steer_s(frac):
        return curve_values(polys,r_of_s(length*frac))['steer_s']
    reverse=clock_for_curve(length,steer_s,quad_n,dense=trace)
    forward=clock_for_curve(x[0],lambda f:np.zeros_like(f),quad_n,dense=trace)
    result={key:forward[key]+reverse[key] for key in (
        'objective','duration_s','acceleration_integral','steering_integral')}
    result.update({key:max(forward[key],reverse[key]) for key in ('max_speed','max_accel','max_steer_rate')})
    result.update(length_m=x[0]+length,forward_length_m=x[0],reverse_length_m=length,
        gear_changes=1,max_steer=float(abs(values['steer']).max()),
        min_curve_parameter_speed=float(values['h'].min()),variables=np.asarray(x).tolist(),
        terminal_yaw_rad=float(x[-1]),polynomials=[p.coef.tolist() for p in polys],
        forward=forward,reverse=reverse)
    if trace:
        fpath=np.c_[np.linspace(0,x[0],max(3,int(x[0]/.001)+1)),np.zeros((max(3,int(x[0]/.001)+1),2)),
                    np.ones(max(3,int(x[0]/.001)+1)),np.zeros(max(3,int(x[0]/.001)+1))]
        rpath=np.c_[values['xy'],values['yaw'],-np.ones(len(r)),values['steer']]
        result['path']=np.vstack((fpath,rpath))
        result['max_curvature']=float(abs(values['curvature']).max())
        result['forward_count']=len(fpath)
        traces=[]
        for gear,phase,L in [(1,forward,x[0]),(-1,reverse,length)]:
            u=np.unique(np.r_[np.linspace(0,1,8001),np.geomspace(1e-10,.01,100),1-np.geomspace(1e-10,.01,100)])
            f,fp,frac=normalized_clock(u); T=phase['duration_s']
            if gear==1:
                xy=np.c_[L*frac,np.zeros_like(u)];yaw=np.zeros_like(u);delta=np.zeros_like(u);delta_s=np.zeros_like(u)
            else:
                v=curve_values(polys,r_of_s(length*frac));xy=v['xy'];yaw=v['yaw'];delta=v['steer'];delta_s=v['steer_s']
            traces.append(np.c_[u*T+(forward['duration_s'] if gear==-1 else 0),xy,yaw,
                        gear*L/T*f,gear*L/T**2*fp,delta,L/T*f*delta_s])
        result['time_trace']=np.vstack(traces)
    return result


def optimize(goal,seed_forward_lengths):
    lo,hi=goal[2]-math.radians(5),goal[2]+math.radians(5)
    bounds=[(2.,8.),(1.,16.),(1.,20.),(-70,70),(-70,70),(-70,70),(-70,70),(lo,hi)]
    def objective(x):
        return evaluate(goal,x)['objective']
    def constraints(x):
        v=curve_values(polynomials(goal,x),np.linspace(0,1,501))
        return np.r_[DMAX-2e-4-abs(v['steer']),v['h']-.15]
    starts=[]
    for d in sorted(set([3.,4.,5.]+[round(v,3) for v in seed_forward_lengths])):
        # Initial reverse tangent aims rearwards, final tangent into the slot.
        starts.append(np.array([d,5.,max(6.,abs(goal[1])*1.4),0,0,0,0,goal[2]]))
    attempts=[]; candidates=[]
    for i,start in enumerate(starts):
        t=time.perf_counter()
        solved=minimize(objective,start,method='SLSQP',bounds=bounds,
            constraints=[{'type':'ineq','fun':constraints}],
            options={'maxiter':180,'ftol':2e-9})
        val=evaluate(goal,solved.x,resolution=1601,quad_n=128)
        attempt=dict(index=i,success=bool(solved.success),message=str(solved.message),
            objective=val['objective'],max_steer=val['max_steer'],variables=solved.x.tolist(),
            seconds=time.perf_counter()-t)
        attempts.append(attempt)
        if val['max_steer']<=DMAX and val['min_curve_parameter_speed']>.14:
            candidates.append(val)
        print('start',i,json.dumps(attempt),flush=True)
    return sorted(candidates,key=lambda c:c['objective']),attempts


def reintegrate(goal,result):
    polys=[P(c) for c in result['polynomials']]
    r=np.linspace(0,1,65537);v=curve_values(polys,r)
    arc=np.r_[0.,cumulative_simpson(v['h'],x=r)];L=float(arc[-1]);r_of_s=PchipInterpolator(arc,r)
    duration=result['reverse']['duration_s']
    def rhs(t,state):
        f,_,frac=normalized_clock(t/duration);vel=-L/duration*float(f)
        steer=float(curve_values(polys,r_of_s(L*frac))['steer'])
        return [vel*np.cos(state[2]),vel*np.sin(state[2]),vel*np.tan(steer)/WB]
    solution=solve_ivp(rhs,[0,duration],[result['forward_length_m'],0,0],method='DOP853',
        rtol=1e-11,atol=1e-12,max_step=.01)
    end=solution.y[:,-1]
    return dict(success=bool(solution.success),endpoint=end.tolist(),
        position_error_m=float(np.linalg.norm(end[:2]-goal[:2])),yaw_error_rad=abs(float(end[2]-result['terminal_yaw_rad'])))


def path_details(path):
    ds=np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1)
    dy=np.arctan2(np.sin(np.diff(path[:,2])),np.cos(np.diff(path[:,2])))
    head=path[:-1,2]+dy/2
    dot=np.sum(np.diff(path[:,:2],axis=0)*np.c_[np.cos(head),np.sin(head)],axis=1)
    gear=np.where(dot>=0,1,-1); moving=ds>1e-6
    sequence=[int(gear[moving][0])]
    for g in gear[moving][1:]:
        if g!=sequence[-1]:sequence.append(int(g))
    cusp=int(np.where((gear<0)&moving)[0][0])
    return dict(sequence=sequence,cusp_index=cusp,cusp_xy=path[cusp,:2].tolist(),
        initial_forward_length_m=float(ds[:cusp].sum()),
        max_forward_heading_deg=float(np.degrees(abs(path[:cusp+1,2])).max()),
        steer_total_variation_deg=float(np.degrees(abs(np.diff(path[:,4])).sum())),
        goal_yaw_deg=float(np.degrees(path[-1,2])))


def rs_reference(goal,scene):
    import importlib.util
    from reeds_shepp_path import generate_paths,sample_path
    maxc=float(np.tan(DMAX)/WB)
    goal=np.asarray(goal,dtype=np.float64)
    candidates=generate_paths([0.,0.,0.],goal,maxc,max_length_m=80)
    rs=min(candidates,key=lambda p:p.total_length)
    sampled=sample_path(rs,[0.,0.,0.],maxc,step_size=.001,max_steer=DMAX)
    p=np.c_[sampled.x,sampled.y,sampled.yaw,sampled.directions,sampled.steers]
    spec=importlib.util.spec_from_file_location('independent_rs',ROOT/'third_party/HOPE/src/env/reeds_shepp.py')
    independent=importlib.util.module_from_spec(spec);spec.loader.exec_module(independent)
    other=independent.calc_optimal_path(0.,0.,0.,*goal,maxc,step_size=.01)
    geom=check_geometry(p,scene,maxc)
    result=dict(length_m=rs.total_length,word=rs.ctypes,signed_segment_lengths_m=rs.lengths,
        gear_changes=rs.gear_switches,independent_length_m=float(other.L),
        independent_difference_m=abs(other.L-rs.total_length),geometry=geom,
        exact_endpoint_position_error_m=float(np.linalg.norm(p[-1,:2]-goal[:2])),
        exact_endpoint_yaw_error_rad=float(abs(np.arctan2(np.sin(p[-1,2]-goal[2]),np.cos(p[-1,2]-goal[2])))),
        details=path_details(p),path=p)
    result['globally_shortest_length_with_obstacles']=bool(geom['collision_free'] and geom['final_inside']
        and geom['conservative_continuous_clearance_m']>=.1 and result['independent_difference_m']<1e-6)
    return result


def refine_existing_paths():
    results=[]
    for label in 'ABC':
        file=OUT/(label+'_paths.npz')
        if not file.exists():return
        with np.load(file) as z:paths={k:z[k] for k in ('gt','k4')}
        row={'label':label}
        for key,path in paths.items():
            vals={}
            for refinement in (16,32):
                val=retime(path,refinement=refinement,keep_trace=refinement==32)
                if refinement==32:
                    (OUT/(label+'_'+key+'_timing32.json')).write_text(json.dumps(val),encoding='utf-8')
                    val.pop('runs')
                vals[str(refinement)]=val
                print('refine',label,key,refinement,val['objective'],val['success'],flush=True)
            row[key]=vals
        results.append(row)
    (OUT/'retiming_convergence.json').write_text(json.dumps(results,indent=2),encoding='utf-8')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--labels',default='ABC')
    parser.add_argument('--retime-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(exist_ok=True,parents=True)
    if args.retime_only:
        refine_existing_paths();return
    from hope_path_transformer.geometry import states_world_to_local
    original={r['uid']:r for r in read_rows(AUDIT/'per_scene.jsonl') if r['uid'] in UIDS}
    for i,uid in enumerate(UIDS):
        label=chr(65+i)
        if label not in args.labels:continue
        row=original[uid]
        with np.load(Path(row['raw_sample_dir'])/'planner_input.npz') as z:scene={k:z[k] for k in z.files}
        goal=states_world_to_local(np.r_[scene['requested_goal_pose'],-1.,0.][None,:],scene['start_state'])[0,:3].astype(float)
        gt,k4=load_paths(row);gt=states_world_to_local(gt,scene['start_state']).astype(float);k4=k4.astype(float)
        result=dict(label=label,uid=uid,raw_sample_dir=row['raw_sample_dir'],goal=goal.tolist(),
            gt_details=path_details(gt),k4_details=path_details(k4))
        arrays=dict(gt=gt,k4=k4,goal=goal)
        print('SCENE',label,uid,goal,flush=True)
        candidates,attempts=optimize(goal,[result[k]['initial_forward_length_m'] for k in ('gt_details','k4_details')])
        result['optimization_attempts']=attempts
        checked=[]
        for c in candidates:
            ref=evaluate(goal,c['variables'],resolution=32769,quad_n=512,trace=True)
            ref['geometry']=check_geometry(ref['path'],scene,ref['max_curvature'])
            checked.append(dict(objective=ref['objective'],geometry=ref['geometry']))
            if not(ref['geometry']['collision_free'] and ref['geometry']['final_inside']
                and ref['geometry']['conservative_continuous_clearance_m']>=.1):continue
            ref['reintegration']=reintegrate(goal,ref)
            refined=evaluate(goal,c['variables'],resolution=16385,quad_n=256)
            ref['quadrature_difference_J']=abs(refined['objective']-ref['objective'])
            ref['verified_feasible']=bool(ref['reintegration']['success'] and ref['reintegration']['position_error_m']<1e-6
                and ref['reintegration']['yaw_error_rad']<1e-6 and ref['max_steer']<=DMAX
                and ref['max_speed']<=VMAX+1e-8 and ref['max_accel']<=AMAX+1e-8 and ref['max_steer_rate']<=OMAX+1e-8)
            arrays['reference']=ref.pop('path');arrays['reference_time_trace']=ref.pop('time_trace')
            result['reference']=ref;break
        result['candidate_geometry_checks']=checked
        rs=rs_reference(goal,scene);arrays['reeds_shepp']=rs.pop('path');result['reeds_shepp']=rs
        for key,p in [('gt',gt),('k4',k4)]:
            result[key+'_fine_retiming']=retime(p,refinement=8,keep_trace=True)
            # Preserve retiming trace separately to keep the human-readable summary small.
            timing=result[key+'_fine_retiming']
            (OUT/(label+'_'+key+'_timing.json')).write_text(json.dumps(timing,indent=2),encoding='utf-8')
            timing.pop('runs',None)
        # Universal rest-to-rest lower bound. Fixed requested XY, yaw +/- 5 degrees.
        D=float(np.linalg.norm(goal[:2]))
        result['unrestricted_J_lower_bound']=4/3*(36*W*D**2)**.25
        # For this *straight-prefix family*, d>=2 implies reverse distance >= |goal_y|.
        # Dropping all other constraints still gives a valid lower bound.
        result['straight_prefix_J_lower_bound']=4/3*(36*W)**.25*(np.sqrt(2.)+np.sqrt(abs(goal[1])))
        np.savez_compressed(OUT/(label+'_paths.npz'),**arrays)
        (OUT/(label+'_result.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
        print('SAVED',label,'reference',result.get('reference',{}).get('objective'),flush=True)
    refine_existing_paths()


if __name__=='__main__':main()
