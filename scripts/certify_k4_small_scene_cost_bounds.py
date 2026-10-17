"""Small-scene global lower/feasible upper bound comparison.

Never labels a local optimizer as globally optimal. The analytic lower bound
applies to every continuous rest-to-rest bicycle trajectory; a smooth polynomial
route gives an independently checked feasible upper bound. Also evaluates the
globally shortest Reeds-Shepp geometry for an exact terminal pose, separately
from the time/comfort objective. Analysis only; no production files are changed.
"""
from __future__ import annotations
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import numpy as np
from numpy.polynomial import Polynomial as P
from scipy.integrate import cumulative_simpson, quad, solve_ivp
from scipy.interpolate import PchipInterpolator
from scipy.optimize import minimize
from numpy.polynomial.legendre import leggauss

from compare_k4_gt_ocp_cost import ROOT, AUDIT, load_paths, retime

OUT=ROOT/"analysis_artifacts/k4_global_reference_20260916"
PREVIOUS=ROOT/"analysis_artifacts/k4_gt_ocp_cost_refined_20260916"
UIDS=["s00-o000006632-36194c0b185f", "s00-o000003821-483a21b46789",
      "s00-o000004818-b956c9fd2e5f", "s00-o000007429-690ffe44f405"]
WB=3.004647
W=.5
VMAX=2.5
AMAX=2.
OMAX=.6
DMAX=.9809930780518399
RAMP=1.e-5  # Continuous acceleration goes to zero at both endpoints.


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def normalized_clock(u):
    """Unit-area velocity f(u), f'(u), and cumulative area; f=f'=0 at ends."""
    u=np.asarray(u,dtype=float)
    e=RAMP; end=np.exp(-1/e); den=1-end
    left=np.exp(-u/e); right=np.exp(-(1-u)/e)
    area=1/6+e*(2*e*den-1-end)/den
    f=(u*(1-u)+e*(left+right-1-end)/den)/area
    fp=(1-2*u+(-left+right)/den)/area
    cumulative=(u*u/2-u**3/3+e*(e*(1-left)+e*(right-end)-(1+end)*u)/den)/area
    return np.maximum(f,0),fp,np.clip(cumulative,0,1)


A_COEFF=quad(lambda u: float(normalized_clock(u)[1])**2,0,1,
    points=[RAMP,10*RAMP,1-10*RAMP,1-RAMP],epsabs=1.e-10,epsrel=1.e-11)[0]


def curve_polynomial(goal, variables):
    x,y,_=goal
    c0,c1,terminal_yaw=variables
    # y, dy/dx, d2y/dx2 match initial/terminal pose and zero steering.
    matrix=np.array([[1,1,1],[3,4,5],[6,12,20]],dtype=float)
    a=np.linalg.solve(matrix,np.array([y,x*np.tan(terminal_yaw),0.]))
    base=P([0,0,0,*a])
    u=P([0,1]); bump=u**3*(1-u)**3
    return base+bump*(c0+c1*(2*u-1))


def curve_values(goal, poly, r):
    x=goal[0]
    y=poly(r); dy=poly.deriv(1)(r); ddy=poly.deriv(2)(r); dddy=poly.deriv(3)(r)
    slope=dy/x; second=ddy/x**2
    h=np.hypot(x,dy)
    curvature=second/(1+slope*slope)**1.5
    curvature_r=dddy/x**2/(1+slope*slope)**1.5-3*second*slope*(ddy/x)/(1+slope*slope)**2.5
    delta=np.arctan(WB*curvature)
    delta_s=WB*curvature_r/(1+(WB*curvature)**2)/h
    return dict(x=x*np.asarray(r),y=y,yaw=np.arctan(slope),steer=delta,steer_s=delta_s,h=h,curvature=curvature)


def evaluate_curve(goal, variables, resolution=1001, quadrature=96, trace=False):
    poly=curve_polynomial(goal,variables)
    r=np.linspace(0,1,resolution); values=curve_values(goal,poly,r)
    arc=np.r_[0,cumulative_simpson(values["h"],x=r)]; length=float(arc[-1])
    r_of_s=PchipInterpolator(arc,r)
    nodes,weights=leggauss(quadrature); u=(nodes+1)/2; weights=weights/2
    f,_,cumulative=normalized_clock(u)
    d=curve_values(goal,poly,r_of_s(length*cumulative))["steer_s"]
    steering_coefficient=float(length**4*np.dot(weights,d*d*f**4))
    acceleration_coefficient=length*length*A_COEFF
    coefficient=acceleration_coefficient+steering_coefficient
    duration=(3*W*coefficient)**.25
    # Uniform time nodes plus boundary-layer nodes check continuous controls.
    u=np.unique(np.r_[np.linspace(0,1,4001),np.geomspace(1.e-10,.01,100),1-np.geomspace(1.e-10,.01,100)])
    f,fp,cumulative=normalized_clock(u)
    timed=curve_values(goal,poly,r_of_s(length*cumulative))
    max_f=float(np.max(f)); max_fp=float(np.max(np.abs(fp)))
    max_rate_factor=float(np.max(np.abs(timed["steer_s"])*f))
    duration=max(duration,length*max_f/VMAX,np.sqrt(length*max_fp/AMAX),length*max_rate_factor/OMAX)
    acc=acceleration_coefficient/duration**3; steer=steering_coefficient/duration**3
    maximum_steer=float(np.max(np.abs(values["steer"])))
    result=dict(objective=duration+W*(acc+steer),duration_s=duration,length_m=length,
        acceleration_integral=acc,steering_integral=steer,
        max_speed=length*max_f/duration,max_accel=length*max_fp/duration**2,
        max_steer_rate=length*max_rate_factor/duration,max_steer=maximum_steer,
        terminal_yaw_rad=float(variables[2]),variables=np.asarray(variables).tolist(),
        polynomial_coefficients=poly.coef.tolist(),gear_changes=0)
    if trace:
        result["path"]=np.c_[values["x"],values["y"],values["yaw"],-np.ones(len(r)),values["steer"]]
        result["time_trace"]=np.c_[u*duration,timed["x"],timed["y"],timed["yaw"],
            -length/duration*f,-length/duration**2*fp,timed["steer"],length/duration*f*timed["steer_s"]]
        result["arc"]=arc
        result["max_curvature"]=float(np.max(np.abs(values["curvature"])))
    return result


def optimize_reference(goal,yaw_tolerance):
    center=float(goal[2]); lo,hi=center-yaw_tolerance,center+yaw_tolerance
    def objective(x):
        result=evaluate_curve(goal,x)
        return result["objective"]+1000*max(0,result["max_steer"]-DMAX)**2
    starts=[np.array([0.,0.,np.clip(1.5*np.arctan(goal[1]/goal[0]),lo,hi)]),np.array([0.,0.,center])]
    best=None; attempts=[]
    for x in starts:
        solved=minimize(objective,x,method="L-BFGS-B",bounds=[(-30,30),(-30,30),(lo,hi)],
            options={"maxiter":120,"ftol":1.e-13,"gtol":1.e-8,"maxls":30})
        attempts.append(dict(success=bool(solved.success),message=str(solved.message),objective=float(solved.fun)))
        if best is None or solved.fun<best.fun: best=solved
    result=evaluate_curve(goal,best.x,resolution=32769,quadrature=768,trace=True)
    alternative=evaluate_curve(goal,best.x,resolution=16385,quadrature=384)
    result["quadrature_refinement_J_difference"]=abs(result["objective"]-alternative["objective"])
    result["optimization_attempts"]=attempts
    result["yaw_tolerance_deg"]=math.degrees(yaw_tolerance)
    return result


def independent_reintegration(goal,result):
    poly=P(result["polynomial_coefficients"])
    r=np.linspace(0,1,65537); values=curve_values(goal,poly,r)
    arc=np.r_[0,cumulative_simpson(values["h"],x=r)]; length=arc[-1]
    r_of_s=PchipInterpolator(arc,r); duration=result["duration_s"]
    def rhs(t,state):
        f,_,fraction=normalized_clock(t/duration)
        v=-length/duration*float(f)
        steer=float(curve_values(goal,poly,r_of_s(length*fraction))["steer"])
        return [v*np.cos(state[2]),v*np.sin(state[2]),v*np.tan(steer)/WB]
    solution=solve_ivp(rhs,[0,duration],[0,0,0],method="DOP853",rtol=1.e-11,atol=1.e-12,max_step=.01)
    end=solution.y[:,-1]; expected=np.array([goal[0],goal[1],result["terminal_yaw_rad"]])
    return dict(success=bool(solution.success),endpoint=end.tolist(),position_error_m=float(np.linalg.norm(end[:2]-expected[:2])),
        yaw_error_rad=float(abs(end[2]-expected[2])))


def check_geometry(path,scene,max_curvature=None):
    import torch
    from compare_k4_gt_paths import enable_pruning
    from hope_path_transformer.worker_api import _live_candidate_geometry,LiveVehicleConfig
    enable_pruning()
    # ~2 mm spacing plus all original points, preserving analytical samples.
    spacing=np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1)
    if len(path)>5000:
        step=max(1,int(.0015/max(spacing.max(),1.e-12)))
        indices=np.unique(np.r_[np.arange(0,len(path),step),len(path)-1]); path=path[indices]
        spacing=np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1)
    valid,clearance,inside=_live_candidate_geometry({"poses":torch.tensor(path[None,None,:,:3],dtype=torch.float32),
        "lengths":torch.tensor([[len(path)]])},scene["obstacle_physical"],scene["start_state"],
        scene["parking_lot_polygon_xy"],LiveVehicleConfig(),scene["target_slot_polygon_xy"])
    # Body displacement between samples <= ds/2 * (1 + R*max_curvature).
    radius=np.hypot(1.41563697+2.39588975,1.081725)
    margin=float(spacing.max()*.5*(1+radius*max_curvature)+2.e-5) if max_curvature is not None else None
    return dict(collision_free=bool(valid[0,0]),final_inside=bool(inside[0,0]),
        sampled_clearance_m=float(clearance[0,0]),max_sample_spacing_m=float(spacing.max()),
        intersample_body_motion_bound_m=margin,
        conservative_continuous_clearance_m=float(clearance[0,0])-margin if margin is not None else None)


def reeds_shepp_reference(goal,scene):
    from reeds_shepp_path import generate_paths,sample_path
    maxc=np.tan(DMAX)/WB
    candidates=generate_paths([0,0,0],goal,maxc)
    candidate=min(candidates,key=lambda p:p.total_length)
    sampled=sample_path(candidate,[0,0,0],maxc,step_size=.001,max_steer=DMAX)
    # sample_path mutates and returns the same path object.
    p=np.c_[sampled.x,sampled.y,sampled.yaw,sampled.directions,sampled.steers]
    endpoint_error=float(np.linalg.norm(p[-1,:2]-goal[:2]))
    yaw_error=float(abs(np.arctan2(np.sin(p[-1,2]-goal[2]),np.cos(p[-1,2]-goal[2]))))
    spec=importlib.util.spec_from_file_location("hope_rs_audit",ROOT/"third_party/HOPE/src/env/reeds_shepp.py")
    independent=importlib.util.module_from_spec(spec); spec.loader.exec_module(independent)
    check=independent.calc_optimal_path(0,0,0,*goal,maxc,step_size=.01)
    result=dict(length_m=candidate.total_length,word=candidate.ctypes,signed_segment_lengths_m=candidate.lengths,
        candidate_count=len(candidates),gear_changes=candidate.gear_switches,
        first_gear=next(1 if x>0 else -1 for x in candidate.lengths if abs(x)>1.e-9),
        endpoint_error_m=endpoint_error,yaw_error_rad=yaw_error,
        independent_hope_length_m=float(check.L),independent_length_difference_m=abs(check.L-candidate.total_length),
        geometry=check_geometry(p,scene,maxc),path=p)
    result["globally_shortest_length_under_exact_pose"]=bool(endpoint_error<1.e-6 and yaw_error<1.e-6
        and result["independent_length_difference_m"]<1.e-7 and result["geometry"]["collision_free"]
        and result["geometry"]["final_inside"] and result["geometry"]["conservative_continuous_clearance_m"]>=.1
        and result["first_gear"]==-1)
    return result


def main():
    OUT.mkdir(exist_ok=False,parents=True)
    from hope_path_transformer.geometry import states_world_to_local
    original={r["uid"]:r for r in read_rows(AUDIT/"per_scene.jsonl")}
    previous={r["uid"]:r for r in read_rows(PREVIOUS/"per_scene.jsonl")}
    rows=[]; started=time.perf_counter()
    for index,uid in enumerate(UIDS):
        row=original[uid]; old=previous[uid]; raw=Path(row["raw_sample_dir"])
        with np.load(raw/"planner_input.npz") as z: scene={k:z[k] for k in z.files}
        # goal_state stores the accepted GT endpoint, including its soft yaw.
        # The network targets requested_goal_pose, so use that common request.
        requested=np.r_[scene["requested_goal_pose"],-1.,0.][None,:]
        goal=states_world_to_local(requested,scene["start_state"])[0,:3].astype(float)
        assert goal[0]<0 and old["start_direction"]==-1 and old["required_maneuver"]=="any"
        gt,k4=load_paths(row); gt=states_world_to_local(gt,scene["start_state"]).astype(float)
        distance=float(np.linalg.norm(goal[:2])); lower=4/3*(36*W*distance**2)**.25
        result=dict(uid=uid,label=chr(ord('A')+index),raw_sample_dir=str(raw),goal=goal.tolist(),
            euclidean_distance_m=distance,global_J_lower_bound=lower,
            gt=old["gt"],k4=old["k4"],gt_goal_yaw_error_deg=row["gt_raw"]["goal_yaw_error_deg"])
        arrays={"gt":gt,"k4":k4,"goal":goal,"start_source":scene["start_state"]}
        for name,tolerance in (("exact_yaw",0.),("yaw_tolerance_5deg",math.radians(5))):
            ref=optimize_reference(goal,tolerance)
            ref["geometry"]=check_geometry(ref["path"],scene,ref["max_curvature"])
            ref["reintegration"]=independent_reintegration(goal,ref)
            # Outward numerical allowance, separate from the analytic bound.
            upper=ref["objective"]+max(1.e-6,10*ref["quadrature_refinement_J_difference"])
            ref["numerical_upper_bound_with_allowance"]=upper
            ref["global_gap_upper_bound_pct"]=100*(upper/lower-1)
            ref["verified_feasible"]=bool(ref["geometry"]["collision_free"] and ref["geometry"]["final_inside"]
                and ref["geometry"]["conservative_continuous_clearance_m"]>=.1
                and ref["max_steer"]<=DMAX and ref["max_steer_rate"]<=OMAX+1.e-8
                and ref["max_speed"]<=VMAX+1.e-8 and ref["max_accel"]<=AMAX+1.e-8
                and ref["reintegration"]["position_error_m"]<1.e-7 and ref["reintegration"]["yaw_error_rad"]<1.e-7)
            arrays[name]=ref.pop("path"); arrays[name+"_time_trace"]=ref.pop("time_trace"); ref.pop("arc")
            result[name]=ref
        rs=reeds_shepp_reference(goal,scene); arrays["reeds_shepp"]=rs.pop("path"); result["reeds_shepp"]=rs
        np.savez_compressed(OUT/(result["label"]+"_paths.npz"),**arrays)
        # Re-evaluate old geometry at the finer resolution; preserve prior values.
        result["gt_fine_retiming"]=retime(gt,refinement=8)
        result["k4_fine_retiming"]=retime(k4,refinement=8)
        rows.append(result)
        (OUT/"results.json").write_text(json.dumps(dict(rows=rows,parameters=dict(weight=W,vmax=VMAX,amax=AMAX,omega_max=OMAX,delta_max=DMAX,wheelbase=WB,clock_ramp=RAMP),elapsed_s=time.perf_counter()-started),indent=2),encoding="utf-8")
        print(json.dumps({k:result[k] for k in ("label","uid","global_J_lower_bound")}),flush=True)
        for name in ("exact_yaw","yaw_tolerance_5deg"):
            print(name,json.dumps({k:result[name][k] for k in ("objective","global_gap_upper_bound_pct","verified_feasible","geometry","reintegration")}),flush=True)
        print("RS",json.dumps(rs),flush=True)
    (OUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (OUT/"manifest.json").write_text(json.dumps(dict(uids=UIDS,selection="near-aligned reverse-start, same audited conditions; intentionally easy certification cases, not representative",source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),previous_audit=str(PREVIOUS)),indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
