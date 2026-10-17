"""Independent integration and continuous-problem lower bounds for final paths."""
from concurrent.futures import ProcessPoolExecutor,as_completed
from search_k4_global_ocp import *


def near_straight_bound(goal,upper):
    # Every forward heading is within alpha, hence the cusp belongs to
    # [F cos(alpha),F] x [-F sin(alpha),F sin(alpha)]. Each component
    # displacement and the required rotation separately lower-bound reverse R.
    alpha=np.deg2rad(15);radius=WB/np.tan(DMAX);limit=(upper/COST_C)**2
    lines=[(radius*max(abs(goal[2])-alpha,0),0.),(abs(goal[1]),-np.sin(alpha)),
           (-goal[0],np.cos(alpha)),(goal[0],-1.),(0.,0.)]
    points=[2.,limit]
    for i,(a,b) in enumerate(lines):
        for c,d in lines[i+1:]:
            if abs(b-d)>1e-12:
                x=(c-a)/(b-d)
                if 2<x<limit:points.append(x)
    vals=[COST_C*(np.sqrt(f)+np.sqrt(max(a+b*f for a,b in lines))) for f in points]
    i=int(np.argmin(vals));lower=float(vals[i]-1e-7)
    return dict(lower_bound=lower,relaxed_forward_length=points[i],upper_bound=upper,
        remaining_gap_upper_pct=100*(upper/lower-1),
        scope='continuous two-phase path; forward heading within 15deg, F>=2m, exact goal pose; obstacles relaxed',
        proof='sqrt(F)+sqrt(max affine lower bounds on R) is concave between affine intersections; minimum occurs at an intersection or endpoint')


def independent_phase(L,delta,gear,start):
    s=L*arc_grid(len(delta)-1)
    def rhs(distance,state):
        steer=np.interp(distance,s,delta)
        return [gear*np.cos(state[2]),gear*np.sin(state[2]),gear*np.tan(steer)/WB]
    sol=solve_ivp(rhs,[0,L],start,method='DOP853',rtol=2e-10,atol=1e-11,max_step=.02)
    return sol.y[:,-1],dict(success=bool(sol.success),nfev=int(sol.nfev))


def run(label):
    tag='_corridor' if label=='A' else ''
    summary=json.loads((OUT/(label+'_two_phase'+tag+'_final.json')).read_text(encoding='utf-8'));best=summary['best']
    assert best is not None and best['verified_feasible']
    p=dict(np.load(OUT/(label+f'_two_phase{tag}_n{best["nf"]}_{best["nr"]}.npz')))
    old=json.loads((PREVIOUS/(label+'_result.json')).read_text(encoding='utf-8'));goal=np.asarray(old['goal'])
    cusp,cf=independent_phase(p['lengths'][0],p['df'],1,[0,0,0])
    end,cr=independent_phase(p['lengths'][1],p['dr'],-1,cusp)
    error=end-goal
    assert cf['success'] and cr['success'] and np.linalg.norm(error[:2])<3e-6 and abs(error[2])<1e-6
    for L,delta,q in zip(p['lengths'],(p['df'],p['dr']),(p['qf'],p['qr'])):
        met=phase_cost(L,delta,q)
        assert met['max_speed']<=VMAX and met['max_accel']<=AMAX and met['max_steer_rate']<=OMAX and met['max_steer']<=DMAX
    assert abs(best['objective']-best['duration_s']-W*(best['acceleration_integral']+best['steering_integral']))<1e-9
    bound=near_straight_bound(goal,best['objective']+1e-5)
    result=dict(label=label,independent_endpoint=end.tolist(),endpoint_error=error.tolist(),
        forward_integrator=cf,reverse_integrator=cr,analytic_segment_control_limits_pass=True,
        continuous_bound=bound,passed=True)
    (OUT/(label+'_independent_validation.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('VALIDATED',label,error.tolist(),'LB',bound['lower_bound'],flush=True);return result


if __name__=='__main__':
    # An exact constant-steering arc detects sign and length-grid mistakes.
    L=4.;delta=np.full(24,.3);got=exact_reverse(L,delta)['endpoint'];k=-np.tan(.3)/WB
    want=np.array([-np.sin(k*L)/k,(np.cos(k*L)-1)/k,k*L]);assert abs(got-want).max()<1e-12
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,l) for l in 'ABC']
        for future in as_completed(futures):future.result()
