"""GT-initialized spatial OCP for an arbitrary fixed sequence of gears.

Uses the fast two-phase formulation and the same continuous validation. It
does not search gear sequences and does not certify global optimality.
"""
import numpy as np
from scipy.optimize import root
import search_k4_two_phase_ocp as ocp
from fast_two_phase_model import build
from validate_k4_global_search import independent_phase


def split_path(path):
    path=np.asarray(path,dtype=float).copy()
    path[:,2]=np.unwrap(path[:,2])
    ds=np.linalg.norm(np.diff(path[:,:2],axis=0),axis=1)
    indices=np.flatnonzero(ds>1e-6)
    if not len(indices):raise ValueError('no moving intervals')
    theta=(path[:-1,2]+path[1:,2])/2
    dot=np.sum(np.diff(path[:,:2],axis=0)*np.c_[np.cos(theta),np.sin(theta)],axis=1)
    signs=np.where(dot[indices]>=0,1,-1)
    # Raw OCP traces can contain millimetre-scale backwards jitter at stops.
    # Remove only sub-centimetre gear runs from the warm-start topology; every
    # output still starts at the original pose and is independently integrated.
    while True:
        runs=np.split(np.arange(len(indices)),np.flatnonzero(signs[1:]!=signs[:-1])+1)
        short=next((i for i,r in enumerate(runs) if ds[indices[r]].sum()<.01),None)
        if short is None or len(runs)==1:break
        neighbour=short-1 if short else 1
        signs[runs[short]]=signs[runs[neighbour][0]]
    groups=np.split(indices,np.flatnonzero(signs[1:]!=signs[:-1])+1)
    parts=[];gears=[]
    for group in groups:
        p=path[np.r_[group,group[-1]+1]].copy()
        s=np.r_[0,np.cumsum(np.linalg.norm(np.diff(p[:,:2],axis=0),axis=1))]
        keep=np.r_[True,np.diff(s)>1e-8]
        p=p[keep];s=s[keep]
        if s[-1]<.005:raise ValueError('total motion shorter than 5 mm')
        parts.append((p,s));gears.append(int(signs[np.searchsorted(indices,group[0])]))
    return parts,gears


def meshes_for(parts,gears):
    # Retain the benchmark's interval allocation for its original scope.
    classic=(gears==[1,-1] and parts[0][1][-1]>=2
             and np.degrees(abs(parts[0][0][:,2])).max()<=15)
    if classic:return [(40,80),(80,160),(160,320)],True
    lengths=np.array([s[-1] for _,s in parts])
    base=np.maximum(12,np.round(120*lengths/lengths.sum()).astype(int))
    return [tuple(base*k) for k in (1,2,4)],False


def pack(lengths,parts):
    changes=[parts[0][1][0],*[b[1][0]-a[1][-1] for a,b in zip(parts,parts[1:])],parts[-1][1][-1]]
    return np.concatenate([np.asarray(lengths),*[v for p,d,q in parts for v in (p.ravel(),d,q)],abs(np.array(changes))/(ocp.OMAX-ocp.EPS)])


def unpack(z,mesh):
    cursor=len(mesh);parts=[]
    for n in mesh:
        p=z[cursor:cursor+3*(n+1)].reshape(-1,3);cursor+=3*(n+1)
        d=z[cursor:cursor+n+1];cursor+=n+1
        q=z[cursor:cursor+n+1];cursor+=n+1
        parts.append((p,d,q))
    return z[:len(mesh)],parts,z[cursor:]


def seed_from_parts(raw,mesh):
    parts=[];lengths=[]
    for (p,s),n in zip(raw,mesh):
        grid=ocp.arc_grid(n);u=grid*s[-1]
        pose=np.column_stack([np.interp(u,s,p[:,j]) for j in range(3)])
        d=np.interp(u,s,p[:,4]);q=1.45*np.sqrt(np.maximum(0,np.sin(np.pi*grid)));q[[0,-1]]=0
        parts.append((pose,d,q));lengths.append(s[-1])
    return pack(lengths,parts)


def remesh(z,oldmesh,mesh):
    lengths,parts,_=unpack(z,oldmesh);new=[]
    for (p,d,q),oldn,n in zip(parts,oldmesh,mesh):
        old=ocp.arc_grid(oldn);u=ocp.arc_grid(n)
        new.append((np.column_stack([np.interp(u,old,p[:,i]) for i in range(3)]),
                    np.interp(u,old,d),np.sqrt(np.interp(u,old,q*q))))
    return pack(lengths,new)


def integrate(lengths,deltas,gears,trace=False):
    start=np.zeros(3);paths=[];arcs=[];nodes=[]
    for L,d,g in zip(lengths,deltas,gears):
        result=ocp.exact_reverse(L,-g*d,trace)
        end=result['endpoint'].copy();end[:2]*=-g
        c,s=np.cos(start[2]),np.sin(start[2]);rot=np.array([[c,-s],[s,c]])
        end[:2]=end[:2]@rot.T+start[:2];end[2]+=start[2]
        if trace:
            p=result['path'].copy();p[:,:2]*=-g
            p[:,:2]=p[:,:2]@rot.T+start[:2];p[:,2]+=start[2];p[:,3]=g;p[:,4]*=-g
            paths.append(p);arcs.append(result['arc'])
            knot=result['node_poses'].copy();knot[:,:2]*=-g
            knot[:,:2]=knot[:,:2]@rot.T+start[:2];knot[:,2]+=start[2];nodes.append(knot)
        start=end
    return (start,paths,arcs,nodes) if trace else start


def validate(z,mesh,gears,scene,goal,classic=False):
    lengths,parts,_=unpack(z,mesh);ds=[d.copy() for _,d,_ in parts]
    # Tiny endpoint correction for RK4 discretization, then fully revalidate.
    # Every accepted correction is bounded; a failed root is never accepted.
    if not classic:
        count=len(mesh)
        def variables(x):return lengths+x[:count],[d+offset for d,offset in zip(ds,x[count:])]
        def residual(x):return integrate(*variables(x),gears)-goal
        x=np.zeros(2*count)
        # A minimum-norm Newton correction avoids the rank deficiency of two
        # adjacent, nearly straight phases. It only removes RK4 error, never
        # changes the acceptance tolerances or bypasses the independent ODE.
        for _ in range(6):
            error=residual(x)
            if np.linalg.norm(error)<1e-10:break
            step=1e-5
            jac=np.column_stack([(residual(x+np.eye(2*count)[j]*step)-error)/step for j in range(2*count)])
            direction=np.linalg.lstsq(jac,-error,rcond=1e-10)[0]
            accepted=False
            for fraction in (1.,.5,.25,.125):
                candidate=x+fraction*direction;ll,dd=variables(candidate)
                if (ll<=1e-6).any() or np.max(abs(candidate[:count]))>=.05 or np.max(abs(candidate[count:]))>=.01:continue
                if np.linalg.norm(residual(candidate))<np.linalg.norm(error):
                    x=candidate;accepted=True;break
            if not accepted:break
        newlen,deltas=variables(x)
        correction_ok=np.linalg.norm(residual(x))<1e-8 and np.max(abs(newlen-lengths))<.05 and max(np.max(abs(a-b)) for a,b in zip(ds,deltas))<.01
    elif len(mesh)>=2:
        def variables(x):
            ll=lengths.copy();ll[-2:]=x[:2];dd=[d.copy() for d in ds];dd[-1]*=x[2]
            return ll,dd
        corrected=root(lambda x:integrate(*variables(x),gears)-goal,[*lengths[-2:],1.],tol=1e-10)
        newlen,deltas=variables(corrected.x)
        correction_ok=corrected.success and np.max(abs(newlen-lengths))<.05 and abs(corrected.x[2]-1)<.01
    else:
        def variables(x):return np.array([x[0]]),[ds[0]+x[1]+x[2]*np.sin(np.pi*ocp.arc_grid(mesh[0]))]
        corrected=root(lambda x:integrate(*variables(x),gears)-goal,[lengths[0],0.,0.],tol=1e-10)
        newlen,deltas=variables(corrected.x)
        correction_ok=corrected.success and abs(newlen[0]-lengths[0])<.05 and np.max(abs(deltas[0]-ds[0]))<.01
    if not np.isfinite(newlen).all() or (newlen<=0).any():
        return dict(verified_feasible=False,reason='invalid_corrected_lengths'),None
    end,paths,arcs,nodes=integrate(newlen,deltas,gears,True)
    metrics=[];speeds=[]
    for L,d,(_,_,q) in zip(newlen,deltas,parts):
        m=ocp.phase_cost(L,d,q)
        slow=max(1.,m['max_speed']/ocp.VMAX,np.sqrt(m['max_accel']/ocp.AMAX),m['max_steer_rate']/ocp.OMAX)*1.00001
        q=q/slow;speeds.append(q);metrics.append(ocp.phase_cost(L,d,q))
    changes=np.r_[deltas[0][0],[b[0]-a[-1] for a,b in zip(deltas,deltas[1:])],deltas[-1][-1]]
    dwell=abs(changes)/(ocp.OMAX-ocp.EPS)
    dense=np.vstack(paths)
    start=dense[0].copy();start[4]=0;finish=dense[-1].copy();finish[4]=0
    geometry=ocp.check_geometry(np.vstack((start,dense,finish)),scene,max(float(abs(np.tan(d)).max()) for d in deltas)/ocp.WB)
    checks=all(m['max_speed']<=ocp.VMAX and m['max_accel']<=ocp.AMAX and m['max_steer_rate']<=ocp.OMAX and m['max_steer']<=ocp.DMAX for m in metrics)
    shape_ok=not classic or (newlen[0]>=2 and np.degrees(abs(paths[0][:,2])).max()<=15)
    terminal=np.linalg.norm(end[:2]-goal[:2])<1e-7 and abs(end[2]-goal[2])<1e-7
    feasible=bool(correction_ok and checks and shape_ok and terminal and (newlen>1e-6).all()
                  and geometry['collision_free'] and geometry['final_inside'] and geometry['conservative_continuous_clearance_m']>=.1)
    independent_ok=False;independent_error=None
    if feasible:
        independent_end=np.zeros(3);independent_ok=True
        for L,d,g in zip(newlen,deltas,gears):
            independent_end,check=independent_phase(L,d,g,independent_end)
            independent_ok &= check['success']
        independent_error=independent_end-goal
        independent_ok &= np.linalg.norm(independent_error[:2])<3e-6 and abs(independent_error[2])<1e-6
    result={k:sum(m[k] for m in metrics) for k in ('objective','duration_s','acceleration_integral','steering_integral')}
    result['objective']+=float(dwell.sum());result['duration_s']+=float(dwell.sum())
    result.update(verified_feasible=bool(feasible and independent_ok),geometry=geometry,control_limits_passed=bool(checks),
        correction_passed=bool(correction_ok),endpoint_error=(end-goal).tolist(),
        independent_endpoint_error=None if independent_error is None else independent_error.tolist(),
        gear_changes=len(gears)-1,gears=gears,length_m=float(newlen.sum()),phases=metrics,
        length_corrections=(newlen-lengths).tolist(),stationary_steer_time_s=float(dwell.sum()))
    if not result['verified_feasible']:return result,None
    # Store controls losslessly and geometry at <=2 cm, keeping phase ends and
    # all steering/speed knots. Dense ~1 mm geometry above is used for checking.
    saved=[];offsets=[0]
    for p,arc,L,n in zip(paths,arcs,newlen,mesh):
        indices=np.unique(np.r_[np.searchsorted(arc,np.arange(0,L,.02)),np.searchsorted(arc,L*ocp.arc_grid(n)),len(p)-1])
        indices=np.minimum(indices,len(p)-1);saved.append(p[indices]);offsets.append(offsets[-1]+len(indices))
    arrays=dict(path_local=np.vstack(saved),phase_path_offsets=np.array(offsets),lengths=newlen,gears=np.array(gears),
                mesh=np.array(mesh),dwell_s=dwell,goal_local=goal,phase_control_offsets=np.r_[0,np.cumsum(np.array(mesh)+1)],
                delta=np.concatenate(deltas),speed=np.concatenate(speeds),poses=np.vstack(nodes))
    return result,arrays
