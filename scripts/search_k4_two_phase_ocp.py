"""Jointly optimize near-straight forward and curved reverse phases.

Forward heading stays within 15 degrees, including the actual sampled paths.
Exact requested terminal pose. Analysis-only multistart, not a global proof.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from scipy.optimize import root
from search_k4_global_ocp import *


def slices(nf,nr):
    cursor=2;out=[]
    for n in (nf,nr):
        p=slice(cursor,cursor+3*(n+1));cursor=p.stop
        de=slice(cursor,cursor+n+1);cursor=de.stop
        q=slice(cursor,cursor+n+1);cursor=q.stop;out.append((p,de,q))
    return out,slice(cursor,cursor+3)


def build(nf,nr,goal,corridors=None):
    ca=casadi();length=ca.SX.sym('L',2);parts=[];variables=[length];lb=[2.,.5];ub=[20.,25.]
    g=[];gl=[];gu=[]
    def cons(v,lo=0.,hi=0.):g.append(v);gl.extend([lo]*v.numel());gu.extend([hi]*v.numel())
    objective=0
    for phase,(n,gear) in enumerate(((nf,1),(nr,-1))):
        poses=ca.SX.sym('p'+str(phase),3,n+1);de=ca.SX.sym('de'+str(phase),n+1);q=ca.SX.sym('q'+str(phase),n+1)
        variables.extend((ca.reshape(poses,-1,1),de,q));parts.append((poses,de,q))
        yawcap=np.deg2rad(14.99) if phase==0 else 12.
        lb.extend(np.tile([-35,-35,-yawcap],n+1));ub.extend(np.tile([35,35,yawcap],n+1))
        lb.extend(np.full(n+1,-DMAX+EPS));ub.extend(np.full(n+1,DMAX-EPS))
        lb.extend(np.r_[0,np.full(n-1,1e-5),0]);ub.extend(np.r_[0,np.full(n-1,VMAX-EPS),0])
        if corridors is not None:
            for j,planes in enumerate(corridors[phase]):
                for normal,bound in planes:
                    for bx in (1.41563697-2.39588975,1.41563697+2.39588975):
                        for by in (-1.081725,1.081725):
                            xx=poses[0,j]+bx*ca.cos(poses[2,j])-by*ca.sin(poses[2,j])
                            yy=poses[1,j]+bx*ca.sin(poses[2,j])+by*ca.cos(poses[2,j])
                            cons(normal[0]*xx+normal[1]*yy,bound,np.inf)
        for i,weight in enumerate(np.diff(arc_grid(n))):
            h=length[phase]*weight;mid=(de[i]+de[i+1])/2
            def rhs(theta,delta):return gear*ca.vertcat(ca.cos(theta),ca.sin(theta),ca.tan(delta)/WB)
            k1=rhs(poses[2,i],de[i]);k2=rhs(poses[2,i]+h*k1[2]/2,mid)
            k3=rhs(poses[2,i]+h*k2[2]/2,mid);k4=rhs(poses[2,i]+h*k3[2],de[i+1])
            cons(poses[:,i+1]-poses[:,i]-h/6*(k1+2*k2+2*k3+k4))
            dt=2*h/(q[i]+q[i+1]);a=(q[i+1]**2-q[i]**2)/(2*h);slope=(de[i+1]-de[i])/h
            cons(a,-AMAX+EPS,AMAX-EPS)
            cons(slope*q[i],-OMAX+EPS,OMAX-EPS);cons(slope*q[i+1],-OMAX+EPS,OMAX-EPS)
            f4=(q[i]**4+q[i]**3*q[i+1]+q[i]**2*q[i+1]**2+q[i]*q[i+1]**3+q[i+1]**4)/5
            objective+=dt*(1+W*(a*a+slope*slope*f4))
    pf,df,qf=parts[0];pr,dr,qr=parts[1]
    cons(pf[:,0]);cons(pr[:,0]-pf[:,-1]);cons(pr[:,-1]-np.asarray(goal))
    dwell=ca.SX.sym('dwell',3);variables.append(dwell);lb.extend([0]*3);ub.extend([4]*3)
    changes=ca.vertcat(df[0],dr[0]-df[-1],dr[-1])
    cons((OMAX-EPS)*dwell-changes,0,np.inf);cons((OMAX-EPS)*dwell+changes,0,np.inf)
    objective+=ca.sum1(dwell);z=ca.vertcat(*variables)
    solver=ca.nlpsol('two_phase','ipopt',dict(x=z,f=objective,g=ca.vertcat(*g)),{
        'print_time':False,'ipopt.print_level':0,'ipopt.sb':'yes','ipopt.tol':2e-8,
        'ipopt.acceptable_tol':1e-6,'ipopt.max_iter':1000,'ipopt.max_cpu_time':120.,
        'ipopt.bound_relax_factor':0.,'ipopt.mu_strategy':'adaptive'})
    evaluate=ca.Function('two_phase_eval',[z],[objective,ca.vertcat(*g)])
    return dict(solver=solver,evaluate=evaluate,lb=np.array(lb),ub=np.array(ub),gl=np.array(gl),gu=np.array(gu))


def pack_two(lengths,parts):
    values=[np.asarray(lengths)]
    for p,d,q in parts:values.extend((p.ravel(),d,q))
    changes=np.array([parts[0][1][0],parts[1][1][0]-parts[0][1][-1],parts[1][1][-1]])
    values.append(abs(changes)/(OMAX-EPS));return np.concatenate(values)


def unpack_two(z,nf,nr):
    ss,dw=slices(nf,nr)
    return z[:2],[(z[p].reshape(-1,3),z[d],z[q]) for p,d,q in ss],z[dw]


def initial(label,nf,nr):
    with np.load(OUT/(label+'_exact_n5120.npz')) as p:
        lf=float(p['d']);lr=float(p['L']);dr=np.interp(arc_grid(nr),arc_grid(5120),p['delta'])
        qr=np.sqrt(np.interp(arc_grid(nr),arc_grid(5120),p['q']**2))
        pr=np.column_stack([np.interp(arc_grid(nr),arc_grid(5120),p['poses'][:,i]) for i in range(3)])
    pf=np.c_[lf*arc_grid(nf),np.zeros((nf+1,2))];df=np.zeros(nf+1)
    qf=1.45*np.sqrt(np.maximum(np.sin(np.pi*arc_grid(nf)),0));qf[[0,-1]]=0
    return pack_two([lf,lr],[(pf,df,qf),(pr,dr,qr)])


def obstacle_corridors(scene,z,nf,nr,*,phase_parts=None,include_far=False):
    from scipy.ndimage import label as label_components
    from scipy.spatial import ConvexHull
    labs,count=label_components(scene['obstacle_physical']);start=scene['start_state'];c,s=np.cos(start[2]),np.sin(start[2])
    polygons=[]
    boundary=scene['parking_lot_polygon_xy']-start[:2]
    boundary=np.c_[c*boundary[:,0]+s*boundary[:,1],-s*boundary[:,0]+c*boundary[:,1]]
    boundary_planes=[]
    center=boundary.mean(axis=0)
    for a,b in zip(boundary,np.roll(boundary,-1,axis=0)):
        edge=b-a;normal=np.array([-edge[1],edge[0]])/np.linalg.norm(edge)
        if np.dot(center-a,normal)<0:normal=-normal
        boundary_planes.append((normal,float(np.dot(a,normal))))
    if include_far:
        # The exact live checker also treats the 20m raster edge as occupied.
        # Parking-lot polygons can extend beyond that observed area.
        raster=np.array([[-10.,-10.],[10.,-10.],[10.,10.],[-10.,10.]])-start[:2]
        raster=np.c_[c*raster[:,0]+s*raster[:,1],-s*raster[:,0]+c*raster[:,1]]
        for a,b in zip(raster,np.roll(raster,-1,axis=0)):
            edge=b-a;normal=np.array([-edge[1],edge[0]])/np.linalg.norm(edge)
            boundary_planes.append((normal,float(np.dot(a,normal))))
    for index in range(1,count+1):
        rr,cc=np.nonzero(labs==index);centers=np.c_[10-(rr+.5)*.1,-10+(cc+.5)*.1]
        points=(centers[:,None,:]+np.array([[-.05,-.05],[-.05,.05],[.05,.05],[.05,-.05]])[None,:,:]).reshape(-1,2)-start[:2]
        points=np.c_[c*points[:,0]+s*points[:,1],-s*points[:,0]+c*points[:,1]]
        polygons.append(points[ConvexHull(points).vertices])
    parts=unpack_two(z,nf,nr)[1] if phase_parts is None else phase_parts
    result=[]
    body=np.array([[bx,by] for bx in (1.41563697-2.39588975,1.41563697+2.39588975) for by in (-1.081725,1.081725)])
    for poses,_,_ in parts:
        phase=[]
        for pose in poses:
            c,s=np.cos(pose[2]),np.sin(pose[2]);rot=np.array([[c,-s],[s,c]])
            corners=body@rot.T+pose[:2];planes=[]
            for normal,threshold in boundary_planes:
                if include_far or np.min(corners@normal)-threshold<3.:
                    planes.append((normal,threshold+.12))
            for polygon in polygons:
                edges=np.roll(polygon,-1,axis=0)-polygon;normals=np.c_[-edges[:,1],edges[:,0]]
                normals=normals/np.linalg.norm(normals,axis=1)[:,None];normals=np.vstack((normals,rot.T,-normals,-rot.T))
                thresholds=(polygon@normals.T).max(axis=0);gap=(corners@normals.T).min(axis=0)-thresholds
                best=int(gap.argmax())
                if include_far or gap[best]<3.:
                    planes.append((normals[best],float(thresholds[best]+.12)))
            phase.append(planes)
        result.append(phase)
    return result


def remesh(z,oldf,oldr,newf,newr):
    lengths,parts,_=unpack_two(z,oldf,oldr);new=[]
    for (p,d,q),oldn,n in zip(parts,(oldf,oldr),(newf,newr)):
        pg=np.column_stack([np.interp(arc_grid(n),arc_grid(oldn),p[:,i]) for i in range(3)])
        new.append((pg,np.interp(arc_grid(n),arc_grid(oldn),d),np.sqrt(np.interp(arc_grid(n),arc_grid(oldn),q*q))))
    return pack_two(lengths,new)


def integrate_pair(lengths,df,dr,trace=False):
    f=exact_reverse(lengths[0],-df,trace);r=exact_reverse(lengths[1],dr,trace)
    f['endpoint'][:2]*=-1;theta=f['endpoint'][2];c,s=np.cos(theta),np.sin(theta);rotation=np.array([[c,-s],[s,c]])
    rend=r['endpoint'].copy();rend[:2]=rotation@rend[:2]+f['endpoint'][:2];rend[2]+=theta
    if trace:
        fp=f['path'];fp[:,:2]*=-1;fp[:,3]=1;fp[:,4]*=-1
        rp=r['path'];rp[:,:2]=rp[:,:2]@rotation.T+f['endpoint'][:2];rp[:,2]+=theta
        end=rp[-1].copy();end[4]=0;start=fp[0].copy();start[4]=0
        return rend,np.vstack((start,fp,rp,end)),f['endpoint']
    return rend


def validate_two(label,z,nf,nr,scene,goal):
    lengths,parts,_=unpack_two(z,nf,nr);pf,df,qf=parts[0];pr,dr,qr=parts[1]
    def residual(p):return integrate_pair(p[:2],df,dr*p[2])-goal
    corrected=root(residual,[*lengths,1.],tol=1e-10);newlen=corrected.x[:2];scale=corrected.x[2];dr=dr*scale
    end,path,cusp=integrate_pair(newlen,df,dr,True);metrics=[];newq=[]
    for L,de,q in zip(newlen,(df,dr),(qf,qr)):
        met=phase_cost(L,de,q);slow=max(1.,met['max_speed']/VMAX,np.sqrt(met['max_accel']/AMAX),met['max_steer_rate']/OMAX)*1.00001
        q=q/slow;newq.append(q);metrics.append(phase_cost(L,de,q))
    dwell=float((abs(df[0])+abs(dr[0]-df[-1])+abs(dr[-1]))/(OMAX-EPS))
    geometry=check_geometry(path,scene,float(max(abs(np.tan(df)).max(),abs(np.tan(dr)).max())/WB))
    moving_forward=path[:,3]>0;maxheading=float(np.degrees(abs(path[moving_forward,2])).max())
    result={k:sum(m[k] for m in metrics) for k in ('objective','duration_s','acceleration_integral','steering_integral')}
    result['objective']+=dwell;result['duration_s']+=dwell
    feasible=bool(corrected.success and np.max(abs(newlen-lengths))<.05 and abs(scale-1)<.01
        and np.linalg.norm(end[:2]-goal[:2])<1e-7 and abs(end[2]-goal[2])<1e-7
        and newlen[0]>=2 and maxheading<=15 and max(abs(df).max(),abs(dr).max())<=DMAX
        and geometry['collision_free'] and geometry['final_inside'] and geometry['conservative_continuous_clearance_m']>=.1)
    result.update(label=label,nf=nf,nr=nr,verified_feasible=feasible,forward=metrics[0],reverse=metrics[1],
        stationary_steer_time_s=dwell,length_m=float(newlen.sum()),forward_length_m=float(newlen[0]),reverse_length_m=float(newlen[1]),
        max_forward_heading_deg=maxheading,cusp=cusp.tolist(),gear_changes=1,geometry=geometry,
        endpoint_error=(end-goal).tolist(),correction=dict(length_changes=(newlen-lengths).tolist(),reverse_delta_scale=float(scale)),
        path=path,df=df,dr=dr,qf=newq[0],qr=newq[1],lengths=newlen)
    return result


def run(label,starts=24,with_obstacles=False):
    row=json.loads((OUT/(label+'_search.json')).read_text(encoding='utf-8'));goal=np.array(row['goal'])
    with np.load(Path(row['raw_sample_dir'])/'planner_input.npz') as p:scene={k:p[k] for k in p.files}
    tag='_corridor' if with_obstacles else ''
    nf,nr=20,40;base=initial(label,nf,nr)
    corridor=obstacle_corridors(scene,base,nf,nr) if with_obstacles else None
    problem=build(nf,nr,goal,corridor);rng=np.random.default_rng(291+ord(label));logs=[];best=None
    for i in range(starts):
        start=base.copy() if i==0 or best is None else best['z'].copy()
        if i:
            ll,pp,_=unpack_two(start,nf,nr)
            for j,n in enumerate((nf,nr)):
                pp[j][1][:]+=rng.normal(0,.22)*np.sin(np.pi*arc_grid(n))+rng.normal(0,.18)*np.sin(2*np.pi*arc_grid(n))
            ll*=rng.uniform(.85,1.2,2);start=pack_two(ll,pp)
        solved=solve(problem,start);z=solved.pop('z');logs.append(solved)
        if solved['success'] and (best is None or solved['objective']<best['objective']):best=dict(z=z,**solved)
        print('TWO',label,i,solved['objective'],solved['status'],flush=True)
    assert best is not None;z=best['z'];checked=[]
    for newf,newr in ((40,80),(160,320),(640,1280),(1280,2560)):
        seed=remesh(z,nf,nr,newf,newr)
        corridor=obstacle_corridors(scene,seed,newf,newr) if with_obstacles else None
        problem=build(newf,newr,goal,corridor);sol=solve(problem,seed);z=sol.pop('z');nf,nr=newf,newr
        print('TWO REFINE',label,nf,nr,sol['objective'],sol['status'],flush=True)
        if not sol['success']:break
        val=validate_two(label,z,nf,nr,scene,goal);arrays={k:val.pop(k) for k in ('path','df','dr','qf','qr','lengths')};val['solver']=sol
        np.savez_compressed(OUT/(label+f'_two_phase{tag}_n{nf}_{nr}.npz'),**arrays,z=z,nf=nf,nr=nr)
        (OUT/(label+f'_two_phase{tag}_n{nf}_{nr}.json')).write_text(json.dumps(val,indent=2),encoding='utf-8');checked.append(val)
    valid=[v for v in checked if v['verified_feasible']]
    result=dict(label=label,starts=logs,refinement=checked,best=min(valid,key=lambda r:r['objective']) if valid else None,
        scope='forward heading within +/-15 degrees, forward length >=2m, one switch, exact requested goal pose',global_optimum_certified=False)
    result['conservative_obstacle_corridor']=with_obstacles
    (OUT/(label+'_two_phase'+tag+'_final.json')).write_text(json.dumps(result,indent=2),encoding='utf-8');return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--labels',default='ABC');parser.add_argument('--starts',type=int,default=24);args=parser.parse_args()
    with ProcessPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(run,l,args.starts,l=='A') for l in args.labels]
        for future in as_completed(futures):future.result()
