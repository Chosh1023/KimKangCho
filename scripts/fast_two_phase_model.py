"""Vectorized construction of the same two-phase spatial RK4 NLP."""
import os
import numpy as np
import search_k4_two_phase_ocp as ocp


def build(nf,nr,goal,corridors=None,*,forward_heading_deg=14.99,phases=None):
    # The original two-phase entry point retains exactly its original bounds.
    generalized=phases is not None
    phases=list(phases) if generalized else [(nf,1),(nr,-1)]
    count=len(phases)
    ca=ocp.casadi();length=ca.SX.sym('L',count);variables=[length]
    lb=[.005]*count if generalized else [2.,.5]
    ub=[40.]*count if generalized else [20.,25.]
    g=[];gl=[];gu=[];parts=[];objective=0
    def cons(v,low=0.,high=0.):
        v=ca.vec(v);g.append(v);gl.extend([low]*v.numel());gu.extend([high]*v.numel())
    for phase,(n,gear) in enumerate(phases):
        p=ca.SX.sym('p'+str(phase),3,n+1);d=ca.SX.sym('d'+str(phase),n+1);q=ca.SX.sym('q'+str(phase),n+1)
        variables.extend([ca.vec(p),d,q]);parts.append((p,d,q))
        yawcap=np.deg2rad(forward_heading_deg) if phase==0 and not generalized else 12.
        lb.extend(np.tile([-35,-35,-yawcap],n+1));ub.extend(np.tile([35,35,yawcap],n+1))
        lb.extend(np.full(n+1,-ocp.DMAX+ocp.EPS));ub.extend(np.full(n+1,ocp.DMAX-ocp.EPS))
        lb.extend(np.r_[0,np.full(n-1,1e-5),0]);ub.extend(np.r_[0,np.full(n-1,ocp.VMAX-ocp.EPS),0])
        if corridors is not None:
            ids=[];normals=[];thresholds=[]
            for j,planes in enumerate(corridors[phase]):
                for normal,bound in planes:ids.append(j);normals.append(normal);thresholds.append(bound)
            if ids:
                normals=np.asarray(normals);theta=p[2,ids].T
                for bx in (1.41563697-2.39588975,1.41563697+2.39588975):
                    for by in (-1.081725,1.081725):
                        xx=p[0,ids].T+bx*ca.cos(theta)-by*ca.sin(theta)
                        yy=p[1,ids].T+bx*ca.sin(theta)+by*ca.cos(theta)
                        cons(ca.DM(normals[:,0])*xx+ca.DM(normals[:,1])*yy-ca.DM(thresholds),0,np.inf)
        h=length[phase]*ca.DM(np.diff(ocp.arc_grid(n)));mid=(d[:-1]+d[1:])/2
        theta=p[2,:-1].T
        def rhs(t,de):return gear*ca.horzcat(ca.cos(t),ca.sin(t),ca.tan(de)/ocp.WB)
        k1=rhs(theta,d[:-1]);k2=rhs(theta+h*k1[:,2]/2,mid)
        k3=rhs(theta+h*k2[:,2]/2,mid);k4=rhs(theta+h*k3[:,2],d[1:])
        cons((p[:,1:]-p[:,:-1]).T-ca.repmat(h,1,3)/6*(k1+2*k2+2*k3+k4))
        dt=2*h/(q[:-1]+q[1:]);a=(q[1:]**2-q[:-1]**2)/(2*h);slope=(d[1:]-d[:-1])/h
        cons(a,-ocp.AMAX+ocp.EPS,ocp.AMAX-ocp.EPS)
        cons(slope*q[:-1],-ocp.OMAX+ocp.EPS,ocp.OMAX-ocp.EPS)
        cons(slope*q[1:],-ocp.OMAX+ocp.EPS,ocp.OMAX-ocp.EPS)
        u=q[:-1];v=q[1:];f4=(u**4+u**3*v+u**2*v**2+u*v**3+v**4)/5
        objective+=ca.sum1(dt*(1+ocp.W*(a*a+slope*slope*f4)))
    cons(parts[0][0][:,0]);cons(parts[-1][0][:,-1]-np.asarray(goal))
    for previous,current in zip(parts,parts[1:]):cons(current[0][:,0]-previous[0][:,-1])
    dwell=ca.SX.sym('dwell',count+1);variables.append(dwell);lb.extend([0]*(count+1));ub.extend([4]*(count+1))
    changes=ca.vertcat(parts[0][1][0],*[b[1][0]-a[1][-1] for a,b in zip(parts,parts[1:])],parts[-1][1][-1])
    cons((ocp.OMAX-ocp.EPS)*dwell-changes,0,np.inf);cons((ocp.OMAX-ocp.EPS)*dwell+changes,0,np.inf)
    objective+=ca.sum1(dwell);z=ca.vertcat(*variables);constraints=ca.vertcat(*g)
    solver=ca.nlpsol('fast_two','ipopt',dict(x=z,f=objective,g=constraints),{
        'print_time':False,'ipopt.print_level':0,'ipopt.sb':'yes','ipopt.tol':2e-8,
        'ipopt.acceptable_tol':1e-6,'ipopt.max_iter':1000,'ipopt.max_cpu_time':float(os.environ.get('FAST_OCP_CPU','120')),
        'ipopt.bound_relax_factor':0.,'ipopt.mu_strategy':'adaptive'})
    evaluate=ca.Function('fast_eval',[z],[objective,constraints])
    return dict(solver=solver,evaluate=evaluate,lb=np.array(lb),ub=np.array(ub),gl=np.array(gl),gu=np.array(gu))
