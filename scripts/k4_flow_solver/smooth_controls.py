"""Compact physical B-spline controls, expanded on the unchanged dense grid."""
from functools import lru_cache
import numpy as np
from scipy.interpolate import BSpline
from .config import DEFAULT
from .numerical import Controls, grid


@lru_cache(maxsize=16)
def spline_basis(coefficients, intervals=512):
    if not 8 <= coefficients <= intervals+1:
        raise ValueError('invalid spline coefficient count')
    # Cosine-spaced breakpoints preserve resolution near stationary phase ends.
    interior=grid(coefficients-3)[1:-1]
    knots=np.r_[np.zeros(4),interior,np.ones(4)]
    basis=BSpline(knots,np.eye(coefficients),3)(grid(intervals))
    basis[np.abs(basis)<1e-15]=0.
    return basis


@lru_cache(maxsize=16)
def fitting_matrix(coefficients, intervals=512):
    basis=spline_basis(coefficients,intervals)
    # Fixed endpoint values; interior least squares is deterministic.
    return np.linalg.pinv(basis[:,1:-1],rcond=1e-10)


def fit_values(values, coefficients):
    values=np.asarray(values,dtype=float)
    basis=spline_basis(coefficients,values.shape[-1]-1)
    result=np.empty((*values.shape[:-1],coefficients))
    result[...,0]=values[...,0];result[...,-1]=values[...,-1]
    residual=values-values[...,:1]*basis[:,0]-values[...,-1:]*basis[:,-1]
    result[...,1:-1]=residual@fitting_matrix(coefficients,values.shape[-1]-1).T
    return result


def fit_controls(controls, coefficients, cfg=DEFAULT):
    dc=np.clip(fit_values(controls.delta,coefficients),-cfg.max_steer,cfg.max_steer)
    bc=np.clip(fit_values(controls.b,coefficients),0.,cfg.max_speed**2)
    bc[:,[0,-1]]=0.
    basis=spline_basis(coefficients,controls.delta.shape[-1]-1)
    return Controls(controls.lengths.copy(),dc@basis.T,bc@basis.T,controls.gears.copy()),dc,bc


def torch_retime(lengths, delta, b, cfg=DEFAULT):
    """Exact differentiable counterpart of numerical.retime, using two scans.

    This enforces speed/acceleration/steering-rate caps; not obstacle/goal limits.
    It is not an optimizer for J. Gradients are piecewise defined at active caps.
    """
    import torch
    u=torch.as_tensor(grid(delta.shape[-1]-1),device=delta.device,dtype=delta.dtype)
    s=lengths[...,None]*u
    distance=lengths[...,None]*torch.diff(u)
    slope=torch.diff(delta,dim=-1).abs()/distance.clamp_min(1e-12)
    cap=((cfg.max_steer_rate-cfg.control_margin)/slope.clamp_min(1e-12)).square()
    inf=torch.full_like(cap[...,:1],float('inf'))
    caps=torch.minimum(torch.cat((cap,inf),-1),torch.cat((inf,cap),-1))
    value=torch.minimum(b.clamp(max=(cfg.max_speed-cfg.control_margin)**2),caps)
    interior=torch.ones(delta.shape[-1],device=delta.device,dtype=delta.dtype)
    interior[[0,-1]]=0
    value=value*interior
    twice_accel=2*(cfg.max_accel-cfg.control_margin)
    forward=twice_accel*s+torch.cummin(value-twice_accel*s,dim=-1).values
    backward=-twice_accel*s+torch.cummin((forward+twice_accel*s).flip(-1),dim=-1).values.flip(-1)
    return backward.clamp_min(0)*interior
