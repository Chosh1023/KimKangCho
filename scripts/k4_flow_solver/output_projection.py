"""Shared training/inference output mapping: bounded shooting then retiming.

The shooting Jacobian is finite-differenced and detached during backpropagation;
the residual, accepted control update, retiming and final J remain differentiable.
This is an approximate unrolled gradient, not exact implicit differentiation.
Independent geometry/dynamics validation is still mandatory.
"""
import math
import torch
from .model import latent_mask
from .physics import decode_physics
from .smooth_controls import torch_retime


def project_output(latent,h,goal,cfg,iterations=2):
    a=latent.double();batch,phases,knots,_=a.shape
    _,valid=latent_mask(h,cfg,a.dtype)
    length=torch.exp(a[:,:,0,0].clamp(-12,math.log(cfg.max_total_length)))
    steering=cfg.max_steer*torch.tanh(a[...,1])
    speed=cfg.max_speed**2*torch.sigmoid(a[...,2])
    u=.5*(1-torch.cos(torch.linspace(0,math.pi,knots,device=a.device,dtype=a.dtype)))
    sine=torch.sin(math.pi*u)
    goal=goal.to(a)

    def build(x,base_length=length,base_steering=steering,base_speed=speed,classes=h):
        n=len(classes)
        _,active=latent_mask(classes,cfg,a.dtype)
        ll=(base_length+x[:,:phases].clamp(-.25,.25)).clamp_min(1e-5)
        correction=(x[:,phases:2*phases,None]+x[:,2*phases:,None]*sine).clamp(-.05,.05)
        dd=(base_steering+correction).clamp(-cfg.max_steer*(1-1e-12),cfg.max_steer*(1-1e-12))
        lengths=torch.cat((ll.log()[...,None],torch.zeros((n,phases,knots-1),device=a.device,dtype=a.dtype)),dim=-1)
        value=torch.stack((lengths,torch.atanh(dd/cfg.max_steer),torch.logit((base_speed/cfg.max_speed**2).clamp(1e-20,1-1e-12))),-1)
        mask,_=latent_mask(classes,cfg,a.dtype)
        return value*mask

    def error(value,classes=h,targets=goal):
        endpoint=decode_physics(value,classes,cfg)['endpoint']
        d=endpoint-targets
        return torch.cat((d[:,:2],torch.atan2(d[:,2:3].sin(),d[:,2:3].cos())),dim=-1)

    x=torch.zeros((batch,3*phases),device=a.device,dtype=a.dtype)
    for _ in range(iterations):
        value=build(x);residual=error(value)
        needs=(residual[:,:2].norm(dim=-1)>=cfg.terminal_position_tolerance_m) | (residual[:,2].abs()>=cfg.terminal_yaw_tolerance_rad)
        if not bool(needs.any()):break
        # Parallel finite differences avoid O(phase count) Python solver calls.
        with torch.no_grad():
            width=3*phases;eps=1e-4
            perturb=x[:,None,:]+eps*torch.eye(width,device=a.device,dtype=a.dtype)[None]
            vv=build(perturb.reshape(-1,width),length.repeat_interleave(width,0),steering.repeat_interleave(width,0),speed.repeat_interleave(width,0),h.repeat_interleave(width))
            ee=error(vv,h.repeat_interleave(width),goal.repeat_interleave(width,0)).reshape(batch,width,3)
            jac=((ee-residual[:,None,:])/eps).transpose(1,2)
            system=jac@jac.transpose(1,2)+1e-5*torch.eye(3,device=a.device,dtype=a.dtype)[None]
        direction=-(jac.transpose(1,2)@torch.linalg.solve(system,residual[...,None]))[...,0]
        chosen=x;best=residual.square().sum(-1)
        for fraction in (1.,.5,.25):
            trial=x+fraction*direction
            bound=torch.cat((torch.full_like(trial[:,:phases],.25),torch.full_like(trial[:,phases:],.05)),dim=-1)
            trial=trial.clamp(-bound,bound)
            e=error(build(trial));score=e.square().sum(-1)
            accept=needs & (score<best)
            chosen=torch.where(accept[:,None],trial,chosen);best=torch.where(accept,score,best)
        x=chosen
    value=build(x)
    ll=torch.exp(value[:,:,0,0]);dd=cfg.max_steer*torch.tanh(value[...,1])
    bb=cfg.max_speed**2*torch.sigmoid(value[...,2])
    bb=torch_retime(ll,dd,bb,cfg)
    mask,_=latent_mask(h,cfg,a.dtype)
    return torch.stack((value[...,0],value[...,1],torch.logit((bb/cfg.max_speed**2).clamp(1e-20,1-1e-12))),-1)*mask
