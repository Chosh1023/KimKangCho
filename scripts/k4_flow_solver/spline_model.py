"""Whole-path Flow in compact spline coefficients, with dense feasible timing."""
from dataclasses import replace
import copy
import math
import torch
from torch import nn
from .whole_model import WholeTrajectoryFlow
from .model import latent_mask
from .smooth_controls import spline_basis,fitting_matrix,torch_retime


class SplineTrajectoryFlow(nn.Module):
    def __init__(self,cfg,normalization,representation):
        super().__init__()
        if representation.get('schema')!='physical-bspline-v1':raise ValueError('unsupported spline contract')
        count=int(representation['coefficients'])
        self.cfg=cfg;self.representation=dict(representation)
        self.sampling_config=replace(cfg,intervals=count-1)
        self.core=WholeTrajectoryFlow(self.sampling_config,normalization)
        self.register_buffer('basis',torch.tensor(spline_basis(count,cfg.intervals),dtype=torch.float64))
        self.register_buffer('fit_matrix',torch.tensor(fitting_matrix(count,cfg.intervals),dtype=torch.float64))

    @property
    def encoder(self):return self.core.encoder

    @property
    def structure(self):return self.core.structure

    def encode(self,batch):return self.core.encode(batch)

    def forward(self,y,t,context,h):return self.core(y,t,context,h)

    def rollout(self,context,h,steps=16,**kwargs):
        y=self.core.rollout(context,h,steps,**kwargs)
        return self.correct(y,context,h) if hasattr(self,'corrector') else y

    def enable_corrector(self,iterations=2):
        if hasattr(self,'corrector'):return
        self.correction_config=dict(schema='supervised-coefficient-correction-v1',iterations=int(iterations))
        self.corrector=copy.deepcopy(self.core)
        for head in (self.corrector.control_head,self.corrector.length_head[-1]):
            nn.init.zeros_(head.weight);nn.init.zeros_(head.bias)
        # Conditions come from the frozen base encoder; unused copies stay frozen.
        self.corrector.encoder.requires_grad_(False);self.corrector.structure.requires_grad_(False)

    def correct(self,y,context,h):
        for _ in range(self.correction_config['iterations']):
            y=y+self.corrector(y,torch.zeros(len(h),device=y.device),context,h)
        return y

    def normalize(self,latent,mask):return self.core.normalize(latent,mask)

    def denormalize(self,y,mask):return self.core.denormalize(y,mask)

    def fit_targets(self,dense,h):
        """No images/goals/teacher costs enter this label-only conversion."""
        cfg=self.cfg;a=dense.double()
        delta=cfg.max_steer*torch.tanh(a[...,1])
        b=cfg.max_speed**2*torch.sigmoid(a[...,2])
        b=b.clone();b[...,[0,-1]]=0
        def fit(values):
            residual=values-values[...,:1]*self.basis[:,0]-values[...,-1:]*self.basis[:,-1]
            middle=residual@self.fit_matrix.T
            return torch.cat((values[...,:1],middle,values[...,-1:]),-1)
        d=fit(delta).clamp(-cfg.max_steer*(1-1e-7),cfg.max_steer*(1-1e-7))
        speed=fit(b).clamp(cfg.max_speed**2*1e-9,cfg.max_speed**2*(1-1e-7))
        mask,_=latent_mask(h,self.sampling_config,a.dtype)
        length=torch.zeros_like(d);length[...,0]=a[:,:,0,0]
        result=torch.stack((length,torch.atanh(d/cfg.max_steer),torch.logit(speed/cfg.max_speed**2)),dim=-1)
        return result.float()*mask.float(),mask.float()

    def dense_latent(self,coefficients,h,*,project=True):
        """Coefficients are unnormalized. Projection is identical at train/infer."""
        cfg=self.cfg;a=coefficients.double()
        mask,valid=latent_mask(h,self.sampling_config,a.dtype)
        lengths=torch.exp(a[:,:,0,0].clamp(-12,math.log(cfg.max_total_length)))*valid
        delta=(cfg.max_steer*torch.tanh(a[...,1]))@self.basis.T
        b=cfg.max_speed**2*torch.sigmoid(a[...,2])
        interior=torch.ones(b.shape[-1],device=b.device,dtype=b.dtype);interior[[0,-1]]=0
        b=(b*interior)@self.basis.T
        if project:b=torch_retime(lengths.clamp_min(1e-6),delta,b,cfg)
        dense_mask,_=latent_mask(h,cfg,a.dtype)
        length=torch.zeros_like(delta);length[...,0]=a[:,:,0,0]
        result=torch.stack((length,torch.atanh((delta/cfg.max_steer).clamp(-1+1e-12,1-1e-12)),
                            torch.logit((b/cfg.max_speed**2).clamp(1e-20,1-1e-12))),dim=-1)
        return result*dense_mask


def decoded_model_samples(model,y,h,cfg,goal=None):
    """Checkpoint-aware representation conversion; old checkpoints unchanged."""
    sampling_cfg=vars(model).get('sampling_config',cfg)
    mask,_=latent_mask(h,sampling_cfg)
    latent=model.denormalize(y,mask)
    if isinstance(model,SplineTrajectoryFlow):
        latent=model.dense_latent(latent,h,project=goal is None)
        if goal is not None:
            from .output_projection import project_output
            latent=project_output(latent,h,goal,cfg,iterations=model.representation.get('projection_iterations',2))
    return latent
