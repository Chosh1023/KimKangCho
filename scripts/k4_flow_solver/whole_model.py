"""Conditional FM over the entire trajectory, with global coarse attention.

Phase boundaries remain physical metadata. All phase knots belong to one
sequence and are updated together at every flow time; no autoregressive decode.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .model import FlowSolver, NumericEncoder, TemporalBlock, latent_mask, time_embedding
from .config import DEFAULT


class WholeTrajectoryFlow(FlowSolver):
    def __init__(self, cfg=DEFAULT, normalization=None):
        nn.Module.__init__(self)
        self.cfg = cfg
        d, w = cfg.context_dim, cfg.temporal_channels
        self.encoder = NumericEncoder(d)
        self.structure = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 2*cfg.max_phases))
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.h_embedding = nn.Embedding(2*cfg.max_phases, d)
        self.down1 = TemporalBlock(9, w, d)
        self.down2 = TemporalBlock(w, w*2, d)
        self.middle = TemporalBlock(w*2, d, d)
        self.global_cross = nn.MultiheadAttention(d, 6, batch_first=True)
        block = nn.TransformerEncoderLayer(d, 6, d*3, dropout=0., batch_first=True,
                                            activation='gelu', norm_first=True)
        self.global_mix = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)
        self.up2 = TemporalBlock(d+w*2, w*2, d)
        self.up1 = TemporalBlock(w*3, w, d)
        self.control_head = nn.Conv1d(w, 2, 1)
        self.length_head = nn.Sequential(nn.Linear(w+d, d), nn.SiLU(), nn.Linear(d, 1))
        for head in (self.control_head, self.length_head[-1]):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        norm = normalization or {'mean':[0, 0, 0], 'std':[1, 1, 1]}
        self.register_buffer('latent_mean', torch.tensor(norm['mean'], dtype=torch.float32))
        self.register_buffer('latent_std', torch.tensor(norm['std'], dtype=torch.float32))
        self.register_buffer('u', .5*(1-torch.cos(torch.linspace(0, math.pi, cfg.intervals+1))))

    def forward(self, y, t, context, h):
        cfg = self.cfg
        batch, phases, knots, _ = y.shape
        mask, valid = latent_mask(h, cfg, y.dtype)
        y = y*mask
        phase = torch.arange(phases, device=y.device)
        gear = torch.where(h < cfg.max_phases, 1., -1.)[:, None]*torch.where(phase%2==0, 1., -1.)[None]
        expand = lambda a: a.expand(batch, phases, knots)
        node_valid = expand(valid[..., None]).to(y.dtype)
        u = expand(self.u[None, None])
        features = torch.stack((y[..., 1], y[..., 2], expand(y[:, :, 0, 0, None]),
            u, expand(gear[..., None]), expand(phase[None, :, None])/cfg.max_phases,
            (u==0).to(y.dtype), (u==1).to(y.dtype), node_valid), dim=-1)
        features = (features*node_valid[..., None]).reshape(batch, phases*knots, 9).transpose(1, 2)
        condition = context.mean(1)+self.h_embedding(h)+self.time(time_embedding(t, cfg.context_dim))
        a = self.down1(features, condition)
        b = self.down2(F.avg_pool1d(a, 4, ceil_mode=True), condition)
        c = self.middle(F.avg_pool1d(b, 4, ceil_mode=True), condition).transpose(1, 2)
        coarse_valid = F.adaptive_max_pool1d(node_valid.reshape(batch, 1, -1), c.shape[1])[:, 0].bool()
        position = time_embedding(torch.linspace(0, 1, c.shape[1], device=y.device), cfg.context_dim)
        c = c+position[None]
        c = c+self.global_cross(c, context, context, need_weights=False)[0]
        c = self.global_mix(c, src_key_padding_mask=~coarse_valid)*coarse_valid[..., None]
        d = self.up2(torch.cat((F.interpolate(c.transpose(1, 2),size=b.shape[-1],mode='linear',align_corners=True),b),1),condition)
        e = self.up1(torch.cat((F.interpolate(d,size=a.shape[-1],mode='linear',align_corners=True),a),1),condition)
        controls = self.control_head(e).transpose(1,2).reshape(batch,phases,knots,2)
        pooled = e.reshape(batch,-1,phases,knots).mean(-1).transpose(1,2)
        lengths = self.length_head(torch.cat((pooled,condition[:,None].expand(-1,phases,-1)),dim=-1))[...,0]
        result = torch.zeros_like(y)
        result[...,1:] = controls
        result[:,:,0,0] = lengths
        return result*mask
