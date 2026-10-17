"""Numeric scene encoder, categorical maneuver head and shared phase 1D U-Net."""
from __future__ import annotations
import math
import torch
from torch import nn
from torch.nn import functional as F
from .config import DEFAULT


def latent_mask(h, cfg=DEFAULT, dtype=torch.float32):
    phase = torch.arange(cfg.max_phases, device=h.device)[None] < (h % cfg.max_phases+1)[:, None]
    mask = torch.zeros((len(h), cfg.max_phases, cfg.intervals+1, 3), device=h.device, dtype=dtype)
    mask[:, :, 0, 0] = phase.to(dtype)
    mask[:, :, :, 1] = phase[:, :, None]
    mask[:, :, 1:-1, 2] = phase[:, :, None]
    return mask, phase


def time_embedding(t, dimension):
    frequency = torch.exp(torch.linspace(0, math.log(1000), dimension//2, device=t.device))
    angle = t[:, None]*frequency[None]*2*math.pi
    return torch.cat((angle.sin(), angle.cos()), dim=-1)


class NumericEncoder(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.scene = nn.Sequential(nn.Linear(10, dimension), nn.SiLU(), nn.Linear(dimension, dimension))
        self.path = nn.Sequential(nn.Conv1d(8, dimension, 5, padding=2), nn.SiLU(),
                                  nn.Conv1d(dimension, dimension, 5, padding=2))
        self.task = nn.Sequential(nn.Linear(5, dimension), nn.SiLU(), nn.Linear(dimension, dimension))
        self.kind = nn.Parameter(torch.randn(2, dimension)*0.02)
        self.queries = nn.Parameter(torch.randn(24, dimension)*0.02)
        self.cross = nn.MultiheadAttention(dimension, 6, batch_first=True)
        block = nn.TransformerEncoderLayer(dimension, 6, dimension*3, dropout=0.0,
                                            batch_first=True, activation='gelu', norm_first=True)
        self.mix = nn.TransformerEncoder(block, 2, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dimension)

    def enable_image(self, image_config, coordinate_scale=20.):
        from .image_encoder import SpatialImageEncoder
        if hasattr(self, 'image_encoder'):
            raise ValueError('image encoder already installed')
        self.image_encoder = SpatialImageEncoder(self.kind.shape[-1], image_config, coordinate_scale).to(self.kind.device)
        self.image_kind = nn.Parameter(torch.randn_like(self.kind[0])*.02)

    def enable_previous_path(self):
        """Opt-in extension, installed after loading a legacy checkpoint.

        Reuse the trained path convolutions for the eight shared features.
        Speed/switch columns start at zero and require history-aware training.
        No-history execution and legacy state dictionaries stay unchanged.
        """
        if hasattr(self,'previous_path'): return
        dimension = self.kind.shape[-1]
        adapter = nn.Sequential(nn.Conv1d(10,dimension,5,padding=2),nn.SiLU(),
                                nn.Conv1d(dimension,dimension,5,padding=2)).to(self.kind.device)
        with torch.no_grad():
            adapter[0].weight.zero_()
            adapter[0].weight[:,:8].copy_(self.path[0].weight)
            adapter[0].bias.copy_(self.path[0].bias)
            adapter[2].load_state_dict(self.path[2].state_dict())
        self.previous_path = adapter
        self.previous_kind = nn.Parameter(self.kind[1].detach().clone())

    def forward(self, batch):
        scene = self.scene(batch['scene_tokens']) + self.kind[0]
        path_mask = batch['k4_tokens_mask'][:, None]
        path = self.path[0]((batch['k4_tokens']*batch['k4_tokens_mask'][..., None]).transpose(1, 2))
        path = self.path[2](self.path[1](path)*path_mask).transpose(1, 2)+self.kind[1]
        task = self.task(batch['task'])
        keys = torch.cat((scene, path, task[:, None]), dim=1)
        valid = torch.cat((batch['scene_tokens_mask'], batch['k4_tokens_mask'],
                           torch.ones((len(task), 1), device=task.device, dtype=torch.bool)), dim=1)
        if hasattr(self, 'image_encoder'):
            if 'bev_image' not in batch or 'bev_start' not in batch:
                raise ValueError('image-conditioned model requires aligned BEV input')
            image, image_valid = self.image_encoder(batch['bev_image'], batch['bev_start'])
            keys = torch.cat((keys, image+self.image_kind), dim=1)
            valid = torch.cat((valid, image_valid), dim=1)
        if 'previous_path_tokens' in batch and bool(batch['previous_path_mask'].any()):
            if not hasattr(self,'previous_path'):
                raise ValueError('enable_previous_path must be called before passing history')
            previous_mask = batch['previous_path_mask']
            raw = torch.where(previous_mask[...,None],batch['previous_path_tokens'],0.)
            previous = self.previous_path[0](raw.transpose(1,2))
            previous = self.previous_path[2](self.previous_path[1](previous)*previous_mask[:,None]).transpose(1,2)
            previous = previous+self.previous_kind
            keys = torch.cat((keys,previous),dim=1)
            valid = torch.cat((valid,previous_mask),dim=1)
        query = self.queries[None].expand(len(task), -1, -1)+task[:, None]
        out = query+self.cross(query, keys, keys, key_padding_mask=~valid, need_weights=False)[0]
        return self.norm(self.mix(out))


class TemporalBlock(nn.Module):
    def __init__(self, inputs, outputs, context):
        super().__init__()
        self.conv1 = nn.Conv1d(inputs, outputs, 5, padding=2)
        self.norm1 = nn.GroupNorm(8, outputs)
        self.condition = nn.Linear(context, outputs*2)
        self.conv2 = nn.Conv1d(outputs, outputs, 5, padding=2)
        self.norm2 = nn.GroupNorm(8, outputs)
        self.skip = nn.Conv1d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, x, context):
        scale, shift = self.condition(context).chunk(2, dim=-1)
        value = self.norm1(self.conv1(x))*(1+scale[..., None])+shift[..., None]
        value = self.norm2(self.conv2(F.silu(value)))
        return F.silu(value+self.skip(x))


class FlowSolver(nn.Module):
    def __init__(self, cfg=DEFAULT, normalization=None):
        super().__init__()
        self.cfg = cfg
        d, w = cfg.context_dim, cfg.temporal_channels
        self.encoder = NumericEncoder(d)
        self.structure = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 2*cfg.max_phases))
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.h_embedding = nn.Embedding(2*cfg.max_phases, d)
        self.phase_embedding = nn.Embedding(cfg.max_phases, d)
        self.summary = nn.Linear(5, d)
        self.phase_cross = nn.MultiheadAttention(d, 6, batch_first=True)
        layer = nn.TransformerEncoderLayer(d, 6, d*2, dropout=0, batch_first=True, norm_first=True)
        self.phase_mix = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.down1 = TemporalBlock(6, w, d)
        self.down2 = TemporalBlock(w, w*2, d)
        self.middle = TemporalBlock(w*2, w*3, d)
        self.up2 = TemporalBlock(w*5, w*2, d)
        self.up1 = TemporalBlock(w*3, w, d)
        self.control_head = nn.Conv1d(w, 2, 1)
        self.length_head = nn.Sequential(nn.Linear(d+w*3, d), nn.SiLU(), nn.Linear(d, 1))
        nn.init.zeros_(self.control_head.weight)
        nn.init.zeros_(self.control_head.bias)
        nn.init.zeros_(self.length_head[-1].weight)
        nn.init.zeros_(self.length_head[-1].bias)
        norm = normalization or {'mean':[0, 0, 0], 'std':[1, 1, 1]}
        self.register_buffer('latent_mean', torch.tensor(norm['mean'], dtype=torch.float32))
        self.register_buffer('latent_std', torch.tensor(norm['std'], dtype=torch.float32))
        self.register_buffer('u', 0.5*(1-torch.cos(torch.linspace(0, math.pi, cfg.intervals+1))))

    def encode(self, batch):
        context = self.encoder(batch)
        return context, self.structure(context.mean(dim=1))

    def normalize(self, latent, mask):
        return (latent-self.latent_mean)/self.latent_std*mask

    def denormalize(self, y, mask):
        return (y*self.latent_std+self.latent_mean)*mask

    def forward(self, y, t, context, h):
        cfg = self.cfg
        batch, phases, knots, _ = y.shape
        mask, valid = latent_mask(h, cfg, y.dtype)
        y = y*mask
        phase_indices = torch.arange(phases, device=y.device)
        gear = torch.where(h < cfg.max_phases, 1., -1.)[:, None]*torch.where(phase_indices % 2 == 0, 1., -1.)[None]
        summary = torch.cat((y[:, :, 0, :1], y[:, :, :, 1:].mean(2), y[:, :, :, 1:].amax(2)), dim=-1)
        cond = self.summary(summary)+self.phase_embedding(phase_indices)[None]
        cond = cond+self.h_embedding(h)[:, None]+self.time(time_embedding(t, cfg.context_dim))[:, None]
        cond = cond+self.phase_cross(cond, context, context, need_weights=False)[0]
        cond = self.phase_mix(cond, src_key_padding_mask=~valid)*valid[..., None]
        length = y[:, :, 0, 0, None].expand(-1, -1, knots)
        features = torch.stack((y[..., 1], y[..., 2], length,
            self.u[None, None].expand(batch, phases, -1), gear[..., None].expand(-1, -1, knots),
            phase_indices[None, :, None].expand(batch, -1, knots)/cfg.max_phases), dim=2)
        features = features.reshape(batch*phases, 6, knots)
        conditioning = cond.reshape(batch*phases, -1)
        a = self.down1(features, conditioning)
        b = self.down2(F.avg_pool1d(a, 2, ceil_mode=True), conditioning)
        c = self.middle(F.avg_pool1d(b, 2, ceil_mode=True), conditioning)
        d = self.up2(torch.cat((F.interpolate(c, size=b.shape[-1], mode='linear', align_corners=True), b), 1), conditioning)
        e = self.up1(torch.cat((F.interpolate(d, size=a.shape[-1], mode='linear', align_corners=True), a), 1), conditioning)
        control = self.control_head(e).transpose(1, 2).reshape(batch, phases, knots, 2)
        length_velocity = self.length_head(torch.cat((conditioning, c.mean(-1)), -1)).reshape(batch, phases)
        result = torch.zeros_like(y)
        result[..., 1:] = control
        result[:, :, 0, 0] = length_velocity
        return result*mask

    def rollout(self, context, h, steps=4, *, noise=None, generator=None, regression=False):
        if steps < 1:
            raise ValueError('steps must be positive')
        mask, _ = latent_mask(h, self.cfg, torch.float32)
        if regression:
            zeros = torch.zeros_like(mask)
            return self(zeros, torch.zeros(len(h), device=h.device), context, h)
        y = torch.randn(mask.shape, device=h.device, dtype=mask.dtype, generator=generator) if noise is None else noise
        y = y*mask
        for k in range(steps):
            t = torch.full((len(h),), k/steps, device=h.device, dtype=context.dtype)
            y = (y+self(y, t, context, h)/steps)*mask
        return y


def channel_mse(prediction, target, mask):
    error = ((prediction-target)**2*mask).sum(dim=(1, 2))
    count = mask.sum(dim=(1, 2)).clamp_min(1)
    return (error/count).mean()


def make_model(cfg=DEFAULT, normalization=None, architecture='phase', image_config=None, representation=None):
    if architecture == 'phase':
        model = FlowSolver(cfg, normalization)
    elif architecture == 'whole':
        from .whole_model import WholeTrajectoryFlow
        model = WholeTrajectoryFlow(cfg, normalization)
    elif architecture == 'spline_whole':
        if representation and representation.get('flow_source') in ('k4_residual_v1', 'warmstart_residual_v1'):
            from .residual_spline import ResidualSplineFlow
            model = ResidualSplineFlow(cfg,normalization,representation)
        else:
            from .spline_model import SplineTrajectoryFlow
            model = SplineTrajectoryFlow(cfg,normalization,representation)
    else:
        raise ValueError(f'unknown solver architecture: {architecture}')
    if image_config is not None:
        from .image_data import validate_image_config
        image_config = validate_image_config(image_config)
        model.encoder.enable_image(image_config, cfg.coordinate_scale)
        model.image_config = dict(image_config)
    return model
