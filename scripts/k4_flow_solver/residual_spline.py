"""Warm-start-centred stochastic transport in the existing physical spline space.

x0 = source(warm start) + noise_scale * eps, x1 = OCP GT (Lipman et al. 2023,
eqs. 21-23 with sigma_min = 0 and a source conditioned on the warm start).

source_kind:
  'tokens'   : the historical K4 contract - steering interpolated from the
               numeric path tokens, triangular speed seed.  Used by
               'k4_residual_v1' and available to 'warmstart_residual_v1'.
  'controls' : the warm start converted to retimed numerical controls
               (``warmstart.controls_from_hybrid_astar`` -> ``encode``) and fitted
               to spline coefficients.  Defined only for the warm start's own
               gear structure; any other structure falls back to 'tokens' and
               ``last_source_used_controls`` records which rows used which.
The source kind is fixed by the checkpoint representation and cannot be
changed at inference.
"""
import math
import torch
from .spline_model import SplineTrajectoryFlow
from .model import latent_mask

FLOW_SOURCES = ('k4_residual_v1', 'warmstart_residual_v1')
SOURCE_KINDS = ('tokens', 'controls')
WARM_START_SOURCES = ('hybrid_astar', 'known_route')
GEAR_MISMATCH_POLICIES = ('exclude', 'uniform_tokens', 'structure_only')


def validate_representation(representation):
    flow_source = representation.get('flow_source')
    if flow_source not in FLOW_SOURCES:
        raise ValueError(f'unsupported residual flow source: {flow_source}')
    kind = representation.get('source_kind', 'tokens')
    if flow_source == 'k4_residual_v1':
        if kind != 'tokens':
            raise ValueError('k4_residual_v1 only defines the token source')
        return kind
    missing = [k for k in ('source_kind', 'warm_start_source', 'residual_noise_scale',
                           'gear_mismatch_policy', 'warm_start_planner') if k not in representation]
    if missing:
        raise ValueError('warm-start representation missing: '+', '.join(missing))
    if kind not in SOURCE_KINDS:
        raise ValueError(f'unknown source_kind: {kind}')
    if representation['warm_start_source'] not in WARM_START_SOURCES:
        raise ValueError('unknown warm_start_source')
    if representation['gear_mismatch_policy'] not in GEAR_MISMATCH_POLICIES:
        raise ValueError('unknown gear_mismatch_policy')
    return kind


class ResidualSplineFlow(SplineTrajectoryFlow):
    def __init__(self, cfg, normalization, representation):
        super().__init__(cfg, normalization, representation)
        self.source_kind = validate_representation(representation)
        self.noise_scale = float(representation['residual_noise_scale'])
        if not 0 < self.noise_scale <= 1:
            raise ValueError('residual noise scale must be in (0,1]')
        self._source_tokens = None
        self._source_controls = None
        self.last_source_used_controls = None

    def encode(self, batch):
        # Runtime sampling expands a single scene into candidate batches.
        # Training sub-batches must supply their own base explicitly.
        self._source_tokens = (batch['k4_tokens'].detach(), batch['k4_tokens_mask'].detach())
        if self.source_kind == 'controls':
            self._source_controls = self._controls_inputs(batch)
        return super().encode(batch)

    def _controls_inputs(self, batch):
        if 'warm_latent' not in batch or 'warm_structure' not in batch:
            raise ValueError("source_kind='controls' requires warm_latent and warm_structure inputs")
        return batch['warm_latent'].detach(), batch['warm_structure'].detach().long().reshape(-1)

    @torch.no_grad()
    def source(self, h, batch=None):
        if self.source_kind == 'tokens':
            self.last_source_used_controls = torch.zeros(len(h), dtype=torch.bool, device=h.device)
            return self._token_source(h, batch)
        if batch is None:
            if self._source_controls is None:
                raise RuntimeError('encode the current warm-start scene before sampling')
            latent, structure = self._source_controls
        else:
            latent, structure = self._controls_inputs(batch)
        if len(latent) == 1:
            latent, structure = latent.expand(len(h), -1, -1, -1), structure.expand(len(h))
        if len(latent) != len(h):
            raise ValueError('source batch mismatch: pass the selected batch explicitly')
        use = structure.to(h.device) == h
        parts = []
        if use.any():
            dense = latent[use].to(h.device).float()
            dense_mask, _ = latent_mask(h[use], self.cfg)
            coefficients, mask = self.fit_targets(dense*dense_mask, h[use])
            parts.append((use, self.normalize(coefficients, mask)))
        if (~use).any():
            parts.append((~use, self._token_source(h[~use], self._subset(batch, ~use))))
        result = torch.zeros((len(h), *parts[0][1].shape[1:]), device=h.device, dtype=parts[0][1].dtype)
        for rows, value in parts:
            result[rows] = value
        if not torch.isfinite(result).all():
            raise ValueError('nonfinite warm-start source')
        self.last_source_used_controls = use
        return result.detach()

    @staticmethod
    def _subset(batch, rows):
        if batch is None:
            return None
        tokens, mask = batch['k4_tokens'], batch['k4_tokens_mask']
        if len(tokens) == 1:
            return dict(k4_tokens=tokens, k4_tokens_mask=mask)
        return dict(k4_tokens=tokens[rows.to(tokens.device)], k4_tokens_mask=mask[rows.to(mask.device)])

    @torch.no_grad()
    def _token_source(self, h, batch=None):
        if batch is None:
            if self._source_tokens is None:
                raise RuntimeError('encode the current K4 scene before sampling')
            tokens, token_mask = self._source_tokens
        else:
            tokens, token_mask = batch['k4_tokens'], batch['k4_tokens_mask']
        if len(tokens) == 1:
            tokens = tokens.expand(len(h), -1, -1)
            token_mask = token_mask.expand(len(h), -1)
        if len(tokens) != len(h):
            raise ValueError('source batch mismatch: pass the selected batch explicitly')
        tokens = tokens.float()
        cfg = self.cfg
        n = token_mask.sum(-1)
        if (n < 2).any() or not torch.isfinite(tokens[token_mask]).all():
            raise ValueError('invalid finite K4 token contract')
        arc = tokens[..., 6] * cfg.coordinate_scale
        last = (n - 1)[:, None]
        total = arc.gather(1, last)[:, 0].clamp(1e-3, cfg.max_total_length)
        phase_ids = (tokens[..., 7] * 12).round().long()
        original_count = phase_ids.gather(1, last)[:, 0] + 1
        count = h % cfg.max_phases + 1
        first_gear = torch.where(h < cfg.max_phases, 1., -1.)
        matching = (original_count == count) & (tokens[:, 0, 5] == first_gear)
        phase = torch.arange(cfg.max_phases, device=h.device)[None, :]
        valid = phase < count[:, None]
        # Retain K4 cusp distances for matching topology. For another topology,
        # partition its progress uniformly; the learned field can move lengths.
        starts = torch.where(
            (phase_ids[:, None, :] == phase[..., None]) & token_mask[:, None, :],
            arc[:, None, :], torch.full_like(arc[:, None, :], float('inf'))).amin(-1)
        starts = torch.where(torch.isfinite(starts), starts, total[:, None])
        ends = torch.cat((starts[:, 1:], total[:, None]), -1)
        ends = torch.where(phase == count[:, None]-1, total[:, None], ends)
        lo = torch.where(matching[:, None], starts, total[:, None]*phase/count[:, None])
        hi = torch.where(matching[:, None], ends, total[:, None]*(phase+1)/count[:, None])
        lengths = (hi-lo).clamp_min(1e-3)
        u = .5*(1-torch.cos(torch.linspace(0, math.pi, cfg.intervals+1, device=h.device)))
        queries = (lo[..., None]+lengths[..., None]*u).flatten(1).contiguous()
        searchable = arc.masked_fill(~token_mask, float('inf')).contiguous()
        right = torch.searchsorted(searchable, queries, right=True)
        right = torch.minimum(right.clamp_min(1), last)
        left = right-1
        a, b = arc.gather(1, left), arc.gather(1, right)
        weight = ((queries-a)/(b-a).clamp_min(1e-8)).clamp(0, 1)
        steering = tokens[..., 4].clamp(-1+1e-5, 1-1e-5)
        d0, d1 = steering.gather(1, left), steering.gather(1, right)
        # Do not interpolate steering across a gear cusp.
        same_phase = phase_ids.gather(1, left) == phase_ids.gather(1, right)
        delta = torch.where(same_phase, d0+(d1-d0)*weight, torch.where(weight < .5, d0, d1))
        old_gear = tokens[..., 5].gather(1, torch.where(weight < .5, left, right)).sign()
        gears = first_gear[:, None]*torch.where(phase % 2 == 0, 1., -1.)
        # On resegmentation preserve the signed curvature as an initial guess.
        delta = delta.reshape(len(h), cfg.max_phases, -1)
        delta = delta*old_gear.reshape_as(delta)*gears[..., None]
        speed_sq = torch.minimum(2*cfg.max_accel*lengths[..., None]*u,
                                 2*cfg.max_accel*lengths[..., None]*(1-u))
        speed_sq = speed_sq.clamp(1e-8, cfg.max_speed**2*(1-1e-5))
        dense = torch.zeros((*delta.shape, 3), device=h.device)
        dense[:, :, 0, 0] = lengths.log()
        dense[..., 1] = torch.atanh(delta.clamp(-1+1e-5, 1-1e-5))
        dense[..., 2] = torch.logit(speed_sq/cfg.max_speed**2)
        dense_mask, _ = latent_mask(h, cfg)
        coefficients, mask = self.fit_targets(dense*dense_mask, h)
        return self.normalize(coefficients, mask).detach()

    def bridge(self, target, h, base, *, generator=None):
        mask, _ = latent_mask(h, self.sampling_config)
        noise = torch.randn(target.shape, device=target.device, generator=generator)
        initial = (base+self.noise_scale*noise)*mask
        t = torch.rand(len(h), device=h.device, generator=generator)
        y = torch.lerp(initial, target, t[:, None, None, None])
        return y, t, (target-initial)*mask

    def rollout(self, context, h, steps=16, *, noise=None, generator=None,
                regression=False, base=None):
        if steps < 1 or regression:
            raise ValueError('residual stochastic Flow requires positive steps and flow mode')
        mask, _ = latent_mask(h, self.sampling_config)
        base = self.source(h) if base is None else base
        if noise is None:
            noise = torch.randn(mask.shape, device=h.device, generator=generator)
        y = (base+self.noise_scale*noise)*mask
        for k in range(steps):
            t = torch.full((len(h),), k/steps, device=h.device, dtype=context.dtype)
            y = (y+self(y, t, context, h)/steps)*mask
        return y
