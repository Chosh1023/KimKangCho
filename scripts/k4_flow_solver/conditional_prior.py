"""K4-centered smooth initial distributions for the unchanged WholeTrajectoryFlow.

No teacher or feasibility admission gate. Full controls remain the action; the
conditional initial distribution is shared by rollout and CFM training.
"""
from dataclasses import replace
import numpy as np
import torch
from .whole_model import WholeTrajectoryFlow
from .model import latent_mask
from .numerical import Controls, from_k4, encode, gears_for, structure_id, grid


PRIOR_CONFIG = dict(kind='k4_smooth_v2', nodes=17, white_fraction=.001,
    local_scale=[.03, .015, .15], alternative_scale=[.12, .05, .30])


class K4PriorFlow(WholeTrajectoryFlow):
    def __init__(self, cfg, normalization, exploration=None):
        super().__init__(cfg, normalization)
        self.exploration = dict(PRIOR_CONFIG if exploration is None else exploration)
        if self.exploration != PRIOR_CONFIG:
            raise ValueError('unsupported conditional prior contract')
        self.prior = None
        self.record = False
        self.role = 'whole'
        self.recordings = []

    def set_k4_prior(self, path):
        """Convert supported finite K4 controls without testing their feasibility.

        If there are too many phases, resample them into the model's phase
        capacity. If conversion fails, retain generic smooth exploration and
        still encode the original raw input in the ordinary inference path.
        """
        cfg = self.cfg
        normalized_mean = self.latent_mean.detach().cpu().numpy()
        normalized_std = self.latent_std.detach().cpu().numpy()
        centers = np.broadcast_to(normalized_mean, (2*cfg.max_phases, cfg.max_phases, cfg.intervals+1, 3)).copy()
        scales = np.tile(np.array(self.exploration['alternative_scale']), (2*cfg.max_phases, 1))
        reference, source = 0, 'generic_smooth_fallback'
        reason = None
        try:
            # A phase-count exception must not become an input-admission gate.
            relaxed = replace(cfg, max_phases=max(cfg.max_phases, len(path)))
            base = from_k4(path, relaxed)
            reference = int(base.gears[0]<0)*cfg.max_phases+min(len(base.gears), cfg.max_phases)-1
            source = 'K4_controls_without_feasibility_filter'
            boundaries = np.r_[0., np.cumsum(base.lengths)]
            u = grid(cfg.intervals)
            for h in range(2*cfg.max_phases):
                gears = gears_for(h, cfg)
                if len(base.gears) <= cfg.max_phases and h == structure_id(base.gears, cfg):
                    proposal = base
                else:
                    # Resample unsigned progress into the requested phase count.
                    # Flip steering when the desired gear differs, retaining the
                    # source's yaw-vs-progress tendency; XY is free to change.
                    lengths = np.full(len(gears), base.lengths.sum()/len(gears))
                    delta, speed = [], []
                    for p, gear in enumerate(gears):
                        arc = (p+u)*lengths[p]
                        index = np.clip(np.searchsorted(boundaries, arc, side='right')-1, 0, len(base.gears)-1)
                        d, b = np.zeros_like(u), np.zeros_like(u)
                        for old in np.unique(index):
                            selected = index == old
                            progress = (arc[selected]-boundaries[old])/base.lengths[old]
                            d[selected] = np.interp(progress, u, base.delta[old])*base.gears[old]/gear
                            b[selected] = np.interp(progress, u, base.b[old])
                        b = np.maximum(b, 1e-8); b[[0, -1]] = 0.
                        delta.append(d); speed.append(b)
                    proposal = Controls(lengths, np.array(delta), np.array(speed), gears)
                centers[h] = encode(proposal, cfg)[0]
        except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
            reason = str(exc)
            # Partial conversion must not leave inconsistent centers behind.
            centers[:] = normalized_mean
            reference, source = 0, 'generic_smooth_fallback'
        scales[reference] = self.exploration['local_scale']
        centers = (centers-normalized_mean)/normalized_std
        device = self.latent_mean.device
        self.prior = dict(center=torch.as_tensor(centers, device=device, dtype=torch.float32),
            scale=torch.as_tensor(scales/normalized_std, device=device, dtype=torch.float32),
            reference_structure=reference, source=source, reason=reason)
        return self.prior

    def initial_samples(self, white, h):
        if self.prior is None:
            raise RuntimeError('set_k4_prior is required before conditional sampling')
        if white.shape != (len(h), self.cfg.max_phases, self.cfg.intervals+1, 3):
            raise ValueError('conditional initial sample shape mismatch')
        n = min(self.exploration['nodes'], self.cfg.intervals+1)
        x = self.u*(n-1)
        lo = x.floor().long().clamp(max=n-2)
        weight = (x-lo)[None, None, :, None]
        coarse = white[:, :, :n, 1:]
        smooth = coarse[:, :, lo]*(1-weight)+coarse[:, :, lo+1]*weight
        mixed = white.clone()
        # A small full-rank component avoids a strictly singular control prior.
        fraction = self.exploration['white_fraction']
        mixed[..., 1:] = (1-fraction)*smooth+fraction*white[..., 1:]
        mask, _ = latent_mask(h, self.cfg, white.dtype)
        return (self.prior['center'][h]+mixed*self.prior['scale'][h, None, None])*mask

    def choose_structures(self, logits, top_k):
        reference = self.prior['reference_structure']
        other = logits[0].clone()
        other[reference] = -float('inf')
        selected = [reference]+other.topk(min(top_k-1, len(other)-1)).indices.tolist()
        return torch.tensor(selected, device=logits.device)

    def rollout(self, context, h, steps=4, *, noise=None, generator=None, regression=False):
        if regression:
            raise ValueError('conditional prior requires a stochastic Flow policy')
        if noise is None:
            mask, _ = latent_mask(h, self.cfg)
            noise = torch.randn(mask.shape, device=h.device, generator=generator)
        initial = self.initial_samples(noise, h)
        result = super().rollout(context, h, steps, noise=initial)
        if self.record:
            self.recordings.append((self.role, result.detach().cpu(), h.detach().cpu()))
        return result


def alternative_probabilities(logits, reference, exploration=.1):
    allowed = torch.ones_like(logits, dtype=torch.bool)
    allowed[:, reference] = False
    masked = logits.masked_fill(~allowed, -float('inf'))
    uniform = allowed.to(logits.dtype)/(logits.shape[-1]-1)
    return (1-exploration)*masked.softmax(-1)+exploration*uniform
