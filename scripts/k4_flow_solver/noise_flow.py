"""Noise -> OCP GT Flow conditioned only on the scene and the start/goal task.

No warm-start or K4 path enters the model.  Standard conditional flow
matching: x0 ~ N(0, I) in the normalized output space, x1 = original OCP GT
(fit to the output representation), x_t = (1 - t) x0 + t x1, target velocity
x1 - x0.  The final trajectory is always: model controls -> shared
projection (bounded shooting + retime caps) -> kinematic bicycle integration.
The model never outputs states.

output_repr
  'bspline64' : cubic B-spline, 64 coefficients per gear phase for steering and
                speed plus one length (the epoch-10 representation)
  'grid{N}'   : N = 64, 128, 513; steering and v^2 values at N cosine knots
                u_k = 0.5(1 - cos(pi k/(N-1))) per phase plus one length.
                Fit: linear sampling of the 513-grid GT (endpoints kept).
                Restore: tanh/sigmoid per knot (limits), then linear
                interpolation in u to the 513 grid (a convex combination, so
                the limits hold); v^2 endpoints are 0.  grid513 is the 513
                grid itself.  Normalization: per-representation train stats.
"""
from __future__ import annotations
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import math
import numpy as np
import torch
from torch import nn

from .config import SolverConfig
from .model import NumericEncoder, latent_mask
from .numerical import grid
from .whole_model import WholeTrajectoryFlow
from .spline_model import SplineTrajectoryFlow, decoded_model_samples
from .output_projection import project_output
from .smooth_controls import torch_retime

SCHEMA = 'noise-scene-flow-v1'
OUTPUT_REPRS = ('bspline64', 'grid64', 'grid128', 'grid513')
GRID_KNOTS = {'grid64': 64, 'grid128': 128, 'grid513': 513}


class SceneTaskEncoder(NumericEncoder):
    """NumericEncoder without the path branch: keys = scene tokens + task token.

    The 24 queries, the 192-d context and the scene/task layers are the
    NumericEncoder ones (loadable from epoch 10); the path convolutions are not
    created.  ``kind`` keeps its (2, d) shape so the checkpoint loads strictly
    for it; only row 0 (scene) is used.
    """
    def __init__(self, dimension):
        super().__init__(dimension)
        del self.path

    def forward(self, batch):
        if 'bev_image' in batch or 'previous_path_tokens' in batch:
            raise ValueError('SceneTaskEncoder takes numeric scene and task inputs only')
        scene = self.scene(batch['scene_tokens'])+self.kind[0]
        task = self.task(batch['task'])
        keys = torch.cat((scene, task[:, None]), dim=1)
        valid = torch.cat((batch['scene_tokens_mask'],
                           torch.ones((len(task), 1), device=task.device, dtype=torch.bool)), dim=1)
        query = self.queries[None].expand(len(task), -1, -1)+task[:, None]
        out = query+self.cross(query, keys, keys, key_padding_mask=~valid, need_weights=False)[0]
        return self.norm(self.mix(out))


class SceneTaskFlowCore(WholeTrajectoryFlow):
    """WholeTrajectoryFlow vector field with the scene/task-only encoder."""
    def __init__(self, cfg, normalization=None):
        super().__init__(cfg, normalization)
        self.encoder = SceneTaskEncoder(cfg.context_dim)


def spline_representation(representation, cfg):
    return dict(schema='physical-bspline-v1', coefficients=int(representation['knots']), degree=3,
                output_intervals=cfg.intervals, projection='bounded_shooting_then_exact_forward_backward_caps',
                projection_iterations=int(representation.get('projection_iterations', 2)))


def noise_representation(output_repr='bspline64', projection_iterations=2):
    if output_repr not in OUTPUT_REPRS:
        raise ValueError(f'unsupported output_repr: {output_repr}')
    if output_repr in GRID_KNOTS:
        return dict(schema=SCHEMA, output_repr=output_repr, knots=GRID_KNOTS[output_repr], grid='cosine',
                    fit='linear sampling of the 513-grid steering and v^2 at the knots (endpoints kept)',
                    restore='tanh/sigmoid per knot, then linear interpolation in u to the 513 grid; v^2 endpoints 0',
                    normalization='per-representation train GT statistics', condition='scene+task', source='gaussian',
                    projection='bounded_shooting_then_exact_forward_backward_caps',
                    projection_iterations=int(projection_iterations), target='original OCP GT (dataset latent)')
    return dict(schema=SCHEMA, output_repr=output_repr, knots=64, condition='scene+task', source='gaussian',
                projection='bounded_shooting_then_exact_forward_backward_caps',
                projection_iterations=int(projection_iterations),
                target='original OCP GT (dataset latent)')


class NoiseSplineFlow(SplineTrajectoryFlow):
    """bspline64 output; the spline fit/decoder of SplineTrajectoryFlow is reused as is."""
    def __init__(self, cfg, normalization, representation):
        if representation.get('schema') != SCHEMA or representation.get('output_repr') != 'bspline64':
            raise ValueError('NoiseSplineFlow requires a noise-scene-flow-v1 bspline64 representation')
        super().__init__(cfg, normalization, spline_representation(representation, cfg))
        self.core = SceneTaskFlowCore(self.sampling_config, normalization)
        self.representation = dict(representation)


def interpolation_matrix(source, target):
    """Linear interpolation weights W (len(target), len(source)): values(target) = W @ values(source)."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    index = np.clip(np.searchsorted(source, target, side='right')-1, 0, len(source)-2)
    weight = np.clip((target-source[index])/(source[index+1]-source[index]), 0., 1.)
    matrix = np.zeros((len(target), len(source)))
    matrix[np.arange(len(target)), index] = 1-weight
    matrix[np.arange(len(target)), index+1] += weight
    return matrix


class NoiseGridFlow(nn.Module):
    """grid{N}: the network outputs steering/v^2 values at N cosine knots per phase.

    Same CFM core (SceneTaskFlowCore = WholeTrajectoryFlow vector field with N
    knots), same projection and validation as bspline64; only the output
    representation and its normalization differ.
    """
    def __init__(self, cfg, normalization, representation):
        super().__init__()
        knots = GRID_KNOTS.get(representation.get('output_repr'))
        if representation.get('schema') != SCHEMA or knots is None or representation.get('knots') != knots:
            raise ValueError('NoiseGridFlow requires a noise-scene-flow-v1 grid representation')
        self.cfg = cfg
        self.representation = dict(representation)
        self.sampling_config = replace(cfg, intervals=knots-1)
        self.core = SceneTaskFlowCore(self.sampling_config, normalization)
        dense, coarse = grid(cfg.intervals), grid(knots-1)
        self.identity = knots == cfg.intervals+1
        self.register_buffer('sample_matrix', torch.tensor(interpolation_matrix(dense, coarse), dtype=torch.float64))
        self.register_buffer('restore_matrix', torch.tensor(interpolation_matrix(coarse, dense), dtype=torch.float64))

    @property
    def encoder(self): return self.core.encoder

    @property
    def structure(self): return self.core.structure

    def encode(self, batch): return self.core.encode(batch)

    def forward(self, y, t, context, h): return self.core(y, t, context, h)

    def rollout(self, context, h, steps=16, **kwargs): return self.core.rollout(context, h, steps, **kwargs)

    def normalize(self, latent, mask): return self.core.normalize(latent, mask)

    def denormalize(self, y, mask): return self.core.denormalize(y, mask)

    def fit_targets(self, dense, h):
        """513-grid GT latent -> knot latent (label only; no goal or teacher cost enters)."""
        cfg = self.cfg
        a = dense.double()
        mask, _ = latent_mask(h, self.sampling_config, a.dtype)
        if self.identity:
            return (a*mask).float(), mask.float()
        delta = cfg.max_steer*torch.tanh(a[..., 1])@self.sample_matrix.T
        speed = cfg.max_speed**2*torch.sigmoid(a[..., 2])
        speed = speed.clone()
        speed[..., [0, -1]] = 0
        speed = speed@self.sample_matrix.T
        delta = delta.clamp(-cfg.max_steer*(1-1e-7), cfg.max_steer*(1-1e-7))
        speed = speed.clamp(cfg.max_speed**2*1e-9, cfg.max_speed**2*(1-1e-7))
        length = torch.zeros_like(delta)
        length[..., 0] = a[:, :, 0, 0]
        result = torch.stack((length, torch.atanh(delta/cfg.max_steer), torch.logit(speed/cfg.max_speed**2)), dim=-1)
        return (result*mask).float(), mask.float()

    def dense_latent(self, values, h, *, project=True):
        """Unnormalized knot latent -> 513-grid latent. Limits are applied per knot before interpolation."""
        cfg = self.cfg
        a = values.double()
        _, valid = latent_mask(h, self.sampling_config, a.dtype)
        lengths = torch.exp(a[:, :, 0, 0].clamp(-12, math.log(cfg.max_total_length)))*valid
        delta = (cfg.max_steer*torch.tanh(a[..., 1]))@self.restore_matrix.T
        speed = cfg.max_speed**2*torch.sigmoid(a[..., 2])
        interior = torch.ones(speed.shape[-1], device=speed.device, dtype=speed.dtype)
        interior[[0, -1]] = 0
        speed = (speed*interior)@self.restore_matrix.T
        if project:
            speed = torch_retime(lengths.clamp_min(1e-6), delta, speed, cfg)
        dense_mask, _ = latent_mask(h, cfg, a.dtype)
        length = torch.zeros_like(delta)
        length[..., 0] = a[:, :, 0, 0]
        result = torch.stack((length, torch.atanh((delta/cfg.max_steer).clamp(-1+1e-12, 1-1e-12)),
                              torch.logit((speed/cfg.max_speed**2).clamp(1e-20, 1-1e-12))), dim=-1)
        return result*dense_mask


def decode_samples(model, y, h, cfg, goal=None):
    """Normalized model output -> 513-grid latent (goal projection when ``goal`` is given).

    bspline64 goes through the unchanged ``decoded_model_samples``; grid models
    restore by interpolation and then use the same ``project_output``.
    """
    if not isinstance(model, NoiseGridFlow):
        return decoded_model_samples(model, y, h, cfg, goal)
    mask, _ = latent_mask(h, model.sampling_config)
    latent = model.dense_latent(model.denormalize(y, mask), h, project=goal is None)
    if goal is not None:
        latent = project_output(latent, h, goal, cfg, iterations=model.representation.get('projection_iterations', 2))
    return latent


def make_noise_model(cfg, normalization, representation):
    if representation.get('output_repr') == 'bspline64':
        return NoiseSplineFlow(cfg, normalization, representation)
    if representation.get('output_repr') in GRID_KNOTS:
        return NoiseGridFlow(cfg, normalization, representation)
    raise ValueError(f"unsupported output_repr: {representation.get('output_repr')}")


def load_init_features(model, checkpoint, *, skip=('core.latent_mean', 'core.latent_std')):
    """Copy shape-compatible EMA tensors from a spline (``core.*``) or whole (top-level) checkpoint.

    ``skip`` keeps the model's own normalization buffers (grid statistics
    differ from the dense statistics of the source).  Returns the loaded,
    skipped and missing key lists for the log.
    """
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    source = data['ema']
    if not any(k.startswith('core.') for k in source):
        source = {'core.'+k: v for k, v in source.items()}
    own = model.state_dict()
    compatible = {k: v for k, v in source.items() if k in own and own[k].shape == v.shape and k not in skip}
    result = model.load_state_dict(compatible, strict=False)
    return dict(checkpoint=str(checkpoint), architecture=data.get('architecture'), epoch=data.get('epoch'),
                loaded=sorted(compatible), skipped_from_checkpoint=sorted(k for k in source if k not in compatible),
                missing_in_checkpoint=sorted(result.missing_keys), dataset_sha256=data['dataset_sha256'],
                source_normalization=data['normalization'], source_representation=data.get('representation'))


def load_epoch10_features(model, checkpoint):
    """Copy every shape-compatible epoch-10 EMA tensor; report loaded/skipped keys."""
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    source = data['ema']
    own = model.state_dict()
    compatible = {k: v for k, v in source.items() if k in own and own[k].shape == v.shape}
    result = model.load_state_dict(compatible, strict=False)
    skipped = sorted(k for k in source if k not in compatible)
    return dict(loaded=sorted(compatible), skipped_from_checkpoint=skipped,
                missing_in_checkpoint=sorted(result.missing_keys), dataset_sha256=data['dataset_sha256'],
                normalization=data['normalization'], representation=data['representation'])


def cfm_pair(target, mask, generator=None):
    """Standard CFM training pair: x0 ~ N(0, I), x_t = (1-t) x0 + t x1, v = x1 - x0."""
    x0 = torch.randn(target.shape, device=target.device, generator=generator)*mask
    t = torch.rand(len(target), device=target.device, generator=generator)
    tt = t[:, None, None, None]
    return (1-tt)*x0+tt*target, t, (target-x0)*mask, x0


def top_k_schedule(structures, candidates, seed):
    """Distribute ``candidates`` over ranked structures round-robin (rank 0 gets the extra ones).

    Each candidate has its own reproducible noise seed keyed by (seed, h, sample).
    """
    if candidates < 1 or not structures:
        raise ValueError('empty candidate schedule')
    if len(set(structures)) != len(structures):
        raise ValueError('structure candidates must be unique')
    counts = {h: 0 for h in structures}
    result = []
    for i in range(candidates):
        rank = i % len(structures)
        h = structures[rank]
        key = f'noise-flow-v1/{int(seed)}/{int(h)}/{counts[h]}'.encode('ascii')
        result.append(dict(structure=int(h), structure_rank=rank, sample_index=counts[h],
                           noise_seed=int.from_bytes(hashlib.sha256(key).digest()[:8], 'little') % (2**63-1)))
        counts[h] += 1
    return result


class FilteredSolverDataset(torch.utils.data.Dataset):
    """In-memory rows of one split, excluding given uids (keeps held-out scenes out of selection)."""
    def __init__(self, root, split, *, exclude=(), limit=None, include_sdf=False):
        root = Path(root)
        manifest = json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        excluded = set(exclude)
        self.samples = []
        for shard in manifest['shards']:
            if shard['split'] != split:
                continue
            with np.load(root/shard['file'], allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files if include_sdf or k != 'sdf'}
            for i, uid in enumerate(arrays['uids']):
                if str(uid) in excluded:
                    continue
                sample = {}
                for key, array in arrays.items():
                    if key in ('scene_tokens_mask', 'k4_tokens_mask', 'k4_tokens'):
                        continue
                    value = array[i]
                    if key == 'scene_tokens':
                        value = value[arrays['scene_tokens_mask'][i]]
                    elif key in ('latent', 'goal', 'target_J'):
                        value = np.asarray(value, dtype=np.float32)
                    sample[key] = str(value) if key == 'uids' else value
                self.samples.append(sample)
                if limit is not None and len(self.samples) >= limit:
                    return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


def load_noise_flow(checkpoint, device='cpu'):
    data = torch.load(checkpoint, map_location=device, weights_only=False)
    representation = data.get('representation') or {}
    if representation.get('schema') != SCHEMA:
        raise ValueError('not a noise-scene-flow-v1 checkpoint')
    cfg = SolverConfig(**data['config'])
    model = make_noise_model(cfg, data['normalization'], representation).to(device)
    model.load_state_dict(data['ema'])
    model.eval().requires_grad_(False)
    return model, data
