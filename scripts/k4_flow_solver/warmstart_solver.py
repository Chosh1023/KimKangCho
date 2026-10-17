"""Experiment 1 solver: Hybrid A* warm start -> warm-start residual Flow.

Candidates are generated from x0 = source(warm start) + sigma*eps by Euler
integration of the learned field, decoded by the shared goal projection,
validated numerically and the minimum-J verified candidate is returned.
No donor split, segment DP, refinement or K4 fallback is used; when no
candidate verifies the report says ``stop_and_replan`` and no trajectory is
returned (the warm start is never substituted).

structure_mode
  'warm_start' (default, experiment 1-A): every candidate uses the Hybrid A*
      gear structure, the structure IPOPT (experiment 2) keeps fixed.
  'predicted' (experiment 1-B): top_k predicted structures x
      samples_per_structure; candidates on another structure start from the
      numeric-token source (recorded per candidate as ``x0_method``).

Timing: CUDA-synchronized wall time split into input encoding, generation and
post-processing/validation.  ``warmup()`` runs one untimed call so the first
measured plan excludes CUDA/kernel initialization (reported as
``warmup_excluded``).  Hybrid A* planning time is not included here.
"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time
import numpy as np
import torch

from .config import SolverConfig
from .conditions import scene_tokens, task_state
from .data import collate, to_device
from .inference import goal_from_scene, choose_valid
from .model import make_model
from .numerical import decode, validate, time_trace
from .sampling import candidate_schedule, sample_candidates
from .spline_model import decoded_model_samples
from .warmstart import warm_start_features, controls_from_hybrid_astar

STRUCTURE_MODES = ('warm_start', 'predicted')


def _validate_candidate(task):
    latent, h, goal, scene, cfg_dict = task
    cfg = SolverConfig(**cfg_dict)
    try:
        controls = decode(latent, h, cfg)
        return validate(controls, goal, scene, cfg), None
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return dict(verified_feasible=False), str(exc)


class WarmStartFlowSolver:
    def __init__(self, checkpoint, device='cuda', *, validation_workers=0):
        torch.set_num_threads(2)
        data = torch.load(checkpoint, map_location=device, weights_only=False)
        if data.get('schema') != 'k4-numeric-flow-checkpoint-v1':
            raise ValueError('unsupported solver checkpoint')
        representation = data.get('representation') or {}
        if representation.get('flow_source') != 'warmstart_residual_v1':
            raise ValueError('WarmStartFlowSolver requires a warmstart_residual_v1 checkpoint')
        self.cfg = SolverConfig(**data['config'])
        self.device = torch.device(device)
        self.model = make_model(self.cfg, data['normalization'], data.get('architecture', 'spline_whole'),
                                data.get('image_config'), representation).to(self.device)
        self.model.load_state_dict(data['ema'])
        self.model.eval().requires_grad_(False)
        self.representation = representation
        self.source_kind = self.model.source_kind          # fixed by the checkpoint
        self.checkpoint = str(Path(checkpoint).resolve())
        self.epoch = data.get('epoch')
        self.validation_workers = int(validation_workers)
        self._pool = ProcessPoolExecutor(self.validation_workers) if self.validation_workers > 0 else None
        self._warmed = False

    def close(self):
        if self._pool is not None:
            self._pool.shutdown()
            self._pool = None

    def set_goal_tolerance(self, position_m=None, yaw_rad=None):
        from dataclasses import replace
        changes = {}
        if position_m is not None:
            changes['terminal_position_tolerance_m'] = float(position_m)
        if yaw_rad is not None:
            changes['terminal_yaw_tolerance_rad'] = float(yaw_rad)
        self.cfg = replace(self.cfg, **changes)
        self.model.cfg = self.cfg

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def warmup(self, scene, warm_start_path, **kwargs):
        """One untimed call (CUDA context, kernels, validation pool start-up)."""
        self.plan(scene, warm_start_path, **kwargs)
        self._warmed = True

    def _inputs(self, scene, warm_start_path):
        features, warm_controls, info = warm_start_features(np.asarray(warm_start_path, dtype=np.float64), self.cfg)
        sample = dict(scene_tokens=scene_tokens(scene, self.cfg), k4_tokens=features['k4_tokens'],
                      task=task_state(scene, self.cfg), warm_latent=features['warm_latent'],
                      warm_structure=features['warm_structure'])
        return to_device(collate([sample]), self.device), int(features['warm_structure']), warm_controls, info

    @torch.inference_mode()
    def plan(self, scene, warm_start_path, *, top_k=2, samples_per_structure=8, steps=16, seed=20260917,
             include_warm_start=False, structure_mode='warm_start', candidate_batch_size=8):
        if structure_mode not in STRUCTURE_MODES:
            raise ValueError(f'unknown structure_mode: {structure_mode}')
        if top_k < 1 or samples_per_structure < 1 or steps < 1:
            raise ValueError('candidate budget must be positive')
        if abs(float(scene['start_state'][4])) > 1e-7:
            raise ValueError('requires zero initial steering, matching the training targets')
        warmup_excluded = self._warmed
        self.sync()
        started = time.perf_counter()
        timings = {}
        batch, warm_h, warm_controls, warm_info = self._inputs(scene, warm_start_path)
        goal = goal_from_scene(scene)
        context, logits = self.model.encode(batch)
        if structure_mode == 'warm_start':
            structures, samples = [warm_h], top_k*samples_per_structure
        else:
            structures = logits.topk(min(top_k, logits.shape[-1]), dim=-1).indices[0].cpu().tolist()
            samples = samples_per_structure
        schedule = candidate_schedule(structures, samples, seed)
        self.sync()
        timings['input_encoding_s'] = time.perf_counter()-started
        tick = time.perf_counter()
        y = sample_candidates(self.model, context, schedule, steps, batch_size=candidate_batch_size)
        hs = torch.tensor([c['structure'] for c in schedule], device=self.device)
        latent = decoded_model_samples(self.model, y, hs, self.cfg,
                                       torch.as_tensor(goal, device=self.device, dtype=torch.float32).expand(len(hs), -1))
        latent = latent.double().cpu().numpy()
        self.sync()
        timings['generation_s'] = time.perf_counter()-tick
        tick = time.perf_counter()
        tasks = [(a, c['structure'], goal, scene, self.cfg.to_dict()) for a, c in zip(latent, schedule)]
        checks = list(self._pool.map(_validate_candidate, tasks)) if self._pool else [_validate_candidate(t) for t in tasks]
        candidates, controls_by_index = [], {}
        for i, ((check, error), c) in enumerate(zip(checks, schedule)):
            method = 'controls' if (self.source_kind == 'controls' and c['structure'] == warm_h) else 'tokens'
            item = dict(index=i, source='flow', structure=c['structure'], structure_rank=c['structure_rank'],
                        sample_index=c['sample_index'], noise_seed=c['noise_seed'], x0_method=method,
                        J=check.get('J'), verified_feasible=bool(check.get('verified_feasible')), validation=check)
            if error:
                item['error'] = error
            candidates.append(item)
        if include_warm_start:
            check = validate(warm_controls, goal, scene, self.cfg)
            candidates.append(dict(index=len(candidates), source='warm_start', structure=warm_h, x0_method=None,
                                   J=check['J'], verified_feasible=bool(check['verified_feasible']), validation=check))
            controls_by_index[len(candidates)-1] = warm_controls
        selected = choose_valid(candidates)
        trajectory = None
        if selected is not None:
            controls = controls_by_index.get(selected['index'])
            if controls is None:
                controls = decode(latent[selected['index']], selected['structure'], self.cfg)
            trajectory = time_trace(controls, self.cfg)
        timings['postprocess_validation_s'] = time.perf_counter()-tick
        self.sync()
        timings['total_s'] = time.perf_counter()-started
        report = dict(
            status='valid_trajectory' if selected else 'stop_and_replan',
            structure_mode=structure_mode, warm_start_structure=warm_h, warm_start_gears=warm_info['gears'],
            selected_index=selected['index'] if selected else None,
            selected_source=selected['source'] if selected else None,
            selected_structure=selected['structure'] if selected else None,
            selected_same_structure_as_warm_start=(selected['structure'] == warm_h) if selected else None,
            selected_J=selected['J'] if selected else None,
            verified_candidates=sum(c['verified_feasible'] for c in candidates),
            candidates=[{k: v for k, v in c.items() if k != 'validation'} for c in candidates],
            timings=timings, device=str(self.device), warmup_excluded=warmup_excluded,
            validation_workers=self.validation_workers, include_warm_start=include_warm_start,
            steps=steps, seed=seed, samples=len(schedule), structures=structures,
            goal_tolerance=dict(position_m=self.cfg.terminal_position_tolerance_m,
                                yaw_rad=self.cfg.terminal_yaw_tolerance_rad),
            checkpoint=self.checkpoint, epoch=self.epoch, source_kind=self.source_kind,
            residual_noise_scale=self.representation.get('residual_noise_scale'),
            timing_scope='warm start supplied; includes input encoding, Flow generation, decoding and validation')
        return trajectory, report
