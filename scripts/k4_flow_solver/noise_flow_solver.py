"""Inference for the scene/task-conditioned noise Flow (no warm start).

plan(): encode scene + task -> top-K structures from the structure head ->
N candidates distributed over them (one batch, independent noise) -> Euler
integration of the learned field -> shared projection -> bicycle-model
integration and numerical validation -> minimum-J verified candidate.
No valid candidate -> ``stop_and_replan`` (nothing is substituted).
Timing covers input encoding through the final selection, CUDA-synchronized;
``warmup()`` runs one untimed call first.
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
from .numerical import decode, validate, time_trace
from .sampling import sample_candidates
from .noise_flow import load_noise_flow, top_k_schedule, decode_samples

FAILURE_CLASSES = ('collision', 'goal_not_reached', 'limit_violation', 'invalid_output', 'numerical_disagreement')


def classify(check, error=None):
    """Failure class of one candidate (None when verified)."""
    if error is not None:
        return 'invalid_output'
    if check.get('verified_feasible'):
        return None
    if not check.get('terminal_ok'):
        return 'goal_not_reached'
    if not check.get('controls_ok'):
        return 'limit_violation'
    geometry = check.get('geometry') or {}
    if not geometry.get('collision_free', True) or (geometry.get('conservative_continuous_clearance_m') or 1.0) < 0.1:
        return 'collision'
    if not geometry.get('final_inside', True):
        return 'goal_not_reached'
    return 'numerical_disagreement'


def _validate(task):
    latent, h, goal, scene, cfg_dict = task
    cfg = SolverConfig(**cfg_dict)
    try:
        controls = decode(latent, h, cfg)
        check = validate(controls, goal, scene, cfg)
        steer_rate = float(check.get('max_steer_rate', np.nan))
        return check, None, steer_rate
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return dict(verified_feasible=False), str(exc), np.nan


class NoiseFlowSolver:
    def __init__(self, checkpoint, device='cuda', *, validation_workers=0):
        torch.set_num_threads(2)
        self.device = torch.device(device)
        self.model, data = load_noise_flow(checkpoint, self.device)
        self.cfg = self.model.cfg
        self.representation = self.model.representation
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

    def warmup(self, scene, **kwargs):
        self.plan(scene, **kwargs)
        self._warmed = True

    @torch.inference_mode()
    def plan(self, scene, *, candidates=16, top_k=3, steps=16, seed=20260917, keep_candidates=True):
        if candidates < 1 or top_k < 1 or steps < 1:
            raise ValueError('candidate budget must be positive')
        if abs(float(scene['start_state'][4])) > 1e-7:
            raise ValueError('requires zero initial steering, matching the training targets')
        warmup_excluded = self._warmed
        self.sync()
        started = time.perf_counter()
        timings = {}
        sample = dict(scene_tokens=scene_tokens(scene, self.cfg), task=task_state(scene, self.cfg))
        batch = to_device(collate([sample]), self.device)
        goal = goal_from_scene(scene)
        context, logits = self.model.encode(batch)
        ranked = logits.topk(min(top_k, logits.shape[-1]), dim=-1).indices[0].cpu().tolist()
        schedule = top_k_schedule(ranked, candidates, seed)
        self.sync()
        timings['input_encoding_s'] = time.perf_counter()-started
        tick = time.perf_counter()
        y = sample_candidates(self.model, context, schedule, steps, batch_size=len(schedule))
        hs = torch.tensor([c['structure'] for c in schedule], device=self.device)
        latent = decode_samples(self.model, y, hs, self.cfg,
                                torch.as_tensor(goal, device=self.device, dtype=torch.float32).expand(len(hs), -1))
        latent = latent.double().cpu().numpy()
        self.sync()
        timings['generation_s'] = time.perf_counter()-tick
        tick = time.perf_counter()
        tasks = [(a, c['structure'], goal, scene, self.cfg.to_dict()) for a, c in zip(latent, schedule)]
        results = list(self._pool.map(_validate, tasks)) if self._pool else [_validate(t) for t in tasks]
        rows = []
        for i, ((check, error, steer_rate), c) in enumerate(zip(results, schedule)):
            rows.append(dict(index=i, structure=c['structure'], structure_rank=c['structure_rank'],
                             sample_index=c['sample_index'], noise_seed=c['noise_seed'], J=check.get('J'),
                             verified_feasible=bool(check.get('verified_feasible')), failure=classify(check, error),
                             max_steer_rate=steer_rate, validation=check, error=error))
        selected = choose_valid(rows)
        trajectory = None
        if selected is not None:
            trajectory = time_trace(decode(latent[selected['index']], selected['structure'], self.cfg), self.cfg)
        timings['postprocess_validation_s'] = time.perf_counter()-tick
        self.sync()
        timings['total_s'] = time.perf_counter()-started
        failures = [r['failure'] for r in rows if r['failure']]
        report = dict(
            status='valid_trajectory' if selected else 'stop_and_replan',
            ranked_structures=ranked, top_k=len(ranked), candidates=len(schedule), steps=steps, seed=seed,
            selected_index=selected['index'] if selected else None,
            selected_structure=selected['structure'] if selected else None,
            selected_J=selected['J'] if selected else None,
            verified_candidates=sum(r['verified_feasible'] for r in rows),
            failure_counts={k: failures.count(k) for k in FAILURE_CLASSES if failures.count(k)},
            scene_failure=None if selected else (max(set(failures), key=failures.count) if failures else 'no_candidate'),
            timings=timings, device=str(self.device), warmup_excluded=warmup_excluded,
            validation_workers=self.validation_workers, checkpoint=self.checkpoint, epoch=self.epoch,
            goal_tolerance=dict(position_m=self.cfg.terminal_position_tolerance_m, yaw_rad=self.cfg.terminal_yaw_tolerance_rad))
        if keep_candidates:
            report['candidate_rows'] = [{k: v for k, v in r.items() if k != 'validation'} for r in rows]
        return trajectory, report
