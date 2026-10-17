"""Whole-trajectory inference with bounded control repair and fail-closed selection."""
from __future__ import annotations
from pathlib import Path
from dataclasses import replace
import time
import numpy as np
import torch
from .config import SolverConfig
from .conditions import scene_tokens, path_tokens, task_state, local_points
from .model import make_model, latent_mask
from .numerical import decode, integrate, cost, repair, validate, from_k4, time_trace, wrap
from .sampling import candidate_schedule, sample_candidates


def goal_from_scene(scene):
    start, goal = scene['start_state'], scene['requested_goal_pose']
    return np.r_[local_points(np.asarray(goal[:2])[None], start)[0], wrap(goal[2]-start[2])]


def model_input(scene, path, cfg, device, image_config=None):
    numeric, k4, task = scene_tokens(scene, cfg), path_tokens(path, cfg), task_state(scene, cfg)
    result = dict(scene_tokens=torch.as_tensor(numeric[None], device=device),
                scene_tokens_mask=torch.ones((1,len(numeric)), dtype=torch.bool, device=device),
                k4_tokens=torch.as_tensor(k4[None], device=device),
                k4_tokens_mask=torch.ones((1,len(k4)), dtype=torch.bool, device=device),
                task=torch.as_tensor(task[None], device=device))
    if image_config is not None:
        if scene.get('image_config') != image_config:
            raise ValueError('missing or incompatible semantic image provenance')
        image = np.asarray(scene['bev_image'], dtype=np.float32)
        start = np.asarray(scene['bev_start'], dtype=np.float32)
        if image.shape != (len(image_config['channels']), *image_config['shape']) or not np.isfinite(image).all():
            raise ValueError('invalid image shape or values')
        if not np.isin(image, [0., 1.]).all() or start.shape != (3,) or not np.allclose(start, scene['start_state'][:3], atol=1e-5, rtol=0):
            raise ValueError('image values or start-frame alignment mismatch')
        result.update(bev_image=torch.as_tensor(image[None], device=device),
                      bev_start=torch.as_tensor(start[None], device=device))
    return result


def choose_valid(candidates):
    """Only independently verified candidates enter the cost comparison."""
    valid = [c for c in candidates if c.get('validation', {}).get('verified_feasible', False)
             and np.isfinite(c['validation'].get('J', float('nan')))]
    return min(valid, key=lambda c:c['validation']['J']) if valid else None


class LearnedTrajectorySolver:
    def __init__(self, checkpoint, device='cuda'):
        torch.set_num_threads(2)
        data = torch.load(checkpoint, map_location=device, weights_only=False)
        if data.get('schema') != 'k4-numeric-flow-checkpoint-v1':
            raise ValueError('unsupported solver checkpoint')
        self.cfg = SolverConfig(**data['config'])
        self.device = torch.device(device)
        self.mode = data['mode']
        self.architecture = data.get('architecture','phase')
        if data.get('exploration') is not None:
            from .conditional_prior import K4PriorFlow
            if self.architecture != 'whole' or self.mode != 'flow':
                raise ValueError('conditional prior requires the whole Flow architecture')
            self.model = K4PriorFlow(self.cfg, data['normalization'], data['exploration']).to(device)
        else:
            self.model = make_model(self.cfg, data['normalization'], self.architecture, data.get('image_config'),data.get('representation')).to(device)
        if data.get('correction_config') is not None:
            self.model.enable_corrector(data['correction_config']['iterations'])
        self.model.load_state_dict(data['ema'])
        self.model.eval().requires_grad_(False)
        self.checkpoint = str(Path(checkpoint).resolve())
        self.epoch = data['epoch']

    def set_goal_tolerance(self, position_m=None, yaw_rad=None):
        changes={}
        if position_m is not None:changes['terminal_position_tolerance_m']=float(position_m)
        if yaw_rad is not None:changes['terminal_yaw_tolerance_rad']=float(yaw_rad)
        self.cfg=replace(self.cfg,**changes)
        self.model.cfg=self.cfg

    def sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    @torch.inference_mode()
    def plan(self, scene, k4_path, *, top_k=2, steps=4, samples_per_structure=1,
             seed=20260917, include_baseline=True, allowed_structures=None,
             repair_time_budget_s=0.15, diagnostic_teacher_structure=None,
             length_correction_bound=0.25, steering_correction_bound=0.05,
             start_speed_mps=0.0, candidate_batch_size=8, retain_candidate_controls=False):
        if abs(float(start_speed_mps)) > 1e-7:
            raise ValueError('first implementation requires stationary start')
        if abs(float(scene['start_state'][4])) > 1e-7:
            raise ValueError('first implementation requires zero initial steering, matching collected teachers')
        if top_k < 1 or samples_per_structure < 1:
            raise ValueError('candidate budget must be positive')
        if self.mode == 'regression' and samples_per_structure != 1:
            raise ValueError('deterministic regression does not provide independent noise samples')
        self.sync()
        started = time.perf_counter()
        timings = {}
        batch = model_input(scene, k4_path, self.cfg, self.device, vars(self.model).get('image_config'))
        goal = goal_from_scene(scene)
        timings['input_seconds'] = time.perf_counter()-started
        tick = time.perf_counter()
        context, logits = self.model.encode(batch)
        conditional_prior = hasattr(type(self.model), 'set_k4_prior')
        if conditional_prior:
            self.model.set_k4_prior(k4_path)
        if allowed_structures is not None:
            allowed = torch.zeros_like(logits, dtype=torch.bool)
            allowed[:, list(allowed_structures)] = True
            if not allowed.any():
                raise ValueError('no allowed structures')
            logits = logits.masked_fill(~allowed, -float('inf'))
            top_k = min(top_k, int(allowed.sum()))
        if diagnostic_teacher_structure is not None:
            # Explicit diagnostic switch; callers must never treat this as deployment performance.
            structures = torch.tensor([diagnostic_teacher_structure], device=self.device)
        elif conditional_prior and allowed_structures is None:
            structures = self.model.choose_structures(logits, top_k)
        else:
            structures = logits.topk(min(top_k, logits.shape[-1]), dim=-1).indices[0]
        schedule = candidate_schedule(structures.cpu().tolist(),samples_per_structure,seed)
        structures = torch.tensor([c['structure'] for c in schedule],device=self.device)
        self.sync()
        timings['encode_seconds'] = time.perf_counter()-tick
        tick = time.perf_counter()
        y = sample_candidates(self.model,context,schedule,steps,batch_size=candidate_batch_size,
                              regression=self.mode=='regression')
        from .spline_model import decoded_model_samples
        latent = decoded_model_samples(self.model,y,structures,self.cfg,
            torch.as_tensor(goal,device=self.device).expand(len(structures),-1)).double().cpu().numpy()
        hs = structures.cpu().tolist()
        self.sync()
        timings['generation_seconds'] = time.perf_counter()-tick
        candidates = []
        raw_controls = []
        for i, (a, h) in enumerate(zip(latent, hs)):
            candidate = dict(source='learned', index=i, **schedule[i])
            tick = time.perf_counter()
            original = None
            try:
                controls = decode(a, h, self.cfg)
                original = controls.copy() if retain_candidate_controls else None
                endpoint = integrate(controls, cfg=self.cfg)[0]
                candidate['raw'] = dict(**cost(controls,self.cfg), endpoint_error=np.r_[endpoint[:2]-goal[:2],wrap(endpoint[2]-goal[2])].tolist())
                if self.architecture=='spline_whole':
                    correction=dict(method='shared_bounded_shooting_and_retime',iterations=vars(self.model)['representation'].get('projection_iterations',2),
                        endpoint_error=candidate['raw']['endpoint_error'],scope='Already applied in the shared train/inference decoder; no second legacy repair')
                else:
                    controls, correction = repair(controls, goal, self.cfg, length_bound=length_correction_bound,
                        steering_bound=steering_correction_bound, time_budget_s=repair_time_budget_s)
                candidate.update(correction=correction, validation=validate(controls,goal,scene,self.cfg), controls=controls)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                candidate.update(error=str(exc), validation=dict(verified_feasible=False))
            candidate['postprocess_seconds'] = time.perf_counter()-tick
            candidates.append(candidate)
            raw_controls.append(original)
        timings['learned_postprocess_seconds'] = sum(c['postprocess_seconds'] for c in candidates)
        tick = time.perf_counter()
        if include_baseline:
            baseline = dict(source='k4_baseline')
            try:
                controls = from_k4(k4_path, self.cfg)
                controls, correction = repair(controls, goal, self.cfg, length_bound=length_correction_bound,
                    steering_bound=steering_correction_bound, time_budget_s=repair_time_budget_s)
                baseline.update(correction=correction, validation=validate(controls,goal,scene,self.cfg),controls=controls)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
                baseline.update(error=str(exc),validation=dict(verified_feasible=False))
            candidates.append(baseline)
        timings['baseline_seconds'] = time.perf_counter()-tick
        selected = choose_valid(candidates)
        trajectory = None
        tick = time.perf_counter()
        if selected is not None:
            trajectory = time_trace(selected['controls'], self.cfg)
        timings['trace_seconds'] = time.perf_counter()-tick
        self.sync()
        timings['total_seconds'] = time.perf_counter()-started
        report = dict(status='valid_trajectory' if selected else 'stop_and_replan',
            raw_metrics_scope=('After shared shooting/retiming decoder; before validation and segment composition' if self.architecture=='spline_whole'
                               else 'Before bounded endpoint repair and retiming'),
            selected_source=selected['source'] if selected else None,
            selected_J=selected['validation']['J'] if selected else None,
            selected_structure=selected.get('structure') if selected else None,
            candidates=[{k:v for k,v in c.items() if k != 'controls'} for c in candidates], timings=timings,
            checkpoint=self.checkpoint, epoch=self.epoch, mode=self.mode, architecture=self.architecture, steps=steps,
            structures=hs, seed=seed, diagnostic_teacher_structure=diagnostic_teacher_structure,
            sampling=dict(version='independent-prefix-v2',samples_per_structure=samples_per_structure,
                          candidate_batch_size=candidate_batch_size),
            goal_tolerance=dict(position_m=self.cfg.terminal_position_tolerance_m,yaw_rad=self.cfg.terminal_yaw_tolerance_rad),
            correction_bounds=dict(length_m=length_correction_bound,steering_rad=steering_correction_bound,
                                   iteration_time_budget_s=repair_time_budget_s),
            timing_scope='K4 path already supplied; includes input encoding, generation, repair, baseline and independent validation')
        if retain_candidate_controls:
            # Explicit diagnostic output; the caller must remove these objects
            # before JSON serialization. Normal runtime reports stay JSON-only.
            report['_candidate_controls'] = [c.get('controls') for c in candidates]
            report['_raw_candidate_controls'] = raw_controls
        return trajectory, report
