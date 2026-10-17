"""Frozen deployed K4 generation, anchor and runtime selection; no GT inputs."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import torch

from hope_path_transformer.inference import load_planner_checkpoint, generate_candidate_output
from hope_path_transformer.metrics import evaluate_inference_batch
from hope_path_transformer.worker_api import _live_candidate_geometry, _select_runtime_candidate, _candidate_control_metrics, LiveVehicleConfig
from compare_k4_gt_paths import anchor_output, enable_pruning, INPUT_KEYS

ROOT = Path(__file__).resolve().parents[2]


class FrozenK4:
    def __init__(self, profile_path=None, device='cuda', workers=4):
        profile_path = profile_path or ROOT/'config/hope_path_transformer_town05_town04_hardcase_v2_feasibility_k4_strict_anchor_inference.json'
        self.profile = json.loads(Path(profile_path).read_text(encoding='utf-8-sig'))
        if self.profile['total_candidates'] != 4 or self.profile.get('mirror_tta') or self.profile.get('additional'):
            raise ValueError('unsupported K4 profile')
        self.checkpoint_path = ROOT/self.profile['primary']['checkpoint']
        self.device = torch.device(device)
        self.model, config, checkpoint = load_planner_checkpoint(self.checkpoint_path, self.device, use_ema=True)
        self.epoch = checkpoint.get('epoch')
        del checkpoint
        self.model.eval().requires_grad_(False)
        self.config = replace(config, model=replace(config.model, hard_goal_terminal_pose=True,
            terminal_anchor_steps=self.profile['runtime_terminal_anchor']['coarse_steps']))
        enable_pruning()
        self.pool = ThreadPoolExecutor(max_workers=workers)

    @staticmethod
    def _geometry(task):
        poses, lengths, scene = task
        return tuple(x.numpy()[0] for x in _live_candidate_geometry(
            {'poses':torch.from_numpy(poses[None]), 'lengths':torch.from_numpy(lengths[None])},
            scene['obstacle_physical'], scene['start_state'], scene['parking_lot_polygon_xy'],
            LiveVehicleConfig(), scene['target_slot_polygon_xy']))

    @torch.inference_mode()
    def predict(self, batch, scenes):
        batch = {k:batch[k].to(self.device) for k in INPUT_KEYS}
        out = generate_candidate_output(self.model, batch, candidate_count=4, mirror_tta=False)
        out = anchor_output(out, batch, self.config, self.config.model.terminal_anchor_steps)
        poses = out['poses'].cpu().numpy()
        lengths = out['lengths'].cpu().numpy()
        checks = list(self.pool.map(self._geometry, zip(poses, lengths, scenes)))
        valid, clearance, inside = (torch.as_tensor(np.stack([c[i] for c in checks]), device=self.device) for i in range(3))
        geometry = evaluate_inference_batch(out, batch, self.config, collision_free_override=valid)
        path_length, switches, variation = _candidate_control_metrics(out, self.config.vehicle.wheelbase_m)
        s = self.profile['runtime_selection']
        weights = {'goal_error':s['goal_error_weight'], 'steering_variation':s['steering_variation_weight'],
                   'gear_switches':s['gear_switch_weight'], 'path_length':s['path_length_weight'], 'clearance':s['clearance_weight']}
        selected, success, _ = _select_runtime_candidate(fallback_selected=geometry['selected'],
            exact_success=geometry['candidate_success'], collision_free=valid, final_inside=inside,
            live_clearance_m=clearance, goal_position_error_m=geometry['candidate_goal_position_error_m'],
            goal_yaw_error_deg=geometry['candidate_goal_yaw_error_deg'], path_length_m=path_length,
            gear_switches=switches, steering_variation_rad=variation, minimum_clearance_m=s['minimum_clearance_m'],
            goal_position_tolerance_m=self.config.data.goal_position_tolerance_m,
            goal_yaw_tolerance_deg=self.config.data.goal_yaw_tolerance_deg, weights=weights)
        selected = selected.cpu().numpy()
        gears, steering = out['gear'].cpu().numpy(), out['steering'].cpu().numpy()
        results = []
        for i, chosen in enumerate(selected):
            n = lengths[i, chosen]
            path = np.c_[poses[i, chosen, :n], 2*gears[i, chosen, :n]-1, steering[i, chosen, :n]]
            results.append((path.astype(np.float32), int(chosen), bool(success[i, chosen])))
        return results

    def close(self):
        self.pool.shutdown()
