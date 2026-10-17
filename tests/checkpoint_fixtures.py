"""Deterministic test checkpoints derived from the released epoch-10 spline model.

No trained 'k4_residual_v1' or 'warmstart_residual_v1' checkpoint exists, so
tests build them exactly like ``train_k4_spline_flow.py --residual`` does at
initialization: epoch-10 EMA weights with the velocity heads reset to zero.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))

EPOCH10 = ROOT/'run-logs/k4_spline_projected_j_v1_20260920/epoch_10.pt'
DATASET = ROOT/'data/k4_numeric_flow_full_v1_20260917'
SCENES = ROOT/'data/flow_scenes'
WARM_CACHE = ROOT/'data/warmstart_cache/hybrid_astar_e3000_v1'
FIXTURES = Path(__file__).resolve().parent/'fixtures'
# A validation scene with a successful Hybrid A* warm start (fixed for regressions).
REGRESSION_UID = 's00-o000000001-11064f7da89b'


def have_release_assets():
    return EPOCH10.is_file() and DATASET.is_dir() and SCENES.is_dir() and WARM_CACHE.is_dir()


def synthetic_checkpoint(out_path, representation_update):
    """Save epoch-10 weights under a residual representation, heads zeroed."""
    import torch
    from k4_flow_solver.config import SolverConfig
    from k4_flow_solver.model import make_model
    data = torch.load(EPOCH10, map_location='cpu', weights_only=False)
    representation = dict(data['representation'], **representation_update)
    model = make_model(SolverConfig(**data['config']), data['normalization'], 'spline_whole', None, representation)
    model.load_state_dict(data['ema'], strict=True)
    for head in (model.core.control_head, model.core.length_head[-1]):
        torch.nn.init.zeros_(head.weight)
        torch.nn.init.zeros_(head.bias)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    checkpoint = {k: data[k] for k in ('schema', 'config', 'mode', 'architecture', 'normalization',
                                       'dataset_sha256', 'epoch')}
    checkpoint.update(representation=representation, ema=state, model=state, image_config=None)
    torch.save(checkpoint, out_path)
    return out_path


K4_RESIDUAL = dict(flow_source='k4_residual_v1', residual_noise_scale=0.15,
                   source_contract='numeric_K4_tokens_only; matching cusps else uniform arc partitions; analytic speed seed')


def regression_inputs():
    import numpy as np
    from k4_flow_solver.prepare import load_scene
    scene = load_scene(SCENES/REGRESSION_UID)
    with np.load(WARM_CACHE/(REGRESSION_UID+'.npz'), allow_pickle=False) as z:
        path = z['path'].copy()
    return scene, path


def raw_candidates(checkpoint, scene, path, seed=20260917, steps=4, top_k=2, samples_per_structure=2):
    """Generated normalized latents (before decoding) for a fixed schedule."""
    import torch
    from k4_flow_solver.inference import LearnedTrajectorySolver, model_input
    from k4_flow_solver.sampling import candidate_schedule, sample_candidates
    solver = LearnedTrajectorySolver(checkpoint, device='cpu')
    with torch.inference_mode():
        batch = model_input(scene, path, solver.cfg, solver.device)
        context, logits = solver.model.encode(batch)
        structures = logits.topk(top_k, dim=-1).indices[0].tolist()
        schedule = candidate_schedule(structures, samples_per_structure, seed)
        y = sample_candidates(solver.model, context, schedule, steps, batch_size=8)
    return y.numpy(), [c['structure'] for c in schedule]
