"""Noise Flow grid representations (grid64 / grid128 / grid513).

Run: python -m unittest tests.test_noise_grid_flow   (CPU is enough; the model
tests need the released epoch-10 / whole epoch-35 checkpoints, the dataset and
experiments/noise_flow_grid_normalization.json).
"""
from pathlib import Path
from collections import Counter
from dataclasses import replace
import copy
import json
import math
import os
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import checkpoint_fixtures as F  # noqa: E402
from k4_flow_solver.config import SolverConfig  # noqa: E402
from k4_flow_solver.numerical import grid  # noqa: E402
from k4_flow_solver.noise_flow import (GRID_KNOTS, NoiseGridFlow, interpolation_matrix,  # noqa: E402
                                       noise_representation, cfm_pair)

EPOCH35 = ROOT/'run-logs/k4_whole_flow_direct_j_scratch_v1_20260919/epoch_35.pt'
NORMALIZATION = ROOT/'experiments/noise_flow_grid_normalization.json'
EVAL_UIDS = ROOT/'experiments/eval_uids.json'


def have_assets():
    return all(p.exists() for p in (F.EPOCH10, EPOCH35, F.DATASET, F.SCENES, NORMALIZATION, EVAL_UIDS))


def reference_round_trip(a, h, cfg, n):
    """Step-0 numpy procedure: linear sampling at n cosine knots, limits per knot, linear interpolation back."""
    u, v = grid(cfg.intervals), grid(n-1)
    m = h % cfg.max_phases+1
    delta = cfg.max_steer*np.tanh(a[..., 1])
    b = cfg.max_speed**2/(1+np.exp(-a[..., 2]))
    b[:, [0, -1]] = 0
    out_delta, out_b = delta.copy(), b.copy()
    for p in range(m):
        d_n = np.clip(np.interp(v, u, delta[p]), -cfg.max_steer*(1-1e-7), cfg.max_steer*(1-1e-7))
        b_n = np.interp(v, u, b[p])
        b_n[[0, -1]] = 0
        out_delta[p] = np.interp(u, v, d_n)
        out_b[p] = np.interp(u, v, b_n)
    return out_delta[:m], out_b[:m]


class Interpolation(unittest.TestCase):
    def test_matrix_is_linear_interpolation(self):
        dense, coarse = grid(512), grid(63)
        w = interpolation_matrix(coarse, dense)
        np.testing.assert_allclose(w.sum(1), 1., atol=1e-12)
        self.assertGreaterEqual(w.min(), 0.)
        f = lambda x: 3*x-1
        np.testing.assert_allclose(w@f(coarse), f(dense), atol=1e-12)
        np.testing.assert_allclose(w@np.sin(coarse), np.interp(dense, coarse, np.sin(coarse)), atol=1e-12)
        np.testing.assert_allclose(interpolation_matrix(dense, dense), np.eye(513), atol=0)

    def test_restore_keeps_limits_for_random_outputs(self):
        cfg = SolverConfig()
        generator = torch.Generator().manual_seed(4)
        for name, n in GRID_KNOTS.items():
            model = NoiseGridFlow(cfg, None, noise_representation(name)).double()
            h = torch.tensor([0, 3, 7, 11])
            values = 6*torch.randn(4, cfg.max_phases, n, 3, generator=generator, dtype=torch.float64)
            values[..., 0, 0] = values[..., 0, 0].clamp(-2, 2)
            for project in (False, True):
                dense = model.dense_latent(values, h, project=project)
                steer = cfg.max_steer*torch.tanh(dense[..., 1])
                speed = cfg.max_speed**2*torch.sigmoid(dense[..., 2])
                self.assertLessEqual(float(steer.abs().max()), cfg.max_steer)
                self.assertTrue(bool(((speed >= 0) & (speed <= cfg.max_speed**2)).all()))
                phase = torch.arange(cfg.max_phases)[None] < (h % cfg.max_phases+1)[:, None]
                # endpoint speeds are masked in the latent (decode sets them to 0); interior stays in (0, vmax^2)
                self.assertEqual(float(dense[..., [0, -1], 2].abs().max()), 0.)
                if not project:
                    knot_speed = cfg.max_speed**2*torch.sigmoid(values[..., 2])
                    knot_speed[..., [0, -1]] = 0
                    expected = knot_speed@model.restore_matrix.T
                    torch.testing.assert_close(speed[..., 1:-1][phase], expected[..., 1:-1][phase], rtol=1e-9, atol=1e-12)
                # convex combination: the 513-point steering stays within the knot extremes per phase
                knot = cfg.max_steer*torch.tanh(values[..., 1])
                self.assertTrue(bool((steer.amax(-1)[phase] <= knot.amax(-1)[phase]+1e-9).all()))
                self.assertTrue(bool((steer.amin(-1)[phase] >= knot.amin(-1)[phase]-1e-9).all()))


@unittest.skipUnless(have_assets(), 'release assets not downloaded')
class Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from k4_flow_solver.noise_flow import FilteredSolverDataset
        manifest = json.loads((F.DATASET/'manifest.json').read_text(encoding='utf-8'))
        cls.cfg = replace(SolverConfig(**manifest['config']), terminal_position_tolerance_m=.1,
                          terminal_yaw_tolerance_rad=np.deg2rad(5))
        cls.manifest = manifest
        cls.stats = json.loads(NORMALIZATION.read_text(encoding='utf-8'))
        held_out = set(json.loads(EVAL_UIDS.read_text(encoding='utf-8'))['uids'])
        cls.rows = FilteredSolverDataset(F.DATASET, 'validation', exclude=held_out, limit=16)
        cls.tmp = tempfile.mkdtemp()

    def model(self, name, init=True):
        from k4_flow_solver.noise_flow import make_noise_model, load_init_features
        model = make_noise_model(self.cfg, self.stats['representations'][name], noise_representation(name)).eval()
        report = load_init_features(model, EPOCH35 if name == 'grid513' else F.EPOCH10) if init else None
        return model, report

    def test_normalization_file_matches_dataset(self):
        from k4_flow_solver.prepare import digest
        self.assertEqual(self.stats['dataset_sha256'], digest(F.DATASET/'manifest.json'))
        self.assertTrue(self.stats['grid513_vs_manifest']['counts_equal'])
        self.assertLess(max(self.stats['grid513_vs_manifest']['mean_abs_difference']), 1e-8)

    def test_round_trip_matches_step0_procedure(self):
        from k4_flow_solver.numerical import decode, integrate, wrap
        worst = {}
        for name, n in GRID_KNOTS.items():
            model, _ = self.model(name, init=False)
            model = model.double()
            errors = []
            for row in self.rows.samples[:8]:
                a = np.asarray(row['latent'], np.float64)
                h = int(row['structure'])
                hh = torch.tensor([h])
                label, mask = model.fit_targets(torch.as_tensor(a)[None], hh)
                back = model.denormalize(model.normalize(label.double(), mask.double()), mask.double())
                rebuilt = model.dense_latent(back, hh, project=False)[0].numpy()
                gt, rec = decode(a, h, self.cfg), decode(rebuilt, h, self.cfg)
                ref_delta, ref_b = reference_round_trip(a, h, self.cfg, n)
                # same numbers as the step-0 measurement (float32 storage only)
                self.assertLess(float(np.abs(rec.delta-ref_delta).max()), 2e-6, name)
                self.assertLess(float(np.abs(np.sqrt(rec.b)-np.sqrt(ref_b)).max()), 2e-5, name)
                e_gt, e_rec = integrate(gt, cfg=self.cfg)[0], integrate(rec, cfg=self.cfg)[0]
                errors.append((float(np.degrees(np.abs(gt.delta-rec.delta)).mean()), float(np.hypot(*(e_gt[:2]-e_rec[:2]))),
                               float(abs(math.degrees(wrap(e_gt[2]-e_rec[2]))))))
            worst[name] = np.max(errors, axis=0)
        # step-0 maxima over 1,000 train samples: grid64 32 mm / 0.22 deg, grid128 9 mm / 0.06 deg, grid513 ~0
        self.assertLess(worst['grid64'][1], .035)
        self.assertLess(worst['grid128'][1], .01)
        self.assertLess(worst['grid513'][1], 1e-5)
        self.assertLess(worst['grid513'][0], 1e-5)

    def test_init_keys(self):
        _, r64 = self.model('grid64')
        _, r128 = self.model('grid128')
        _, r513 = self.model('grid513')
        path = {'core.encoder.path.0.weight', 'core.encoder.path.0.bias', 'core.encoder.path.2.weight', 'core.encoder.path.2.bias'}
        norm = {'core.latent_mean', 'core.latent_std'}
        spline = {'basis', 'fit_matrix'}  # epoch-10 B-spline buffers (not part of a grid model)
        self.assertEqual(set(r64['skipped_from_checkpoint']), path | norm | spline)
        self.assertEqual(set(r128['skipped_from_checkpoint']), path | norm | spline | {'core.u'})
        self.assertEqual(set(r513['skipped_from_checkpoint']), path | norm)
        self.assertEqual(r513['architecture'], 'whole')
        for r, extra in ((r64, set()), (r128, {'core.u'}), (r513, set())):
            self.assertEqual(set(r['missing_in_checkpoint']), norm | extra | {'sample_matrix', 'restore_matrix'})
        model, _ = self.model('grid64')
        torch.testing.assert_close(model.core.latent_mean.double(),
                                   torch.tensor(self.stats['representations']['grid64']['mean'], dtype=torch.float64))

    def test_bspline64_decoding_unchanged(self):
        from k4_flow_solver.noise_flow import make_noise_model, decode_samples
        from k4_flow_solver.spline_model import decoded_model_samples
        model = make_noise_model(self.cfg, self.manifest['normalization'], noise_representation('bspline64')).eval()
        h = torch.tensor([1, 8])
        y = torch.randn(2, self.cfg.max_phases, 64, 3, generator=torch.Generator().manual_seed(2))
        goal = torch.tensor([[6., 2., 0.3], [5., -1., -0.2]])
        for g in (None, goal):
            torch.testing.assert_close(decode_samples(model, y, h, self.cfg, g), decoded_model_samples(model, y, h, self.cfg, g),
                                       rtol=0, atol=0)

    def test_pipeline_with_grid_checkpoints(self):
        from k4_flow_solver.noise_flow_solver import NoiseFlowSolver, FAILURE_CLASSES
        from k4_flow_solver.prepare import load_scene
        scene = load_scene(F.SCENES/self.rows[0]['uids'])
        for name in ('grid64', 'grid513'):
            model, report = self.model(name)
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            path = os.path.join(self.tmp, f'{name}.pt')
            torch.save(dict(schema='k4-numeric-flow-checkpoint-v1', config=self.cfg.to_dict(),
                            normalization=self.stats['representations'][name], mode='flow', architecture='noise_scene_flow',
                            representation=noise_representation(name), ema=state, model=state, epoch=0,
                            dataset_sha256=report['dataset_sha256']), path)
            solver = NoiseFlowSolver(path, device='cpu')
            trajectory, out = solver.plan(scene, candidates=6, top_k=3, steps=2, seed=7)
            solver.close()
            self.assertEqual(out['candidates'], 6)
            self.assertEqual(Counter(r['structure'] for r in out['candidate_rows']),
                             Counter({h: 2 for h in out['ranked_structures']}))
            self.assertTrue(set(out['failure_counts']) <= set(FAILURE_CLASSES))
            self.assertEqual(out['verified_candidates']+sum(out['failure_counts'].values()), 6)
            self.assertIn(out['status'], ('valid_trajectory', 'stop_and_replan'))
            self.assertEqual(out['status'] == 'valid_trajectory', trajectory is not None)

    def test_overfit_smoke(self):
        """A few hundred GT samples: FM loss on those samples drops (fixed evaluation noise).

        grid64 starts from the epoch-10 features; grid513 starts from scratch
        because the whole epoch-35 initializer already sits near the CFM loss
        floor on these samples (0.070 -> 0.051 after 256 steps).
        """
        from k4_flow_solver.data import collate
        from k4_flow_solver.model import channel_mse
        from k4_flow_solver.noise_flow import FilteredSolverDataset
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        count = 256 if device.type == 'cuda' else 64
        rows = FilteredSolverDataset(F.DATASET, 'train', limit=count)
        batches = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in collate(rows.samples[i:i+16]).items()}
                   for i in range(0, count, 16)]
        for name, init in (('grid64', True), ('grid513', False)):
            model = copy.deepcopy(self.model(name, init=init)[0]).to(device).train()

            def evaluate():
                model.eval()
                losses = []
                with torch.no_grad():
                    for i, batch in enumerate(batches):
                        h = batch['structure'].long()
                        label, mask = model.fit_targets(batch['latent'], h)
                        target = model.normalize(label, mask)
                        context, _ = model.encode(batch)
                        y, t, velocity, _ = cfm_pair(target, mask, torch.Generator(device=device).manual_seed(1000+i))
                        losses.append(float(channel_mse(model(y, t, context, h), velocity, mask)))
                model.train()
                return float(np.mean(losses))

            before = evaluate()
            optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.)
            for _ in range(8 if device.type == 'cuda' else 4):
                for batch in batches:
                    h = batch['structure'].long()
                    with torch.no_grad():
                        label, mask = model.fit_targets(batch['latent'], h)
                        target = model.normalize(label, mask)
                    context, logits = model.encode(batch)
                    y, t, velocity, _ = cfm_pair(target, mask)
                    loss = channel_mse(model(y, t, context, h), velocity, mask)+torch.nn.functional.cross_entropy(logits, h)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    optimizer.step()
            after = evaluate()
            print(json.dumps(dict(representation=name, initialized=init, overfit_samples=count, device=str(device), FM_before=before, FM_after=after)))
            self.assertTrue(np.isfinite(after))
            self.assertLess(after, .8*before, name)


if __name__ == '__main__':
    unittest.main()
