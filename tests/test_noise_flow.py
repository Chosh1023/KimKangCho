"""Noise -> OCP GT Flow conditioned on the scene and start/goal only (no warm start).

Run: python -m unittest tests.test_noise_flow   (CPU is enough; the model tests
use epoch-10 features loaded strict=False, so no trained noise-flow checkpoint
is required).
"""
from pathlib import Path
from collections import Counter
import json
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
from k4_flow_solver.noise_flow import cfm_pair, top_k_schedule  # noqa: E402

EVAL_UIDS = ROOT/'experiments/eval_uids.json'


def have_assets():
    return F.EPOCH10.is_file() and F.DATASET.is_dir() and F.SCENES.is_dir() and EVAL_UIDS.is_file()


class Formula(unittest.TestCase):
    def test_standard_cfm_bridge(self):
        generator = torch.Generator().manual_seed(3)
        target = torch.randn(512, 6, 64, 3, generator=generator)
        mask = torch.zeros_like(target)
        mask[:, :2, :, 1:] = 1
        mask[:, :2, 0, 0] = 1
        target = target*mask
        y, t, velocity, x0 = cfm_pair(target, mask, torch.Generator().manual_seed(5))
        tt = t[:, None, None, None]
        torch.testing.assert_close(y, (1-tt)*x0+tt*target, rtol=0, atol=1e-6)
        torch.testing.assert_close(velocity, target-x0, rtol=0, atol=1e-6)
        # x0 ~ N(0, I) on valid entries only, zero elsewhere; t ~ U(0, 1).
        self.assertEqual(float((x0*(1-mask)).abs().max()), 0.)
        valid = x0[mask.bool()]
        self.assertLess(abs(float(valid.mean())), .01)
        self.assertLess(abs(float(valid.std())-1), .01)
        self.assertTrue(bool(((t >= 0) & (t < 1)).all()))
        self.assertLess(abs(float(t.mean())-.5), .05)
        # Euler along the exact straight path lands on x1 from x0 for any step count.
        for steps in (1, 4, 16):
            z = x0.clone()
            for _ in range(steps):
                z = z+velocity/steps
            torch.testing.assert_close(z, target, rtol=0, atol=1e-5)

    def test_top_k_distribution(self):
        schedule = top_k_schedule([5, 2, 9], 16, seed=11)
        self.assertEqual(Counter(c['structure'] for c in schedule), Counter({5: 6, 2: 5, 9: 5}))
        self.assertEqual([c['structure_rank'] for c in schedule[:6]], [0, 1, 2, 0, 1, 2])
        for h in (5, 2, 9):
            self.assertEqual([c['sample_index'] for c in schedule if c['structure'] == h],
                             list(range(sum(c['structure'] == h for c in schedule))))
        self.assertEqual(len({c['noise_seed'] for c in schedule}), 16)
        self.assertEqual(schedule, top_k_schedule([5, 2, 9], 16, seed=11))
        self.assertEqual(top_k_schedule([5, 2, 9], 8, seed=11), schedule[:8])
        self.assertNotEqual([c['noise_seed'] for c in top_k_schedule([5, 2, 9], 16, seed=12)],
                            [c['noise_seed'] for c in schedule])
        self.assertEqual({c['structure'] for c in top_k_schedule([4], 16, seed=11)}, {4})
        with self.assertRaises(ValueError):
            top_k_schedule([1, 1], 4, seed=0)
        with self.assertRaises(ValueError):
            top_k_schedule([1], 0, seed=0)


@unittest.skipUnless(have_assets(), 'release assets not downloaded')
class Model(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from dataclasses import replace
        from k4_flow_solver.config import SolverConfig
        from k4_flow_solver.noise_flow import (make_noise_model, noise_representation, load_epoch10_features,
                                               FilteredSolverDataset)
        torch.manual_seed(0)
        manifest = json.loads((F.DATASET/'manifest.json').read_text(encoding='utf-8'))
        cls.cfg = replace(SolverConfig(**manifest['config']), terminal_position_tolerance_m=.1,
                          terminal_yaw_tolerance_rad=np.deg2rad(5))
        cls.normalization = manifest['normalization']
        cls.representation = noise_representation('bspline64')
        cls.model = make_noise_model(cls.cfg, cls.normalization, cls.representation).eval()
        cls.init = load_epoch10_features(cls.model, F.EPOCH10)
        cls.held_out = set(json.loads(EVAL_UIDS.read_text(encoding='utf-8'))['uids'])
        cls.rows = FilteredSolverDataset(F.DATASET, 'validation', exclude=cls.held_out, limit=16)
        cls.tmp = tempfile.mkdtemp()

    def batch(self, count=4):
        from k4_flow_solver.data import collate
        return collate([self.rows[i] for i in range(count)])

    def test_held_out_scenes_excluded(self):
        self.assertTrue(self.held_out)
        self.assertFalse({r['uids'] for r in self.rows} & self.held_out)
        self.assertNotIn('k4_tokens', self.rows[0])

    def test_epoch10_init_skips_only_the_path_branch(self):
        own = self.model.state_dict()
        self.assertEqual(self.init['missing_in_checkpoint'], [])
        self.assertEqual(len(self.init['loaded']), len(own))
        self.assertTrue(self.init['skipped_from_checkpoint'])
        self.assertTrue(all(k.startswith('core.encoder.path.') for k in self.init['skipped_from_checkpoint']))
        self.assertFalse([k for k in own if '.path.' in k])
        self.assertEqual(self.init['normalization'], self.normalization)

    def test_encoder_uses_scene_and_task_only(self):
        batch = self.batch()
        with torch.no_grad():
            context, logits = self.model.encode(batch)
            self.assertEqual(tuple(context.shape), (4, 24, 192))
            self.assertEqual(tuple(logits.shape), (4, 2*self.cfg.max_phases))
            # A path in the batch is not read: the context is bit-identical.
            extra = dict(batch, k4_tokens=torch.randn(4, 10, 7), k4_tokens_mask=torch.ones(4, 10, dtype=torch.bool))
            self.assertTrue(torch.equal(self.model.encode(extra)[0], context))
            # Changing the task changes the context.
            moved = dict(batch, task=batch['task']+.1)
            self.assertFalse(torch.equal(self.model.encode(moved)[0], context))
            with self.assertRaises(ValueError):
                self.model.encode(dict(batch, bev_image=torch.zeros(4, 1, 8, 8)))

    def test_forward_and_rollout_from_noise(self):
        from k4_flow_solver.model import latent_mask
        from k4_flow_solver.spline_model import decoded_model_samples
        batch = self.batch()
        h = batch['structure'].long()
        with torch.no_grad():
            label, mask = self.model.fit_targets(batch['latent'], h)
            target = self.model.normalize(label, mask)
            self.assertEqual(tuple(target.shape), (4, self.cfg.max_phases, 64, 3))
            context, _ = self.model.encode(batch)
            y, t, velocity, _ = cfm_pair(target, mask, torch.Generator().manual_seed(1))
            prediction = self.model(y, t, context, h)
            self.assertEqual(prediction.shape, velocity.shape)
            self.assertEqual(float((prediction*(1-mask)).abs().max()), 0.)
            generated = self.model.rollout(context, h, 4, generator=torch.Generator().manual_seed(2))
            self.assertEqual(tuple(generated.shape), (4, self.cfg.max_phases, 64, 3))
            self.assertTrue(bool(torch.isfinite(generated).all()))
            dense = decoded_model_samples(self.model, generated, h, self.cfg, batch['goal'])
            dense_mask, _ = latent_mask(h, self.cfg)
            self.assertEqual(tuple(dense.shape), (4, self.cfg.max_phases, self.cfg.intervals+1, 3))
            self.assertTrue(bool(torch.isfinite(dense).all()))
            self.assertEqual(float((dense*(1-dense_mask.to(dense))).abs().max()), 0.)

    def test_gt_round_trip_bspline64(self):
        """GT dense controls -> 64 coefficients -> dense: grid points and endpoint stay within mm."""
        from k4_flow_solver.numerical import decode, integrate, validate, wrap
        from k4_flow_solver.output_projection import project_output
        from k4_flow_solver.prepare import load_scene
        from k4_flow_solver.inference import goal_from_scene
        import copy
        model = copy.deepcopy(self.model).double()
        worst = dict(point=0., endpoint=0., yaw=0., steer=0., speed=0.)
        ratios = []
        for row in self.rows.samples[:8]:
            a = np.asarray(row['latent'], np.float64)
            h = int(row['structure'])
            hh = torch.tensor([h])
            coefficients, _ = model.fit_targets(torch.as_tensor(a)[None], hh)
            rebuilt = model.dense_latent(coefficients.double(), hh, project=False)[0].numpy()
            gt, back = decode(a, h, self.cfg), decode(rebuilt, h, self.cfg)
            end_gt, phases_gt, _ = integrate(gt, cfg=self.cfg)
            end_back, phases_back, _ = integrate(back, cfg=self.cfg)
            worst['point'] = max(worst['point'], max(np.linalg.norm(p[:, :2]-q[:, :2], axis=1).max()
                                                     for p, q in zip(phases_gt, phases_back)))
            worst['endpoint'] = max(worst['endpoint'], float(np.linalg.norm(end_gt[:2]-end_back[:2])))
            worst['yaw'] = max(worst['yaw'], float(abs(wrap(end_gt[2]-end_back[2]))))
            worst['steer'] = max(worst['steer'], float(np.abs(gt.delta-back.delta).max()))
            worst['speed'] = max(worst['speed'], float(np.abs(np.sqrt(gt.b)-np.sqrt(back.b)).max()))
            # The inference decoding path (goal projection) keeps the GT verified at the GT cost.
            scene = load_scene(F.SCENES/row['uids'])
            goal = goal_from_scene(scene)
            projected = project_output(model.dense_latent(coefficients.double(), hh, project=False), hh,
                                       torch.as_tensor(goal)[None], self.cfg, iterations=2)[0].numpy()
            check = validate(decode(projected, h, self.cfg), goal, scene, self.cfg)
            self.assertTrue(check['verified_feasible'], row['uids'])
            ratios.append(check['J']/validate(gt, goal, scene, self.cfg)['J'])
        self.assertLess(worst['point'], .02)
        self.assertLess(worst['endpoint'], .02)
        self.assertLess(worst['yaw'], np.deg2rad(.5))
        self.assertLess(worst['steer'], .02)
        self.assertLess(worst['speed'], .01)
        self.assertLess(max(abs(r-1) for r in ratios), .05)

    def test_solver_top_k_candidates_in_one_batch(self):
        from k4_flow_solver.noise_flow_solver import NoiseFlowSolver, FAILURE_CLASSES
        from k4_flow_solver.prepare import load_scene
        state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
        path = os.path.join(self.tmp, 'noise.pt')
        torch.save(dict(schema='k4-numeric-flow-checkpoint-v1', config=self.cfg.to_dict(), normalization=self.normalization,
                        mode='flow', architecture='noise_scene_flow', representation=self.representation,
                        ema=state, model=state, epoch=0, dataset_sha256=self.init['dataset_sha256']), path)
        solver = NoiseFlowSolver(path, device='cpu')
        scene = load_scene(F.SCENES/self.rows[0]['uids'])
        trajectory, report = solver.plan(scene, candidates=6, top_k=3, steps=2, seed=7)
        self.assertEqual(len(report['ranked_structures']), 3)
        self.assertEqual(report['candidates'], 6)
        rows = report['candidate_rows']
        self.assertEqual(Counter(r['structure'] for r in rows), Counter({h: 2 for h in report['ranked_structures']}))
        self.assertEqual({r['structure_rank'] for r in rows}, {0, 1, 2})
        self.assertTrue(set(report['failure_counts']) <= set(FAILURE_CLASSES))
        self.assertEqual(report['verified_candidates']+sum(report['failure_counts'].values()), 6)
        for key in ('input_encoding_s', 'generation_s', 'postprocess_validation_s', 'total_s'):
            self.assertGreaterEqual(report['timings'][key], 0.)
        if report['status'] == 'valid_trajectory':
            verified = [r['J'] for r in rows if r['verified_feasible']]
            self.assertEqual(report['selected_J'], min(verified))
            self.assertIsNotNone(trajectory)
        else:
            self.assertEqual(report['status'], 'stop_and_replan')
            self.assertIsNone(trajectory)
        _, again = solver.plan(scene, candidates=6, top_k=3, steps=2, seed=7)
        self.assertEqual([r['J'] for r in again['candidate_rows']], [r['J'] for r in rows])
        solver.close()

    def test_overfit_smoke(self):
        """A few hundred GT samples: FM loss on those samples drops (fixed evaluation noise)."""
        import copy
        from k4_flow_solver.data import collate
        from k4_flow_solver.model import channel_mse
        from k4_flow_solver.noise_flow import FilteredSolverDataset
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        count = 256 if device.type == 'cuda' else 64
        rows = FilteredSolverDataset(F.DATASET, 'train', limit=count)
        model = copy.deepcopy(self.model).to(device).train()
        batches = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in collate(rows.samples[i:i+16]).items()}
                   for i in range(0, count, 16)]
        fixed = [(torch.Generator(device=device).manual_seed(i)) for i in range(len(batches))]

        def evaluate():
            model.eval()
            losses, correct = [], 0
            with torch.no_grad():
                for batch, generator in zip(batches, fixed):
                    generator.manual_seed(1000)
                    h = batch['structure'].long()
                    label, mask = model.fit_targets(batch['latent'], h)
                    target = model.normalize(label, mask)
                    context, logits = model.encode(batch)
                    y, t, velocity, _ = cfm_pair(target, mask, generator)
                    losses.append(float(channel_mse(model(y, t, context, h), velocity, mask)))
                    correct += int((logits.argmax(-1) == h).sum())
            model.train()
            return float(np.mean(losses)), correct/count

        before, accuracy_before = evaluate()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.)
        for _ in range(8 if device.type == 'cuda' else 6):
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
        after, accuracy_after = evaluate()
        print(json.dumps(dict(overfit_samples=count, device=str(device), FM_before=before, FM_after=after,
                              structure_accuracy_before=accuracy_before, structure_accuracy_after=accuracy_after)))
        self.assertTrue(np.isfinite(after))
        self.assertLess(after, .8*before)
        self.assertGreaterEqual(accuracy_after, accuracy_before)


if __name__ == '__main__':
    unittest.main()
