"""Train the scene/task-conditioned noise -> OCP GT Flow (standard CFM, no warm start).

x0 ~ N(0, I) in the normalized output space, x1 = original OCP GT fitted to
the output representation, x_t = (1 - t) x0 + t x1, target x1 - x0 (MSE),
plus structure CE (teacher-forced GT structure) and the physical auxiliaries
of the warm-start run (from epoch 3, every 4 steps, 4 samples, 16-step
rollout from pure noise, endpoint 0.2 / collision 0.05 / limits 0.01 / J
cost_weight, ramped).  Initialization: shape-compatible epoch-10 EMA tensors.

Per epoch: validation FM loss and structure accuracy on validation scenes
outside the 500 evaluation scenes, and sample/scene success rate and mean J
on 16 of those scenes (full inference: top-K structures, projection,
integration, validation).

  python scripts/train_noise_flow.py --output run-logs/noise_flow_gt_bspline64_v1
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(key, '1')
import argparse
import copy
import json
import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from k4_flow_solver.prepare import dump, digest, replace_with_retry, load_scene
from k4_flow_solver.config import SolverConfig
from k4_flow_solver.data import StreamingSolverDataset, collate, to_device
from k4_flow_solver.model import channel_mse, latent_mask
from k4_flow_solver.physics import auxiliary_losses
from k4_flow_solver.inference import goal_from_scene, choose_valid
from k4_flow_solver.sampling import sample_candidates
from k4_flow_solver.noise_flow import (make_noise_model, noise_representation, load_epoch10_features, cfm_pair,
                                       top_k_schedule, FilteredSolverDataset, OUTPUT_REPRS, GRID_KNOTS,
                                       load_init_features, decode_samples)
from k4_flow_solver.noise_flow_solver import _validate, classify


@torch.no_grad()
def fm_diagnostics(model, loader, device):
    model.eval()
    losses, correct, count = [], 0, 0
    generator = torch.Generator(device=device).manual_seed(99173)
    for raw in loader:
        batch = to_device(raw, device)
        h = batch['structure'].long()
        label, mask = model.fit_targets(batch['latent'], h)
        target = model.normalize(label, mask)
        context, logits = model.encode(batch)
        y, t, velocity, _ = cfm_pair(target, mask, generator)
        losses.append(float(channel_mse(model(y, t, context, h), velocity, mask)))
        correct += int((logits.argmax(-1) == h).sum())
        count += len(h)
    return dict(val_FM_loss=float(np.mean(losses)), structure_accuracy=correct/max(count, 1), val_samples=count)


@torch.no_grad()
def panel(model, rows, scenes, pool, *, candidates, top_k, steps, seed=20260917):
    """Full inference on fixed validation scenes: sample and scene success, mean J."""
    model.eval()
    device = next(model.parameters()).device
    cfg = model.cfg
    verified = total = 0
    scene_ok, js, ratios, structure_hit = 0, [], [], 0
    failures = {}
    for i, row in enumerate(rows):
        scene = load_scene(scenes/row['uids'])
        goal = goal_from_scene(scene)
        batch = to_device(collate([{k: row[k] for k in ('scene_tokens', 'task')}]), device)
        context, logits = model.encode(batch)
        ranked = logits.topk(top_k, dim=-1).indices[0].cpu().tolist()
        structure_hit += int(int(row['structure']) in ranked)
        schedule = top_k_schedule(ranked, candidates, seed+i)
        y = sample_candidates(model, context, schedule, steps, batch_size=len(schedule))
        hs = torch.tensor([c['structure'] for c in schedule], device=device)
        latent = decode_samples(model, y, hs, cfg, torch.as_tensor(goal, device=device, dtype=torch.float32)
                                       .expand(len(hs), -1)).double().cpu().numpy()
        results = list(pool.map(_validate, [(a, c['structure'], goal, scene, cfg.to_dict()) for a, c in zip(latent, schedule)]))
        rowsc = [dict(J=r[0].get('J'), verified_feasible=bool(r[0].get('verified_feasible')), validation=r[0],
                      failure=classify(r[0], r[1])) for r in results]
        for r in rowsc:
            if r['failure']:
                failures[r['failure']] = failures.get(r['failure'], 0)+1
        verified += sum(r['verified_feasible'] for r in rowsc)
        total += len(rowsc)
        best = choose_valid(rowsc)
        if best is not None:
            scene_ok += 1
            js.append(best['J'])
            ratios.append(best['J']/float(row['target_J']))
    return dict(panel_scenes=len(rows), sample_success_rate=verified/max(total, 1), scene_success_rate=scene_ok/max(len(rows), 1),
                mean_J=float(np.mean(js)) if js else None, mean_J_over_gt=float(np.mean(ratios)) if ratios else None,
                gt_structure_in_topk=structure_hit/max(len(rows), 1), candidate_failures=failures)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', type=Path, default=Path('data/k4_numeric_flow_full_v1_20260917'))
    p.add_argument('--scenes', type=Path, default=Path('data/flow_scenes'))
    p.add_argument('--eval-uids', type=Path, default=Path('experiments/eval_uids.json'),
                   help='held-out evaluation scenes; excluded from validation/selection')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--output-repr', default='bspline64', choices=OUTPUT_REPRS)
    p.add_argument('--init', type=Path, help='default: epoch_10 (bspline64, grid64, grid128), whole epoch_35 (grid513)')
    p.add_argument('--normalization', type=Path, default=Path('experiments/noise_flow_grid_normalization.json'),
                   help='grid representations: per-representation train statistics (tools/compute_grid_normalization.py)')
    p.add_argument('--smooth-weight', type=float, default=0.0,
                   help='weight of the mean squared adjacent difference of the decoded 513-point steering (physics steps)')
    p.add_argument('--no-init', action='store_true')
    p.add_argument('--resume', type=Path)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--learning-rate', type=float, default=5e-5)
    p.add_argument('--steps', type=int, default=16)
    p.add_argument('--train-limit', type=int)
    p.add_argument('--validation-limit', type=int, default=128)
    p.add_argument('--panel-scenes', type=int, default=16)
    p.add_argument('--panel-candidates', type=int, default=16)
    p.add_argument('--panel-top-k', type=int, default=3)
    p.add_argument('--panel-workers', type=int, default=8)
    p.add_argument('--physical-start', type=int, default=3)
    p.add_argument('--physical-every', type=int, default=4)
    p.add_argument('--physical-batch', type=int, default=4)
    p.add_argument('--cost-weight', type=float, default=.05)
    p.add_argument('--projection-iterations', type=int, default=2)
    p.add_argument('--save-every', type=int, default=5)
    p.add_argument('--seed', type=int, default=20260929)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    if args.init is None:
        args.init = Path('run-logs/k4_whole_flow_direct_j_scratch_v1_20260919/epoch_35.pt' if args.output_repr == 'grid513'
                         else 'run-logs/k4_spline_projected_j_v1_20260920/epoch_10.pt')
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'latest.pt').exists() and not args.resume:
        raise ValueError('existing training requires resume')
    held_out = set(json.loads(args.eval_uids.read_text(encoding='utf-8'))['uids'])
    train = StreamingSolverDataset(args.dataset, 'train', args.train_limit, shuffle=True, seed=args.seed)
    val = FilteredSolverDataset(args.dataset, 'validation', exclude=held_out, limit=args.validation_limit)
    panel_pool_rows = FilteredSolverDataset(args.dataset, 'validation', exclude=held_out)
    order = np.random.default_rng(20260917).permutation(len(panel_pool_rows))[:args.panel_scenes]
    panel_rows = [panel_pool_rows[int(i)] for i in order]
    del panel_pool_rows
    manifest = train.manifest
    data_hash = digest(args.dataset/'manifest.json')
    cfg = replace(SolverConfig(**manifest['config']), terminal_position_tolerance_m=.1, terminal_yaw_tolerance_rad=math.radians(5))
    representation = noise_representation(args.output_repr, args.projection_iterations)
    if args.output_repr in GRID_KNOTS:
        stats = json.loads(args.normalization.read_text(encoding='utf-8'))
        if stats['dataset_sha256'] != data_hash:
            raise ValueError('grid normalization was computed from a different dataset')
        normalization = stats['representations'][args.output_repr]
    else:
        normalization = manifest['normalization']
    model = make_noise_model(cfg, normalization, representation).to(device)
    init_report = None
    start_epoch, best_score = 0, None
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if state['representation'] != representation or state['config'] != cfg.to_dict():
            raise ValueError('resume contract mismatch')
        model.load_state_dict(state['model'])
    elif not args.no_init and args.output_repr == 'bspline64':
        init_report = load_epoch10_features(model, args.init)
        if init_report['dataset_sha256'] != data_hash:
            raise ValueError('initializer dataset mismatch')
        if init_report['normalization']['mean'] != manifest['normalization']['mean']:
            raise ValueError('initializer normalization differs from the dataset statistics')
    elif not args.no_init:
        init_report = load_init_features(model, args.init)
        if init_report['dataset_sha256'] != data_hash:
            raise ValueError('initializer dataset mismatch')
        source = init_report['source_representation'] or {}
        init_report['note'] = (
            f"{init_report['architecture']} epoch {init_report['epoch']} -> {args.output_repr}: shape-compatible tensors only; "
            'normalization buffers kept from the grid statistics. '
            + ('The source outputs 64 B-spline coefficients per phase; grid knot values have a different meaning.'
               if source.get('schema') == 'physical-bspline-v1' else
               'The source outputs the same 513-grid latent (log length, atanh steering, logit v^2).'))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    if args.resume:
        optimizer.load_state_dict(state['optimizer'])
        ema.load_state_dict(state['ema'])
        start_epoch, best_score = state['epoch'], state['best_score']
    loader = DataLoader(train, batch_size=args.batch_size, num_workers=0, collate_fn=collate, pin_memory=device.type == 'cuda')
    validation = DataLoader(val, batch_size=args.batch_size, num_workers=0, collate_fn=collate)
    contract = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    train_samples=len(train), validation_samples=len(val), panel_uids=[r['uids'] for r in panel_rows],
                    held_out_excluded=len(held_out), dataset_sha256=data_hash, representation=representation,
                    model_inputs=['numeric scene tokens', 'task (start/goal)'], source='N(0, I)',
                    initialization=(None if init_report is None else dict(
                        checkpoint=str(args.init), loaded=len(init_report['loaded']),
                        skipped_from_checkpoint=init_report['skipped_from_checkpoint'],
                        missing_in_checkpoint=init_report['missing_in_checkpoint'], note=init_report.get('note'))),
                    normalization=dict(normalization, source=('dataset manifest' if args.output_repr == 'bspline64'
                                                              else str(args.normalization))),
                    checkpoint_selection='best.pt: highest panel scene success, then lowest mean J/GT',
                    source_hashes={q.name: digest(q) for q in [Path(__file__), *Path('scripts/k4_flow_solver').glob('noise_flow*.py')]})
    dump(args.output/'run_config.json', contract)
    if init_report is not None:
        (args.output/'init_keys.json').write_text(json.dumps(dict(loaded=init_report['loaded'],
            skipped_from_checkpoint=init_report['skipped_from_checkpoint'],
            missing_in_checkpoint=init_report['missing_in_checkpoint'], note=init_report.get('note')), indent=1),
            encoding='utf-8')
    snapshot = args.output/'source_snapshot'
    if not snapshot.exists():
        import shutil
        snapshot.mkdir()
        shutil.copy2(__file__, snapshot/Path(__file__).name)
        shutil.copytree(Path('scripts/k4_flow_solver'), snapshot/'k4_flow_solver', ignore=shutil.ignore_patterns('__pycache__'))

    def payload(epoch, metrics):
        return dict(schema='k4-numeric-flow-checkpoint-v1', model=model.state_dict(), ema=ema.state_dict(),
                    optimizer=optimizer.state_dict(), config=cfg.to_dict(), normalization=normalization,
                    mode='flow', architecture='noise_scene_flow', representation=representation, epoch=epoch,
                    dataset_sha256=data_hash, metrics=metrics, best_score=best_score, run_config=contract,
                    deployment_approved=False)

    def save(name, value):
        path = args.output/name
        temporary = path.with_suffix('.tmp.pt')
        torch.save(value, temporary)
        replace_with_retry(temporary, path)

    pool = ProcessPoolExecutor(args.panel_workers)
    started = time.perf_counter()
    global_step = start_epoch*len(loader)
    print(json.dumps(dict(event='training_start', train_samples=len(train), validation_samples=len(val),
                          init_loaded=None if init_report is None else len(init_report['loaded']),
                          init_skipped=None if init_report is None else init_report['skipped_from_checkpoint'])), flush=True)
    for epoch in range(start_epoch, args.epochs):
        train.epoch = epoch
        model.train()
        total, updates, physical, last, first_gradient = 0., 0, {}, 0., None
        epoch_start = time.perf_counter()
        for index, raw in enumerate(loader):
            batch = to_device(raw, device)
            h = batch['structure'].long()
            with torch.no_grad():
                label, mask = model.fit_targets(batch['latent'], h)
                target = model.normalize(label, mask)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                context, logits = model.encode(batch)
                y, t, velocity, _ = cfm_pair(target, mask)
                fitting = channel_mse(model(y, t, context, h), velocity, mask)
                loss = fitting+torch.nn.functional.cross_entropy(logits, h)
                if epoch+1 >= args.physical_start and global_step % args.physical_every == 0:
                    selected = torch.randperm(len(h), device=device)[:args.physical_batch]
                    small = {k: v[selected] for k, v in batch.items() if torch.is_tensor(v)}
                    hh = h[selected]
                    generated = model.rollout(context[selected], hh, args.steps)
                    with torch.autocast(device_type=device.type, enabled=False):
                        dense = decode_samples(model, generated.float(), hh, cfg, small['goal'])
                        auxiliaries, _ = auxiliary_losses(dense, hh, small, cfg, cost_objective='normalized_j')
                        if args.smooth_weight > 0:
                            steer = cfg.max_steer*torch.tanh(dense[..., 1])
                            _, active = latent_mask(hh, cfg, steer.dtype)
                            auxiliaries['smooth'] = (((torch.diff(steer, dim=-1)**2).mean(-1)*active).sum()
                                                     / active.sum().clamp_min(1))
                    ramp = min(1., (epoch+2-args.physical_start)/3)
                    loss = loss+ramp*(.2*auxiliaries['endpoint']+.05*auxiliaries['collision']+
                                      .01*auxiliaries['limits']+args.cost_weight*auxiliaries['cost'])
                    if args.smooth_weight > 0:
                        loss = loss+args.smooth_weight*auxiliaries['smooth']
                    if first_gradient is None:
                        gradients = torch.autograd.grad(auxiliaries['cost'], tuple(model.parameters()),
                                                        retain_graph=True, allow_unused=True)
                        first_gradient = float(torch.sqrt(sum(g.detach().float().square().sum() for g in gradients if g is not None)))
                        if not np.isfinite(first_gradient) or first_gradient <= 0:
                            raise FloatingPointError('projected J has no finite nonzero neural gradient')
                        print(json.dumps(dict(event='projected_J_gradient', epoch=epoch+1, norm=first_gradient)), flush=True)
                    updates += 1
                    for k, v in auxiliaries.items():
                        physical[k] = physical.get(k, 0.)+float(v.detach())
            if not torch.isfinite(loss):
                raise FloatingPointError(f'nonfinite training loss at epoch {epoch+1} batch {index+1}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            with torch.no_grad():
                for dst, src in zip(ema.parameters(), model.parameters()):
                    dst.lerp_(src, .01)
            total += float(loss.detach())
            global_step += 1
            if index == 0 or time.perf_counter()-last > 30:
                progress = dict(stage='training', epoch=epoch+1, epochs=args.epochs, batch=index+1, batches=len(loader),
                                loss=total/(index+1), grad_norm=float(norm), physical_updates=updates,
                                elapsed_s=time.perf_counter()-started,
                                cuda_peak_gb=torch.cuda.max_memory_allocated()/1e9 if device.type == 'cuda' else 0.)
                dump(args.output/'progress.json', progress)
                print(json.dumps(progress), flush=True)
                last = time.perf_counter()
        train_s = time.perf_counter()-epoch_start
        metrics = fm_diagnostics(ema, validation, device)
        metrics['panel'] = panel(ema, panel_rows, args.scenes, pool, candidates=args.panel_candidates,
                                 top_k=args.panel_top_k, steps=args.steps)
        score = [-metrics['panel']['scene_success_rate'], metrics['panel']['mean_J_over_gt'] or 1e12]
        if best_score is None or score < best_score:
            best_score = score
            save('best.pt', payload(epoch+1, metrics))
        if (epoch+1) % args.save_every == 0 or epoch+1 == args.epochs:
            save(f'epoch_{epoch+1:02d}.pt', payload(epoch+1, metrics))
        save('latest.pt', payload(epoch+1, metrics))
        report = dict(stage='epoch_complete', epoch=epoch+1, epochs=args.epochs, train_loss=total/len(loader),
                      validation=metrics, physics_mean={k: v/max(updates, 1) for k, v in physical.items()},
                      projected_J_gradient=first_gradient, physical_updates=updates, best_score=best_score,
                      train_s=train_s, mean_step_s=train_s/max(len(loader), 1), epoch_s=time.perf_counter()-epoch_start, elapsed_s=time.perf_counter()-started,
                      cuda_peak_gb=torch.cuda.max_memory_allocated()/1e9 if device.type == 'cuda' else 0.)
        with (args.output/'metrics.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(report, allow_nan=False)+'\n')
        dump(args.output/'progress.json', report)
        print(json.dumps(report), flush=True)
    pool.shutdown()
    dump(args.output/'completion.json', dict(state='complete', epochs=args.epochs, elapsed_s=time.perf_counter()-started,
                                              best_score=best_score))


if __name__ == '__main__':
    main()
