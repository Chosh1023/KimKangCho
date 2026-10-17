"""Resumable numeric-scene/K4/teacher overlay, preserving source splits."""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np

from .config import SolverConfig
from .conditions import scene_tokens, path_tokens, task_state, scene_sdf
from .numerical import from_teacher, encode, decode, structure_id, repair, validate, cost

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def replace_with_retry(source, destination):
    """Windows readers/sync clients can briefly deny atomic replacement."""
    for attempt in range(12):
        try:
            os.replace(source, destination)
            return
        except PermissionError:
            if attempt == 11:
                raise
            time.sleep(min(.05 * 2 ** attempt, .5))


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    replace_with_retry(temp, path)


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            sha.update(block)
    return sha.hexdigest()


def rows(path):
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def load_scene(source):
    with np.load(Path(source)/'planner_input.npz', allow_pickle=False) as z:
        # known_feasible_route and goal_state are deliberately not read.
        return {k:z[k] for k in ('obstacle_physical', 'start_state', 'requested_goal_pose',
                                 'parking_lot_polygon_xy', 'target_slot_polygon_xy')}


def initialize_worker():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def prepare_one(task):
    row, cache_file, teacher_root, output, cfg_dict = task
    cfg = SolverConfig(**cfg_dict)
    uid = row['uid']
    directory = Path(output)/'samples'/uid[:3]
    directory.mkdir(parents=True, exist_ok=True)
    report = directory/(uid+'.json')
    target = directory/(uid+'.npz')
    if report.is_file():
        old = read(report)
        if not old.get('success') or target.is_file():
            return old
    result = dict(uid=uid, split=row['processed_split'], success=False,
                  source_sample_dir=row['sample_dir'], label_origin='gt_warm_start',
                  source_input_sha256=row.get('input_sha256'), source_gt_sha256=row.get('gt_sha256'))
    tick = time.perf_counter()
    try:
        scene = load_scene(row['sample_dir'])
        if abs(float(scene['start_state'][4])) > 1e-7:
            raise ValueError('nonzero_start_steering_not_in_teacher_contract')
        with np.load(cache_file, allow_pickle=False) as a:
            k4path = a['path'].copy()
        teacher_dir = Path(teacher_root)/'samples'/uid[:3]/uid
        original = read(teacher_dir/'result.json')
        if original['split'] != row['processed_split'] or not original['success']:
            raise ValueError('teacher_provenance_mismatch')
        controls, goal = from_teacher(teacher_dir/'trajectory.npz', cfg)
        h = structure_id(controls.gears, cfg)
        # Check the actual inverse-transformed target, including numerical clips.
        latent, _ = encode(controls, cfg)
        controls = decode(latent, h, cfg)
        controls, correction = repair(controls, goal, cfg, time_budget_s=1.0)
        checks = validate(controls, goal, scene, cfg)
        result.update(teacher_J=float(original['best']['objective']), target_J=checks['J'],
                      structure=h, gears=controls.gears.tolist(), correction=correction, validation=checks)
        if not checks['verified_feasible']:
            raise ValueError('converted_teacher_failed_validation')
        task_vector = task_state(scene, cfg)
        if np.linalg.norm(task_vector[:2]*cfg.coordinate_scale-goal[:2]) > 2e-5:
            raise ValueError('goal_coordinate_mismatch')
        numeric_scene = scene_tokens(scene, cfg)
        k4 = path_tokens(k4path, cfg)
        latent, mask = encode(controls, cfg)
        temporary = target.with_suffix('.tmp.npz')
        np.savez_compressed(temporary, latent=latent, latent_mask=mask, structure=np.int64(h),
            scene_tokens=numeric_scene, k4_tokens=k4, task=task_vector, goal=goal,
            sdf=scene_sdf(scene).astype(np.float16), scene_start=scene['start_state'].astype(np.float32),
            target_J=np.float64(checks['J']), lengths=controls.lengths,
            delta=controls.delta, b=controls.b, gears=controls.gears)
        os.replace(temporary, target)
        result.update(success=True, scene_token_count=len(numeric_scene), k4_token_count=len(k4),
                      sample=str(target.relative_to(Path(output))))
    except Exception as exc:
        result.update(reason=str(exc), error_type=type(exc).__name__, traceback=traceback.format_exc())
    result['seconds'] = time.perf_counter()-tick
    dump(report, result)
    return result


def pack_dataset(output, reports, cfg, provenance):
    output = Path(output)
    good = [r for r in reports if r['success']]
    counts = Counter(r['split'] for r in good)
    sums, squares, n = np.zeros(3), np.zeros(3), np.zeros(3)
    shards = []
    for split in ('train', 'validation', 'test'):
        selected = [r for r in good if r['split'] == split]
        for begin in range(0, len(selected), 128):
            group = selected[begin:begin+128]
            samples = []
            for row in group:
                with np.load(output/row['sample'], allow_pickle=False) as z:
                    samples.append({k:z[k] for k in ('latent', 'latent_mask', 'structure', 'scene_tokens',
                        'k4_tokens', 'task', 'goal', 'sdf', 'scene_start', 'target_J')})
            arrays = {}
            for key in samples[0]:
                if key in ('scene_tokens', 'k4_tokens'):
                    size = max(len(a[key]) for a in samples)
                    arrays[key] = np.zeros((len(samples), size, samples[0][key].shape[-1]), np.float32)
                    arrays[key+'_mask'] = np.zeros((len(samples), size), bool)
                    for i, a in enumerate(samples):
                        arrays[key][i, :len(a[key])] = a[key]
                        arrays[key+'_mask'][i, :len(a[key])] = True
                else:
                    arrays[key] = np.stack([a[key] for a in samples])
            if split == 'train':
                v, m = arrays['latent'], arrays['latent_mask']
                sums += np.where(m, v, 0).sum(axis=(0, 1, 2))
                squares += np.where(m, v*v, 0).sum(axis=(0, 1, 2))
                n += m.sum(axis=(0, 1, 2))
            arrays['uids'] = np.asarray([r['uid'] for r in group])
            file = f'{split}_{begin//128:04d}.npz'
            np.savez_compressed(output/file, **arrays)
            shards.append(dict(file=file, split=split, count=len(group)))
    if not counts['train'] or not counts['validation']:
        raise RuntimeError('both train and validation targets are required')
    mean = sums/np.maximum(n, 1)
    std = np.sqrt(np.maximum(squares/np.maximum(n, 1)-mean*mean, 1e-4))
    manifest = dict(schema='k4-numeric-flow-dataset-v1', config=cfg.to_dict(), shards=shards,
        counts=dict(counts), attempted=len(reports), rejected=len(reports)-len(good),
        failures=dict(Counter(r.get('reason') for r in reports if not r['success'])),
        normalization=dict(mean=mean.tolist(), std=std.tolist(), source_split='train', counts=n.tolist()),
        provenance=provenance, model_inputs=['scene_tokens', 'k4_tokens', 'task'],
        scene_representation='numeric lossless occupied-cell rectangles and boundary/slot edges; no image encoder',
        label_scope='fixed-GT-structure local OCP teacher; K4 topology may differ',
        state='complete')
    dump(output/'manifest.json', manifest)
    return manifest


def select_rows(teacher_root, train_count, val_count, seed):
    accepted = {r['uid'] for r in rows(teacher_root/'accepted.jsonl')}
    grouped = defaultdict(list)
    for row in rows(teacher_root/'source_selection.jsonl'):
        if row['uid'] in accepted and row['processed_split'] in ('train', 'validation'):
            grouped[(row['processed_split'], row['processed_shard'])].append(row)
    rng = np.random.default_rng(seed)
    selection = []
    for split, limit in (('train', train_count), ('validation', val_count)):
        keys = sorted(k for k in grouped if k[0] == split)
        rng.shuffle(keys)
        remaining = limit
        for key in keys:
            group = grouped[key]
            rng.shuffle(group)
            use = group[:remaining] if remaining is not None else group
            selection.extend(use)
            if remaining is not None:
                remaining -= len(use)
                if not remaining:
                    break
    return selection


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--teacher-root', type=Path, default=ROOT/'data/k4_fast_ocp_gt_v2_20260916')
    p.add_argument('--k4-dataset', type=Path, default=ROOT/'data/town05_town04_physx_combined_dense_v4_hardcase_v2')
    p.add_argument('--train-count', type=int, default=6000)
    p.add_argument('--validation-count', type=int, default=512)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--seed', type=int, default=20260917)
    p.add_argument('--k4-cache', type=Path)
    p.add_argument('--device', default='cuda')
    args = p.parse_args()
    import torch
    from torch.utils.data._utils.collate import default_collate
    from hope_path_transformer.dataset import ProcessedShardDataset, processed_manifest_entries
    from materialize_supervised_k8_carla_eval_paths import resolve_shard
    from .k4 import FrozenK4
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    cfg = SolverConfig()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    k4 = FrozenK4(device=args.device, workers=args.workers)
    provenance = dict(teacher_root=str(args.teacher_root.resolve()), k4_dataset=str(args.k4_dataset.resolve()),
        checkpoint=str(k4.checkpoint_path), checkpoint_sha256=digest(k4.checkpoint_path), profile=k4.profile,
        teacher_source_hashes=read(args.teacher_root/'source_hashes.json'),
        k4_manifest_sha256=digest(args.k4_dataset/'manifest.json'), seed=args.seed,
        train_count=args.train_count, validation_count=args.validation_count, config=cfg.to_dict(),
        selection='seeded shuffle of processed shards then rows; source splits preserved; test untouched by training',
        preparation_sources={name:digest(Path(__file__).parent/name) for name in
                             ('prepare.py','config.py','conditions.py','numerical.py','k4.py')})
    contract = output/'preparation_contract.json'
    if contract.is_file() and read(contract) != provenance:
        raise RuntimeError('resume contract mismatch; choose a new output directory')
    dump(contract, provenance)
    selection_path = output/'selection.jsonl'
    if selection_path.is_file():
        selection = list(rows(selection_path))
    else:
        selection = select_rows(args.teacher_root, args.train_count, args.validation_count, args.seed)
        selection_path.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selection), encoding='utf-8')
    desired = defaultdict(dict)
    for row in selection:
        desired[row['processed_shard']][row['uid']] = row
    entries = processed_manifest_entries(args.k4_dataset)
    entries_by_stem = {resolve_shard(args.k4_dataset, e).stem:e for e in entries}
    started, reports, futures = time.perf_counter(), [], []
    cache_root = args.k4_cache.resolve() if args.k4_cache else output/'k4_cache'
    cache_root.mkdir(parents=True, exist_ok=True)
    # Reused caches must belong to the exact same frozen checkpoint/profile.
    cache_contract = dict(checkpoint_sha256=provenance['checkpoint_sha256'], profile=k4.profile,
                          k4_manifest_sha256=provenance['k4_manifest_sha256'])
    cache_contract_path = cache_root/'contract.json'
    if cache_contract_path.is_file() and read(cache_contract_path) != cache_contract:
        raise RuntimeError('K4 cache contract mismatch')
    if not cache_contract_path.is_file() and any(cache_root.glob('*.npz')):
        prior = read(cache_root.parent/'preparation_contract.json')
        if any(prior[k] != v for k, v in cache_contract.items()):
            raise RuntimeError('unverified legacy K4 cache')
    dump(cache_contract_path, cache_contract)
    cached_count = sum(1 for _ in cache_root.glob('*.npz'))
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize_worker) as pool:
        for stem, wanted in desired.items():
            path = resolve_shard(args.k4_dataset, entries_by_stem[stem])
            dataset = ProcessedShardDataset(path, k4.config, mirror=False)
            indices = [i for i, uid in enumerate(dataset.arrays['source_uid']) if str(uid) in wanted]
            if len(indices) != len(wanted):
                raise RuntimeError('processed UID join incomplete')
            for begin in range(0, len(indices), args.batch_size):
                group = indices[begin:begin+args.batch_size]
                missing = [i for i in group if not (cache_root/(str(dataset.arrays['source_uid'][i])+'.npz')).is_file()]
                if missing:
                    scenes = [load_scene(wanted[str(dataset.arrays['source_uid'][i])]['sample_dir']) for i in missing]
                    batch = default_collate([dataset[i] for i in missing])
                    for i, (trajectory, chosen, success) in zip(missing, k4.predict(batch, scenes)):
                        uid = str(dataset.arrays['source_uid'][i])
                        temp = cache_root/(uid+'.tmp.npz')
                        np.savez_compressed(temp, path=trajectory, selected=chosen, k4_runtime_success=success)
                        os.replace(temp, cache_root/(uid+'.npz'))
                        cached_count += 1
                for i in group:
                    uid = str(dataset.arrays['source_uid'][i])
                    futures.append(pool.submit(prepare_one, (wanted[uid], str(cache_root/(uid+'.npz')),
                        str(args.teacher_root), str(output), cfg.to_dict())))
                ready = [f for f in futures if f.done()]
                if len(futures) >= max(128, args.workers*32) and not ready:
                    ready = list(wait(futures, return_when=FIRST_COMPLETED).done)
                for f in ready:
                    reports.append(f.result())
                    futures.remove(f)
                if begin % (args.batch_size*4) == 0:
                    status = dict(stage='preparing', shard=stem, cached=cached_count,
                        completed=len(reports), accepted=sum(r['success'] for r in reports),
                        pending=len(futures), requested=len(selection), elapsed_s=time.perf_counter()-started)
                    dump(output/'progress.json', status)
                    print(json.dumps(status), flush=True)
            del dataset
        k4.close()
        for i, future in enumerate(futures):
            reports.append(future.result())
            if i % 32 == 0:
                status = dict(stage='validating', completed=len(reports), accepted=sum(r['success'] for r in reports),
                              requested=len(selection), elapsed_s=time.perf_counter()-started)
                dump(output/'progress.json', status)
                print(json.dumps(status), flush=True)
    (output/'sample_reports.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in reports), encoding='utf-8')
    dump(output/'progress.json', dict(stage='packing', completed=len(reports), requested=len(selection),
                                     elapsed_s=time.perf_counter()-started))
    manifest = pack_dataset(output, reports, cfg, provenance)
    dump(output/'progress.json', dict(stage='complete', counts=manifest['counts'], rejected=manifest['rejected'], elapsed_s=time.perf_counter()-started))
    print(json.dumps({'counts':manifest['counts'], 'rejected':manifest['rejected'], 'failures':manifest['failures']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
