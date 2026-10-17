"""In-memory pilot dataset; numeric condition padding is applied per batch."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import IterableDataset


class SolverDataset:
    def __init__(self, root, split, limit=None, image_cache=None):
        self.root = Path(root)
        self.image_cache, self.split = image_cache, split
        self.manifest = json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        if self.manifest['state'] != 'complete':
            raise ValueError('dataset preparation is incomplete')
        self.samples = []
        for shard in self.manifest['shards']:
            if shard['split'] != split:
                continue
            with np.load(self.root/shard['file'], allow_pickle=False) as z:
                arrays = {k:z[k] for k in z.files}
            for i in range(shard['count']):
                sample = {}
                for k, a in arrays.items():
                    if k in ('scene_tokens_mask', 'k4_tokens_mask'):
                        continue
                    value = a[i]
                    if k in ('scene_tokens', 'k4_tokens'):
                        value = value[arrays[k+'_mask'][i]].copy()
                    elif k in ('latent', 'goal', 'target_J'):
                        value = np.asarray(value, dtype=np.float32)
                    else:
                        value = np.array(value, copy=True)
                    sample[k] = str(value) if k == 'uids' else value
                self.samples.append(sample)
                if limit is not None and len(self.samples) >= limit:
                    return

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        if self.image_cache is not None:
            sample = dict(sample, **self.image_cache.sample(sample['uids'], sample['scene_start'], self.split))
        return sample


class StreamingSolverDataset(IterableDataset):
    """Bounded-memory traversal: every selected scene once per epoch.

    Shuffle shard order and rows within each shard. Intended for num_workers=0
    so batching never loses per-worker tails and epoch length stays exact.
    """
    def __init__(self, root, split, limit=None, *, shuffle=False, seed=20260917, include_sdf=True, image_cache=None):
        super().__init__()
        self.root=Path(root)
        self.image_cache,self.split=image_cache,split
        self.manifest=json.loads((self.root/'manifest.json').read_text(encoding='utf-8'))
        if self.manifest['state']!='complete': raise ValueError('dataset preparation is incomplete')
        if limit is not None and limit<1: raise ValueError('positive scene limit required')
        self.shards=[]; self.count=0
        for shard in self.manifest['shards']:
            if shard['split']!=split: continue
            size=shard['count'] if limit is None else min(shard['count'],limit-self.count)
            if size<=0: break
            self.shards.append((shard['file'],size)); self.count+=size
        if not self.count: raise ValueError('empty split')
        self.shuffle,self.seed,self.epoch,self.include_sdf=shuffle,seed,0,include_sdf

    def __len__(self): return self.count

    def __iter__(self):
        if torch.utils.data.get_worker_info() is not None:
            raise ValueError('StreamingSolverDataset requires num_workers=0')
        rng=np.random.default_rng(self.seed+self.epoch)
        order=rng.permutation(len(self.shards)) if self.shuffle else range(len(self.shards))
        for shard_index in order:
            file,size=self.shards[shard_index]
            with np.load(self.root/file,allow_pickle=False) as z:
                arrays={k:z[k] for k in z.files if self.include_sdf or k!='sdf'}
            indices=rng.permutation(size) if self.shuffle else range(size)
            for i in indices:
                sample={}
                for key,array in arrays.items():
                    if key in ('scene_tokens_mask','k4_tokens_mask'): continue
                    value=array[i]
                    if key in ('scene_tokens','k4_tokens'): value=value[arrays[key+'_mask'][i]]
                    elif key in ('latent','goal','target_J'): value=np.asarray(value,dtype=np.float32)
                    sample[key]=str(value) if key=='uids' else value
                if self.image_cache is not None:
                    sample.update(self.image_cache.sample(sample['uids'], sample['scene_start'], self.split))
                yield sample
            del arrays


def collate(samples):
    result = {}
    for key in samples[0]:
        if key == 'uids':
            result[key] = [s[key] for s in samples]
        elif key in ('scene_tokens', 'k4_tokens'):
            count = max(len(s[key]) for s in samples)
            value = np.zeros((len(samples), count, samples[0][key].shape[-1]), np.float32)
            mask = np.zeros((len(samples), count), bool)
            for i, s in enumerate(samples):
                value[i, :len(s[key])] = s[key]
                mask[i, :len(s[key])] = True
            result[key] = torch.from_numpy(value)
            result[key+'_mask'] = torch.from_numpy(mask)
        else:
            result[key] = torch.from_numpy(np.stack([s[key] for s in samples]))
    return result


def to_device(batch, device):
    return {k:v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in batch.items()}
