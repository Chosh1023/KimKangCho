"""Nested independent-noise candidates, reproducible across sample budgets."""
import hashlib
import torch


def candidate_schedule(structures, samples_per_structure, seed):
    if samples_per_structure < 1 or not structures:
        raise ValueError('empty candidate schedule')
    if len(set(structures)) != len(structures):
        raise ValueError('structure candidates must be unique')
    result = []
    for sample in range(samples_per_structure):
        for rank, h in enumerate(structures):
            key = f'k4-flow-v2/{int(seed)}/{int(h)}/{sample}'.encode('ascii')
            noise_seed = int.from_bytes(hashlib.sha256(key).digest()[:8],'little') % (2**63-1)
            result.append(dict(structure=int(h),structure_rank=rank,sample_index=sample,noise_seed=noise_seed))
    return result


def candidate_noise(schedule, cfg, device):
    shape = (cfg.max_phases,cfg.intervals+1,3)
    # CPU RNG fixes the same initial vectors for CPU/GPU and all sample counts.
    return torch.stack([torch.randn(shape,generator=torch.Generator().manual_seed(c['noise_seed'])) for c in schedule]).to(device)


def sample_candidates(model, context, schedule, steps, *, batch_size=8, regression=False):
    if batch_size < 1 or not schedule:
        raise ValueError('invalid generation budget')
    result = []
    for begin in range(0,len(schedule),batch_size):
        items = schedule[begin:begin+batch_size]
        size = len(items)
        # Padding keeps the first candidate's numerical batch shape unchanged
        # when comparing smaller/larger prefixes with the same batch_size.
        padded = items+[items[-1]]*(batch_size-size)
        h = torch.tensor([c['structure'] for c in padded],device=context.device)
        noise = None if regression else candidate_noise(padded,vars(model).get('sampling_config',model.cfg),context.device)
        result.append(model.rollout(context.expand(batch_size,-1,-1),h,steps,
            noise=noise,regression=regression)[:size])
    return torch.cat(result,dim=0)
