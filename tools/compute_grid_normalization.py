"""Per-channel normalization statistics of the noise-Flow grid representations.

Every train GT latent is converted with NoiseGridFlow.fit_targets (the exact
training conversion) and the masked channel mean / std are accumulated like the
dataset manifest (population std, variance floor 1e-4).  grid513 is the 513
grid itself, so its statistics are compared with the manifest as a check.

  python tools/compute_grid_normalization.py   -> experiments/noise_flow_grid_normalization.json
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from k4_flow_solver.config import SolverConfig  # noqa: E402
from k4_flow_solver.prepare import digest  # noqa: E402
from k4_flow_solver.noise_flow import GRID_KNOTS, NoiseGridFlow, noise_representation  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', type=Path, default=ROOT/'data/k4_numeric_flow_full_v1_20260917')
    p.add_argument('--output', type=Path, default=ROOT/'experiments/noise_flow_grid_normalization.json')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f'{args.output} exists; not overwriting')
    manifest = json.loads((args.dataset/'manifest.json').read_text(encoding='utf-8'))
    cfg = SolverConfig(**manifest['config'])
    models = {r: NoiseGridFlow(cfg, None, noise_representation(r)) for r in GRID_KNOTS}
    sums = {r: np.zeros(3) for r in GRID_KNOTS}
    squares = {r: np.zeros(3) for r in GRID_KNOTS}
    counts = {r: np.zeros(3) for r in GRID_KNOTS}
    samples = 0
    for shard in manifest['shards']:
        if shard['split'] != 'train':
            continue
        with np.load(args.dataset/shard['file'], allow_pickle=False) as z:
            latent, structure = torch.as_tensor(z['latent']), torch.as_tensor(z['structure']).long()
        samples += len(latent)
        with torch.no_grad():
            for r, model in models.items():
                value, mask = model.fit_targets(latent, structure)
                v, m = value.double().numpy(), mask.bool().numpy()
                sums[r] += np.where(m, v, 0).sum(axis=(0, 1, 2))
                squares[r] += np.where(m, v*v, 0).sum(axis=(0, 1, 2))
                counts[r] += m.sum(axis=(0, 1, 2))
    out = dict(schema='noise-flow-grid-normalization-v1', dataset=str(args.dataset), dataset_sha256=digest(args.dataset/'manifest.json'),
               train_samples=samples, method='masked population mean/std of fit_targets output, variance floor 1e-4 (as the manifest)',
               representations={})
    for r in GRID_KNOTS:
        mean = sums[r]/np.maximum(counts[r], 1)
        std = np.sqrt(np.maximum(squares[r]/np.maximum(counts[r], 1)-mean*mean, 1e-4))
        out['representations'][r] = dict(mean=mean.tolist(), std=std.tolist(), source_split='train', counts=counts[r].tolist())
    reference = manifest['normalization']
    out['grid513_vs_manifest'] = dict(
        mean_abs_difference=np.abs(np.array(out['representations']['grid513']['mean'])-reference['mean']).tolist(),
        std_abs_difference=np.abs(np.array(out['representations']['grid513']['std'])-reference['std']).tolist(),
        counts_equal=out['representations']['grid513']['counts'] == reference['counts'])
    args.output.write_bytes(json.dumps(out, indent=1).encode('utf-8'))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
