"""Time/J figure of the noise-Flow sweep with the HA* + IPOPT reference point.

x = mean inference time over all 500 scenes (log scale), y = mean J / GT over
each configuration's successful scenes.  Color = top-K, marker = Euler steps,
hollow = success rate more than 3 %p below HA* + IPOPT.  Reads
exp2_noise_flow_{repr}_summary.json and writes exp2_noise_flow_{repr}_sweep.png.

Colors: the first three categorical slots of the dataviz reference palette
(validated all-pairs for scatter forms); the reference point is neutral ink
with a direct label, so identity never relies on color alone.
"""
from pathlib import Path
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT/'experiments/results'
K_COLORS = {1: '#2a78d6', 3: '#eb6834', 5: '#1baf7a'}
STEP_MARKERS = {4: 'o', 8: 's', 16: '^'}
INK, INK_2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--repr', default='bspline64')
    p.add_argument('--out-dir', type=Path, default=RESULTS)
    args = p.parse_args()
    summary = json.loads((args.out_dir/f'exp2_noise_flow_{args.repr}_summary.json').read_text(encoding='utf-8'))
    ref = summary['hastar_ipopt_reference']
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'xtick.color': INK_2,
                         'ytick.color': INK_2, 'axes.linewidth': .8})
    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=200, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.grid(True, which='major', color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    for e in summary['sweep']:
        if e['mean_J_over_gt'] is None:
            continue
        color = K_COLORS[e['K']]
        hollow = e['success_3pp_below_hastar_ipopt']
        ax.scatter(e['mean_time_s_all'], e['mean_J_over_gt'], s=18+4*e['N'] ** .9, marker=STEP_MARKERS[e['steps']],
                   facecolors='none' if hollow else color, edgecolors=color if hollow else SURFACE,
                   linewidths=1.2 if hollow else .8, zorder=3)
    ax.scatter(ref['mean_time_s_all'], ref['mean_J_over_gt'], s=110, marker='*', color=INK, edgecolors=SURFACE,
               linewidths=.8, zorder=4)
    ax.annotate(f"HA* + IPOPT\n{ref['mean_time_s_all']:.2f} s, J/GT {ref['mean_J_over_gt']:.3f}\n"
                f"success {100*ref['success_rate']:.1f}%",
                (ref['mean_time_s_all'], ref['mean_J_over_gt']), xytext=(-10, 10), textcoords='offset points',
                ha='right', va='bottom', fontsize=8, color=INK)
    main = next((e for e in summary['sweep'] if (e['N'], e['steps'], e['K']) == (16, 16, 3)), None)
    if main and main['mean_J_over_gt'] is not None:
        ax.annotate(f"N16 / 16 steps / K3\n{main['mean_time_s_all']:.2f} s, J/GT {main['mean_J_over_gt']:.3f}",
                    (main['mean_time_s_all'], main['mean_J_over_gt']), xytext=(8, -4), textcoords='offset points',
                    ha='left', va='top', fontsize=8, color=INK)
    ax.set_xscale('log')
    ax.set_xlabel('mean inference time per scene [s] (500 scenes, log scale)')
    ax.set_ylabel('mean J / OCP GT (successful scenes)')
    ax.set_title(f"Noise Flow ({args.repr}) sweep vs HA* + IPOPT", loc='left', fontsize=10, color=INK)
    handles = [Line2D([], [], marker='o', linestyle='', color=K_COLORS[k], label=f'top-K = {k}') for k in (1, 3, 5)]
    handles += [Line2D([], [], marker=STEP_MARKERS[s], linestyle='', color=INK_2, label=f'{s} Euler steps') for s in (4, 8, 16)]
    handles += [Line2D([], [], marker='o', linestyle='', markerfacecolor='none', markeredgecolor=INK_2,
                       label='success < HA*+IPOPT - 3%p'),
                Line2D([], [], marker='*', linestyle='', color=INK, markersize=10, label='HA* + IPOPT')]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, fontsize=8,
              title='marker size = candidates N', title_fontsize=8)
    fig.tight_layout()
    out = args.out_dir/f'exp2_noise_flow_{args.repr}_sweep.png'
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == '__main__':
    main()
