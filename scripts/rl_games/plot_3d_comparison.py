#!/usr/bin/env python3
"""Publication-quality 3D comparison figure.

X-axis : Method  (Nav-RL | PANTHER | Ours PA)
Y-axis : ORB feature count  (box plot per method)
Z-axis : Success rate  (scatter + 95% CI bar per method)

Each trial is a shape marker; filled = success, hollow = failure.
Box plots show ORB distribution (Q1-Q3 box, median, whiskers = 1.5×IQR).

Usage
-----
    python3 scripts/rl_games/plot_3d_comparison.py \\
        --results_dir results/quick_test \\
        --output     figures/fig_3d_comparison.pdf

    # compare multiple method prefixes (one label each)
    python3 scripts/rl_games/plot_3d_comparison.py \\
        --results_dir results/quick_test \\
        --methods pa navrl panther \\
        --labels  "Ours (PA)" "Nav-RL" "PANTHER" \\
        --output  figures/fig_3d_comparison.pdf
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D          # noqa: F401 (registers 3d projection)
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        9,
    'axes.labelsize':   10,
    'axes.titlesize':   11,
    'xtick.labelsize':  8,
    'ytick.labelsize':  8,
    'figure.dpi':       300,
    'pdf.fonttype':     42,   # embed fonts as TrueType (required by IEEE/ACM)
    'ps.fonttype':      42,
})

# ── Method config ─────────────────────────────────────────────────────────────
DEFAULT_METHODS = ['navrl', 'panther', 'pa']
DEFAULT_LABELS  = ['Nav-RL', 'PANTHER', 'Ours (PA)']
COLORS  = ['#1565c0', '#6a1b9a', '#2e7d32']
MARKERS = ['o', 's', 'D']
X_POS   = [0, 1, 2]           # integer X position per method


# ── Data loading ──────────────────────────────────────────────────────────────

def load_trials(results_dir: str, method: str) -> list[dict]:
    """Load all trial JSONs for a method prefix."""
    pattern = os.path.join(results_dir, f'{method}_trial_*.json')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'[WARN] No files found for method="{method}" in {results_dir}')
        return []
    trials = []
    for f in files:
        with open(f) as fh:
            d = json.load(fh)
        trials.append({
            'success': 1 if d.get('result') == 'success' else 0,
            'orb':     float(d.get('orb_mean', 0)),
            'time':    float(d.get('flight_time_s', 0)),
            'result':  d.get('result', '?'),
        })
    return trials


# ── Statistics ────────────────────────────────────────────────────────────────

def box_stats(values: list[float]) -> dict:
    """Tukey box plot statistics (whiskers = 1.5×IQR, clamped to data)."""
    a  = np.array(values)
    q1, med, q3 = np.percentile(a, [25, 50, 75])
    iqr = q3 - q1
    lo  = a[a >= q1 - 1.5 * iqr]
    hi  = a[a <= q3 + 1.5 * iqr]
    return dict(
        q1=q1, med=med, q3=q3,
        wlo=lo.min() if len(lo) else q1,
        whi=hi.max() if len(hi) else q3,
        outliers=a[(a < q1 - 1.5*iqr) | (a > q3 + 1.5*iqr)].tolist(),
        mean=float(a.mean()),
    )


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a proportion."""
    if n == 0:
        return 0.0, 0.0
    p = successes / n
    denom  = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# ── 3D box plot helper ────────────────────────────────────────────────────────

def draw_boxplot_3d(ax, x: float, z: float, stats: dict, color: str,
                    box_width: float = 0.12):
    """Draw a vertical box plot at position (x, *, z) in 3D space.

    Y axis = ORB feature count.
    """
    bw = box_width / 2
    q1, med, q3 = stats['q1'], stats['med'], stats['q3']
    wlo, whi    = stats['wlo'], stats['whi']

    # IQR box (filled polygon)
    verts = [[(x-bw, q1, z), (x+bw, q1, z),
               (x+bw, q3, z), (x-bw, q3, z)]]
    poly = Poly3DCollection(verts, alpha=0.22, facecolor=color,
                            edgecolor=color, linewidth=1.2, zorder=3)
    ax.add_collection3d(poly)

    # Median line
    ax.plot([x-bw, x+bw], [med, med], [z, z],
            color=color, linewidth=2.2, zorder=5)

    # Whiskers (solid, prominent)
    ax.plot([x, x], [whi, q3], [z, z], color=color,
            linewidth=2.0, linestyle='-', alpha=1.0, zorder=4)
    ax.plot([x, x], [q1, wlo], [z, z], color=color,
            linewidth=2.0, linestyle='-', alpha=1.0, zorder=4)

    # Whisker caps
    cap = bw * 0.7
    for y_cap in [whi, wlo]:
        ax.plot([x-cap, x+cap], [y_cap, y_cap], [z, z],
                color=color, linewidth=2.0, zorder=4)

    # Outliers
    for ov in stats['outliers']:
        ax.scatter([x], [ov], [z], marker='x', color=color, s=20,
                   linewidths=1, zorder=6)


# ── Main plot ─────────────────────────────────────────────────────────────────

def make_figure(all_data: list[dict], output: str, elev: float, azim: float):
    """
    all_data: list of dicts with keys:
        label, color, marker, x_pos, trials, box, success_rate, ci
    """
    fig = plt.figure(figsize=(7.2, 5.4))   # IEEE double-column width ~7.2"
    ax  = fig.add_subplot(111, projection='3d')

    rng = np.random.default_rng(42)

    # ── Scatter points ────────────────────────────────────────────────────────
    for m in all_data:
        x0 = m['x_pos']
        for t in m['trials']:
            jx = rng.uniform(-0.08, 0.08)
            jz = rng.uniform(-0.018, 0.018)
            z  = m['success_rate'] + jz
            filled = t['success'] == 1
            ax.scatter([x0 + jx], [t['orb']], [z],
                       color=m['color'] if filled else 'none',
                       edgecolors=m['color'],
                       marker=m['marker'],
                       s=28 if filled else 24,
                       linewidths=0.9,
                       alpha=0.85 if filled else 0.55,
                       zorder=4 if filled else 3,
                       depthshade=True)

    # ── Box plots ─────────────────────────────────────────────────────────────
    for m in all_data:
        draw_boxplot_3d(ax, m['x_pos'], m['success_rate'], m['box'],
                        m['color'], box_width=0.30)

    # ── Success rate CI bars ──────────────────────────────────────────────────
    ORB_CI_Y = min(t['orb'] for m in all_data for t in m['trials']) - 18
    for m in all_data:
        lo, hi = m['ci']
        ax.plot([m['x_pos'], m['x_pos']], [ORB_CI_Y, ORB_CI_Y], [lo, hi],
                color=m['color'], linewidth=2.0, zorder=5)
        # CI caps
        cap = 0.06
        for zv in [lo, hi]:
            ax.plot([m['x_pos']-cap, m['x_pos']+cap],
                    [ORB_CI_Y, ORB_CI_Y], [zv, zv],
                    color=m['color'], linewidth=1.5, zorder=5)
        # Center dot
        ax.scatter([m['x_pos']], [ORB_CI_Y], [m['success_rate']],
                   color=m['color'], s=28, zorder=6, depthshade=False)

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xticks(X_POS[:len(all_data)])
    ax.set_xticklabels([m['label'] for m in all_data], fontsize=9)
    ax.set_xlabel('Method', labelpad=8)

    ax.set_ylabel('ORB Feature Count', labelpad=8)

    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel('Success Rate', rotation=90, labelpad=8)
    ax.set_zlim(0, 1.0)
    ax.zaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f'{int(v*100)}%'))
    ax.set_zticks([0, 0.25, 0.50, 0.75, 1.0])

    # ── View ──────────────────────────────────────────────────────────────────
    ax.view_init(elev=elev, azim=azim)
    ax.set_box_aspect([1.4, 1.8, 1.2])

    # Pane colours
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor('#cccccc')
    ax.yaxis.pane.set_edgecolor('#cccccc')
    ax.zaxis.pane.set_edgecolor('#cccccc')
    ax.grid(True, alpha=0.25, linewidth=0.5)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = []
    for m in all_data:
        legend_elements.append(
            Line2D([0], [0], marker=m['marker'], color='w',
                   markerfacecolor=m['color'], markeredgecolor=m['color'],
                   markersize=7, label=m['label']))
    legend_elements += [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='#555', markeredgecolor='#555',
               markersize=6, label='Success'),
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='none', markeredgecolor='#555',
               markersize=6, label='Failure'),
        mpatches.Patch(facecolor='#888', alpha=0.22,
                       edgecolor='#888', label='IQR box (ORB)'),
        Line2D([0], [0], color='#888', linewidth=1.2,
               linestyle='--', label='Whisker'),
    ]
    ax.legend(handles=legend_elements, loc='upper left',
              bbox_to_anchor=(0.0, 0.92), fontsize=8,
              framealpha=0.85, edgecolor='#ccc')

    # ── Caption (as figure suptitle) ──────────────────────────────────────────
    fig.text(0.5, 0.01,
             'Fig. Comparison of navigation methods across 15 trials '
             '(3 environments × 5 trials). '
             'Shapes: ● Nav-RL  ■ PANTHER  ◆ Ours.\n'
             'Filled = success, hollow = failure. '
             'Vertical box: ORB IQR (Q1–Q3), whiskers = 1.5×IQR. '
             'Horizontal bar: 95% Wilson CI on success rate.',
             ha='center', va='bottom', fontsize=7.5, color='#444',
             style='italic', wrap=True)

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, bbox_inches='tight', dpi=300)
    print(f'Saved: {output}')
    plt.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--results_dir', default='results/quick_test')
    ap.add_argument('--methods', nargs='+', default=DEFAULT_METHODS)
    ap.add_argument('--labels',  nargs='+', default=DEFAULT_LABELS)
    ap.add_argument('--colors',  nargs='+', default=COLORS)
    ap.add_argument('--output',  default='figures/fig_3d_comparison.pdf')
    ap.add_argument('--elev',    type=float, default=22,
                    help='3D elevation angle (degrees)')
    ap.add_argument('--azim',    type=float, default=-55,
                    help='3D azimuth angle (degrees)')
    args = ap.parse_args()

    if len(args.methods) != len(args.labels):
        ap.error('--methods and --labels must have the same length')

    all_data = []
    for i, (method, label) in enumerate(zip(args.methods, args.labels)):
        trials = load_trials(args.results_dir, method)
        if not trials:
            continue
        orbs = [t['orb'] for t in trials]
        nsuc = sum(t['success'] for t in trials)
        n    = len(trials)
        sr   = nsuc / n
        ci   = wilson_ci(nsuc, n)
        all_data.append(dict(
            label=label,
            color=args.colors[i % len(args.colors)],
            marker=MARKERS[i % len(MARKERS)],
            x_pos=i,
            trials=trials,
            box=box_stats(orbs),
            success_rate=sr,
            ci=ci,
            n=n,
        ))
        print(f'{label:15s}  n={n:3d}  success={sr*100:.0f}%  '
              f'ORB={np.mean(orbs):.0f}±{np.std(orbs):.0f}  '
              f'CI=[{ci[0]*100:.0f}%,{ci[1]*100:.0f}%]')

    if not all_data:
        sys.exit('No data loaded — check --results_dir and --methods.')

    make_figure(all_data, args.output, args.elev, args.azim)


if __name__ == '__main__':
    main()
