"""Aggregate per-trial JSON results into Table I (latency) and Table II (success rate).

Reads all <method>_trial_<N>.json files from results_dir/<method>/ and prints
the formatted tables from the IROS paper.

Usage
-----
  python3 gz_aggregate_results.py \\
      --results_dir results/ \\
      --methods pa nopa navrl panther \\
      --n_trials 20
"""

import argparse
import json
import math
import os


def _load_trials(results_dir: str, method: str, n_trials: int) -> list[dict]:
    trials = []
    method_dir = os.path.join(results_dir, method)
    for i in range(n_trials):
        p = os.path.join(method_dir, f'{method}_trial_{i}.json')
        if os.path.exists(p):
            with open(p) as fh:
                trials.append(json.load(fh))
    return trials


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else float('nan')


def _std(vals: list[float]) -> float:
    if len(vals) < 2:
        return 0.0
    m = _mean(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _worst(vals: list[float]) -> float:
    return max(vals) if vals else float('nan')


METHOD_LABELS = {
    'pa':      'Proposed method',
    'nopa':    'No-PA',
    'navrl':   'NAV-RL [9]',
    'panther': 'PANTHER [10]',
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--results_dir', type=str,   default='results')
    parser.add_argument('--methods',     nargs='+',  default=['pa', 'nopa', 'navrl', 'panther'])
    parser.add_argument('--n_trials',    type=int,   default=20)
    parser.add_argument('--save_csv',    type=str,   default=None,
                        help='Also save summary CSV to this path')
    args = parser.parse_args()

    rows_latency  = []
    rows_success  = []
    rows_orb      = []

    for method in args.methods:
        trials = _load_trials(args.results_dir, method, args.n_trials)
        if not trials:
            print(f'[WARN] No trials found for method "{method}" in {args.results_dir}')
            continue

        label = METHOD_LABELS.get(method, method)
        n     = len(trials)

        # ── Success rate ──────────────────────────────────────────────────
        successes  = sum(1 for t in trials if t.get('result') == 'success')
        collisions = sum(1 for t in trials if t.get('result') == 'collision')
        timeouts   = sum(1 for t in trials if t.get('result') == 'timeout')
        crashes    = sum(1 for t in trials if t.get('result') == 'crash')
        sr = 100.0 * successes / n
        rows_success.append({
            'label': label, 'N': n,
            'success': successes, 'collision': collisions,
            'timeout': timeouts, 'crash': crashes,
            'rate': sr,
        })

        # ── Latency ───────────────────────────────────────────────────────
        lat_means  = [t['latency_mean_ms']  for t in trials if t.get('latency_samples', 0) > 0]
        lat_worsts = [t['latency_worst_ms'] for t in trials if t.get('latency_samples', 0) > 0]
        if lat_means:
            rows_latency.append({
                'label': label,
                'mean':  _mean(lat_means),
                'std':   _std(lat_means),
                'worst': _mean(lat_worsts),
            })
        else:
            rows_latency.append({
                'label': label,
                'mean': float('nan'), 'std': float('nan'), 'worst': float('nan'),
            })

        # ── ORB ───────────────────────────────────────────────────────────
        orb_means = [t['orb_mean'] for t in trials if t.get('orb_frames', 0) > 0]
        rows_orb.append({
            'label': label,
            'mean':  _mean(orb_means),
            'std':   _std(orb_means),
            'n':     len(orb_means),
        })

    # ── Print Table I: Control Latency ───────────────────────────────────────
    print()
    print('Table I: Control latency comparison over five repeated runs.')
    print(f'{"Method":<22}  {"Mean (ms)":>10}  {"Std. (ms)":>10}  {"Worst (ms)":>11}')
    print('-' * 60)
    for r in rows_latency:
        m = f'{r["mean"]:.1f}'  if not math.isnan(r["mean"])  else 'N/A'
        s = f'{r["std"]:.1f}'   if not math.isnan(r["std"])   else 'N/A'
        w = f'{r["worst"]:.1f}' if not math.isnan(r["worst"]) else 'N/A'
        print(f'{r["label"]:<22}  {m:>10}  {s:>10}  {w:>11}')
    print()

    # ── Print Table II: Success Rate ─────────────────────────────────────────
    print('Table II: Success rate comparison across randomized obstacle layouts.')
    print(f'{"Method":<22}  {"Trials":>6}  {"Success":>8}  {"Rate (%)":>9}'
          f'  {"Collision":>9}  {"Timeout":>8}')
    print('-' * 70)
    for r in rows_success:
        print(f'{r["label"]:<22}  {r["N"]:>6}  {r["success"]:>8}  {r["rate"]:>8.1f}%'
              f'  {r["collision"]:>9}  {r["timeout"]:>8}')
    print()

    # ── Print ORB feature summary ─────────────────────────────────────────────
    print('ORB Feature-Point Comparison (perception-awareness metric)')
    print(f'{"Method":<22}  {"Mean keypoints":>15}  {"Std":>8}  {"Trials w/ ORB":>14}')
    print('-' * 65)
    for r in rows_orb:
        m = f'{r["mean"]:.1f}' if not math.isnan(r["mean"]) else 'N/A (cv2 missing)'
        s = f'{r["std"]:.1f}'  if not math.isnan(r["std"])  else ''
        print(f'{r["label"]:<22}  {m:>15}  {s:>8}  {r["n"]:>14}')
    print()

    # ── Optional CSV save ─────────────────────────────────────────────────────
    if args.save_csv:
        import csv
        rows = []
        for method in args.methods:
            trials = _load_trials(args.results_dir, method, args.n_trials)
            for t in trials:
                rows.append({
                    'method':           method,
                    'trial_id':         t.get('trial_id', ''),
                    'result':           t.get('result', ''),
                    'flight_time_s':    t.get('flight_time_s', ''),
                    'latency_mean_ms':  t.get('latency_mean_ms', ''),
                    'latency_std_ms':   t.get('latency_std_ms', ''),
                    'latency_worst_ms': t.get('latency_worst_ms', ''),
                    'orb_mean':         t.get('orb_mean', ''),
                    'orb_std':          t.get('orb_std', ''),
                })
        if rows:
            with open(args.save_csv, 'w', newline='') as fh:
                writer = csv.DictWriter(fh, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f'CSV saved to {args.save_csv}')


if __name__ == '__main__':
    main()
