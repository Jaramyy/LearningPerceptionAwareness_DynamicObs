"""Analyze crash logs from student_ros2_node_icp.py / gz_eval_monitor.py.

For each failed trial in a results directory the script reads the result JSON
(crash position, nearest obstacle) and the per-step CSV (last-5-s crash buffer),
then prints a structured diagnosis and writes a summary table.

Usage
-----
    # Analyze all trials in a results directory:
    python3 gz_log_analyzer.py results/quick_test/

    # Analyze a single trial explicitly:
    python3 gz_log_analyzer.py --result results/quick_test/pa_trial_2.json

    # Write CSV summary table:
    python3 gz_log_analyzer.py results/quick_test/ --summary crash_summary.csv
"""

import argparse
import csv
import json
import math
import os
import glob
from typing import Optional

LIDAR_RANGE = 5.0


# ── CSV log helpers ───────────────────────────────────────────────────────────

def _load_csv(path: str) -> list[dict]:
    if not path or not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def _float(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError, TypeError):
        return default


def _stats(values: list[float]) -> dict:
    if not values:
        return {'min': None, 'max': None, 'mean': None}
    return {
        'min':  round(min(values), 4),
        'max':  round(max(values), 4),
        'mean': round(sum(values) / len(values), 4),
    }


# ── Per-trial analysis ────────────────────────────────────────────────────────

def analyze_trial(json_path: str, verbose: bool = True) -> dict:
    """Return a dict summarising one trial; prints diagnosis when verbose=True."""
    with open(json_path) as f:
        result = json.load(f)

    trial_id = result.get('trial_id', '?')
    outcome  = result.get('result', '?')
    t_flight = result.get('flight_time_s', 0.0)

    # Locate crash-buffer CSV (written by student node on any shutdown)
    stem = json_path.replace('.json', '')
    # Try both naming conventions
    csv_candidates = [
        stem.replace('_trial_', '_trial_').replace('.json', '') + '_steps_last5s.csv',
        stem + '_steps_last5s.csv',
        os.path.join(os.path.dirname(json_path),
                     os.path.basename(stem) + '_steps_last5s.csv'),
    ]
    csv_path = next((p for p in csv_candidates if os.path.exists(p)), None)

    if verbose:
        sep = '=' * 64
        print(f'\n{sep}')
        print(f'Trial {trial_id}  |  result={outcome}  |  flight={t_flight:.1f}s')
        print(sep)

    summary = {
        'trial_id':   trial_id,
        'result':     outcome,
        'flight_s':   t_flight,
        'crash_E':    None, 'crash_N': None, 'crash_alt': None,
        'near_obs_type': None, 'near_obs_dist_m': None,
        'near_obs_x': None,   'near_obs_y': None,
        'd_fwd_min':  None, 'd_all_min': None,
        'speed_at_crash': None, 'W1_at_crash': None,
        'X0_at_crash': None,   'X1_at_crash': None,
        'lateral_frac': None,
        'icp_active': None,
        'diagnosis': '',
    }

    # ── Crash position from monitor JSON ─────────────────────────────────────
    crash_pos = result.get('crash_pos_enu')
    near_obs  = result.get('nearest_obstacle')
    if crash_pos:
        summary['crash_E'], summary['crash_N'], summary['crash_alt'] = crash_pos
        if verbose:
            print(f'  Crash position: E={crash_pos[0]:.2f}  N={crash_pos[1]:.2f}  '
                  f'alt={crash_pos[2]:.2f} m')
    if near_obs and near_obs.get('type'):
        summary['near_obs_type']   = near_obs['type']
        summary['near_obs_dist_m'] = near_obs.get('dist_m')
        summary['near_obs_x']      = near_obs.get('x')
        summary['near_obs_y']      = near_obs.get('y')
        if verbose:
            print(f'  Nearest obstacle: {near_obs["type"]} at '
                  f'({near_obs.get("x"):.1f}, {near_obs.get("y"):.1f}) ENU  '
                  f'dist={near_obs.get("dist_m"):.3f} m')

    if outcome == 'success':
        if verbose:
            print('  SUCCESS — no crash analysis needed.')
        summary['diagnosis'] = 'success'
        return summary

    if outcome == 'timeout':
        if verbose:
            print('  TIMEOUT — drone did not crash but did not reach goal.')
        summary['diagnosis'] = 'timeout'
        return summary

    # ── Per-step CSV analysis ─────────────────────────────────────────────────
    rows = _load_csv(csv_path)
    if not rows:
        if verbose:
            print(f'  [!] No crash-buffer CSV found near {json_path}')
            print(f'      Expected: {csv_candidates[0]}')
        summary['diagnosis'] = 'crash:no_csv'
        return summary

    if verbose:
        print(f'  CSV: {csv_path}  ({len(rows)} rows)')

    # Last 100 rows ≈ 1 s; last 500 rows ≈ 5 s (all available)
    tail_1s  = rows[-100:] if len(rows) >= 100 else rows
    tail_all = rows

    d_fwds  = [_float(r, 'd_fwd_m') for r in tail_1s]
    d_alls  = [_float(r, 'd_all_m') for r in tail_1s]
    speeds  = [_float(r, 'speed_scale') for r in tail_1s]
    W1s     = [_float(r, 'W1') for r in tail_1s]
    X0s     = [_float(r, 'X0') for r in tail_1s]
    X1s     = [_float(r, 'X1') for r in tail_1s]
    vxs     = [_float(r, 'a_vx') for r in tail_1s]
    vys     = [_float(r, 'a_vy') for r in tail_1s]

    stat_d_fwd = _stats(d_fwds)
    stat_d_all = _stats(d_alls)
    stat_spd   = _stats(speeds)
    stat_W1    = _stats(W1s)

    # Lateral motion fraction
    lat_fracs = [abs(vy) / (abs(vx) + abs(vy) + 1e-6) for vx, vy in zip(vxs, vys)]
    lat_frac  = sum(lat_fracs) / len(lat_fracs) if lat_fracs else 0.0

    # ICP was active if speed was ever reduced significantly
    icp_active = stat_spd['min'] is not None and stat_spd['min'] < 0.8

    summary['d_fwd_min']     = stat_d_fwd['min']
    summary['d_all_min']     = stat_d_all['min']
    summary['speed_at_crash'] = stat_spd['min']
    summary['W1_at_crash']   = stat_W1['mean']
    summary['X0_at_crash']   = _stats(X0s)['max']
    summary['X1_at_crash']   = _stats(X1s)['max']
    summary['lateral_frac']  = round(lat_frac, 3)
    summary['icp_active']    = icp_active

    if verbose:
        print(f'\n  Last 1 s ({len(tail_1s)} steps) statistics:')
        print(f'    d_fwd : min={stat_d_fwd["min"]:.3f}  '
              f'mean={stat_d_fwd["mean"]:.3f}  max={stat_d_fwd["max"]:.3f} m')
        print(f'    d_all : min={stat_d_all["min"]:.3f}  '
              f'mean={stat_d_all["mean"]:.3f}  max={stat_d_all["max"]:.3f} m')
        print(f'    speed : min={stat_spd["min"]:.3f}  '
              f'mean={stat_spd["mean"]:.3f}')
        print(f'    W1    : mean={stat_W1["mean"]:.4f}')
        print(f'    X0max : {_stats(X0s)["max"]:.3f}')
        print(f'    X1max : {_stats(X1s)["max"]:.3f}')
        print(f'    lateral_frac: {lat_frac:.2f}  '
              f'({"mostly lateral" if lat_frac > 0.45 else "mostly forward"})')

        # Last 10 rows
        print(f'\n  Last 10 steps:')
        hdr = ('   t_mono   alt  d_fwd  d_all  speed    W1    X0    X1  '
               'b_R  b_sR  b_C  b_sL  b_L  fsm')
        print(hdr)
        for r in tail_1s[-10:]:
            b = [_float(r, k) for k in
                 ('b_right', 'b_slR', 'b_ctr', 'b_slL', 'b_left')]
            print(
                f'  {_float(r,"t_mono"):8.2f}'
                f'  {_float(r,"alt_m"):4.2f}'
                f'  {_float(r,"d_fwd_m"):5.3f}'
                f'  {_float(r,"d_all_m"):5.3f}'
                f'  {_float(r,"speed_scale"):5.3f}'
                f'  {_float(r,"W1"):5.3f}'
                f'  {_float(r,"X0"):4.2f}'
                f'  {_float(r,"X1"):4.2f}'
                f'  {b[0]:4.2f} {b[1]:4.2f} {b[2]:4.2f} {b[3]:4.2f} {b[4]:4.2f}'
                f'  {r.get("fsm","?")}'
            )

    # ── Crash diagnosis ───────────────────────────────────────────────────────
    diagnosis_parts = []

    d_all_min  = stat_d_all['min'] or 999.0
    d_fwd_min  = stat_d_fwd['min'] or 999.0
    speed_min  = stat_spd['min']   or 1.0
    X0_max     = _stats(X0s)['max'] or 0.0

    if d_fwd_min > 1.2 and d_all_min < 0.5:
        diagnosis_parts.append('SIDE_BLIND_SPOT: forward sectors clear but side obstacle inside 0.5 m')
    elif d_fwd_min < 0.8 and speed_min > 0.5:
        diagnosis_parts.append('ICP_INSUFFICIENT: forward obstacle close but speed not reduced enough')
    elif d_fwd_min < 0.8 and speed_min < 0.2:
        diagnosis_parts.append('SPEED_ZERO_BUT_CRASHED: ICP stopped but drone still drifted into obstacle')
    elif X0_max < 0.1 and d_all_min < 0.8:
        diagnosis_parts.append('X0_NOT_TRIGGERED: obstacle within 0.8 m but reflexive neuron never fired')

    if lat_frac > 0.45:
        diagnosis_parts.append(f'LATERAL_MOTION: {lat_frac:.0%} lateral commands during last 1 s')

    if not diagnosis_parts:
        diagnosis_parts.append(f'GENERAL: d_fwd_min={d_fwd_min:.2f} d_all_min={d_all_min:.2f} '
                                f'speed_min={speed_min:.2f}')

    diagnosis = ' | '.join(diagnosis_parts)
    summary['diagnosis'] = diagnosis

    if verbose:
        print(f'\n  DIAGNOSIS: {diagnosis}')

    return summary


# ── Summary table ─────────────────────────────────────────────────────────────

def write_summary(summaries: list[dict], path: str) -> None:
    if not summaries:
        return
    fields = list(summaries[0].keys())
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(summaries)
    print(f'\nSummary → {path}')


def print_summary_table(summaries: list[dict]) -> None:
    crashes = [s for s in summaries if s['result'] not in ('success', 'timeout')]
    successes = sum(1 for s in summaries if s['result'] == 'success')
    timeouts  = sum(1 for s in summaries if s['result'] == 'timeout')

    print(f'\n{"="*64}')
    print(f'OVERALL: {len(summaries)} trials  '
          f'success={successes}  collision={len(crashes)}  timeout={timeouts}')
    print(f'{"="*64}')

    if not crashes:
        print('No crashes to report.')
        return

    print(f'\nCRASH BREAKDOWN ({len(crashes)} collisions):')
    pattern_counts: dict[str, int] = {}
    for s in crashes:
        diag = s['diagnosis']
        # Bucket by leading keyword
        key = diag.split(':')[0] if ':' in diag else diag[:30]
        pattern_counts[key] = pattern_counts.get(key, 0) + 1

    for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
        print(f'  {count:3d}x  {pattern}')

    # Side-blind-spot rate
    side_blind = sum(1 for s in crashes
                     if 'SIDE_BLIND_SPOT' in s.get('diagnosis', ''))
    if side_blind:
        print(f'\n  Side-blind-spot crashes: {side_blind}/{len(crashes)}  '
              f'({100*side_blind/len(crashes):.0f}%)')
        print('  → d_all_min column reveals obstacles in sectors 11/15 (LEFT/RIGHT)')
        print('    that were not seen by the forward 3-sector ICP check.')


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('results_dir', nargs='?', default=None,
                        help='Directory containing *_trial_*.json result files')
    parser.add_argument('--result', type=str, default=None,
                        help='Analyze a single result JSON file')
    parser.add_argument('--summary', type=str, default=None,
                        help='Write summary CSV to this path')
    parser.add_argument('--quiet', action='store_true',
                        help='Only print the summary table, not per-trial details')
    args = parser.parse_args()

    json_files: list[str] = []
    if args.result:
        json_files = [args.result]
    elif args.results_dir:
        json_files = sorted(glob.glob(os.path.join(args.results_dir, '*_trial_*.json')))
    else:
        parser.error('Provide either results_dir or --result')

    if not json_files:
        print(f'No *_trial_*.json files found in {args.results_dir}')
        return

    summaries = []
    for jf in json_files:
        try:
            s = analyze_trial(jf, verbose=not args.quiet)
            summaries.append(s)
        except Exception as e:
            print(f'[ERROR] {jf}: {e}')

    print_summary_table(summaries)

    if args.summary:
        write_summary(summaries, args.summary)


if __name__ == '__main__':
    main()
