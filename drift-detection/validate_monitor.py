"""
Null-Experiment Calibration Check for the Drift Monitor
=======================================================

The question this answers
-------------------------
Before trusting any drift study, you should know how often the monitor alarms
when **nothing has changed**. That is hard to measure on a real stream, because
a baseline-vs-week comparison confounds monitor artifacts with genuine drift.

The null experiment removes the confound: take a single window and split it into
two random halves — one large (standing in for the reference window), one small
(standing in for a monitored window), preserving the size asymmetry of the real
setting. Two random halves of the same data cannot differ systematically, so
**every alarm is an artifact** and the alarm rate is the monitor's
false-positive rate.

A correctly specified monitor should land at or below its nominal significance
level (~5% of features at alpha = 0.05). Substantially more than that means the
pipeline is measuring itself.

What this caught
----------------
Run against the original stateless feature pipeline, this reports drift in
roughly one feature in four. Three distinct causes, none visible without the
null:

1. Encoders (ordinal codes, frequency maps, PCA rotation, redundancy filter)
   refit independently on each window.
2. A per-window relative frequency feature, whose resolution floor of 1/n makes
   it window-size dependent even though it is a proportion (reached KS D = 0.88).
3. Entity-keyed aggregates on a near-unique id, undefined for most rows of any
   future window; the resulting point mass gives KS D equal to the unseen-entity
   rate (0.61) no matter what happened (reached KS D = 0.61).

Usage
-----
    python validate_monitor.py --data_dir ./dataset
    python validate_monitor.py --data_dir ./dataset --compare_legacy
"""

import argparse
import logging

import numpy as np
import pandas as pd

from data_processing import load_data
from feature_engineering import (
    FeatureEngineer,
    add_causal_sequence_features,
    apply_feature_engineering,
)
from feature_selection import select_monitoring_features
from model_training import train_model
from drift_engine import calculate_ks_stat, ks_test, psi_with_null, calculate_psi

logger = logging.getLogger(__name__)

BASELINE_DAYS = 90
DAY = 24 * 3600


def _to_numeric(df):
    X = df.drop(columns=['isFraud'], errors='ignore').copy()
    for c in X.columns:
        if X[c].dtype == object:
            X[c] = pd.factorize(X[c].astype(str))[0]
        X[c] = pd.to_numeric(X[c], errors='coerce').fillna(0)
    return X.astype(np.float32)


def _alarm_rates(P, Q, cols, calibrated_psi=True):
    """Fraction of features alarming under each rule."""
    ks_p = ks_eff = psi_fixed = psi_cal = n = 0
    offenders = []

    for c in cols:
        p_, q_ = P[c].values, Q[c].values
        if np.nanstd(p_) < 1e-12 and np.nanstd(q_) < 1e-12:
            continue                      # constant in both halves: untestable
        n += 1

        ks_p += calculate_ks_stat(p_, q_)[2]
        res = ks_test(p_, q_)
        ks_eff += res['drift']
        if res['drift']:
            offenders.append((c, round(res['statistic'], 3)))

        psi_fixed += calculate_psi(p_, q_)[1]
        if calibrated_psi:
            psi_cal += psi_with_null(p_, q_, n_bootstrap=40)['drift']

    n = max(n, 1)
    return {
        'n_testable': n,
        'ks_pvalue_only': ks_p / n,
        'ks_with_effect_size': ks_eff / n,
        'psi_fixed_threshold': psi_fixed / n,
        'psi_calibrated_null': psi_cal / n if calibrated_psi else float('nan'),
        'offenders': sorted(offenders, key=lambda x: -x[1])[:8],
    }


def run_null_experiment(data_dir, small_fraction=1 / 12, compare_legacy=False,
                        random_state=7):
    tr_id, tr_trn, _, _ = load_data(data_dir)
    raw = pd.merge(tr_trn, tr_id, on='TransactionID', how='left').sort_values('TransactionDT')
    raw = add_causal_sequence_features(raw)

    t0 = raw['TransactionDT'].min()
    window = raw[raw['TransactionDT'] < t0 + BASELINE_DAYS * DAY]

    rng = np.random.default_rng(random_state)
    perm = rng.permutation(len(window))
    n_small = max(int(len(window) * small_fraction), 50)
    small, big = window.iloc[perm[:n_small]], window.iloc[perm[n_small:]]

    print(f"\nNULL EXPERIMENT — one window split at random")
    print(f"  reference half : {len(big):>7,} rows")
    print(f"  monitored half : {len(small):>7,} rows")
    print(f"  No drift can exist between these. Every alarm below is an artifact.\n")

    results = {}

    if compare_legacy:
        legacy_a = _to_numeric(apply_feature_engineering(big.copy()))
        legacy_b = _to_numeric(apply_feature_engineering(small.copy()))
        shared = [c for c in legacy_a.columns if c in legacy_b.columns]
        results['stateless (legacy)'] = _alarm_rates(legacy_a, legacy_b, shared)

    fe = FeatureEngineer()
    frozen_a = _to_numeric(fe.fit_transform(big))
    frozen_b = _to_numeric(fe.transform(small))
    shared = [c for c in frozen_a.columns if c in frozen_b.columns]
    results['frozen (current)'] = _alarm_rates(frozen_a, frozen_b, shared)

    print(f"{'configuration':<22}{'n':>5}{'KS p-only':>12}{'KS +effect':>12}"
          f"{'PSI fixed':>12}{'PSI calib':>12}")
    print("-" * 75)
    for name, r in results.items():
        print(f"{name:<22}{r['n_testable']:>5}"
              f"{100 * r['ks_pvalue_only']:>11.1f}%"
              f"{100 * r['ks_with_effect_size']:>11.1f}%"
              f"{100 * r['psi_fixed_threshold']:>11.1f}%"
              f"{100 * r['psi_calibrated_null']:>11.1f}%")

    current = results['frozen (current)']
    print()
    if current['offenders']:
        print("Features still alarming on null data (feature, KS statistic):")
        for f, d in current['offenders']:
            print(f"    {f:<40} D = {d}")
        print()

    nominal = 0.05
    verdict = current['ks_with_effect_size'] <= nominal
    print(f"VERDICT: {'PASS' if verdict else 'FAIL'} — "
          f"{100 * current['ks_with_effect_size']:.1f}% of features alarm on data "
          f"with no drift (nominal {100 * nominal:.0f}%).")
    if not verdict:
        print("  The monitor is measuring itself. Investigate the features listed above\n"
              "  before drawing any conclusion from a drift study built on this pipeline.")
    return results


def calibrate_consensus_threshold(data_dir, top_k=20, n_bags=3, n_trials=30,
                                  small_fraction=1 / 12, quantile=0.99, random_state=7):
    """Null-calibrate MIN_FEATURE_DRIFT_FRACTION instead of picking it by convention.

    The per-feature KS/PSI thresholds were null-calibrated in ``run_null_experiment``
    above; the *consensus* statistic built on top of them — "what fraction of the
    monitored set must individually cross before the vote counts as drift" — was
    not. This repeats the same null split many times, restricted to the actual
    monitoring set the pipeline would select, and reports the empirical
    distribution of the crossed-fraction statistic under provable non-drift. The
    threshold should sit above this distribution's upper tail, not at a round
    number chosen by eye.
    """
    tr_id, tr_trn, _, _ = load_data(data_dir)
    raw = pd.merge(tr_trn, tr_id, on='TransactionID', how='left').sort_values('TransactionDT')
    raw = add_causal_sequence_features(raw)

    t0 = raw['TransactionDT'].min()
    window = raw[raw['TransactionDT'] < t0 + BASELINE_DAYS * DAY]

    # Select the monitoring set the same way the real pipeline would, on the
    # full baseline window, so the calibration targets the actual feature set.
    fe0 = FeatureEngineer()
    processed = fe0.fit_transform(window)
    ref_X = _to_numeric(processed)
    ref_y = processed['isFraud'].reset_index(drop=True)
    monitored, _, _ = select_monitoring_features(
        ref_X, ref_y, train_fn=lambda X, y, seed: train_model(X, y, seed=seed),
        reference_model=train_model(ref_X, ref_y, seed=random_state),
        top_k=top_k, n_bags=n_bags, random_state=random_state,
    )
    print(f"\nCalibrating the consensus threshold for {len(monitored)} monitored features:")
    print(f"  {monitored}\n")

    rng = np.random.default_rng(random_state)
    ks_fractions, psi_fractions = [], []
    for trial in range(n_trials):
        perm = rng.permutation(len(window))
        n_small = max(int(len(window) * small_fraction), 50)
        small, big = window.iloc[perm[:n_small]], window.iloc[perm[n_small:]]

        fe = FeatureEngineer()
        A = _to_numeric(fe.fit_transform(big))
        B = _to_numeric(fe.transform(small))

        ks_hits = psi_hits = 0
        for feat in monitored:
            if feat not in A.columns or feat not in B.columns:
                continue
            ks_hits += ks_test(A[feat], B[feat], alpha=0.05, min_effect=0.10)['drift']
            psi_hits += psi_with_null(A[feat], B[feat], threshold=0.20, quantile=0.99)['drift']
        ks_fractions.append(ks_hits / len(monitored))
        psi_fractions.append(psi_hits / len(monitored))
        logger.info(f"  trial {trial + 1}/{n_trials}: KS frac={ks_fractions[-1]:.2f}  "
                    f"PSI frac={psi_fractions[-1]:.2f}")

    ks_arr, psi_arr = np.array(ks_fractions), np.array(psi_fractions)
    combined = np.concatenate([ks_arr, psi_arr])

    def _pct(arr, q):
        return float(np.quantile(arr, q)) if len(arr) else 0.0

    print(f"{'statistic':<28}{'mean':>8}{'p90':>8}{'p95':>8}{'p99':>8}{'max':>8}")
    print("-" * 68)
    for name, arr in [('KS crossed-fraction', ks_arr), ('PSI crossed-fraction', psi_arr),
                       ('combined', combined)]:
        print(f"{name:<28}{arr.mean():>8.2f}{_pct(arr,0.90):>8.2f}"
              f"{_pct(arr,0.95):>8.2f}{_pct(arr,0.99):>8.2f}{arr.max():>8.2f}")

    raw_suggestion = _pct(combined, quantile)
    if combined.max() == 0.0:
        print(f"\nNULL DISTRIBUTION IS DEGENERATE: 0 of {len(monitored)} monitored features "
              f"crossed their own threshold in ANY of the {n_trials} null trials (KS or PSI).")
        print("  This means the per-feature tests already have an essentially 0% false-positive")
        print("  rate on this monitoring set — there is no null-noise floor to calibrate the")
        print("  consensus bar against. Any fixed MIN_FEATURE_DRIFT_FRACTION > 0 is 'safe' from a")
        print("  false-positive standpoint; the real question is how much genuine (non-null,")
        print("  possibly seasonal) per-feature movement should count as 'consensus drift'. That")
        print("  is a policy decision informed by the ACTUAL replay's non-null crossed-fraction")
        print("  distribution (see run_drift_analysis.py's method_week_matrix.csv), not by this")
        print("  null check. Do not apply this function's numeric suggestion blindly in that case.")
        return {'ks_fractions': ks_fractions, 'psi_fractions': psi_fractions,
                'monitored_features': monitored, 'suggested_threshold': None, 'degenerate': True}

    # Round up to the nearest 0.05 so the bar sits strictly above the tail
    # rather than exactly on a sampled value.
    suggested = float(np.ceil(raw_suggestion * 20) / 20)
    suggested = min(suggested, 0.95)
    print(f"\nSUGGESTED MIN_FEATURE_DRIFT_FRACTION: {suggested:.2f} "
          f"(p{int(quantile*100)} of the null distribution = {raw_suggestion:.3f}, rounded up)")
    return {'ks_fractions': ks_fractions, 'psi_fractions': psi_fractions,
            'monitored_features': monitored, 'suggested_threshold': suggested, 'degenerate': False}


if __name__ == '__main__':
    logging.disable(logging.INFO)
    parser = argparse.ArgumentParser(description='Null-experiment calibration check')
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--compare_legacy', action='store_true',
                        help='Also run the stateless (per-window refit) pipeline for contrast')
    parser.add_argument('--calibrate_consensus', action='store_true',
                        help='Null-calibrate MIN_FEATURE_DRIFT_FRACTION on the real monitoring set '
                             'instead of running the per-feature check')
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--n_bags', type=int, default=5)
    parser.add_argument('--n_trials', type=int, default=30)
    parser.add_argument('--quantile', type=float, default=0.99)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if args.calibrate_consensus:
        calibrate_consensus_threshold(args.data_dir, top_k=args.top_k, n_bags=args.n_bags,
                                      n_trials=args.n_trials, quantile=args.quantile,
                                      random_state=args.seed)
    else:
        run_null_experiment(args.data_dir, compare_legacy=args.compare_legacy,
                            random_state=args.seed)
