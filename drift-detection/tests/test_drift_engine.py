"""
Behavioural tests for the drift detectors.

These are not unit tests of arithmetic — they assert the *properties* each
correction was made to guarantee, on synthetic data where the ground truth is
known by construction. Each test pairs a null case (must stay quiet) with a
signal case (must fire), because a detector that never fires and a detector
that always fires are both useless and only the pair distinguishes them.

Two of these tests caught real defects during development:

* ``test_balanced_error_stream_amplifies_recall_collapse`` caught an
  implementation that reweighted instances and then clipped the weights back
  into [0, 1], which silently undid the rebalancing entirely.
* ``test_ks_effect_size_gate_suppresses_large_n_false_positive`` encodes the
  reason the original pipeline reported drift in 14 of 14 weeks.

Run with pytest, or directly:  python tests/test_drift_engine.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drift_engine import (  # noqa: E402
    benjamini_hochberg,
    build_error_stream,
    calculate_js_distance,
    calculate_ks_stat,
    DriftConfirmation,
    HDDMTracker,
    ks_test,
    PrequentialAUCDetector,
    psi_with_null,
)

SEED = 0


# ══════════════════════════════════════════════
# KS: significance vs. effect size
# ══════════════════════════════════════════════
def test_ks_effect_size_gate_suppresses_large_n_false_positive():
    """A 100k vs 10k comparison of identical distributions must not flag.

    The p-value alone cannot deliver this: at these sizes the critical KS
    statistic is ~0.015, so a 1.5% shift in probability mass is 'significant'.
    """
    rng = np.random.default_rng(SEED)
    ref, curr = rng.normal(0, 1, 100_000), rng.normal(0, 1, 10_000)

    result = ks_test(ref, curr)
    assert not result['drift'], f"false positive on identical distributions: {result}"
    assert result['statistic'] < 0.05
    assert result['d_critical'] < 0.02, "critical value should be tiny at this n"


def test_ks_still_detects_real_shift():
    rng = np.random.default_rng(SEED)
    ref, curr = rng.normal(0, 1, 100_000), rng.normal(0.5, 1, 10_000)

    result = ks_test(ref, curr)
    assert result['drift'], f"missed a genuine 0.5-sigma shift: {result}"
    assert result['statistic'] > 0.15


def test_ks_legacy_rule_is_the_one_that_saturates():
    """Documents *why* the effect-size gate exists, so the fix is not reverted."""
    rng = np.random.default_rng(SEED)
    ref = rng.normal(0, 1, 50_000)
    curr = rng.normal(0.02, 1, 50_000)          # 0.02 sigma — operationally nil

    _, p_value, legacy_drift = calculate_ks_stat(ref, curr)
    gated = ks_test(ref, curr)

    assert legacy_drift and p_value < 0.05, "legacy rule should flag this trivial shift"
    assert not gated['drift'], "gated rule should not"


# ══════════════════════════════════════════════
# Multiple testing
# ══════════════════════════════════════════════
def test_benjamini_hochberg_is_monotone_and_selective():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.9])
    rejected, adjusted = benjamini_hochberg(p, alpha=0.05)

    ordered = adjusted[np.argsort(p)]
    assert np.all(np.diff(ordered) >= -1e-12), "step-up procedure must be monotone"
    assert rejected[0], "smallest p-value should survive correction"
    assert not rejected[-1], "the null should not be rejected"
    assert np.all(adjusted >= p - 1e-12), "adjusted p-values cannot shrink"


def test_benjamini_hochberg_handles_empty_input():
    rejected, adjusted = benjamini_hochberg([], alpha=0.05)
    assert len(rejected) == 0 and len(adjusted) == 0


# ══════════════════════════════════════════════
# PSI calibration
# ══════════════════════════════════════════════
def test_psi_bootstrap_null_suppresses_small_window_false_positive():
    """PSI's expected value under the null scales as (B-1)/n.

    A small window therefore shows 'moderate shift' PSI from noise alone. The
    calibrated null must absorb that.
    """
    rng = np.random.default_rng(SEED)
    ref = rng.normal(0, 1, 50_000)
    curr = rng.choice(ref, 300, replace=False)   # drawn FROM the reference

    result = psi_with_null(ref, curr, n_bootstrap=60)
    assert not result['drift'], f"false positive at n=300: {result}"
    assert result['psi_null_q'] > 0.01, "null band should be non-trivial at n=300"


def test_psi_still_detects_real_shift():
    rng = np.random.default_rng(SEED)
    ref, curr = rng.normal(0, 1, 50_000), rng.normal(1.2, 1, 3_000)

    result = psi_with_null(ref, curr, n_bootstrap=60)
    assert result['drift'] and result['psi'] > 0.5


# ══════════════════════════════════════════════
# Jensen-Shannon
# ══════════════════════════════════════════════
def test_js_distance_is_bounded_and_symmetric():
    rng = np.random.default_rng(SEED)
    a, b = rng.normal(0, 1, 20_000), rng.normal(0.5, 1, 20_000)

    same, diff, reverse = (calculate_js_distance(a, a),
                           calculate_js_distance(a, b),
                           calculate_js_distance(b, a))

    assert 0.0 <= same <= 1.0 and 0.0 <= diff <= 1.0, "JS distance must lie in [0,1]"
    assert abs(diff - reverse) < 1e-9, "JS distance must be symmetric"
    assert diff > same, "JS must separate drift from no-drift"


def test_js_distance_finite_on_disjoint_support():
    """KL would be +inf here; JS must not be."""
    a = np.zeros(1000)
    b = np.ones(1000)
    assert np.isfinite(calculate_js_distance(a, b))


# ══════════════════════════════════════════════
# Class-balanced error stream
# ══════════════════════════════════════════════
def _imbalanced_predictions(n=20_000, prevalence=0.035, seed=SEED):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < prevalence).astype(int)
    good = np.where(y == 1, rng.uniform(0.6, 0.99, n), rng.uniform(0.0, 0.4, n))
    collapsed = rng.uniform(0.0, 0.4, n)        # never ranks fraud above legit
    return y, good, collapsed


def test_balanced_error_stream_amplifies_recall_collapse():
    """The defect this catches: reweighting then clipping to [0,1] is a no-op.

    Under 3.5% prevalence a total recall collapse moves the 0/1 error rate by
    only the prevalence. The balanced stream must respond far more strongly,
    or DDM/EDDM/HDDM cannot see the failure that matters most.
    """
    y, good, collapsed = _imbalanced_predictions()

    zero_one_delta = (build_error_stream(y, collapsed, 'zero_one').mean()
                      - build_error_stream(y, good, 'zero_one').mean())
    balanced_delta = (build_error_stream(y, collapsed, 'balanced').mean()
                      - build_error_stream(y, good, 'balanced').mean())

    assert zero_one_delta < 0.05, "0/1 error barely moves — that is the problem"
    assert balanced_delta > 0.4, f"balanced stream must respond strongly, got {balanced_delta}"
    assert balanced_delta > 5 * zero_one_delta


def test_balanced_error_stream_stays_bernoulli():
    """DDM's variance term and HDDM's Hoeffding bound both require {0,1}."""
    y, good, _ = _imbalanced_predictions()
    stream = build_error_stream(y, good, 'balanced')
    assert set(np.unique(stream)).issubset({0.0, 1.0})
    assert len(stream) == 2 * int(y.sum()), "should be all positives + equal negatives"


def test_brier_stream_is_bounded():
    y, good, _ = _imbalanced_predictions()
    stream = build_error_stream(y, good, 'brier')
    assert stream.min() >= 0.0 and stream.max() <= 1.0


# ══════════════════════════════════════════════
# HDDM
# ══════════════════════════════════════════════
def test_hddm_quiet_on_stable_stream():
    rng = np.random.default_rng(SEED)
    tracker = HDDMTracker()
    tracker.process_batch((rng.random(5_000) < 0.05).astype(float))
    result = tracker.process_batch((rng.random(5_000) < 0.05).astype(float))
    assert not result['drift_detected']


def test_hddm_fires_on_error_rate_step():
    rng = np.random.default_rng(SEED)
    tracker = HDDMTracker()
    tracker.process_batch((rng.random(5_000) < 0.05).astype(float))
    result = tracker.process_batch((rng.random(5_000) < 0.35).astype(float))
    assert result['drift_detected']


# ══════════════════════════════════════════════
# Prequential AUC
# ══════════════════════════════════════════════
def test_prequential_auc_quiet_when_nothing_changed():
    y, good, _ = _imbalanced_predictions()
    detector = PrequentialAUCDetector()
    detector.set_reference(y, good)
    assert not detector.evaluate_drift(y, good)['drift_detected']


def test_prequential_auc_fires_on_ranking_degradation():
    rng = np.random.default_rng(SEED)
    y, good, _ = _imbalanced_predictions()
    detector = PrequentialAUCDetector()
    detector.set_reference(y, good)

    n = len(y)
    degraded = np.where(y == 1, rng.uniform(0.2, 0.7, n), rng.uniform(0.1, 0.6, n))
    result = detector.evaluate_drift(y, degraded)
    assert result['drift_detected'] and result['auc_drop'] > 0.02


# ══════════════════════════════════════════════
# Persistence gate
# ══════════════════════════════════════════════
def test_persistence_gate_requires_consecutive_alarms():
    gate = DriftConfirmation(k=2, n=2)
    observed = [gate.update(f) for f in [True, False, True, True, False]]
    assert observed == [False, False, False, True, False], (
        "an isolated alarm must not trigger; two consecutive must")


def test_persistence_gate_reset_clears_history():
    gate = DriftConfirmation(k=2, n=2)
    gate.update(True)
    gate.reset()
    assert not gate.update(True), "after a retrain the history must not carry over"


def test_persistence_gate_rejects_invalid_config():
    try:
        DriftConfirmation(k=3, n=2)
    except ValueError:
        return
    raise AssertionError("k > n should be rejected")


# ══════════════════════════════════════════════
# Standalone runner
# ══════════════════════════════════════════════
if __name__ == '__main__':
    import logging
    logging.disable(logging.WARNING)

    tests = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith('test_') and callable(fn)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                      # noqa: BLE001
            failures.append((name, exc))
            print(f"  FAIL  {name}\n          {exc}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
