"""
Drift Detection Engine
======================
Implements 10 drift detection methods for tabular fraud detection models.

Methods
-------
1. KS Stats          — Kolmogorov-Smirnov two-sample test
2. PSI               — Population Stability Index
3. KL Divergence     — Kullback-Leibler divergence on binned densities
4. DDM               — Drift Detection Method (Gama et al., 2004)
5. EDDM              — Early Drift Detection Method (Baena-García et al., 2006)
6. ADWIN             — Adaptive Windowing (Bifet & Gavaldà, 2007)
7. SHAP              — SHAP value distribution shift
8. Clustering        — K-Means centroid distance & cluster assignment shift
9. Autoencoder       — Reconstruction error (MSE) shift via bottleneck MLP
10. Champion vs Challenger — Performance gap between incumbent & retrained model

Every public function / class returns a result dict that always contains
``drift_detected`` (bool) and ``warning_detected`` (bool).
"""

import logging
import warnings

import numpy as np
import pandas as pd
import shap
from scipy.stats import ks_2samp, entropy
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, roc_auc_score, f1_score
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# River (streaming drift detectors) — optional dependency
try:
    from river import drift
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

__all__ = [
    'calculate_ks_stat',
    'ks_test',
    'benjamini_hochberg',
    'calculate_psi',
    'psi_with_null',
    'calculate_kl_divergence',
    'calculate_js_distance',
    'build_error_stream',
    'DDMTracker',
    'EDDMTracker',
    'HDDMTracker',
    'PrequentialAUCDetector',
    'DriftConfirmation',
    'run_ddm_batch',
    'run_eddm_batch',
    'run_adwin_batch',
    'SHAPDriftDetector',
    'ClusteringDriftDetector',
    'AutoencoderDriftDetector',
    'evaluate_champion_vs_challenger',
]


# ═══════════════════════════════════════════════
# 1. KS STATS (Kolmogorov-Smirnov)
# ═══════════════════════════════════════════════
def calculate_ks_stat(reference, current, alpha=0.05):
    """Two-sample KS test between reference and current distributions.

    Returns:
        (ks_statistic, p_value, is_drift)
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)
    ref_clean = ref[~np.isnan(ref)]
    curr_clean = curr[~np.isnan(curr)]

    if len(ref_clean) == 0 or len(curr_clean) == 0:
        return 0.0, 1.0, False

    stat, p_val = ks_2samp(ref_clean, curr_clean)
    return float(stat), float(p_val), bool(p_val < alpha)


def ks_test(reference, current, alpha=0.05, min_effect=0.10,
            max_ref_sample=20_000, max_curr_sample=20_000, random_state=42):
    """Two-sample KS test gated on BOTH significance and effect size.

    Why the p-value alone is not usable here
    ----------------------------------------
    The KS p-value answers "could this difference have arisen by chance?" — and
    with a 90-day reference window (~10^5 rows) against a 7-day window (~10^4),
    the answer is essentially always "no". The critical KS statistic at
    alpha=0.05 for those sizes is

        D_crit = 1.358 * sqrt((n+m)/(n*m)) ≈ 0.015

    so a shift of 1.5% of probability mass — utterly irrelevant to a fraud
    model — is "statistically significant". This is exactly why the original
    pipeline reported KS drift in 14 of 14 weeks: it was measuring sample size,
    not drift. Any monitoring rule built on unqualified significance testing
    over large windows degenerates the same way.

    The fix has two parts:

    1. **Effect-size floor.** Require D >= ``min_effect`` (default 0.10, i.e.
       the CDFs must separate by at least 10 percentage points somewhere).
       D is itself an interpretable effect size, bounded in [0, 1], so this
       is a statement about practical magnitude, not about n.
    2. **Bounded samples.** Subsample both windows to at most ~20k rows so the
       p-value retains some discriminating power instead of saturating at 0.
       The point estimate of D is unbiased under subsampling; only its variance
       grows, which is the honest trade.

    Returns:
        dict with statistic, p_value, d_critical, min_effect, significant,
        practically_significant, and drift (= both conditions met).
    """
    rng = np.random.default_rng(random_state)

    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]

    if len(ref) == 0 or len(curr) == 0:
        return {'statistic': 0.0, 'p_value': 1.0, 'd_critical': 1.0,
                'min_effect': min_effect, 'significant': False,
                'practically_significant': False, 'drift': False,
                'n_ref': int(len(ref)), 'n_curr': int(len(curr))}

    n_ref_full, n_curr_full = len(ref), len(curr)
    if len(ref) > max_ref_sample:
        ref = rng.choice(ref, size=max_ref_sample, replace=False)
    if len(curr) > max_curr_sample:
        curr = rng.choice(curr, size=max_curr_sample, replace=False)

    stat, p_val = ks_2samp(ref, curr)
    n, m = len(ref), len(curr)
    d_critical = 1.358 * np.sqrt((n + m) / (n * m))

    significant = bool(p_val < alpha)
    practical = bool(stat >= min_effect)

    return {
        'statistic': float(stat),
        'p_value': float(p_val),
        'd_critical': float(d_critical),
        'min_effect': float(min_effect),
        'significant': significant,
        'practically_significant': practical,
        'drift': bool(significant and practical),
        'n_ref': int(n_ref_full),
        'n_curr': int(n_curr_full),
        'n_ref_used': int(n),
        'n_curr_used': int(m),
    }


def benjamini_hochberg(p_values, alpha=0.05):
    """Benjamini-Hochberg FDR control across the monitored feature set.

    Monitoring K features means running K simultaneous hypothesis tests each
    week. At alpha=0.05 with K=10 and no correction, the probability of at
    least one false positive in a stable week is 1-(0.95)^10 = 40%; across 14
    weeks it is a near-certainty. Bonferroni would control that but is far too
    conservative when features are correlated (and monitored features usually
    are). BH controls the *expected proportion* of false discoveries instead,
    which is the right target when the decision rule is "what fraction of
    features drifted?".

    Returns:
        (rejected: np.ndarray[bool], adjusted_p: np.ndarray[float])
    """
    p = np.asarray(p_values, dtype=float)
    k = len(p)
    if k == 0:
        return np.array([], dtype=bool), np.array([], dtype=float)

    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * k / np.arange(1, k + 1)
    # Enforce monotonicity of the step-up procedure
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)

    out_adjusted = np.empty(k, dtype=float)
    out_adjusted[order] = adjusted
    return out_adjusted < alpha, out_adjusted


# ═══════════════════════════════════════════════
# 2. PSI (Population Stability Index)
# ═══════════════════════════════════════════════
def calculate_psi(reference, current, num_buckets=10, threshold=0.2):
    """Population Stability Index between reference and current.

    Interpretation:
        PSI < 0.10  — No significant shift
        0.10 ≤ PSI < 0.20  — Moderate shift (warning)
        PSI ≥ 0.20  — Significant drift

    Returns:
        (psi_value, is_drift)
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)
    ref_clean = ref[~np.isnan(ref)]
    curr_clean = curr[~np.isnan(curr)]

    if len(ref_clean) < 10 or len(curr_clean) < 10:
        return 0.0, False

    try:
        percentiles = np.linspace(0, 100, num_buckets + 1)
        breakpoints = np.unique(np.percentile(ref_clean, percentiles))
        if len(breakpoints) < 2:
            return 0.0, False

        breakpoints[0] = -np.inf
        breakpoints[-1] = np.inf

        ref_counts, _ = np.histogram(ref_clean, bins=breakpoints)
        curr_counts, _ = np.histogram(curr_clean, bins=breakpoints)

        eps = 1e-4
        ref_pct = np.where(ref_counts == 0, eps, ref_counts / len(ref_clean))
        curr_pct = np.where(curr_counts == 0, eps, curr_counts / len(curr_clean))

        psi_val = float(np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct)))
        return psi_val, bool(psi_val >= threshold)
    except Exception as e:
        logger.error(f"PSI calculation error: {e}")
        return 0.0, False


def psi_with_null(reference, current, num_buckets=10, threshold=0.2,
                  n_bootstrap=100, quantile=0.99, random_state=42):
    """PSI compared against a bootstrap null calibrated to the window size.

    Why the 0.10 / 0.20 thresholds are not sufficient
    -------------------------------------------------
    The canonical PSI bands (<0.10 stable, 0.10-0.20 moderate, >0.20 drift) are
    industry folklore from credit scorecards with a particular sample size in
    mind. PSI has a known sampling distribution: under the null of no drift it
    is asymptotically ``chi2(B-1) / n_curr`` for B buckets, so its expected
    value scales as ``(B-1)/n_curr``. A 1,000-row window is therefore expected
    to show PSI ~0.009 with *no drift at all*, while a 100-row window expects
    ~0.09 — nearly the "moderate shift" band, purely from noise. Comparing a
    weekly window against a fixed constant conflates drift with window size.

    This function resamples ``n_curr`` rows from the reference window
    ``n_bootstrap`` times and computes PSI of each resample against the
    reference. That empirical null says what PSI looks like when nothing has
    changed *at this window size*, and drift is declared only when the observed
    PSI exceeds both the folklore threshold and the null's upper quantile.

    Returns:
        dict with psi, psi_null_q, psi_ratio, exceeds_threshold,
        exceeds_null, drift.
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]

    psi_val, _ = calculate_psi(ref, curr, num_buckets=num_buckets, threshold=threshold)

    if len(ref) < 10 or len(curr) < 10:
        return {'psi': psi_val, 'psi_null_q': 0.0, 'psi_ratio': 0.0,
                'exceeds_threshold': False, 'exceeds_null': False, 'drift': False,
                'chi2_expected_psi': 0.0}

    rng = np.random.default_rng(random_state)
    n_curr = len(curr)
    null_vals = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        resample = rng.choice(ref, size=n_curr, replace=True)
        null_vals[b], _ = calculate_psi(ref, resample, num_buckets=num_buckets,
                                        threshold=threshold)

    psi_null_q = float(np.quantile(null_vals, quantile))
    exceeds_threshold = bool(psi_val >= threshold)
    exceeds_null = bool(psi_val > psi_null_q)

    return {
        'psi': float(psi_val),
        'psi_null_q': psi_null_q,
        'psi_null_quantile': float(quantile),
        'psi_ratio': float(psi_val / max(psi_null_q, 1e-9)),
        # Asymptotic reference point: E[PSI] ≈ (B-1)/n under the null.
        'chi2_expected_psi': float((num_buckets - 1) / max(n_curr, 1)),
        'exceeds_threshold': exceeds_threshold,
        'exceeds_null': exceeds_null,
        'drift': bool(exceeds_threshold and exceeds_null),
    }


def calculate_js_distance(reference, current, num_bins=20):
    """Jensen-Shannon distance between two binned empirical distributions.

    Preferred over raw KL for monitoring because it is (a) symmetric, (b)
    always finite — KL is unbounded and blows up to +inf whenever the current
    window puts mass where the reference had none, which for a rare-category
    feature happens routinely and forces the epsilon-smoothing hack that makes
    KL's absolute value depend on the arbitrary epsilon — and (c) bounded in
    [0, 1] after taking the square root, so a threshold means the same thing
    for every feature regardless of its scale or entropy.

    Returns:
        (js_distance in [0, 1], is_drift) — drift left to the caller's threshold.
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)
    ref = ref[~np.isnan(ref)]
    curr = curr[~np.isnan(curr)]

    if len(ref) < 10 or len(curr) < 10:
        return 0.0

    combined = np.concatenate([ref, curr])
    lo, hi = np.min(combined), np.max(combined)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return 0.0

    bins = np.linspace(lo, hi, num_bins + 1)
    p, _ = np.histogram(curr, bins=bins)
    q, _ = np.histogram(ref, bins=bins)

    p = p / max(p.sum(), 1)
    q = q / max(q.sum(), 1)
    m = 0.5 * (p + q)

    def _kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    js_divergence = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(np.sqrt(max(js_divergence, 0.0)))


# ═══════════════════════════════════════════════
# 3. KL DIVERGENCE (Kullback-Leibler)
# ═══════════════════════════════════════════════
def calculate_kl_divergence(reference, current, num_bins=10, threshold=0.5):
    """KL divergence D_KL(current ‖ reference) computed on binned densities.

    Returns:
        (kl_value, is_drift)
    """
    ref = np.asarray(reference, dtype=float)
    curr = np.asarray(current, dtype=float)
    ref_clean = ref[~np.isnan(ref)]
    curr_clean = curr[~np.isnan(curr)]

    if len(ref_clean) < 10 or len(curr_clean) < 10:
        return 0.0, False

    try:
        combined = np.concatenate([ref_clean, curr_clean])
        bins = np.linspace(np.min(combined), np.max(combined), num_bins + 1)
        if len(np.unique(bins)) < 2:
            return 0.0, False

        ref_density, _ = np.histogram(ref_clean, bins=bins, density=True)
        curr_density, _ = np.histogram(curr_clean, bins=bins, density=True)

        eps = 1e-4
        ref_density = np.where(ref_density <= 0, eps, ref_density)
        curr_density = np.where(curr_density <= 0, eps, curr_density)

        p = curr_density / np.sum(curr_density)
        q = ref_density / np.sum(ref_density)

        kl_val = float(entropy(p, q))
        return kl_val, bool(kl_val >= threshold)
    except Exception as e:
        logger.error(f"KL divergence calculation error: {e}")
        return 0.0, False


# ═══════════════════════════════════════════════
# Error-signal construction for performance-aware detectors
# ═══════════════════════════════════════════════
def build_error_stream(y_true, probs, mode='balanced', threshold=0.5, random_state=42):
    """Turn (labels, predicted probabilities) into the stream DDM/EDDM/HDDM watch.

    Why the choice of signal matters more than the detector
    -------------------------------------------------------
    DDM, EDDM and HDDM are all defined over a Bernoulli error stream and assume
    that a rise in P(error) indicates degradation. Under severe class imbalance
    that assumption breaks down. Fraud prevalence here is ~3.5%, so the 0/1
    error stream is ~96% determined by how the model treats legitimate
    transactions. A model can lose *all* of its fraud-catching ability and move
    the raw error rate by only 3.5 percentage points — well inside the noise
    band that DDM's ``p_min + 3*s_min`` rule tolerates — while a trivial shift
    in the legitimate-transaction mix swamps the signal. Detectors fed this
    stream are effectively monitoring the majority class.

    Modes:
        'zero_one'  — classic 1[ŷ != y]. Provided for baseline comparison.
        'balanced'  — the error stream of a **class-balanced subsample**: every
                      positive, plus an equal number of randomly drawn
                      negatives, kept in time order. The stream's mean is then
                      the balanced error rate 0.5*(FNR + FPR), so a collapse in
                      recall moves it by up to 0.5 instead of by the prevalence.
                      Crucially it stays a genuine Bernoulli stream in {0, 1},
                      which is what DDM's variance term and HDDM's Hoeffding
                      bound both assume — instance *reweighting* would break
                      that (weights of 1/0.035 ≈ 29 are not bounded by 1, and
                      clipping them back to 1 undoes the rebalancing entirely).
        'brier'     — per-instance squared error (y - p)^2, a proper scoring
                      rule. Continuous rather than binary, so it reacts to
                      confidence erosion *before* any label flips — the "early"
                      behaviour EDDM aims for but obtains only indirectly.

    Note that 'balanced' returns a *shorter* stream (2 x n_positives) than the
    input. That is intended: the discarded negatives carried almost no
    information about fraud-detection quality, and the trackers only need the
    stream's statistics, not a one-row-per-transaction correspondence.

    Returns:
        np.ndarray of per-instance error signal in [0, 1].
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(probs, dtype=float)

    if mode == 'brier':
        return np.clip((y - p) ** 2, 0.0, 1.0)

    errors = ((p >= threshold).astype(int) != y.astype(int)).astype(float)
    if mode == 'zero_one':
        return errors

    if mode != 'balanced':
        raise ValueError(f"Unknown error stream mode: {mode!r}")

    pos_idx = np.flatnonzero(y == 1)
    neg_idx = np.flatnonzero(y == 0)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return errors

    rng = np.random.default_rng(random_state)
    n_neg = min(len(pos_idx), len(neg_idx))
    sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False)
    # Sort so the temporal ordering survives — EDDM's inter-error distances and
    # ADWIN's windowing are both order-dependent.
    keep = np.sort(np.concatenate([pos_idx, sampled_neg]))
    return errors[keep]


# ═══════════════════════════════════════════════
# 4. DDM (Drift Detection Method)
#    Gama et al., "Learning with Drift Detection", 2004
# ═══════════════════════════════════════════════
class DDMTracker:
    """Online, persistent DDM tracker that monitors the error rate stream.

    Drift is signalled when  p + s  exceeds  p_min + drift_level * s_min,
    where p is the running error rate and s its standard deviation.

    This tracker is meant to be created ONCE per method and fed successive
    weekly batches via ``process_batch`` — ``p_min``/``s_min`` accumulate
    across the tracker's whole lifetime, exactly as the DDM algorithm
    intends. Recreating a fresh tracker every week (as an earlier version of
    this pipeline did) resets ``min_instances`` gating and ``p_min``/``s_min``
    every time; since fraud models are usually highly accurate, it's easy for
    the first ~30 predictions of a week to contain zero errors, collapsing
    p_min/s_min to 0 and making the very next error look like "drift" even
    though the model is fine — call ``.reset()`` only when the champion model
    itself has actually changed (i.e. this method just retrained).
    """

    def __init__(self, min_instances=30, warning_level=2.0, drift_level=3.0):
        self.min_instances = min_instances
        self.warning_level = warning_level
        self.drift_level = drift_level
        self.reset()

    def reset(self):
        self.n = 0
        self.p = 0.0         # running error rate
        self.s = 0.0         # running std
        self.p_min = float('inf')
        self.s_min = float('inf')
        self.ps_min = float('inf')
        self.warning_detected = False
        self.drift_detected = False

    def update(self, is_error):
        self.n += 1
        self.p += (is_error - self.p) / self.n
        self.s = np.sqrt(self.p * (1.0 - self.p) / self.n) if self.n > 0 else 0.0

        if self.n < self.min_instances:
            return

        ps = self.p + self.s
        if ps < self.ps_min:
            self.p_min = self.p
            self.s_min = self.s
            self.ps_min = ps

        drift_bound = self.p_min + self.drift_level * self.s_min
        warn_bound = self.p_min + self.warning_level * self.s_min

        if ps > drift_bound:
            self.drift_detected = True
            self.warning_detected = False
        elif ps > warn_bound:
            self.warning_detected = True
            self.drift_detected = False
        else:
            self.warning_detected = False
            self.drift_detected = False

    def process_batch(self, error_stream):
        """Feed a batch (e.g. one week) of binary error indicators (1=wrong,
        0=correct) through the tracker, continuing from its prior state.

        Returns:
            dict with drift_detected/warning_detected/drift_occurrences for
            THIS batch, plus the tracker's current (whole-lifetime) p_min,
            s_min, and the thresholds derived from them.
        """
        drift_count = 0
        warning_count = 0

        for err in error_stream:
            self.update(err)
            if self.drift_detected:
                drift_count += 1
            if self.warning_detected:
                warning_count += 1

        mean_err = float(np.mean(error_stream)) if len(error_stream) > 0 else 0.0
        p_min = self.p_min if self.p_min != float('inf') else mean_err
        s_min = self.s_min if self.s_min != float('inf') else 0.0
        drift_threshold = float(p_min + self.drift_level * s_min)
        warning_threshold = float(p_min + self.warning_level * s_min)

        return {
            'drift_detected': bool(drift_count > 0),
            'warning_detected': bool(warning_count > 0),
            'drift_occurrences': drift_count,
            'mean_error_rate': mean_err,
            'p_min': float(p_min),
            's_min': float(s_min),
            'drift_level': float(self.drift_level),
            'warning_level': float(self.warning_level),
            'drift_threshold': drift_threshold,
            'warning_threshold': warning_threshold,
        }


def run_ddm_batch(error_stream):
    """One-shot convenience wrapper: run DDM on a single batch with a fresh
    tracker (no cross-batch history). For the drift pipeline, use
    ``DDMTracker`` directly and keep it alive across weeks instead.

    Returns:
        dict with drift_detected, warning_detected, mean_error_rate,
        p_min, s_min, drift_level, and computed drift_threshold.
    """
    return DDMTracker().process_batch(error_stream)


# ═══════════════════════════════════════════════
# 5. EDDM (Early Drift Detection Method)
#    Baena-García et al., 2006
#
#    Tracks the distance between classification errors.
#    Monitors  p' + 2·s'  (mean + 2·std of inter-error distances).
#    Drift when current value drops below  β · max_historical_value.
# ═══════════════════════════════════════════════
class EDDMTracker:
    """Online, persistent EDDM tracker.

    Monitors the distance (in samples) between consecutive prediction
    errors. The metric ``p' + 2·s'`` (mean + 2*std of inter-error distances,
    computed via Welford's online algorithm so it never needs to replay
    history) is tracked cumulatively, and drift is signalled when it drops
    below ``beta_drift * max_metric_ever_seen``.

    Like ``DDMTracker``, this is meant to be created ONCE per method and fed
    successive weekly batches via ``process_batch`` — an earlier version of
    this pipeline recreated the tracker every week, which reset
    ``max_metric`` and the error-distance history each time and made the
    method far less meaningful (each week judged only against itself).
    Call ``.reset()`` only when the champion model itself has changed (i.e.
    this method just retrained).
    """

    def __init__(self, beta_drift=0.90, beta_warning=0.95, min_errors=10):
        self.beta_drift = beta_drift
        self.beta_warning = beta_warning
        self.min_errors = min_errors
        self.reset()

    def reset(self):
        self.n_distances = 0          # number of inter-error distances seen
        self.mean_d = 0.0             # running mean (Welford)
        self._m2 = 0.0                # running sum of squared deviations (Welford)
        self.samples_since_last_error = None  # None until the first error ever seen
        self.max_metric = 0.0
        self.drift_detected = False
        self.warning_detected = False

    def _current_metric(self):
        std_d = np.sqrt(self._m2 / self.n_distances) if self.n_distances > 0 else 0.0
        return self.mean_d + 2.0 * std_d

    def _observe_distance(self, distance):
        self.n_distances += 1
        delta = distance - self.mean_d
        self.mean_d += delta / self.n_distances
        delta2 = distance - self.mean_d
        self._m2 += delta * delta2

        metric = self._current_metric()
        if metric > self.max_metric:
            self.max_metric = metric

        if self.n_distances >= self.min_errors and self.max_metric > 0:
            if metric < self.beta_drift * self.max_metric:
                self.drift_detected = True
                self.warning_detected = False
            elif metric < self.beta_warning * self.max_metric:
                self.warning_detected = True
                self.drift_detected = False
            else:
                self.drift_detected = False
                self.warning_detected = False

    def process_batch(self, error_stream):
        """Feed a batch (e.g. one week) of binary error indicators through
        the tracker, continuing from its prior state.

        Returns:
            dict with drift_detected/warning_detected for THIS batch, plus
            the tracker's current (whole-lifetime) metric/max_metric and
            the thresholds derived from them.
        """
        errors = np.asarray(error_stream, dtype=int)
        mean_err = float(np.mean(errors)) if len(errors) > 0 else 0.0

        drift_this_batch = False
        warning_this_batch = False
        for e in errors:
            if self.samples_since_last_error is None:
                # No error seen yet at all (ever) — start counting once the first arrives.
                if e == 1:
                    self.samples_since_last_error = 0
                continue
            if e == 1:
                distance = self.samples_since_last_error + 1
                self._observe_distance(distance)
                self.samples_since_last_error = 0
                if self.drift_detected:
                    drift_this_batch = True
                if self.warning_detected:
                    warning_this_batch = True
            else:
                self.samples_since_last_error += 1

        return {
            'drift_detected': drift_this_batch,
            'warning_detected': warning_this_batch and not drift_this_batch,
            'drift_occurrences': 1 if drift_this_batch else 0,
            'mean_error_rate': mean_err,
            'metric_value': float(self._current_metric()) if self.n_distances > 0 else 0.0,
            'max_metric': float(self.max_metric),
            'drift_threshold': float(self.beta_drift * self.max_metric) if self.max_metric > 0 else 0.0,
            'warning_threshold': float(self.beta_warning * self.max_metric) if self.max_metric > 0 else 0.0,
        }


def run_eddm_batch(error_stream, beta_drift=0.90, beta_warning=0.95, min_errors=10):
    """One-shot convenience wrapper: run EDDM on a single batch with a fresh
    tracker (no cross-batch history). For the drift pipeline, use
    ``EDDMTracker`` directly and keep it alive across weeks instead.

    Returns:
        dict with drift_detected, warning_detected, mean_error_rate,
        metric_value, max_metric, drift_threshold, warning_threshold.
    """
    return EDDMTracker(beta_drift=beta_drift, beta_warning=beta_warning, min_errors=min_errors).process_batch(error_stream)


# ═══════════════════════════════════════════════
# 6. ADWIN (Adaptive Windowing)
#    Bifet & Gavaldà, "Learning from Time-Changing Data
#    with Adaptive Windowing", 2007
# ═══════════════════════════════════════════════
def _two_sample_z(w1, w2, delta):
    """Welch's z-test approximation between two independent samples.

    Returns:
        dict with estimation, z_score, z_threshold, drift_detected, warning_detected.
    """
    from scipy.stats import norm

    z_threshold = float(norm.ppf(1 - delta / 2))

    if len(w1) < 2 or len(w2) < 2:
        return {
            'estimation': float(np.mean(w2)) if len(w2) > 0 else 0.0,
            'z_score': 0.0,
            'z_threshold': z_threshold,
            'drift_detected': False,
            'warning_detected': False,
        }

    mean1, mean2 = np.mean(w1), np.mean(w2)
    var1, var2 = np.var(w1, ddof=1), np.var(w2, ddof=1)
    n1, n2 = len(w1), len(w2)

    pooled_se = np.sqrt(var1 / n1 + var2 / n2) if (var1 + var2) > 0 else 1e-10
    z_score = abs(mean1 - mean2) / pooled_se

    is_drift = bool(z_score > z_threshold)
    is_warning = bool(z_score > z_threshold * 0.8) and not is_drift

    return {
        'estimation': float(mean2),
        'z_score': float(z_score),
        'z_threshold': z_threshold,
        'drift_detected': is_drift,
        'warning_detected': is_warning,
    }


def run_adwin_batch(data_stream, delta=0.002, reference_stream=None):
    """Detect a mean shift in a continuous-valued stream (e.g. model predictions)
    relative to a reference/baseline stream.

    ``reference_stream`` should be the same stream captured at the reference
    point (e.g. baseline predictions) — without it, ADWIN can only compare
    early-vs-late values *within* ``data_stream`` itself, which cannot detect
    a shift relative to history if ``data_stream`` is internally stable (as a
    single week's predictions from an already-stable model typically are).

    When River is available, the reference values are fed into ADWIN first to
    establish its window, then the current values are fed in and any change
    point raised while processing them counts as drift. Regardless of
    backend, a two-sample z-test between reference and current is also
    computed and included in the result so downstream consumers can recompute
    the decision live for a different ``delta`` without replaying the stream.

    Args:
        data_stream: Array-like of continuous values for the current window.
        delta: ADWIN confidence parameter (lower = more sensitive).
        reference_stream: Array-like of continuous values from the reference/
            baseline window. If omitted, falls back to an early-vs-late split
            of ``data_stream`` itself.

    Returns:
        dict with drift_detected, warning_detected, estimation, drift_occurrences,
        delta, z_score, z_threshold, backend.
    """
    arr = np.asarray(data_stream, dtype=float)

    if reference_stream is not None and len(reference_stream) > 0:
        ref = np.asarray(reference_stream, dtype=float)
    else:
        half = len(arr) // 2
        ref, arr = arr[:half], arr[half:]

    window_stats = _two_sample_z(ref, arr, delta)

    if RIVER_AVAILABLE and hasattr(drift, 'ADWIN'):
        adwin = drift.ADWIN(delta=delta)
        for val in ref:
            adwin.update(val)
        drift_count = 0
        for val in arr:
            adwin.update(val)
            if adwin.drift_detected:
                drift_count += 1
        return {
            'drift_detected': bool(drift_count > 0),
            'warning_detected': False,
            'drift_occurrences': drift_count,
            'estimation': float(adwin.estimation),
            'delta': float(delta),
            'z_score': window_stats['z_score'],
            'z_threshold': window_stats['z_threshold'],
            'backend': 'river',
        }

    return {
        'drift_detected': window_stats['drift_detected'],
        'warning_detected': window_stats['warning_detected'],
        'drift_occurrences': 1 if window_stats['drift_detected'] else 0,
        'estimation': window_stats['estimation'],
        'z_score': window_stats['z_score'],
        'z_threshold': window_stats['z_threshold'],
        'delta': float(delta),
        'backend': 'fallback',
    }


# ═══════════════════════════════════════════════
# 7. SHAP-BASED DRIFT DETECTION
# ═══════════════════════════════════════════════
class SHAPDriftDetector:
    """Monitors drift via SHAP value distribution shifts between reference and current data."""

    def __init__(self, model):
        self.model = model
        try:
            self.explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
        except Exception as e:
            logger.warning(f"SHAP TreeExplainer init failed: {e}. SHAP drift detection disabled.")
            self.explainer = None

    def _compute_shap_values(self, X_sample):
        """Compute SHAP values, handling list/3D array returns from binary classifiers."""
        if self.explainer is None:
            return np.zeros_like(X_sample)

        shap_vals = self.explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):
            return shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
            return shap_vals[:, :, 1]
        return shap_vals

    def evaluate_drift(self, X_ref, X_curr, top_features, alpha=0.05,
                       min_effect=0.10, sample_size=1000, random_state=42):
        """Compute per-feature SHAP drift between reference and current batches.

        Improvements over a plain per-feature KS on SHAP values:

        * **Random sampling instead of ``.iloc[:200]``.** Taking the first 200
          rows of a time-ordered window samples one particular slice of days —
          for the reference window that means only the earliest days of the
          baseline. Any weekly/seasonal structure then registers as "SHAP
          drift" forever. Rows are now drawn at random.
        * **Benjamini-Hochberg across the monitored set**, because K
          simultaneous KS tests per week otherwise guarantee false positives.
        * **Effect-size gate** on the KS statistic, for the same reason as
          ``ks_test``: significance on thousands of SHAP values is automatic.

        Returns:
            dict with overall_shap_drift_detected, drifted_features_count, and
            per-feature details in feature_shap_drift.
        """
        rng = np.random.default_rng(random_state)

        def _sample(df):
            n = min(sample_size, len(df))
            if n == len(df):
                return df
            idx = rng.choice(len(df), size=n, replace=False)
            return df.iloc[np.sort(idx)]

        ref_shap = self._compute_shap_values(_sample(X_ref))
        curr_shap = self._compute_shap_values(_sample(X_curr))

        all_cols = X_ref.columns.tolist()
        stats, p_values, importances, names = [], [], [], []

        for feat in top_features:
            if feat not in all_cols:
                continue
            idx = all_cols.index(feat)
            ref_col = ref_shap[:, idx] if ref_shap.ndim > 1 else ref_shap
            curr_col = curr_shap[:, idx] if curr_shap.ndim > 1 else curr_shap

            ks_stat, p_val = ks_2samp(ref_col, curr_col)
            names.append(feat)
            stats.append(float(ks_stat))
            p_values.append(float(p_val))
            importances.append((float(np.mean(np.abs(ref_col))),
                                float(np.mean(np.abs(curr_col)))))

        rejected, adjusted = benjamini_hochberg(p_values, alpha=alpha)

        feature_shap_drift = {}
        drift_count = 0
        for i, feat in enumerate(names):
            ref_imp, curr_imp = importances[i]
            is_drift = bool(rejected[i] and stats[i] >= min_effect)
            if is_drift:
                drift_count += 1
            feature_shap_drift[feat] = {
                'ks_stat': stats[i],
                'p_value': p_values[i],
                'p_value_adjusted': float(adjusted[i]),
                'significant_fdr': bool(rejected[i]),
                'is_drift': is_drift,
                'ref_importance': ref_imp,
                'curr_importance': curr_imp,
                'importance_shift': float(curr_imp - ref_imp),
            }

        return {
            'drift_detected': bool(drift_count > 0),
            'warning_detected': bool(drift_count > 0 and drift_count <= 2),
            'overall_shap_drift_detected': bool(drift_count > 0),
            'drifted_features_count': drift_count,
            'alpha': float(alpha),
            'min_effect': float(min_effect),
            'feature_shap_drift': feature_shap_drift,
        }


# ═══════════════════════════════════════════════
# 8. CLUSTERING-BASED DRIFT DETECTION
# ═══════════════════════════════════════════════
class ClusteringDriftDetector:
    """K-Means clustering drift detector.

    Monitors centroid distance shift and cluster assignment distribution
    changes (via PSI) between reference and incoming batches.

    Features are standardized (zero mean, unit variance, fit on the
    reference batch) before K-Means sees them. K-Means uses Euclidean
    distance, so without scaling, whichever raw feature happens to have the
    largest numeric scale (e.g. a dollar amount vs. a small count/flag
    column) dominates every distance calculation — the "clusters" end up
    splitting almost entirely along that one axis, and if it happens to stay
    stable, distance_ratio sits at ~1.0 regardless of real drift elsewhere.
    """

    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.is_fitted = False
        self.ref_cluster_dist = None
        self.ref_mean_dist = 0.0
        self.feature_mean = None
        self.feature_std = None

    def _scale(self, X_clean):
        return (X_clean - self.feature_mean) / self.feature_std

    def fit_reference(self, X_ref):
        """Standardize, fit K-Means on reference data, and store baseline statistics."""
        self.feature_names = list(X_ref.columns)
        X_clean = X_ref.fillna(0).values
        self.feature_mean = np.mean(X_clean, axis=0)
        self.feature_std = np.std(X_clean, axis=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0
        X_scaled = self._scale(X_clean)

        self.kmeans.fit(X_scaled)

        counts = np.bincount(self.kmeans.labels_, minlength=self.n_clusters)
        self.ref_cluster_counts = counts.astype(int)
        self.ref_n_rows = int(len(X_scaled))
        self.ref_cluster_dist = counts / len(X_scaled)

        dists = np.min(self.kmeans.transform(X_scaled), axis=1)
        self.ref_mean_dist = float(np.mean(dists))

        # Per-feature contribution to distance-from-centroid, on the
        # reference: how much of the average squared distance each monitored
        # column is responsible for. This is what evaluate_drift compares
        # against to explain *which features* are driving a distance-ratio
        # change, rather than just reporting one aggregate number.
        assigned_centroids = self.kmeans.cluster_centers_[self.kmeans.labels_]
        self.ref_per_dim_sqdev = np.mean((X_scaled - assigned_centroids) ** 2, axis=0)

        self.is_fitted = True

    def evaluate_drift(self, X_curr, distance_threshold=1.5, psi_threshold=0.2, top_n_features=5):
        """Compare current batch against the fitted reference.

        Returns:
            dict with drift_detected, warning_detected, distance_ratio, cluster_psi,
            per-cluster record counts (reference and current), and the
            monitored features contributing most to any distance shift.
        """
        if not self.is_fitted:
            return {
                'drift_detected': False,
                'warning_detected': False,
                'distance_ratio': 1.0,
                'cluster_psi': 0.0,
            }

        X_clean = self._scale(X_curr.fillna(0).values)
        curr_dists = np.min(self.kmeans.transform(X_clean), axis=1)
        curr_mean_dist = float(np.mean(curr_dists))
        distance_ratio = curr_mean_dist / max(self.ref_mean_dist, 1e-6)

        # Cluster distribution shift via PSI
        curr_labels = self.kmeans.predict(X_clean)
        curr_counts = np.bincount(curr_labels, minlength=self.n_clusters)
        curr_cluster_dist = curr_counts / len(X_clean)

        eps = 1e-4
        p_ref = np.where(self.ref_cluster_dist == 0, eps, self.ref_cluster_dist)
        p_curr = np.where(curr_cluster_dist == 0, eps, curr_cluster_dist)
        cluster_psi = float(np.sum((p_curr - p_ref) * np.log(p_curr / p_ref)))

        is_drift = bool(distance_ratio >= distance_threshold or cluster_psi >= psi_threshold)
        is_warning = bool(
            (distance_ratio >= distance_threshold * 0.8 or cluster_psi >= psi_threshold * 0.5)
            and not is_drift
        )

        # Which monitored features are driving the distance-ratio change:
        # same per-dimension squared-deviation-from-assigned-centroid measure
        # as the reference, compared feature by feature.
        assigned_centroids = self.kmeans.cluster_centers_[curr_labels]
        curr_per_dim_sqdev = np.mean((X_clean - assigned_centroids) ** 2, axis=0)
        per_dim_shift = curr_per_dim_sqdev - self.ref_per_dim_sqdev
        top_idx = np.argsort(per_dim_shift)[::-1][:top_n_features]
        top_contributing_features = [
            {'feature': self.feature_names[i],
             'ref_sqdev': float(self.ref_per_dim_sqdev[i]),
             'curr_sqdev': float(curr_per_dim_sqdev[i]),
             'shift': float(per_dim_shift[i])}
            for i in top_idx
        ]

        return {
            'drift_detected': is_drift,
            'warning_detected': is_warning,
            'distance_ratio': float(distance_ratio),
            'ref_mean_dist': float(self.ref_mean_dist),
            'curr_mean_dist': curr_mean_dist,
            'cluster_psi': float(cluster_psi),
            'n_clusters': int(self.n_clusters),
            'ref_n_rows': self.ref_n_rows,
            'ref_cluster_counts': self.ref_cluster_counts.tolist(),
            'curr_n_rows': int(len(X_clean)),
            'curr_cluster_counts': curr_counts.tolist(),
            'top_contributing_features': top_contributing_features,
        }


# ═══════════════════════════════════════════════
# 9. AUTOENCODER-BASED DRIFT DETECTION
# ═══════════════════════════════════════════════
class AutoencoderDriftDetector:
    """Bottleneck MLP autoencoder drift detector.

    Trains on reference data and monitors reconstruction error (RMSE) shift
    on incoming batches.

    Features are standardized (zero mean, unit variance, fit on the reference
    batch) before being fed to the autoencoder. Without this, features on
    much larger raw scales (e.g. a dollar amount vs. a small integer code)
    dominate the reconstruction error, inflating RMSE and making the z-score
    unstable/uninterpretable regardless of whether real drift occurred.

    Drift is decided purely from the RMSE z-score (an effect-size measure).
    An earlier version also OR'd in a KS-test p-value on the error
    distributions, but with thousands of rows a KS test flags "significant"
    for almost any nonzero difference — it made the method fire on nearly
    every batch. The KS stat/p-value are still reported for visibility, just
    not used to decide drift.
    """

    def __init__(self, random_state=42):
        self.autoencoder = None
        self.is_fitted = False
        self.feature_mean = None
        self.feature_std = None
        self.ref_rmse_mean = 0.0
        self.ref_rmse_std = 0.0
        self.ref_rmses = None
        self.random_state = random_state

    def _scale(self, X_clean):
        return (X_clean - self.feature_mean) / self.feature_std

    def fit_reference(self, X_ref):
        """Standardize, train the autoencoder on reference data, and record
        the baseline RMSE distribution."""
        self.feature_names = list(X_ref.columns)
        X_clean = X_ref.fillna(0).values
        self.feature_mean = np.mean(X_clean, axis=0)
        self.feature_std = np.std(X_clean, axis=0)
        self.feature_std[self.feature_std < 1e-8] = 1.0
        X_scaled = self._scale(X_clean)

        n_features = X_scaled.shape[1]
        bottleneck = max(2, n_features // 2)

        self.autoencoder = MLPRegressor(
            hidden_layer_sizes=(bottleneck, n_features),
            activation='relu',
            solver='adam',
            max_iter=100,
            random_state=self.random_state,
        )
        self.autoencoder.fit(X_scaled, X_scaled)

        ref_recon = self.autoencoder.predict(X_scaled)
        self.ref_rmses = np.sqrt(np.mean((X_scaled - ref_recon) ** 2, axis=1))
        self.ref_rmse_mean = float(np.mean(self.ref_rmses))
        self.ref_rmse_std = float(np.std(self.ref_rmses))
        # Per-feature reconstruction error on the reference — what each
        # monitored column contributes to the aggregate RMSE, so a drifted
        # week can be explained in terms of *which* columns the autoencoder
        # started reconstructing badly, not just "error went up."
        self.ref_per_feature_mse = np.mean((X_scaled - ref_recon) ** 2, axis=0)
        self.is_fitted = True

    def evaluate_drift(self, X_curr, z_score_threshold=3.0, top_n_features=5):
        """Compare current batch reconstruction error (RMSE) against baseline.

        Drift is detected when the mean RMSE z-score exceeds the threshold.

        Returns:
            dict with drift_detected, warning_detected, mse_z_score, ks_stat,
            and the monitored features contributing most to any reconstruction-
            error increase.
        """
        if not self.is_fitted:
            return {
                'drift_detected': False,
                'warning_detected': False,
                'curr_rmse_mean': 0.0,
                'mse_z_score': 0.0,
                'ks_stat': 0.0,
                'p_value': 1.0,
            }

        X_clean = X_curr.fillna(0).values
        X_scaled = self._scale(X_clean)
        curr_recon = self.autoencoder.predict(X_scaled)
        curr_rmses = np.sqrt(np.mean((X_scaled - curr_recon) ** 2, axis=1))

        curr_rmse_mean = float(np.mean(curr_rmses))
        rmse_z_score = (curr_rmse_mean - self.ref_rmse_mean) / max(self.ref_rmse_std, 1e-6)

        ks_stat, p_val = ks_2samp(self.ref_rmses, curr_rmses)

        is_drift = bool(rmse_z_score > z_score_threshold)
        is_warning = bool(rmse_z_score > z_score_threshold * 0.67) and not is_drift

        curr_per_feature_mse = np.mean((X_scaled - curr_recon) ** 2, axis=0)
        per_feature_shift = curr_per_feature_mse - self.ref_per_feature_mse
        top_idx = np.argsort(per_feature_shift)[::-1][:top_n_features]
        top_contributing_features = [
            {'feature': self.feature_names[i],
             'ref_mse': float(self.ref_per_feature_mse[i]),
             'curr_mse': float(curr_per_feature_mse[i]),
             'shift': float(per_feature_shift[i])}
            for i in top_idx
        ]

        return {
            'drift_detected': is_drift,
            'warning_detected': is_warning,
            'ref_rmse_mean': float(self.ref_rmse_mean),
            'curr_rmse_mean': curr_rmse_mean,
            'mse_z_score': float(rmse_z_score),
            'ks_stat': float(ks_stat),
            'p_value': float(p_val),
            'top_contributing_features': top_contributing_features,
        }


# ═══════════════════════════════════════════════
# 10. CHAMPION VS CHALLENGER MODEL DRIFT
# ═══════════════════════════════════════════════
def evaluate_champion_vs_challenger(
    champion_model,
    challenger_model,
    X_curr,
    y_curr,
    baseline_auc=None,
    auc_degradation_thresh=0.05,
    auc_gap_thresh=0.03,
    challenger_probs=None,
):
    """Compare the incumbent champion against a freshly-trained challenger.

    Evaluation bias — why ``challenger_probs`` exists
    -------------------------------------------------
    The natural implementation ("train a challenger on this week, score both on
    this week") is not a fair comparison. The champion is evaluated strictly
    out-of-sample, but the challenger is scored on data it was just fitted to.
    A 500-tree LightGBM memorises a 10k-row weekly window to a substantial
    degree, so its in-sample AUC is inflated by several points — often more
    than the 0.03 ``auc_gap_thresh``. The method therefore fires on the
    challenger's overfitting rather than on the champion's staleness, and it
    fires in *every* week, drifting or not, because the bias is constant.

    Callers should pass ``challenger_probs`` containing **out-of-fold**
    predictions (see ``oof_predictions`` in the pipeline), which places both
    models on out-of-sample footing and makes ``auc_gap`` an unbiased estimate
    of the benefit of retraining. ``challenger_model`` is still used as a
    fallback so the older calling convention keeps working, with the caveat above.

    Drift is detected when:
      1. The champion's AUC has dropped more than ``auc_degradation_thresh``
         from its baseline, OR
      2. The challenger beats the champion by more than ``auc_gap_thresh``
         AND that gap is larger than its own bootstrap standard error.

    Returns:
        dict with drift_detected, champion_auc, challenger_auc, auc_gap, etc.
    """
    probs_champ = champion_model.predict(X_curr, predict_disable_shape_check=True)
    if challenger_probs is not None:
        probs_chall = np.asarray(challenger_probs, dtype=float)
        challenger_eval = 'out_of_fold'
    else:
        probs_chall = challenger_model.predict(X_curr, predict_disable_shape_check=True)
        challenger_eval = 'in_sample_biased'

    has_both_classes = len(np.unique(y_curr)) > 1

    auc_champ = float(roc_auc_score(y_curr, probs_champ)) if has_both_classes else 0.5
    auc_chall = float(roc_auc_score(y_curr, probs_chall)) if has_both_classes else 0.5

    preds_champ = (probs_champ >= 0.5).astype(int)
    preds_chall = (probs_chall >= 0.5).astype(int)
    f1_champ = float(f1_score(y_curr, preds_champ, zero_division=0))
    f1_chall = float(f1_score(y_curr, preds_chall, zero_division=0))

    # Prediction distribution divergence
    ks_stat, ks_p = ks_2samp(probs_champ, probs_chall)
    psi_val, _ = calculate_psi(probs_champ, probs_chall)

    # Gap: positive means challenger is better
    auc_gap = auc_chall - auc_champ

    # Degradation from baseline
    auc_degradation = 0.0
    if baseline_auc is not None:
        auc_degradation = baseline_auc - auc_champ

    # Bootstrap standard error of the AUC gap. A gap of 0.04 on a week with 200
    # fraud cases has an SE of roughly the same size — declaring drift from it
    # is reading noise. Requiring the gap to clear its own uncertainty is what
    # turns this from a point comparison into a test.
    gap_se = _paired_auc_gap_se(y_curr, probs_champ, probs_chall) if has_both_classes else 0.0
    gap_is_significant = bool(auc_gap > max(auc_gap_thresh, gap_se))

    is_drift = bool(
        auc_degradation > auc_degradation_thresh   # champion has degraded from baseline
        or gap_is_significant                       # challenger reliably outperforms champion
    )
    is_warning = bool(
        (auc_degradation > auc_degradation_thresh * 0.5 or auc_gap > auc_gap_thresh * 0.5)
        and not is_drift
    )

    return {
        'drift_detected': is_drift,
        'warning_detected': is_warning,
        'champion_auc': auc_champ,
        'challenger_auc': auc_chall,
        'auc_gap': float(auc_gap),
        'auc_gap_se': float(gap_se),
        'auc_gap_significant': gap_is_significant,
        'challenger_evaluation': challenger_eval,
        'auc_degradation': float(auc_degradation),
        'baseline_auc': float(baseline_auc) if baseline_auc is not None else None,
        'champion_f1': f1_champ,
        'challenger_f1': f1_chall,
        'prediction_ks_stat': float(ks_stat),
        'prediction_psi': float(psi_val),
    }


def _paired_auc_gap_se(y, probs_a, probs_b, n_boot=200, random_state=42):
    """Bootstrap SE of (AUC_b - AUC_a), resampling rows paired across models."""
    y = np.asarray(y)
    n = len(y)
    if n < 50:
        return 0.0

    rng = np.random.default_rng(random_state)
    gaps = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if len(np.unique(yb)) < 2:
            continue
        gaps.append(roc_auc_score(yb, probs_b[idx]) - roc_auc_score(yb, probs_a[idx]))
    return float(np.std(gaps)) if len(gaps) > 1 else 0.0


# ═══════════════════════════════════════════════
# 11. HDDM (Hoeffding's-bound Drift Detection Method)
#     Frías-Blanco et al., IEEE TKDE 2015
# ═══════════════════════════════════════════════
class HDDMTracker:
    """Drift detection via Hoeffding's inequality on the error stream.

    DDM's control limits assume the error stream is Bernoulli with a variance
    that shrinks as 1/n, which makes it progressively harder to trigger the
    longer a model has been stable — precisely the regime where drift is most
    likely. HDDM replaces the normal-approximation limits with a
    distribution-free Hoeffding bound, which holds for any bounded stream. That
    matters here because ``build_error_stream`` can emit a continuous signal
    (Brier residuals, balanced weights) for which DDM's Bernoulli variance term
    is simply wrong, while Hoeffding's bound only requires values in [0, 1].

    Backed by ``river.drift.binary.HDDM_A`` when available; otherwise a direct
    implementation of the same two-window Hoeffding test.
    """

    def __init__(self, drift_confidence=0.001, warning_confidence=0.005):
        self.drift_confidence = drift_confidence
        self.warning_confidence = warning_confidence
        self._river = None
        if RIVER_AVAILABLE:
            try:
                from river.drift.binary import HDDM_A
                self._river = HDDM_A(drift_confidence=drift_confidence,
                                     warning_confidence=warning_confidence)
                self._river_cls = HDDM_A
            except Exception as e:
                logger.warning(f"river HDDM_A unavailable ({e}); using internal implementation.")
        self.reset()

    def reset(self):
        self.n = 0
        self.total = 0.0
        self.n_min = 0
        self.total_min = 0.0
        self.drift_detected = False
        self.warning_detected = False
        if self._river is not None:
            self._river = self._river_cls(
                drift_confidence=self.drift_confidence,
                warning_confidence=self.warning_confidence,
            )

    @staticmethod
    def _hoeffding_bound(n, confidence):
        """epsilon such that P(|mean - E[mean]| > epsilon) <= confidence."""
        return float(np.sqrt(np.log(2.0 / confidence) / (2.0 * max(n, 1))))

    def _update_internal(self, value):
        self.n += 1
        self.total += value
        mean = self.total / self.n

        # Track the window with the lowest observed mean-plus-bound: the
        # "best the model has ever looked" reference point.
        if self.n_min == 0:
            self.n_min, self.total_min = self.n, self.total
        else:
            mean_min = self.total_min / self.n_min
            if mean + self._hoeffding_bound(self.n, self.drift_confidence) <= \
               mean_min + self._hoeffding_bound(self.n_min, self.drift_confidence):
                self.n_min, self.total_min = self.n, self.total

        mean_min = self.total_min / self.n_min
        eps_drift = (self._hoeffding_bound(self.n_min, self.drift_confidence)
                     + self._hoeffding_bound(self.n, self.drift_confidence))
        eps_warn = (self._hoeffding_bound(self.n_min, self.warning_confidence)
                    + self._hoeffding_bound(self.n, self.warning_confidence))

        self.drift_detected = bool(mean - mean_min > eps_drift)
        self.warning_detected = bool(not self.drift_detected and mean - mean_min > eps_warn)

    def process_batch(self, error_stream):
        """Feed one window of bounded error values, continuing prior state."""
        errors = np.clip(np.asarray(error_stream, dtype=float), 0.0, 1.0)
        drift_count = warning_count = 0

        for value in errors:
            if self._river is not None:
                self._river.update(int(value >= 0.5))
                if self._river.drift_detected:
                    drift_count += 1
                    self._river = self._river_cls(
                        drift_confidence=self.drift_confidence,
                        warning_confidence=self.warning_confidence,
                    )
                elif getattr(self._river, 'warning_detected', False):
                    warning_count += 1
            else:
                self._update_internal(value)
                if self.drift_detected:
                    drift_count += 1
                elif self.warning_detected:
                    warning_count += 1

        mean_err = float(np.mean(errors)) if len(errors) else 0.0
        mean_min = (self.total_min / self.n_min) if self.n_min else mean_err
        return {
            'drift_detected': bool(drift_count > 0),
            'warning_detected': bool(warning_count > 0 and drift_count == 0),
            'drift_occurrences': drift_count,
            'mean_error_rate': mean_err,
            'reference_mean': float(mean_min),
            'hoeffding_bound': self._hoeffding_bound(max(self.n, 1), self.drift_confidence),
            'drift_confidence': float(self.drift_confidence),
            'backend': 'river' if self._river is not None else 'internal',
        }


# ═══════════════════════════════════════════════
# 12. PREQUENTIAL AUC DEGRADATION
# ═══════════════════════════════════════════════
class PrequentialAUCDetector:
    """Monitors the champion's out-of-sample AUC window over window.

    Every error-stream detector (DDM/EDDM/ADWIN/HDDM) is a proxy for the thing
    an operator actually cares about: has ranking quality dropped? Under 3.5%
    prevalence and a 0.5 decision threshold those proxies are dominated by the
    majority class, and threshold-free ranking quality can degrade materially
    with almost no movement in 0/1 error. This detector measures the quantity
    of interest directly, and — unlike champion-vs-challenger — without paying
    to train a second model every window.

    Drift is declared when the current window's AUC falls below the reference
    AUC by more than ``min_drop`` AND by more than ``n_sigma`` bootstrap
    standard errors, so noisy low-fraud weeks cannot trigger retraining on
    their own.
    """

    def __init__(self, min_drop=0.02, n_sigma=2.0, n_boot=200, random_state=42):
        self.min_drop = min_drop
        self.n_sigma = n_sigma
        self.n_boot = n_boot
        self.random_state = random_state
        self.reference_auc = None
        self.history = []

    def set_reference(self, y, probs):
        """Anchor the detector at the incumbent model's out-of-sample AUC."""
        self.reference_auc = self._auc(y, probs)
        self.history = []
        return self.reference_auc

    @staticmethod
    def _auc(y, probs):
        y = np.asarray(y)
        return float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.5

    def _auc_se(self, y, probs):
        y = np.asarray(y)
        n = len(y)
        if n < 50:
            return 0.0
        rng = np.random.default_rng(self.random_state)
        vals = []
        for _ in range(self.n_boot):
            idx = rng.integers(0, n, size=n)
            if len(np.unique(y[idx])) < 2:
                continue
            vals.append(roc_auc_score(y[idx], probs[idx]))
        return float(np.std(vals)) if len(vals) > 1 else 0.0

    def evaluate_drift(self, y, probs):
        curr_auc = self._auc(y, probs)
        if self.reference_auc is None:
            self.reference_auc = curr_auc

        se = self._auc_se(y, probs)
        drop = self.reference_auc - curr_auc
        significant = bool(drop > self.n_sigma * se) if se > 0 else bool(drop > self.min_drop)

        is_drift = bool(drop > self.min_drop and significant)
        is_warning = bool(not is_drift and drop > self.min_drop * 0.5)

        self.history.append(curr_auc)
        return {
            'drift_detected': is_drift,
            'warning_detected': is_warning,
            'current_auc': curr_auc,
            'reference_auc': float(self.reference_auc),
            'auc_drop': float(drop),
            'auc_se': float(se),
            'min_drop': float(self.min_drop),
            'n_sigma': float(self.n_sigma),
            'n_positives': int(np.sum(np.asarray(y) == 1)),
        }


# ═══════════════════════════════════════════════
# 13. PERSISTENCE / HYSTERESIS WRAPPER
# ═══════════════════════════════════════════════
class DriftConfirmation:
    """Requires drift to persist for k of the last n windows before acting.

    Retraining is expensive and irreversible in the sense that it resets every
    reference statistic downstream. A single-window trigger therefore converts
    each detector's per-window false-positive rate directly into a retraining
    rate — which is how the original pipeline reached "DDM retrained in 14 of
    14 weeks". Requiring k-of-n agreement across consecutive windows is the
    standard control-chart remedy: for independent windows it reduces the false
    alarm probability from p to roughly C(n,k) p^k, at the cost of delaying a
    genuine detection by at most (k-1) windows.

    With the default k=2, n=2 a detector must fire in two consecutive weeks —
    turning a 20% weekly false-positive rate into ~4%, while a real, persistent
    drift is still acted on with only one week of delay.
    """

    def __init__(self, k=2, n=2):
        if k > n:
            raise ValueError("k must not exceed n")
        self.k = k
        self.n = n
        self.window = []

    def update(self, raw_flag):
        """Record this window's raw detector output; return the confirmed flag."""
        self.window.append(bool(raw_flag))
        if len(self.window) > self.n:
            self.window = self.window[-self.n:]
        return bool(sum(self.window) >= self.k)

    def reset(self):
        """Clear history — call after a retrain, when the model has changed."""
        self.window = []

    @property
    def state(self):
        return {'k': self.k, 'n': self.n, 'recent_flags': list(self.window),
                'confirmed': bool(sum(self.window) >= self.k)}
