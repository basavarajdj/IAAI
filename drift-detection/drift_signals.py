"""
Drift Signal Matrix — the agent's observation space
====================================================

Every detector in drift_engine.py compares a *reference window* to a *current
window*. In the RL setting the reference changes whenever the agent retrains,
so a signal is a function of two things: which model version is the reference,
and which week is being observed.

This module precomputes that whole matrix — `signals[reference_week][week]` —
so the environment can look up any state instantly instead of recomputing
detectors during PPO rollouts.

What the agent sees, and why these signals
------------------------------------------
The signal vector deliberately mixes detector *families*, because the central
claim of this work is that they fail in different, complementary ways:

  distributional (KS / PSI / JS)  — see covariate shift, blind to concept drift,
                                    need no labels
  attribution   (gradient x input) — sees the model's *reliance* shifting, even
                                    when the raw feature marginals are stable
  representation (clustering / AE) — see joint-geometry changes a per-feature
                                    test misses
  performance   (error rate / AUC) — see what actually matters, but only after
                                    labels arrive

A single detector must pick one of these trade-offs. The agent does not have to:
it observes all of them and learns which to trust in which regime. That is the
entire argument for combining them.

Signals are stored as continuous statistics, not booleans. A thresholded flag
throws away exactly the information the agent needs — "PSI is 0.19" and "PSI is
0.02" are both "no drift" to a fixed rule, and obviously different to a learner.
"""

import logging

import numpy as np
from scipy.stats import ks_2samp

from drift_engine import (
    benjamini_hochberg,
    calculate_js_distance,
    ks_test,
    psi_with_null,
    ClusteringDriftDetector,
    AutoencoderDriftDetector,
)

logger = logging.getLogger(__name__)

# Order matters: this is the agent's input layout and the explainer reads it.
SIGNAL_NAMES = [
    'ks_drift_fraction',        # share of monitored features failing KS (FDR + effect size)
    'ks_mean_statistic',        # mean KS D — magnitude, not just count
    'psi_drift_fraction',
    'psi_mean_ratio',           # PSI / its bootstrap null, so 1.0 means "exactly noise"
    'js_mean_distance',
    'attribution_drift_fraction',
    'attribution_mean_shift',
    'cluster_distance_ratio',
    'cluster_psi',
    'autoencoder_z_score',
    'weeks_since_reference',    # how stale the reference itself is
]

N_SIGNALS = len(SIGNAL_NAMES)


def _feature_signals(ref_X, curr_X, features, n_bootstrap=15):
    """KS / PSI / JS summarised across the monitored feature set.

    `n_bootstrap` is lower than the pipeline default. This runs for every
    (reference, week) pair — a few hundred times — and the bootstrap null only
    needs to locate a quantile roughly, not precisely; the agent sees the
    continuous ratio rather than a threshold decision, so a slightly noisy null
    costs far less here than it would in a fixed-threshold detector.
    """
    ks_stats, ks_pvals, psi_ratios, psi_flags, js_values = [], [], [], [], []

    for feat in features:
        r, c = ref_X[feat].values, curr_X[feat].values

        ks = ks_test(r, c)
        ks_stats.append(ks['statistic'])
        ks_pvals.append(ks['p_value'])

        psi = psi_with_null(r, c, n_bootstrap=n_bootstrap)
        psi_ratios.append(min(psi['psi_ratio'], 20.0))     # clip: ratios can explode
        psi_flags.append(psi['drift'])

        js_values.append(calculate_js_distance(r, c))

    rejected, _ = benjamini_hochberg(ks_pvals, alpha=0.05)
    ks_drift = [bool(rejected[i] and ks_stats[i] >= 0.10) for i in range(len(features))]
    n = max(len(features), 1)

    return {
        'ks_drift_fraction': sum(ks_drift) / n,
        'ks_mean_statistic': float(np.mean(ks_stats)),
        'psi_drift_fraction': sum(psi_flags) / n,
        'psi_mean_ratio': float(np.mean(psi_ratios)),
        'js_mean_distance': float(np.mean(js_values)),
    }


def _attribution_signals(model, ref_X, curr_X, features, alpha=0.05, min_effect=0.10):
    """Has the model's *reliance* on each monitored feature shifted?

    Uses gradient x input rather than exact SHAP — see
    NeuralFraudModel.gradient_attributions for why. The decision rule is the
    same one the SHAP detector uses: KS between reference and current
    attribution distributions, FDR-corrected, with an effect-size floor.
    """
    cols = list(ref_X.columns)
    ref_attr = model.gradient_attributions(ref_X.values)
    curr_attr = model.gradient_attributions(curr_X.values)

    stats, pvals, shifts = [], [], []
    for feat in features:
        j = cols.index(feat)
        stat, p = ks_2samp(ref_attr[:, j], curr_attr[:, j])
        stats.append(float(stat))
        pvals.append(float(p))
        shifts.append(float(np.abs(curr_attr[:, j]).mean() - np.abs(ref_attr[:, j]).mean()))

    rejected, _ = benjamini_hochberg(pvals, alpha=alpha)
    drifted = [bool(rejected[i] and stats[i] >= min_effect) for i in range(len(features))]

    return {
        'attribution_drift_fraction': sum(drifted) / max(len(features), 1),
        'attribution_mean_shift': float(np.mean(np.abs(shifts))),
    }


class DriftSignalMatrix:
    """Precomputes every detector signal for every (reference, week) pair."""

    def __init__(self, ref_X, ref_y, weekly_X, weekly_y, features, lattice,
                 sample_size=8000, seed=42):
        self.ref_X, self.ref_y = ref_X, ref_y
        self.weekly_X, self.weekly_y = weekly_X, weekly_y
        self.features = features
        self.lattice = lattice
        self.weeks = sorted(weekly_X)
        self.sample_size = sample_size
        self.rng = np.random.default_rng(seed)
        self.signals = {}          # reference_week -> {week -> np.array(N_SIGNALS)}

    def _subsample(self, X):
        """Cap a window's size. Both windows are capped so the matrix stays
        affordable across a few hundred (reference, week) pairs; 8k rows resolve
        a KS statistic to ~0.02, well below the effect sizes that matter."""
        if len(X) <= self.sample_size:
            return X.reset_index(drop=True)
        idx = np.sort(self.rng.choice(len(X), self.sample_size, replace=False))
        return X.iloc[idx].reset_index(drop=True)

    def _reference_window(self, ref_week):
        """The data the model with this reference was trained on, subsampled."""
        import pandas as pd
        frames = [self.ref_X]
        for w in self.weeks:
            if w <= ref_week:
                frames.append(self.weekly_X[w])
        return self._subsample(pd.concat(frames, ignore_index=True))

    def build(self):
        logger.info(f"Building drift signal matrix over {len(self.lattice.full_models)} "
                    f"references x {len(self.weeks)} weeks...")

        for ref_week, model in self.lattice.full_models.items():
            ref_X = self._reference_window(ref_week)

            # Representation detectors are fitted once per reference, not per week.
            cluster = ClusteringDriftDetector(n_clusters=5)
            cluster.fit_reference(ref_X[self.features])
            auto = AutoencoderDriftDetector()
            auto.fit_reference(ref_X[self.features])

            per_week = {}
            for w in self.weeks:
                if w <= ref_week:
                    continue
                curr_X = self._subsample(self.weekly_X[w])

                values = {}
                values.update(_feature_signals(ref_X, curr_X, self.features))
                values.update(_attribution_signals(model, ref_X, curr_X, self.features))

                c = cluster.evaluate_drift(curr_X[self.features])
                values['cluster_distance_ratio'] = c['distance_ratio']
                values['cluster_psi'] = c['cluster_psi']

                a = auto.evaluate_drift(curr_X[self.features])
                values['autoencoder_z_score'] = float(np.clip(a['mse_z_score'], -10, 10))

                values['weeks_since_reference'] = float(w - ref_week)

                per_week[w] = np.array([values[k] for k in SIGNAL_NAMES], dtype=np.float32)

            self.signals[ref_week] = per_week
            logger.info(f"  reference week {ref_week}: {len(per_week)} weeks of signals")

        return self

    def get(self, ref_week, week):
        vec = self.signals.get(ref_week, {}).get(week)
        return np.zeros(N_SIGNALS, dtype=np.float32) if vec is None else vec

    def as_frame(self, ref_week=0):
        """Signals against a fixed reference, for inspection and plotting."""
        import pandas as pd
        rows = self.signals.get(ref_week, {})
        return pd.DataFrame(
            [dict(zip(SIGNAL_NAMES, v), week=w) for w, v in sorted(rows.items())]
        ).set_index('week')
