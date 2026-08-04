"""
Retraining-Policy Evaluation
============================

A drift detector is only interesting as a *retraining policy*: a rule that, at
each window, says which model version should be in force. This module evaluates
those policies on the outcome that matters — the quality of the model that was
actually serving traffic — rather than on detection accuracy against a
hypothetical ground-truth change point.

The version lattice
-------------------
Under cumulative retraining a model version is fully determined by the week it
was trained at, so the complete space of reachable models is

    v0  = baseline (first 90 days)
    vw  = all data through week w,   w = 1 .. n_weeks

That is at most ``1 + n_weeks`` models — 15 for a 14-week replay. Once all of
them exist, **every** policy is a lookup rather than a training run: a policy is
just a map from week to version id, and its performance is read out of a
precomputed (version x week) performance matrix. This is what makes the random
control below affordable; evaluating 200 random policies by retraining would be
thousands of fits, but against the matrix it is instantaneous.

Only strictly out-of-sample cells are populated. A version trained through week
*w* has seen weeks <= w, so its performance is recorded only for weeks > w.

The random control
------------------
This is the control the comparison lives or dies on. A detector that triggers
six retrains is only informative if it beats a policy that retrains six times at
*randomly chosen* weeks. Without that comparison, apparent detector skill may be
entirely explained by retraining frequency — more retraining generally helps, so
any trigger-happy detector looks good on quality alone. We report each
detector's percentile against its own frequency-matched random ensemble.
"""

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score

logger = logging.getLogger(__name__)

__all__ = [
    'build_performance_matrix',
    'evaluate_policy',
    'reference_policies',
    'detector_policies',
    'random_policy_ensemble',
    'agreement_matrix',
    'run_policy_evaluation',
]


# ══════════════════════════════════════════════
# Performance matrix
# ══════════════════════════════════════════════
def build_performance_matrix(registry, weekly_X, weekly_y, get_threshold):
    """Score every model version on every week it did not train on.

    Args:
        registry: ``ModelRegistry`` holding the materialised version lattice.
        weekly_X / weekly_y: dicts keyed by week index (1-based).
        get_threshold: ``model -> float`` decision threshold accessor.

    Returns:
        (auc_df, f1_df) indexed by version_id, columns = week. NaN marks cells
        where the version had already trained on that week.
    """
    weeks = sorted(weekly_X)
    version_ids = sorted(registry.versions)

    auc = pd.DataFrame(np.nan, index=version_ids, columns=weeks, dtype=float)
    f1 = pd.DataFrame(np.nan, index=version_ids, columns=weeks, dtype=float)

    for vid in version_ids:
        version = registry.versions[vid]
        cut = get_threshold(version.model)
        for w in weeks:
            if w <= version.week:
                continue                      # in-sample for this version
            X, y = weekly_X[w], weekly_y[w]
            probs = version.model.predict(X, predict_disable_shape_check=True)
            if y.nunique() > 1:
                auc.loc[vid, w] = float(roc_auc_score(y, probs))
            f1.loc[vid, w] = float(f1_score(y, (probs >= cut).astype(int), zero_division=0))

    logger.info(f"Performance matrix built: {len(version_ids)} versions x {len(weeks)} weeks "
                f"({int(auc.notna().sum().sum())} out-of-sample cells).")
    return auc, f1


# ══════════════════════════════════════════════
# Policies
# ══════════════════════════════════════════════
def _policy_from_retrain_weeks(retrain_weeks, weeks):
    """Map each week to the version in force, given the weeks a policy retrained.

    A retrain at week w produces version vw, which can only serve weeks *after*
    w — the decision to retrain is made once week w has already been scored.
    """
    retrain_weeks = sorted(set(retrain_weeks))
    in_force, current = {}, 0
    for w in weeks:
        in_force[w] = current
        if w in retrain_weeks:
            current = w
    return in_force


def evaluate_policy(in_force, auc_df, f1_df, name=''):
    """Read a policy's prequential performance out of the matrix."""
    weeks = sorted(in_force)
    aucs, f1s, missing = [], [], 0

    for w in weeks:
        vid = in_force[w]
        if vid not in auc_df.index or w not in auc_df.columns:
            missing += 1
            continue
        a, f = auc_df.loc[vid, w], f1_df.loc[vid, w]
        if not np.isnan(a):
            aucs.append(float(a))
        if not np.isnan(f):
            f1s.append(float(f))

    n_retrains = len(set(in_force.values()) - {0})
    return {
        'policy': name,
        'n_retrains': n_retrains,
        'mean_auc': float(np.mean(aucs)) if aucs else float('nan'),
        'min_auc': float(np.min(aucs)) if aucs else float('nan'),
        'mean_f1': float(np.mean(f1s)) if f1s else float('nan'),
        'min_f1': float(np.min(f1s)) if f1s else float('nan'),
        'weeks_evaluated': len(aucs),
        'weeks_unavailable': missing,
        'versions_in_force': dict(in_force),
    }


def reference_policies(weeks):
    """The three policies every detector must be judged against."""
    return {
        # Floor: the baseline model forever.
        'never_retrain': {w: 0 for w in weeks},
        # Ceiling and cost upper bound: retrain after every window.
        'always_retrain': _policy_from_retrain_weeks(weeks, weeks),
    }


def detector_policies(retrain_triggers, weeks, methods):
    """Reconstruct each detector's realised policy from its retrain triggers."""
    out = {}
    for m in methods:
        rw = sorted({t['week'] for t in retrain_triggers if t['method'] == m})
        out[m] = _policy_from_retrain_weeks(rw, weeks)
    return out


def random_policy_ensemble(n_retrains, weeks, n_samples=200, random_state=42):
    """Frequency-matched random policies — the control for retraining count."""
    rng = np.random.default_rng(random_state)
    candidates = [w for w in weeks]
    n_retrains = min(n_retrains, len(candidates))
    if n_retrains == 0:
        return [{w: 0 for w in weeks}]

    policies = []
    for _ in range(n_samples):
        chosen = rng.choice(candidates, size=n_retrains, replace=False)
        policies.append(_policy_from_retrain_weeks(chosen, weeks))
    return policies


# ══════════════════════════════════════════════
# Detector agreement
# ══════════════════════════════════════════════
def agreement_matrix(weekly_records, methods, flag_key='drift_method_flags'):
    """Pairwise Jaccard agreement over the weeks each detector flagged.

    Twelve detectors that always agree are not twelve independent opinions.
    High agreement between detectors from *different* families (distributional
    vs. performance-aware) is the interesting case — it suggests the drift is
    visible in both the marginals and the concept.
    """
    flagged = {
        m: {r['week'] for r in weekly_records
            if not r.get('is_baseline') and r.get(flag_key, {}).get(m)}
        for m in methods
    }

    mat = pd.DataFrame(np.nan, index=methods, columns=methods, dtype=float)
    for m in methods:
        mat.loc[m, m] = 1.0
    for a, b in combinations(methods, 2):
        union = flagged[a] | flagged[b]
        j = len(flagged[a] & flagged[b]) / len(union) if union else np.nan
        mat.loc[a, b] = mat.loc[b, a] = j
    return mat, {m: sorted(v) for m, v in flagged.items()}


# ══════════════════════════════════════════════
# Orchestration
# ══════════════════════════════════════════════
def run_policy_evaluation(registry, weekly_X, weekly_y, weekly_records,
                          retrain_triggers, methods, get_threshold,
                          n_random=200, random_state=42):
    """Full policy comparison: detectors vs. references vs. random controls."""
    weeks = sorted(weekly_X)
    auc_df, f1_df = build_performance_matrix(registry, weekly_X, weekly_y, get_threshold)

    rows = []
    for name, pol in reference_policies(weeks).items():
        rows.append(evaluate_policy(pol, auc_df, f1_df, name=name))

    det_policies = detector_policies(retrain_triggers, weeks, methods)
    random_summaries = {}

    for m, pol in det_policies.items():
        result = evaluate_policy(pol, auc_df, f1_df, name=m)

        # Frequency-matched random control
        ensemble = random_policy_ensemble(result['n_retrains'], weeks,
                                          n_samples=n_random, random_state=random_state)
        rand_aucs = [evaluate_policy(p, auc_df, f1_df)['mean_auc'] for p in ensemble]
        rand_aucs = [a for a in rand_aucs if not np.isnan(a)]

        # With zero retrains the "random ensemble" is just the never-retrain
        # policy, so a percentile of 1.0 would be arithmetic, not evidence.
        if result['n_retrains'] == 0:
            rand_aucs = []

        if rand_aucs and not np.isnan(result['mean_auc']):
            pct = float(np.mean([result['mean_auc'] >= a for a in rand_aucs]))
            result['random_control_percentile'] = pct
            result['random_control_mean_auc'] = float(np.mean(rand_aucs))
            result['beats_random_control'] = bool(pct >= 0.95)
        else:
            result['random_control_percentile'] = None
            result['random_control_mean_auc'] = None
            result['beats_random_control'] = None

        random_summaries[m] = {
            'n_samples': len(rand_aucs),
            'mean': float(np.mean(rand_aucs)) if rand_aucs else None,
            'std': float(np.std(rand_aucs)) if rand_aucs else None,
        }
        rows.append(result)

    summary = pd.DataFrame(rows).sort_values('mean_auc', ascending=False)
    agree, flagged_weeks = agreement_matrix(weekly_records, methods)

    logger.info("Retraining-policy comparison (mean out-of-sample AUC):")
    for _, r in summary.iterrows():
        ctrl = r.get('random_control_percentile')
        ctrl_txt = f", random-control pct={ctrl:.2f}" if isinstance(ctrl, float) else ""
        logger.info(f"  {r['policy']:24s} retrains={int(r['n_retrains']):2d} "
                    f"mean_auc={r['mean_auc']:.4f} min_auc={r['min_auc']:.4f}{ctrl_txt}")

    return {
        'summary': summary,
        'auc_matrix': auc_df,
        'f1_matrix': f1_df,
        'agreement': agree,
        'flagged_weeks': flagged_weeks,
        'random_controls': random_summaries,
    }
