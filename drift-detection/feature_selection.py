"""
Feature Selection for Drift Monitoring
======================================

Selecting *which* features to monitor is not the same problem as selecting
which features to train on, and the original pipeline conflated them: it took
the top-K by a single LightGBM gain ranking and monitored those.

Three things go wrong with a single-fit gain ranking:

1. **It is unstable.** Gain is computed from one stochastic fit (row/column
   subsampling, a seed). Re-fit with a different seed and the top-10 can change
   substantially. A monitoring set that is itself arbitrary makes the
   "fraction of monitored features that drifted" statistic arbitrary too.
2. **It is redundant.** Gain is split among correlated features essentially at
   random, and correlated features drift *together*. Ten features that are
   really three underlying signals give a "6 of 10 features drifted" vote that
   is not 6 independent pieces of evidence — it inflates the apparent
   agreement of a consensus rule and understates its variance.
3. **It ignores monitorability.** A near-constant or very low-cardinality
   feature can carry real gain yet be untestable: a KS test on a column with
   three distinct values is dominated by ties, and PSI bucketing collapses.

``select_monitoring_features`` addresses all three:

    Stage 1 — Bagged importance. Fit ``n_bags`` LightGBM models on bootstrap
              resamples with distinct seeds; rank features within each fit.
              Aggregate by mean reciprocal rank and record how often each
              feature lands in the top-M ("selection frequency").
    Stage 2 — SHAP corroboration. Global mean |SHAP| on a sample from the
              reference model, as a second, split-count-independent view of
              importance. Gain is biased toward high-cardinality features;
              SHAP attribution is not, so disagreement between the two is a
              useful red flag.
    Stage 3 — Monitorability filter. Drop features that no distributional test
              can meaningfully evaluate (too few distinct values, or
              effectively zero variance in the reference window).
    Stage 4 — Redundancy pruning. Walk the ranking top-down and skip any
              feature whose |Spearman ρ| against an already-selected feature
              exceeds ``redundancy_rho``. This is what makes the K monitored
              features approximately independent tests.

It also reports Nogueira's stability index for the bagged selection, which is
the quantity to cite when claiming the monitoring set is reproducible.
"""

import logging
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata

logger = logging.getLogger(__name__)

__all__ = ['select_monitoring_features', 'nogueira_stability']


def _feature_family(name):
    """Group a feature with its numbered siblings (``_freq_ref_C1`` and
    ``_freq_ref_C14`` share a family; ``D2`` and ``D15`` share a different one).

    Pairwise Spearman pruning alone under-corrects for these families: a chain
    of moderately-correlated siblings (C1 vs C5 at 0.75, C5 vs C14 at 0.75) can
    all clear a single-pair 0.90 threshold while the group as a whole still
    votes as one redundant block. Stripping the trailing digits collapses each
    numbered family to one key, so a per-family cap can catch what pairwise
    correlation misses.
    """
    return re.sub(r'\d+$', '', name)


def nogueira_stability(selection_matrix):
    """Nogueira et al. (2018) stability index for a set of feature selections.

    Args:
        selection_matrix: (n_bags, n_features) binary matrix; entry (i, j) is 1
            if bag i selected feature j.

    Returns:
        Stability in (-inf, 1]. 1 = every bag chose the identical set; ~0 = no
        more agreement than chance.
    """
    Z = np.asarray(selection_matrix, dtype=float)
    n_bags, n_features = Z.shape
    if n_bags < 2 or n_features == 0:
        return 1.0

    p_hat = Z.mean(axis=0)
    # Unbiased per-feature Bernoulli variance across bags
    var = (n_bags / (n_bags - 1.0)) * p_hat * (1.0 - p_hat)
    k_bar = Z.sum(axis=1).mean()
    denom = (k_bar / n_features) * (1.0 - k_bar / n_features)
    if denom <= 0:
        return 1.0
    return float(1.0 - var.mean() / denom)


def _bagged_importance(X, y, train_fn, n_bags, top_m, random_state):
    """Fit ``n_bags`` models on bootstrap resamples; collect per-fit rankings."""
    rng = np.random.default_rng(random_state)
    features = list(X.columns)
    n_features = len(features)

    rank_matrix = np.zeros((n_bags, n_features))
    gain_matrix = np.zeros((n_bags, n_features))
    selection_matrix = np.zeros((n_bags, n_features), dtype=int)

    for b in range(n_bags):
        idx = rng.choice(len(X), size=len(X), replace=True)
        Xb, yb = X.iloc[idx], y.iloc[idx]
        if yb.nunique() < 2:                      # degenerate resample
            idx = np.arange(len(X))
            Xb, yb = X, y

        model = train_fn(Xb, yb, seed=int(rng.integers(1, 10_000)))
        gains = np.asarray(model.feature_importance(importance_type='gain'), dtype=float)

        gain_matrix[b] = gains
        # rankdata gives 1 = smallest, so negate for "1 = most important"
        rank_matrix[b] = rankdata(-gains, method='average')
        selection_matrix[b, np.argsort(-gains)[:top_m]] = 1

        logger.info(f"  bag {b + 1}/{n_bags} fitted for importance ranking")

    mean_reciprocal_rank = (1.0 / rank_matrix).mean(axis=0)
    return {
        'features': features,
        'mean_gain': gain_matrix.mean(axis=0),
        'gain_cv': np.divide(
            gain_matrix.std(axis=0), gain_matrix.mean(axis=0),
            out=np.zeros(n_features), where=gain_matrix.mean(axis=0) > 0,
        ),
        'mean_reciprocal_rank': mean_reciprocal_rank,
        'selection_frequency': selection_matrix.mean(axis=0),
        'selection_matrix': selection_matrix,
    }


def _shap_importance(model, X, sample_size, random_state):
    """Global mean |SHAP| per feature, or None if SHAP is unavailable."""
    try:
        import shap
        explainer = shap.TreeExplainer(model, feature_perturbation='tree_path_dependent')
    except Exception as e:
        logger.warning(f"SHAP corroboration skipped: {e}")
        return None

    n = min(sample_size, len(X))
    sample = X.sample(n=n, random_state=random_state) if n < len(X) else X
    try:
        vals = explainer.shap_values(sample)
    except Exception as e:
        logger.warning(f"SHAP corroboration failed: {e}")
        return None

    if isinstance(vals, list):
        vals = vals[1] if len(vals) > 1 else vals[0]
    if isinstance(vals, np.ndarray) and vals.ndim == 3:
        vals = vals[:, :, 1]
    return np.abs(vals).mean(axis=0)


def _monitorability(X, min_distinct, min_std):
    """Per-feature flags for whether a distributional test can say anything."""
    n_distinct = X.nunique().values
    stds = X.std(axis=0).values
    return n_distinct, stds, (n_distinct >= min_distinct) & (stds > min_std)


def select_monitoring_features(
    X,
    y,
    train_fn,
    reference_model=None,
    top_k=10,
    n_bags=5,
    top_m=25,
    redundancy_rho=0.90,
    max_per_family=2,
    min_distinct=10,
    min_std=1e-8,
    shap_sample=500,
    random_state=42,
):
    """Choose a stable, non-redundant, monitorable set of ``top_k`` features.

    Args:
        X: Reference-window feature matrix (numeric, aligned).
        y: Reference-window labels.
        train_fn: ``(X, y, seed) -> LightGBM Booster``.
        reference_model: Optional already-fitted model reused for SHAP.
        top_k: Size of the final monitoring set.
        n_bags: Bootstrap fits used for the stability estimate.
        top_m: Cut-off defining "selected" inside each bag, for stability.
        redundancy_rho: |Spearman ρ| above which a candidate is treated as a
            duplicate of an already-selected feature.
        max_per_family: Cap on how many features sharing a numbered family
            (``_freq_ref_C1`` .. ``_freq_ref_C14``, ``D2`` .. ``D15``, ...) may
            appear in the monitoring set. Pairwise correlation pruning alone
            can under-catch these: each pair can clear ``redundancy_rho``
            while the family still votes as one block (see ``_feature_family``).
        min_distinct / min_std: Monitorability floor.

    Returns:
        (selected_features, diagnostics_df, report_dict)
    """
    logger.info(
        f"Selecting {top_k} monitoring features "
        f"(n_bags={n_bags}, redundancy_rho={redundancy_rho})..."
    )

    bag = _bagged_importance(X, y, train_fn, n_bags, top_m, random_state)
    features = bag['features']

    stability = nogueira_stability(bag['selection_matrix'])
    logger.info(f"Nogueira stability of the bagged top-{top_m}: {stability:.3f}")

    shap_imp = _shap_importance(reference_model, X, shap_sample, random_state) if reference_model is not None else None
    n_distinct, stds, monitorable = _monitorability(X, min_distinct, min_std)

    diag = pd.DataFrame({
        'feature': features,
        'mean_gain': bag['mean_gain'],
        'gain_cv': bag['gain_cv'],
        'mean_reciprocal_rank': bag['mean_reciprocal_rank'],
        'selection_frequency': bag['selection_frequency'],
        'shap_importance': shap_imp if shap_imp is not None else np.nan,
        'n_distinct': n_distinct,
        'std': stds,
        'monitorable': monitorable,
    })

    # Composite score: agreement across bags (selection_frequency) is the
    # primary signal — a feature that every bag ranks highly is a feature whose
    # importance is a property of the data, not of one fit's randomness.
    # Mean reciprocal rank breaks ties among features with equal frequency.
    mrr = diag['mean_reciprocal_rank']
    mrr_norm = (mrr - mrr.min()) / max(mrr.max() - mrr.min(), 1e-12)
    diag['score'] = 0.7 * diag['selection_frequency'] + 0.3 * mrr_norm
    diag = diag.sort_values('score', ascending=False).reset_index(drop=True)

    # ── Stage 3+4: monitorability filter, then greedy redundancy pruning ──
    candidates = diag[diag['monitorable']]['feature'].tolist()
    pool = candidates[: max(top_k * 6, 60)]           # bound the correlation matrix
    if pool:
        corr = X[pool].corr(method='spearman').abs().fillna(0.0)
    else:
        corr = pd.DataFrame()

    selected, rejected = [], []
    family_counts = {}
    for feat in candidates:
        if len(selected) >= top_k:
            break
        if feat in corr.columns and selected:
            peers = [s for s in selected if s in corr.columns]
            if peers:
                worst = corr.loc[feat, peers].max()
                if worst >= redundancy_rho:
                    twin = corr.loc[feat, peers].idxmax()
                    rejected.append({'feature': feat, 'reason': 'redundant',
                                     'with': twin, 'rho': float(worst)})
                    continue
        family = _feature_family(feat)
        if family_counts.get(family, 0) >= max_per_family:
            rejected.append({'feature': feat, 'reason': 'family_cap',
                              'with': family, 'rho': None})
            continue
        selected.append(feat)
        family_counts[family] = family_counts.get(family, 0) + 1

    # If pruning was aggressive enough to starve the set, top up by score —
    # respecting the family cap first, and only breaking it if the candidate
    # pool itself is too small to fill top_k any other way.
    if len(selected) < top_k:
        for feat in candidates:
            if len(selected) >= top_k:
                break
            if feat in selected:
                continue
            family = _feature_family(feat)
            if family_counts.get(family, 0) < max_per_family:
                selected.append(feat)
                family_counts[family] = family_counts.get(family, 0) + 1
    if len(selected) < top_k:
        for feat in candidates:
            if len(selected) >= top_k:
                break
            if feat not in selected:
                selected.append(feat)

    dropped_unmonitorable = diag[~diag['monitorable']]['feature'].tolist()
    diag['selected'] = diag['feature'].isin(selected)

    report = {
        'selected_features': selected,
        'stability_index': float(stability),
        'n_bags': n_bags,
        'top_m': top_m,
        'redundancy_rho': redundancy_rho,
        'max_per_family': max_per_family,
        'redundant_rejections': rejected,
        'n_unmonitorable_dropped': len(dropped_unmonitorable),
        'family_counts_selected': family_counts,
        'max_pairwise_rho_among_selected': float(
            corr.loc[[f for f in selected if f in corr.columns],
                     [f for f in selected if f in corr.columns]]
            .where(~np.eye(len([f for f in selected if f in corr.columns]), dtype=bool))
            .max().max()
        ) if len([f for f in selected if f in corr.columns]) > 1 else 0.0,
    }

    logger.info(f"Monitoring set ({len(selected)}): {selected}")
    logger.info(
        f"  stability={stability:.3f}, "
        f"redundant rejections={len(rejected)}, "
        f"unmonitorable dropped={len(dropped_unmonitorable)}, "
        f"max ρ among selected={report['max_pairwise_rho_among_selected']:.3f}"
    )
    return selected, diag, report
