"""
Drift Analysis Pipeline — Shared-Registry Edition
=================================================

Orchestrates 12 drift-detection methods over a weekly replay of the IEEE-CIS
fraud dataset, with a *single shared model registry* rather than one private
model per detector.

Design
------
1. **One representation.** Feature-engineering encoders are fitted once, on the
   baseline window, and frozen for the whole replay (``FeatureEngineer``).
   Retraining changes the *model*, never the feature space — otherwise model
   versions would not be comparable and the observed "drift" would be partly an
   artifact of refitting encoders (see feature_engineering.py).

2. **One baseline model.** All detectors start pointing at version 0.

3. **At most one new model per week.** When k detectors flag drift in week w,
   the registry trains a single model on the cumulative data through week w and
   hands the identical Booster to all k. Detectors that did not flag drift keep
   their existing pointer. Upper bound on distinct models: ``1 + n_weeks``
   (15 for a 14-week replay), regardless of detector count.

4. **Persistence gating.** A detector must fire in k of the last n windows
   before a retrain is triggered, so a detector's per-window false-positive
   rate no longer translates one-for-one into a retraining rate.

Workflow
--------
    Phase 1  Load, merge, sort; add causal sequence features over the full frame.
    Phase 2  Fit the frozen feature pipeline + select the monitoring set on the
             first 90 days; train the single baseline model.
    Phase 3  Fit per-version detector artefacts (SHAP / clustering / autoencoder).
    Phase 4  Stream weekly windows: detect → confirm → retrain-or-reuse.
    Phase 5  Emit JSON + CSV reports for the dashboard.
"""

import os
import gc
import json
import logging
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import StratifiedKFold

from data_processing import load_data
from feature_engineering import FeatureEngineer, add_causal_sequence_features
from feature_selection import select_monitoring_features
from model_training import train_model, evaluate, get_decision_threshold
from model_registry import ModelRegistry, BASELINE_VERSION
from policy_evaluation import run_policy_evaluation

from drift_engine import (
    ks_test,
    benjamini_hochberg,
    psi_with_null,
    calculate_kl_divergence,
    calculate_js_distance,
    build_error_stream,
    DDMTracker,
    EDDMTracker,
    HDDMTracker,
    PrequentialAUCDetector,
    DriftConfirmation,
    run_adwin_batch,
    SHAPDriftDetector,
    ClusteringDriftDetector,
    AutoencoderDriftDetector,
    evaluate_champion_vs_challenger,
)

# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/unified_drift_pipeline.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DRIFT_METHODS = [
    'ks_stats', 'psi', 'kl_divergence',
    'ddm', 'eddm', 'adwin', 'hddm',
    'shap', 'clustering', 'autoencoder',
    'prequential_auc', 'champion_vs_challenger',
]

# Detectors whose method-level decision is a vote over the monitored features.
FEATURE_VOTE_METHODS = ['ks_stats', 'psi', 'kl_divergence', 'shap']

WEEK_SECONDS = 7 * 24 * 3600
THREE_MONTHS_SECONDS = 90 * 24 * 3600

# Reference windows are subsampled to this many rows for distributional tests.
# 20k rows resolve a KS statistic to ~0.01, far finer than any threshold we
# apply, while keeping per-version memory bounded so the registry can hold all
# 15 versions at once.
REF_SAMPLE_SIZE = 20_000

# Every raw metric behind these decisions is stored per week in the report, so
# a dashboard can re-derive drift flags for different thresholds without
# rerunning the pipeline.
METHOD_THRESHOLDS = {
    # Significance AND a practical effect size — see drift_engine.ks_test.
    # min_feature_fraction lowered from 0.6 to 0.3 (6 of 20 features): a
    # supermajority-of-features consensus is appropriate for PSI (folklore
    # bands calibrated per-feature, no persistence-tested gate above them),
    # but for KS/JS/SHAP it meant a single genuinely drifting feature was
    # worth nothing until eleven others agreed — a bar high enough that the
    # method-level pros/cons discussion (§3.2) treated "never confirms" as
    # close to guaranteed regardless of what the data does. 0.3 keeps the
    # persistence gate (2 consecutive weeks) as the actual noise filter,
    # rather than doubling up two conservative gates on the same signal.
    'ks_stats': {'alpha': 0.05, 'min_effect': 0.10, 'fdr': True, 'min_feature_fraction': 0.3},
    # Folklore band AND the bootstrap null calibrated to this window's size.
    'psi': {'warning': 0.10, 'drift': 0.20, 'null_quantile': 0.99, 'min_feature_fraction': 0.6},
    # Decision statistic is Jensen-Shannon distance (bounded, symmetric);
    # raw KL is still reported for continuity with the earlier reports.
    'kl_divergence': {'js_drift': 0.10, 'js_warning': 0.05, 'kl_drift': 0.50,
                      'min_feature_fraction': 0.3},
    'ddm': {'warning_level': 2.0, 'drift_level': 3.0, 'error_signal': 'balanced'},
    'eddm': {'beta_warning': 0.85, 'beta_drift': 0.75, 'min_errors': 100,
             'error_signal': 'balanced'},
    'adwin': {'delta': 0.002},
    'hddm': {'drift_confidence': 0.001, 'warning_confidence': 0.005,
             'error_signal': 'balanced'},
    'shap': {'alpha': 0.05, 'min_effect': 0.10, 'fdr': True, 'min_feature_fraction': 0.3},
    'clustering': {'distance_ratio': 1.5, 'cluster_psi': 0.20},
    'autoencoder': {'z_score': 3.0},
    'prequential_auc': {'min_drop': 0.02, 'n_sigma': 2.0},
    'champion_vs_challenger': {'auc_degradation': 0.05, 'auc_gap': 0.03, 'oof': True},
}

# Per-method feature-vote consensus fraction, read from METHOD_THRESHOLDS so
# there is exactly one place each method's bar is set (previously a single
# global 0.6 silently overrode the per-method values above, which were dead
# config — see the commit that fixed this).
FEATURE_VOTE_FRACTION = {m: METHOD_THRESHOLDS[m]['min_feature_fraction'] for m in FEATURE_VOTE_METHODS}

# Persistence gate: fire in k of the last n windows before retraining.
CONFIRM_K, CONFIRM_N = 2, 2


# ══════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════
def preprocess_and_align(df_no_target, feature_schema):
    """Cast to numeric float32, fill NaNs, and align to the frozen schema.

    Any column still non-numeric at this point is a bug in FeatureEngineer, not
    something to quietly paper over: factorising here is *per window*, which
    reintroduces exactly the encoder-instability artifact the frozen pipeline
    exists to remove. We fall back so the run does not crash, but we say so
    loudly — a silent fallback here previously masked a real defect, because
    pandas 3.x gives string columns a ``str`` dtype rather than ``object`` and
    FeatureEngineer's object-only check skipped them all.
    """
    X = df_no_target.copy()
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            logger.warning(
                f"Column '{col}' reached preprocess_and_align as {X[col].dtype}; "
                f"falling back to a PER-WINDOW factorize, which is not comparable "
                f"across windows. Fix the encoding in FeatureEngineer."
            )
            X[col] = pd.factorize(X[col].astype(str))[0]
        X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
    X = X.astype(np.float32)
    return X.reindex(columns=feature_schema, fill_value=0).fillna(0).astype(np.float32)


def oof_predictions(X, y, n_splits=2, random_state=42):
    """Out-of-fold probabilities for a model trained on (X, y).

    Used for the challenger in champion-vs-challenger. Scoring a challenger on
    the very rows it was fitted to inflates its AUC by more than the drift
    threshold, so the comparison would trigger every week on overfitting alone
    (see drift_engine.evaluate_champion_vs_challenger). Two folds keep the cost
    identical to the single extra fit the previous version already paid.

    Returns:
        (oof_probs, one_fitted_model) — the model is reused for the weekly
        feature-importance snapshot so no third fit is needed.
    """
    y_arr = np.asarray(y)
    oof = np.zeros(len(X), dtype=float)
    first_model = None

    if len(np.unique(y_arr)) < 2 or len(X) < 2 * n_splits:
        model = train_model(X, y)
        return model.predict(X, predict_disable_shape_check=True), model

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for fold, (tr, va) in enumerate(skf.split(X, y_arr)):
        model = train_model(X.iloc[tr], y.iloc[tr])
        oof[va] = model.predict(X.iloc[va], predict_disable_shape_check=True)
        if fold == 0:
            first_model = model
    return oof, first_model


def _subsample(X, y, probs, size=REF_SAMPLE_SIZE, random_state=42):
    """Bounded reference snapshot kept alongside each model version."""
    if len(X) <= size:
        return X.copy(), y.copy(), np.asarray(probs).copy()
    rng = np.random.default_rng(random_state)
    idx = np.sort(rng.choice(len(X), size=size, replace=False))
    return X.iloc[idx].copy(), y.iloc[idx].copy(), np.asarray(probs)[idx].copy()


# ══════════════════════════════════════════════
# Phase 1: Load & prepare
# ══════════════════════════════════════════════
def _load_and_prepare_data(data_dir):
    logger.info("Loading dataset...")
    tr_id, tr_trn, _, _ = load_data(data_dir)
    raw = pd.merge(tr_trn, tr_id, on='TransactionID', how='left')
    raw = raw.sort_values('TransactionDT')

    # Per-entity lags computed over the FULL frame — strictly causal, but not
    # truncated to a 7-day horizon the way a per-window computation would be.
    raw = add_causal_sequence_features(raw)

    start_time = raw['TransactionDT'].min()
    end_time = raw['TransactionDT'].max()
    n_weeks = int(np.ceil((end_time - start_time - THREE_MONTHS_SECONDS) / WEEK_SECONDS))
    logger.info(f"Dataset spans ~{max(n_weeks, 0)} monitored weekly windows after the baseline.")
    return raw, start_time, end_time


# ══════════════════════════════════════════════
# Phase 2: Baseline — frozen representation, monitoring set, single model
# ══════════════════════════════════════════════
def _build_baseline(raw, start_time, top_k, n_bags):
    cutoff = start_time + THREE_MONTHS_SECONDS
    train_raw = raw[raw['TransactionDT'] < cutoff]
    logger.info(f"Building 90-day baseline ({len(train_raw)} raw rows)...")

    fe = FeatureEngineer()
    processed = fe.fit_transform(train_raw)
    feature_schema = fe.feature_schema

    ref_X = preprocess_and_align(processed.drop('isFraud', axis=1), feature_schema)
    ref_y = processed['isFraud'].reset_index(drop=True)
    ref_X = ref_X.reset_index(drop=True)

    registry = ModelRegistry(train_fn=train_model, evaluate_fn=evaluate, methods=DRIFT_METHODS)
    baseline_version = registry.register_baseline(ref_X, ref_y)

    top_features, feat_diag, selection_report = select_monitoring_features(
        ref_X, ref_y,
        train_fn=lambda Xb, yb, seed: train_model(Xb, yb, seed=seed),
        reference_model=baseline_version.model,
        top_k=top_k,
        n_bags=n_bags,
    )

    return fe, ref_X, ref_y, feature_schema, top_features, feat_diag, selection_report, registry, cutoff


# ══════════════════════════════════════════════
# Phase 3: Per-version detector artefacts
# ══════════════════════════════════════════════
def _fit_version_artifacts(version, ref_X, ref_y, ref_probs, top_features, methods_to_run,
                           cluster_pca=None):
    """Fit reference-dependent detector state ONCE per model version.

    Multiple detectors can adopt the same version; fitting the autoencoder or
    K-Means per detector would repeat identical work and, worse, produce
    slightly different reference statistics for detectors that are supposed to
    be looking at the same reference.
    """
    ref_sample_X, ref_sample_y, ref_sample_probs = _subsample(ref_X, ref_y, ref_probs)

    artifacts = {
        'ref_X': ref_sample_X,
        'ref_y': ref_sample_y,
        'ref_predictions': ref_sample_probs,
    }

    if 'shap' in methods_to_run:
        artifacts['shap'] = SHAPDriftDetector(version.model)

    if 'clustering' in methods_to_run:
        det = ClusteringDriftDetector(n_clusters=5)
        det.fit_reference(ref_sample_X[top_features])
        artifacts['clustering'] = det

        if cluster_pca is not None:
            sample = ref_sample_X[top_features].fillna(0).iloc[:300]
            scaled = det._scale(sample.values)
            artifacts['pca_coords'] = cluster_pca.fit_transform(scaled).tolist()
            artifacts['pca_labels'] = det.kmeans.predict(scaled).tolist()
            artifacts['pca_centroids'] = cluster_pca.transform(det.kmeans.cluster_centers_).tolist()

    if 'autoencoder' in methods_to_run:
        det = AutoencoderDriftDetector()
        det.fit_reference(ref_sample_X[top_features])
        artifacts['autoencoder'] = det

    version.artifacts.update(artifacts)
    return version


def _initialize_method_state(methods_to_run):
    """Per-method trackers and the persistence gate. These are NOT shared —
    each detector's decision history is its own, even when the model is shared."""
    thr = METHOD_THRESHOLDS
    state = {}
    for m in methods_to_run:
        entry = {'confirm': DriftConfirmation(k=CONFIRM_K, n=CONFIRM_N)}
        if m == 'ddm':
            entry['tracker'] = DDMTracker(warning_level=thr['ddm']['warning_level'],
                                          drift_level=thr['ddm']['drift_level'])
        elif m == 'eddm':
            entry['tracker'] = EDDMTracker(beta_drift=thr['eddm']['beta_drift'],
                                           beta_warning=thr['eddm']['beta_warning'],
                                           min_errors=thr['eddm']['min_errors'])
        elif m == 'hddm':
            entry['tracker'] = HDDMTracker(
                drift_confidence=thr['hddm']['drift_confidence'],
                warning_confidence=thr['hddm']['warning_confidence'])
        elif m == 'prequential_auc':
            entry['tracker'] = PrequentialAUCDetector(
                min_drop=thr['prequential_auc']['min_drop'],
                n_sigma=thr['prequential_auc']['n_sigma'])
        state[m] = entry
    return state


# ══════════════════════════════════════════════
# Phase 4a: Feature-level drift
# ══════════════════════════════════════════════
def _compute_feature_drift(top_features, registry, X_aligned, methods_to_run):
    """KS / PSI / KL+JS per feature, each against ITS OWN method's reference.

    Two detectors that retrained in different weeks legitimately have different
    references, so these cannot be computed once and shared — unlike the model,
    which can.
    """
    thr = METHOD_THRESHOLDS
    ks_results, psi_results, kl_results = {}, {}, {}

    ref_ks = registry.version_for('ks_stats').artifacts['ref_X']
    ref_psi = registry.version_for('psi').artifacts['ref_X']
    ref_kl = registry.version_for('kl_divergence').artifacts['ref_X']

    # ── KS: BH-corrected across the monitored set, then effect-size gated ──
    ks_raw, ks_names = [], []
    for feat in top_features:
        res = ks_test(ref_ks[feat], X_aligned[feat],
                      alpha=thr['ks_stats']['alpha'],
                      min_effect=thr['ks_stats']['min_effect'],
                      max_ref_sample=REF_SAMPLE_SIZE)
        ks_raw.append(res)
        ks_names.append(feat)

    rejected, adjusted = benjamini_hochberg([r['p_value'] for r in ks_raw],
                                            alpha=thr['ks_stats']['alpha'])
    for i, feat in enumerate(ks_names):
        r = ks_raw[i]
        ks_results[feat] = {
            'stat': r['statistic'],
            'p_value': r['p_value'],
            'p_value_adjusted': float(adjusted[i]),
            'significant_fdr': bool(rejected[i]),
            'd_critical': r['d_critical'],
            'min_effect': r['min_effect'],
            'practically_significant': r['practically_significant'],
            # Drift requires BOTH FDR-controlled significance and a real effect.
            'drift': bool(rejected[i] and r['practically_significant']),
        }

    # ── PSI: folklore band AND bootstrap null for this window size ──
    for feat in top_features:
        res = psi_with_null(ref_psi[feat], X_aligned[feat],
                            threshold=thr['psi']['drift'],
                            quantile=thr['psi']['null_quantile'])
        psi_results[feat] = {
            'psi': res['psi'],
            'psi_null_q': res['psi_null_q'],
            'psi_ratio': res['psi_ratio'],
            'chi2_expected_psi': res['chi2_expected_psi'],
            'exceeds_threshold': res['exceeds_threshold'],
            'exceeds_null': res['exceeds_null'],
            'drift': res['drift'],
        }

    # ── KL (reported) + JS distance (the decision statistic) ──
    for feat in top_features:
        kl_val, kl_flag = calculate_kl_divergence(ref_kl[feat], X_aligned[feat],
                                                  threshold=thr['kl_divergence']['kl_drift'])
        js = calculate_js_distance(ref_kl[feat], X_aligned[feat])
        kl_results[feat] = {
            'kl_div': kl_val,
            'kl_drift_legacy': kl_flag,
            'js_distance': js,
            'drift': bool(js >= thr['kl_divergence']['js_drift']),
        }

    return ks_results, psi_results, kl_results


# ══════════════════════════════════════════════
# Phase 4b: Error-stream detectors
# ══════════════════════════════════════════════
def _compute_stream_drift(method_state, method_error_streams, method_preds,
                          registry, y_curr, methods_to_run):
    """DDM / EDDM / HDDM / ADWIN / prequential AUC, each on its own stream."""
    no_drift = {'drift_detected': False, 'warning_detected': False}
    out = {}

    for m in ('ddm', 'eddm', 'hddm'):
        out[m] = (method_state[m]['tracker'].process_batch(method_error_streams[m])
                  if m in methods_to_run else dict(no_drift))

    out['adwin'] = (
        run_adwin_batch(method_preds['adwin'],
                        delta=METHOD_THRESHOLDS['adwin']['delta'],
                        reference_stream=registry.version_for('adwin').artifacts['ref_predictions'])
        if 'adwin' in methods_to_run else dict(no_drift)
    )

    if 'prequential_auc' in methods_to_run:
        out['prequential_auc'] = method_state['prequential_auc']['tracker'].evaluate_drift(
            y_curr, method_preds['prequential_auc'])
    else:
        out['prequential_auc'] = dict(no_drift, current_auc=0.0, reference_auc=0.0, auc_drop=0.0)

    return out


# ══════════════════════════════════════════════
# Phase 4c: Structural + performance detectors
# ══════════════════════════════════════════════
def _compute_advanced_drift(methods_to_run, registry, X_aligned, y_curr, top_features,
                            challenger_model, challenger_oof):
    no_drift = {'drift_detected': False, 'warning_detected': False}

    if 'shap' in methods_to_run:
        v = registry.version_for('shap')
        shap_res = v.artifacts['shap'].evaluate_drift(
            v.artifacts['ref_X'], X_aligned, top_features,
            alpha=METHOD_THRESHOLDS['shap']['alpha'],
            min_effect=METHOD_THRESHOLDS['shap']['min_effect'],
        )
    else:
        shap_res = {
            'drift_detected': False, 'warning_detected': False,
            'overall_shap_drift_detected': False, 'drifted_features_count': 0,
            'feature_shap_drift': {
                f: {'ref_importance': 0.0, 'curr_importance': 0.0, 'importance_shift': 0.0,
                    'is_drift': False, 'ks_stat': 0.0, 'p_value': 1.0, 'p_value_adjusted': 1.0}
                for f in top_features
            },
        }

    if 'clustering' in methods_to_run:
        clustering_res = registry.version_for('clustering').artifacts['clustering'].evaluate_drift(
            X_aligned[top_features],
            distance_threshold=METHOD_THRESHOLDS['clustering']['distance_ratio'],
            psi_threshold=METHOD_THRESHOLDS['clustering']['cluster_psi'],
        )
    else:
        clustering_res = dict(no_drift, distance_ratio=1.0, cluster_psi=0.0)

    if 'autoencoder' in methods_to_run:
        ae_res = registry.version_for('autoencoder').artifacts['autoencoder'].evaluate_drift(
            X_aligned[top_features],
            z_score_threshold=METHOD_THRESHOLDS['autoencoder']['z_score'],
        )
    else:
        ae_res = dict(no_drift, mse_z_score=0.0, curr_rmse_mean=0.0, ref_rmse_mean=0.0, p_value=1.0)

    if 'champion_vs_challenger' in methods_to_run:
        version = registry.version_for('champion_vs_challenger')
        cvc_res = evaluate_champion_vs_challenger(
            version.model, challenger_model, X_aligned, y_curr,
            baseline_auc=version.artifacts.get('holdout_auc', version.train_auc),
            auc_degradation_thresh=METHOD_THRESHOLDS['champion_vs_challenger']['auc_degradation'],
            auc_gap_thresh=METHOD_THRESHOLDS['champion_vs_challenger']['auc_gap'],
            challenger_probs=challenger_oof,
        )
    else:
        cvc_res = dict(no_drift, champion_auc=0.0, challenger_auc=0.0, auc_gap=0.0)

    return shap_res, clustering_res, ae_res, cvc_res


# ══════════════════════════════════════════════
# Phase 4d: Assemble flags (raw → confirmed)
# ══════════════════════════════════════════════
def _assemble_drift_flags(ks_results, psi_results, kl_results, stream_res,
                          shap_res, clustering_res, ae_res, cvc_res,
                          top_features, method_state, methods_to_run):
    """Combine detector outputs into raw and persistence-confirmed flags.

    The feature-vote methods require a supermajority of monitored features to
    drift. With the redundancy-pruned monitoring set those votes are
    approximately independent, so the fraction is a meaningful consensus rather
    than one underlying signal counted several times.
    """
    n = max(len(top_features), 1)

    raw_flags = {
        'ks_stats': bool(sum(r['drift'] for r in ks_results.values()) / n >= FEATURE_VOTE_FRACTION['ks_stats']),
        'psi': bool(sum(r['drift'] for r in psi_results.values()) / n >= FEATURE_VOTE_FRACTION['psi']),
        'kl_divergence': bool(sum(r['drift'] for r in kl_results.values()) / n >= FEATURE_VOTE_FRACTION['kl_divergence']),
        'ddm': bool(stream_res['ddm']['drift_detected']),
        'eddm': bool(stream_res['eddm']['drift_detected']),
        'adwin': bool(stream_res['adwin']['drift_detected']),
        'hddm': bool(stream_res['hddm']['drift_detected']),
        'shap': bool(shap_res.get('drifted_features_count', 0) / n >= FEATURE_VOTE_FRACTION['shap']),
        'clustering': bool(clustering_res['drift_detected']),
        'autoencoder': bool(ae_res['drift_detected']),
        'prequential_auc': bool(stream_res['prequential_auc']['drift_detected']),
        'champion_vs_challenger': bool(cvc_res['drift_detected']),
    }

    confirmed_flags = {}
    confirm_state = {}
    for m in DRIFT_METHODS:
        if m in methods_to_run:
            confirmed_flags[m] = method_state[m]['confirm'].update(raw_flags[m])
            confirm_state[m] = method_state[m]['confirm'].state
        else:
            confirmed_flags[m] = False
            confirm_state[m] = {'k': CONFIRM_K, 'n': CONFIRM_N, 'recent_flags': [], 'confirmed': False}

    return raw_flags, confirmed_flags, confirm_state


# ══════════════════════════════════════════════
# Phase 4e: Retrain — one model, shared by every method that drifted
# ══════════════════════════════════════════════
def _retrain_drifted_methods(week_idx, drifted_methods, registry, cum_X, cum_y,
                             top_features, methods_to_run, method_state,
                             cluster_pca, reasons):
    """Train at most ONE model for this week and share it across all drifters."""
    if not drifted_methods:
        return [], None

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  WEEK {week_idx} — drift confirmed by: {drifted_methods}")
    logger.info("=" * 60)

    version, was_trained = registry.get_or_train(week_idx, cum_X, cum_y)

    # The registry already scored the model on its own training data when it
    # created the version; reuse that rather than paying for a second full
    # predict over the cumulative window.
    probs = version.train_predictions

    if was_trained:
        # Detector artefacts are fitted against the version, not the detector,
        # so they are built once here even if several detectors adopt it.
        _fit_version_artifacts(version, cum_X, cum_y, probs, top_features,
                               methods_to_run, cluster_pca=cluster_pca)

    triggers = []
    for mn in drifted_methods:
        registry.adopt(mn, version, week_idx, reason=reasons.get(mn))

        # A detector's own history refers to the model it was watching. That
        # model just changed, so the history is no longer comparable.
        state = method_state[mn]
        state['confirm'].reset()
        tracker = state.get('tracker')
        if tracker is not None:
            if isinstance(tracker, PrequentialAUCDetector):
                tracker.set_reference(cum_y, probs)
            else:
                tracker.reset()

        triggers.append({
            'method': mn,
            'week': week_idx,
            'model_version': version.version_id,
            'model_label': version.label,
            'shared_with': [m for m in drifted_methods if m != mn],
            'newly_trained': bool(was_trained),
            'cumulative_rows': int(version.n_train_rows),
            'new_auc': float(version.train_auc),
            'new_f1': float(version.train_f1),
            'retrain_count': registry.retrain_count(mn),
            'reason': reasons.get(mn, 'Drift confirmed.'),
            'trigger_metrics': reasons.get(f'{mn}__metrics', {}),
        })

    return triggers, version


def _build_reasons(drifted_methods, top_features, ks_results, psi_results, kl_results,
                   stream_res, shap_res, clustering_res, ae_res, cvc_res, confirm_state):
    """Human-readable justification + the raw numbers that crossed the line."""
    n = max(len(top_features), 1)
    reasons = {}

    def _vote(results, key, label, method_name):
        drifted = [f for f in top_features if results[f]['drift']]
        pct = 100 * len(drifted) / n
        frac_pct = FEATURE_VOTE_FRACTION[method_name]
        return (f"{len(drifted)}/{len(top_features)} monitored features ({pct:.0f}%) show "
                f"{label} — at or above the {frac_pct:.0%} consensus "
                f"threshold: {drifted}"), {
            'drifted_features': drifted, 'drifted_count': len(drifted),
            'drifted_fraction': len(drifted) / n,
            'per_feature': {f: results[f].get(key) for f in top_features},
        }

    for mn in drifted_methods:
        if mn == 'ks_stats':
            reason, metrics = _vote(ks_results, 'stat', 'KS drift (FDR-corrected, D>=min_effect)', mn)
        elif mn == 'psi':
            reason, metrics = _vote(psi_results, 'psi', 'PSI drift (>=0.20 and above the bootstrap null)', mn)
        elif mn == 'kl_divergence':
            reason, metrics = _vote(kl_results, 'js_distance', 'Jensen-Shannon distance drift', mn)
        elif mn == 'shap':
            fsd = shap_res.get('feature_shap_drift', {})
            drifted = [f for f in top_features if fsd.get(f, {}).get('is_drift')]
            reason = (f"{len(drifted)}/{len(top_features)} monitored features "
                      f"({100 * len(drifted) / n:.0f}%) show SHAP-attribution drift — at or above "
                      f"the {FEATURE_VOTE_FRACTION['shap']:.0%} consensus threshold: {drifted}")
            metrics = {'drifted_features': drifted, 'drifted_count': len(drifted)}
        elif mn == 'ddm':
            r = stream_res['ddm']
            reason = (f"DDM: balanced error rate {r.get('mean_error_rate', 0):.4f} crossed "
                      f"p_min+3s_min = {r.get('drift_threshold', 0):.4f}")
            metrics = {k: r.get(k) for k in ('mean_error_rate', 'drift_threshold', 'p_min', 's_min')}
        elif mn == 'eddm':
            r = stream_res['eddm']
            reason = (f"EDDM: inter-error distance metric {r.get('metric_value', 0):.4f} fell below "
                      f"{r.get('drift_threshold', 0):.4f} (max seen {r.get('max_metric', 0):.4f})")
            metrics = {k: r.get(k) for k in ('metric_value', 'drift_threshold', 'max_metric')}
        elif mn == 'hddm':
            r = stream_res['hddm']
            reason = (f"HDDM: error mean {r.get('mean_error_rate', 0):.4f} exceeded the reference "
                      f"mean {r.get('reference_mean', 0):.4f} by more than the Hoeffding bound "
                      f"{r.get('hoeffding_bound', 0):.4f}")
            metrics = {k: r.get(k) for k in ('mean_error_rate', 'reference_mean', 'hoeffding_bound')}
        elif mn == 'adwin':
            r = stream_res['adwin']
            reason = (f"ADWIN: mean shift in the prediction stream "
                      f"(z={r.get('z_score', 0):.2f} vs {r.get('z_threshold', 0):.2f}, "
                      f"backend={r.get('backend', 'n/a')})")
            metrics = {k: r.get(k) for k in ('z_score', 'z_threshold', 'estimation', 'delta')}
        elif mn == 'prequential_auc':
            r = stream_res['prequential_auc']
            reason = (f"Prequential AUC fell {r.get('auc_drop', 0):.4f} "
                      f"({r.get('reference_auc', 0):.4f} -> {r.get('current_auc', 0):.4f}), "
                      f"more than {METHOD_THRESHOLDS['prequential_auc']['n_sigma']}x its "
                      f"bootstrap SE of {r.get('auc_se', 0):.4f}")
            metrics = {k: r.get(k) for k in ('current_auc', 'reference_auc', 'auc_drop', 'auc_se')}
        elif mn == 'clustering':
            reason = (f"Clustering: distance_ratio={clustering_res.get('distance_ratio', 0):.2f} "
                      f"(>=1.5) or cluster_psi={clustering_res.get('cluster_psi', 0):.4f} (>=0.20)")
            metrics = {k: clustering_res.get(k) for k in ('distance_ratio', 'cluster_psi')}
        elif mn == 'autoencoder':
            reason = (f"Autoencoder reconstruction RMSE z-score="
                      f"{ae_res.get('mse_z_score', 0):.2f} exceeded 3.0 "
                      f"({ae_res.get('ref_rmse_mean', 0):.4f} -> {ae_res.get('curr_rmse_mean', 0):.4f})")
            metrics = {k: ae_res.get(k) for k in ('mse_z_score', 'curr_rmse_mean', 'ref_rmse_mean')}
        elif mn == 'champion_vs_challenger':
            reason = (f"Out-of-fold challenger beat the champion by "
                      f"{cvc_res.get('auc_gap', 0):.4f} AUC (SE {cvc_res.get('auc_gap_se', 0):.4f}), "
                      f"or the champion degraded {cvc_res.get('auc_degradation', 0):.4f} from baseline")
            metrics = {k: cvc_res.get(k) for k in
                       ('auc_gap', 'auc_gap_se', 'auc_degradation', 'champion_auc', 'challenger_auc')}
        else:
            reason, metrics = "Drift confirmed.", {}

        flags = confirm_state.get(mn, {}).get('recent_flags', [])
        reason += f" | persistence gate: fired in {sum(flags)} of the last {len(flags)} window(s)"
        reasons[mn] = reason
        reasons[f'{mn}__metrics'] = metrics
        logger.info(f"  [{mn}] {reason}")

    return reasons


# ══════════════════════════════════════════════
# Phase 5: Reports
# ══════════════════════════════════════════════
def _build_weekly_entry(week_idx, X_aligned, top_features, curr_top_features,
                        raw_flags, confirmed_flags, confirm_state,
                        ks_results, psi_results, kl_results, stream_res,
                        shap_res, clustering_res, ae_res, cvc_res,
                        pca_data, registry, week_triggers, week_performance):
    return {
        'week': week_idx,
        'is_baseline': False,
        'rows': len(X_aligned),
        'top_features': top_features,
        'curr_top_features': curr_top_features,
        'overall_drift_flag': any(confirmed_flags.values()),
        'drift_method_flags': confirmed_flags,
        'drift_method_flags_raw': raw_flags,
        'confirmation_state': confirm_state,
        'method_model_version': dict(registry.method_versions),
        'method_status': {
            'ks_stats': {'drift': confirmed_flags['ks_stats'],
                         'details': {k: ks_results[k] for k in top_features}},
            'psi': {'drift': confirmed_flags['psi'],
                    'details': {k: psi_results[k] for k in top_features}},
            'kl_divergence': {'drift': confirmed_flags['kl_divergence'],
                              'details': {k: kl_results[k] for k in top_features}},
            'ddm': stream_res['ddm'],
            'eddm': stream_res['eddm'],
            'adwin': stream_res['adwin'],
            'hddm': stream_res['hddm'],
            'prequential_auc': stream_res['prequential_auc'],
            'shap': shap_res,
            'clustering': clustering_res,
            'autoencoder': ae_res,
            'champion_vs_challenger': cvc_res,
        },
        'feature_metrics': {
            feat: {
                'ks_stat': ks_results[feat]['stat'],
                'ks_p': ks_results[feat]['p_value'],
                'ks_p_adjusted': ks_results[feat]['p_value_adjusted'],
                'ks_drift': ks_results[feat]['drift'],
                'psi': psi_results[feat]['psi'],
                'psi_null_q': psi_results[feat]['psi_null_q'],
                'psi_drift': psi_results[feat]['drift'],
                'kl_div': kl_results[feat]['kl_div'],
                'js_distance': kl_results[feat]['js_distance'],
                'kl_drift': kl_results[feat]['drift'],
                'ref_importance': shap_res.get('feature_shap_drift', {}).get(feat, {}).get('ref_importance', 0.0),
                'curr_importance': shap_res.get('feature_shap_drift', {}).get(feat, {}).get('curr_importance', 0.0),
                'shap_shift': shap_res.get('feature_shap_drift', {}).get(feat, {}).get('importance_shift', 0.0),
                'shap_drift': shap_res.get('feature_shap_drift', {}).get(feat, {}).get('is_drift', False),
            }
            for feat in top_features
        },
        # Kept for dashboard compatibility; 'model_auc' is now the version's
        # training AUC and 'holdout_auc' the honest out-of-sample number.
        'method_retrain_info': {
            m: {
                'retrain_count': registry.retrain_count(m),
                'model_version': registry.method_versions[m],
                'model_auc': float(registry.version_for(m).train_auc),
                'model_f1': float(registry.version_for(m).train_f1),
                'holdout_auc': float(week_performance.get(m, {}).get('auc', 0.0)),
                'holdout_f1': float(week_performance.get(m, {}).get('f1', 0.0)),
            }
            for m in DRIFT_METHODS
        },
        'pca_cluster_data': pca_data,
        'retrain_triggers_this_week': week_triggers,
    }


# Per-method (value, threshold) extractors for the week x method matrix below.
# Each returns the single scalar the retrain decision actually turns on, paired
# with the threshold it is compared against — the same numbers `_build_reasons`
# already narrates in prose, just tabulated instead.
_KEY_METRIC = {
    'ddm': lambda s: (s['ddm'].get('mean_error_rate'), s['ddm'].get('drift_threshold')),
    'eddm': lambda s: (s['eddm'].get('metric_value'), s['eddm'].get('drift_threshold')),
    'hddm': lambda s: (s['hddm'].get('mean_error_rate'), s['hddm'].get('reference_mean', 0.0) +
                       s['hddm'].get('hoeffding_bound', 0.0)),
    'adwin': lambda s: (s['adwin'].get('z_score'), s['adwin'].get('z_threshold')),
    'prequential_auc': lambda s: (s['prequential_auc'].get('auc_drop'),
                                  METHOD_THRESHOLDS['prequential_auc']['n_sigma'] *
                                  s['prequential_auc'].get('auc_se', 0.0)),
    'clustering': lambda s: (s['clustering'].get('distance_ratio'),
                             METHOD_THRESHOLDS['clustering']['distance_ratio']),
    'autoencoder': lambda s: (s['autoencoder'].get('mse_z_score'),
                              METHOD_THRESHOLDS['autoencoder']['z_score']),
    'champion_vs_challenger': lambda s: (s['champion_vs_challenger'].get('auc_gap'),
                                         METHOD_THRESHOLDS['champion_vs_challenger']['auc_gap']),
}


def _build_method_matrix(weekly_records, methods_to_run, top_features):
    """Tidy week x method table: threshold, whether it crossed, and how many
    of the monitored features crossed it (for the feature-vote detectors).

    One row per (week, method). This is the raw material for both the
    dashboard's matrix view and the paper's per-method results table — every
    number here is already computed during the replay, this just flattens it.
    """
    n_feat = max(len(top_features), 1)
    rows = []
    for rec in weekly_records:
        if rec['is_baseline']:
            continue
        status = rec['method_status']
        for m in methods_to_run:
            row = {
                'week': rec['week'],
                'method': m,
                'model_version': rec['method_model_version'].get(m),
                'raw_flag': rec['drift_method_flags_raw'].get(m, False),
                'confirmed_flag': rec['drift_method_flags'].get(m, False),
                'retrained_this_week': m in [t['method'] for t in rec.get('retrain_triggers_this_week', [])],
                'features_total': None,
                'features_crossed': None,
                'features_crossed_fraction': None,
                'drifted_feature_names': None,
                'key_metric_value': None,
                'key_metric_threshold': None,
            }
            if m in FEATURE_VOTE_METHODS:
                if m == 'shap':
                    fsd = status['shap'].get('feature_shap_drift', {})
                    names = [f for f in top_features if fsd.get(f, {}).get('is_drift')]
                else:
                    key = {'ks_stats': 'ks_drift', 'psi': 'psi_drift', 'kl_divergence': 'kl_drift'}[m]
                    names = [f for f in top_features if rec['feature_metrics'].get(f, {}).get(key)]
                row['features_total'] = n_feat
                row['features_crossed'] = len(names)
                row['features_crossed_fraction'] = len(names) / n_feat
                row['drifted_feature_names'] = ';'.join(names)
            elif m in _KEY_METRIC:
                value, threshold = _KEY_METRIC[m](status)
                row['key_metric_value'] = value
                row['key_metric_threshold'] = threshold
            rows.append(row)
    return pd.DataFrame(rows)


def _save_reports(output_dir, methods_to_run, model_meta, all_triggers, weekly_records,
                  top_k, top_features, registry, selection_report, feat_diag,
                  policy_results=None):
    json_path = os.path.join(output_dir, 'unified_drift_report.json')
    csv_path = os.path.join(output_dir, 'unified_drift_report.csv')
    matrix_path = os.path.join(output_dir, 'method_week_matrix.csv')

    method_matrix = _build_method_matrix(weekly_records, methods_to_run, top_features)
    method_matrix.to_csv(matrix_path, index=False)
    logger.info(f"Week x method matrix saved to {matrix_path} "
                f"({len(method_matrix)} rows = {method_matrix['week'].nunique()} weeks "
                f"x {method_matrix['method'].nunique()} methods)")

    report = {
        'model_info_per_method': model_meta,
        'retrain_triggers': all_triggers,
        'top_k': top_k,
        'top_features': top_features,
        'weekly_records': weekly_records,
        'methods_run': methods_to_run,
        'method_thresholds': {m: METHOD_THRESHOLDS[m] for m in methods_to_run},
        'confirmation_rule': {'k': CONFIRM_K, 'n': CONFIRM_N},
        'model_registry': registry.sharing_summary(),
        'feature_selection': selection_report,
        'method_week_matrix': method_matrix.to_dict(orient='records'),
    }

    if policy_results is not None:
        report['policy_comparison'] = {
            'summary': policy_results['summary'].to_dict(orient='records'),
            'agreement_jaccard': policy_results['agreement'].to_dict(),
            'flagged_weeks': policy_results['flagged_weeks'],
            'random_controls': policy_results['random_controls'],
        }

    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    feat_diag.to_csv(os.path.join(output_dir, 'feature_selection_diagnostics.csv'), index=False)

    csv_rows = []
    for rec in weekly_records:
        if rec['is_baseline']:
            csv_rows.append({'week': rec['week'], 'event': 'Baseline', 'overall_drift': False})
            continue
        triggers = [t['method'] for t in rec.get('retrain_triggers_this_week', [])]
        row = {'week': rec['week'], 'event': 'Inference',
               'overall_drift': rec['overall_drift_flag']}
        for m in DRIFT_METHODS:
            key = m.replace('champion_vs_challenger', 'champ_chall')
            row[f'{key}_drift'] = rec['drift_method_flags'].get(m, False)
            row[f'{key}_version'] = rec['method_model_version'].get(m)
        row['retrain_count'] = len(triggers)
        row['triggered_methods'] = ';'.join(triggers)
        row['models_trained_this_week'] = sum(
            1 for t in rec.get('retrain_triggers_this_week', []) if t.get('newly_trained'))
        csv_rows.append(row)

    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    logger.info(f"Reports saved to {json_path} and {csv_path}")


# ══════════════════════════════════════════════
# Main orchestrator
# ══════════════════════════════════════════════
def run_analysis(data_dir='./dataset', top_k=20, output_dir='./reports',
                 method='all', n_bags=5, save_models=True,
                 evaluate_policies=True, materialize_all=True):
    from sklearn.decomposition import PCA

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    methods_to_run = DRIFT_METHODS if method == 'all' else [method]
    logger.info("=" * 60)
    logger.info(f"DRIFT ANALYSIS — shared registry (top_k={top_k}, methods={methods_to_run})")
    logger.info("=" * 60)

    # ── Phase 1 ──
    raw, start_time, end_time = _load_and_prepare_data(data_dir)

    # ── Phase 2 ──
    (fe, ref_X, ref_y, feature_schema, top_features, feat_diag,
     selection_report, registry, cutoff) = _build_baseline(raw, start_time, top_k, n_bags)

    baseline_version = registry.versions[BASELINE_VERSION]

    # ── Phase 3 ──
    cluster_pca = PCA(n_components=2, random_state=42)
    _fit_version_artifacts(baseline_version, ref_X, ref_y,
                           baseline_version.train_predictions, top_features,
                           methods_to_run, cluster_pca=cluster_pca)
    method_state = _initialize_method_state(methods_to_run)

    if 'prequential_auc' in methods_to_run:
        method_state['prequential_auc']['tracker'].set_reference(
            ref_y, baseline_version.train_predictions)

    pca_data = {
        'ref_pca_x': [p[0] for p in baseline_version.artifacts.get('pca_coords', [[0.0, 0.0]])],
        'ref_pca_y': [p[1] for p in baseline_version.artifacts.get('pca_coords', [[0.0, 0.0]])],
        'ref_labels': baseline_version.artifacts.get('pca_labels', [0]),
        'curr_pca_x': [p[0] for p in baseline_version.artifacts.get('pca_coords', [[0.0, 0.0]])],
        'curr_pca_y': [p[1] for p in baseline_version.artifacts.get('pca_coords', [[0.0, 0.0]])],
        'curr_labels': baseline_version.artifacts.get('pca_labels', [0]),
        'centroids_x': [c[0] for c in baseline_version.artifacts.get('pca_centroids', [[0.0, 0.0]])],
        'centroids_y': [c[1] for c in baseline_version.artifacts.get('pca_centroids', [[0.0, 0.0]])],
    }

    model_meta = {
        m: {
            'model_type': 'LightGBM GBDT Classifier (shared registry)',
            'training_window': 'Initial 90 days (shared baseline v0)',
            'baseline_rows': len(ref_X),
            'total_features_trained': len(feature_schema),
            'top_k_features_count': top_k,
            'baseline_auc': float(baseline_version.train_auc),
            'baseline_f1': float(baseline_version.train_f1),
            'retrain_count': 0,
            'retrain_triggers': [],
        }
        for m in DRIFT_METHODS
    }

    weekly_records = [{
        'week': 0,
        'is_baseline': True,
        'rows': len(ref_X),
        'top_features': top_features,
        'curr_top_features': top_features,
        'overall_drift_flag': False,
        'drift_method_flags': {m: False for m in DRIFT_METHODS},
        'drift_method_flags_raw': {m: False for m in DRIFT_METHODS},
        'method_model_version': dict(registry.method_versions),
        'method_status': {m: {'drift': False, 'details': 'Baseline'} for m in DRIFT_METHODS},
        'feature_metrics': {
            feat: {'ks_stat': 0.0, 'ks_p': 1.0, 'ks_p_adjusted': 1.0, 'ks_drift': False,
                   'psi': 0.0, 'psi_null_q': 0.0, 'psi_drift': False,
                   'kl_div': 0.0, 'js_distance': 0.0, 'kl_drift': False,
                   'ref_importance': 0.0, 'curr_importance': 0.0,
                   'shap_shift': 0.0, 'shap_drift': False}
            for feat in top_features
        },
        'method_retrain_info': {
            m: {'retrain_count': 0, 'model_version': BASELINE_VERSION,
                'model_auc': float(baseline_version.train_auc),
                'model_f1': float(baseline_version.train_f1),
                'holdout_auc': 0.0, 'holdout_f1': 0.0}
            for m in DRIFT_METHODS
        },
        'pca_cluster_data': pca_data,
        'retrain_triggers_this_week': [],
    }]
    all_triggers = []

    # Transformed windows accumulate so cumulative retraining data is assembled
    # by concatenation rather than by re-running feature engineering over a
    # growing raw slice every single time a detector drifts.
    cumulative_X = [ref_X]
    cumulative_y = [ref_y]
    # Retained per week so the policy comparison can score any version on any
    # window afterwards without re-running feature engineering.
    weekly_X, weekly_y = {}, {}

    # ── Phase 4: weekly replay ──
    current_time, week_idx = cutoff, 1
    while current_time < end_time:
        next_time = current_time + WEEK_SECONDS
        chunk = raw[(raw['TransactionDT'] >= current_time) & (raw['TransactionDT'] < next_time)]
        if len(chunk) == 0:
            current_time = next_time
            continue

        logger.info(f"\n--- Week {week_idx} ({len(chunk)} rows) ---")

        processed = fe.transform(chunk)            # frozen encoders — no refit
        y_curr = processed['isFraud'].reset_index(drop=True)
        X_aligned = preprocess_and_align(
            processed.drop('isFraud', axis=1), feature_schema).reset_index(drop=True)

        # One out-of-fold challenger fit serves both the CvC comparison and the
        # weekly feature-importance snapshot.
        challenger_oof, challenger_model = oof_predictions(X_aligned, y_curr)
        curr_top_features = (
            pd.DataFrame({'f': feature_schema,
                          'i': challenger_model.feature_importance(importance_type='gain')})
            .sort_values('i', ascending=False).head(top_k)['f'].tolist()
            if challenger_model is not None else top_features
        )

        # Per-method predictions on this week — methods sharing a version share
        # a prediction vector, so score each distinct version only once.
        version_preds = {}
        method_preds, method_error_streams, week_performance = {}, {}, {}
        for mn in DRIFT_METHODS:
            vid = registry.method_versions[mn]
            if vid not in version_preds:
                version_preds[vid] = registry.versions[vid].model.predict(
                    X_aligned, predict_disable_shape_check=True)
            probs = version_preds[vid]
            method_preds[mn] = probs
            # The error stream must be cut at the operating point the model was
            # calibrated to, not at 0.5 — otherwise every error-stream detector
            # is watching a model that predicts "not fraud" for everything.
            cut = get_decision_threshold(registry.versions[vid].model)
            signal = METHOD_THRESHOLDS.get(mn, {}).get('error_signal', 'zero_one')
            method_error_streams[mn] = build_error_stream(
                y_curr, probs, mode=signal, threshold=cut)

            has_both = y_curr.nunique() > 1
            week_performance[mn] = {
                'auc': float(roc_auc_score(y_curr, probs)) if has_both else 0.5,
                'f1': float(f1_score(y_curr, (probs >= cut).astype(int), zero_division=0)),
                'threshold': float(cut),
            }

        ks_results, psi_results, kl_results = _compute_feature_drift(
            top_features, registry, X_aligned, methods_to_run)
        stream_res = _compute_stream_drift(
            method_state, method_error_streams, method_preds, registry, y_curr, methods_to_run)
        shap_res, clustering_res, ae_res, cvc_res = _compute_advanced_drift(
            methods_to_run, registry, X_aligned, y_curr, top_features,
            challenger_model, challenger_oof)

        raw_flags, confirmed_flags, confirm_state = _assemble_drift_flags(
            ks_results, psi_results, kl_results, stream_res,
            shap_res, clustering_res, ae_res, cvc_res,
            top_features, method_state, methods_to_run)

        drifted = [m for m in methods_to_run if confirmed_flags.get(m)]
        reasons = _build_reasons(drifted, top_features, ks_results, psi_results, kl_results,
                                 stream_res, shap_res, clustering_res, ae_res, cvc_res,
                                 confirm_state)

        cumulative_X.append(X_aligned)
        cumulative_y.append(y_curr)
        weekly_X[week_idx], weekly_y[week_idx] = X_aligned, y_curr

        week_triggers = []
        if drifted:
            cum_X = pd.concat(cumulative_X, ignore_index=True)
            cum_y = pd.concat(cumulative_y, ignore_index=True)
            week_triggers, _ = _retrain_drifted_methods(
                week_idx, drifted, registry, cum_X, cum_y, top_features,
                methods_to_run, method_state, cluster_pca, reasons)
            del cum_X, cum_y
            gc.collect()

        all_triggers.extend(week_triggers)
        for t in week_triggers:
            model_meta[t['method']]['retrain_count'] = registry.retrain_count(t['method'])
            model_meta[t['method']]['retrain_triggers'].append(t)

        # PCA snapshot against the clustering method's current reference
        cluster_version = registry.version_for('clustering')
        det = cluster_version.artifacts.get('clustering')
        if det is not None:
            sample = X_aligned[top_features].fillna(0).iloc[:300]
            scaled = det._scale(sample.values)
            curr_coords = cluster_pca.transform(scaled).tolist()
            curr_labels = det.kmeans.predict(scaled).tolist()
        else:
            curr_coords, curr_labels = [[0.0, 0.0]], [0]

        ref_coords = cluster_version.artifacts.get('pca_coords', [[0.0, 0.0]])
        centroids = cluster_version.artifacts.get('pca_centroids', [[0.0, 0.0]])
        current_pca_data = {
            'ref_pca_x': [p[0] for p in ref_coords], 'ref_pca_y': [p[1] for p in ref_coords],
            'ref_labels': cluster_version.artifacts.get('pca_labels', [0]),
            'curr_pca_x': [p[0] for p in curr_coords], 'curr_pca_y': [p[1] for p in curr_coords],
            'curr_labels': curr_labels,
            'centroids_x': [c[0] for c in centroids], 'centroids_y': [c[1] for c in centroids],
        }

        weekly_records.append(_build_weekly_entry(
            week_idx, X_aligned, top_features, curr_top_features,
            raw_flags, confirmed_flags, confirm_state,
            ks_results, psi_results, kl_results, stream_res,
            shap_res, clustering_res, ae_res, cvc_res,
            current_pca_data, registry, week_triggers, week_performance))

        current_time = next_time
        week_idx += 1
        gc.collect()

    # ── Phase 5: materialise the remaining lattice, then compare policies ──
    policy_results = None
    if evaluate_policies:
        if materialize_all:
            # Fill in the weeks no detector requested. The registry is capped at
            # 1 + n_weeks versions either way, so completing the lattice costs at
            # most that bound — and once complete, EVERY retraining policy
            # (always-retrain, and the frequency-matched random controls) becomes
            # a table lookup instead of thousands of additional fits.
            missing = [w for w in sorted(weekly_X) if w not in registry._week_index]
            logger.info(f"Materialising {len(missing)} remaining lattice versions "
                        f"so all policies can be evaluated without retraining...")
            for w in missing:
                cum_X = pd.concat([ref_X] + [weekly_X[i] for i in sorted(weekly_X) if i <= w],
                                  ignore_index=True)
                cum_y = pd.concat([ref_y] + [weekly_y[i] for i in sorted(weekly_y) if i <= w],
                                  ignore_index=True)
                registry.get_or_train(w, cum_X, cum_y)
                del cum_X, cum_y
                gc.collect()

        policy_results = run_policy_evaluation(
            registry, weekly_X, weekly_y, weekly_records, all_triggers,
            [m for m in DRIFT_METHODS if m in methods_to_run],
            get_threshold=get_decision_threshold,
        )
        policy_results['summary'].to_csv(
            os.path.join(output_dir, 'policy_comparison.csv'), index=False)
        policy_results['agreement'].to_csv(
            os.path.join(output_dir, 'detector_agreement.csv'))
        policy_results['auc_matrix'].to_csv(
            os.path.join(output_dir, 'version_week_auc_matrix.csv'))

    if save_models:
        registry.save_all()
    _save_reports(output_dir, methods_to_run, model_meta, all_triggers, weekly_records,
                  top_k, top_features, registry, selection_report, feat_diag,
                  policy_results=policy_results)

    summary = registry.sharing_summary()
    logger.info("=" * 60)
    logger.info(f"COMPLETE — {len(all_triggers)} adoption events across {len(methods_to_run)} detectors")
    logger.info(f"  distinct models trained : {summary['distinct_models_trained']} "
                f"(upper bound 1 + n_weeks)")
    logger.info(f"  per-detector equivalent : {summary['naive_per_method_train_calls']} fits")
    logger.info(f"  training reduction      : {summary['training_reduction_ratio']:.1f}x")
    for m in methods_to_run:
        logger.info(f"  [{m:24s}] retrains={registry.retrain_count(m):2d} "
                    f"final=v{registry.method_versions[m]}")
    logger.info("=" * 60)
    return registry


# ══════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Drift analysis with a shared model registry')
    parser.add_argument('--top_k', type=int, default=20, help='Size of the monitoring feature set')
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--output_dir', type=str, default='./reports')
    parser.add_argument('--method', type=str, default='all', choices=['all'] + DRIFT_METHODS)
    parser.add_argument('--n_bags', type=int, default=5,
                        help='Bootstrap fits used for stability-aware feature selection')
    parser.add_argument('--no_save_models', action='store_true')
    parser.add_argument('--no_policy_eval', action='store_true',
                        help='Skip the retraining-policy comparison')
    parser.add_argument('--no_materialize', action='store_true',
                        help='Do not fill in unrequested lattice versions '
                             '(always-retrain and random controls become partial)')
    args = parser.parse_args()

    run_analysis(data_dir=args.data_dir, top_k=args.top_k, method=args.method,
                 output_dir=args.output_dir, n_bags=args.n_bags,
                 save_models=not args.no_save_models,
                 evaluate_policies=not args.no_policy_eval,
                 materialize_all=not args.no_materialize)
