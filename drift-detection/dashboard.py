"""
Fraud Model & Feature Drift Monitoring Dashboard
=================================================
Multi-page Streamlit dashboard consuming the unified_drift_report.json
produced by run_drift_analysis.py.

Pages:
- Overview             — all-methods summary, drift heatmaps, AUC/F1 trends
- One page per method  — trend chart, adjustable thresholds (recomputed live
                         from stored raw metrics), current-week status,
                         per-feature detail, retrain history with reasons
- Retrain Events        — full cross-method retraining timeline & metadata

Threshold sliders on each method page recompute that method's drift decision
live from the raw metrics already stored in the report (no pipeline rerun
needed). The values the pipeline actually used when the data was generated
are shown alongside for comparison.
"""

import os
import re
import json

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
DRIFT_METHODS = [
    'ks_stats', 'psi', 'kl_divergence', 'ddm', 'eddm', 'adwin', 'hddm',
    'shap', 'clustering', 'autoencoder', 'prequential_auc', 'champion_vs_challenger',
]

# Methods with a dedicated detail page. The remainder still appear in the
# Overview heatmaps, retrain timeline and policy comparison — they simply do
# not have a bespoke chart yet.
METHODS_WITH_PAGES = [
    'ks_stats', 'psi', 'kl_divergence', 'ddm', 'eddm', 'adwin',
    'shap', 'clustering', 'autoencoder', 'champion_vs_challenger',
]

METHOD_LABELS = {
    'ks_stats': 'KS Statistics',
    'psi': 'PSI (Population Stability Index)',
    'kl_divergence': 'KL / Jensen–Shannon Divergence',
    'ddm': 'DDM (Drift Detection Method)',
    'eddm': 'EDDM (Early Drift Detection Method)',
    'adwin': 'ADWIN (Adaptive Windowing)',
    'hddm': 'HDDM (Hoeffding Drift Detection)',
    'shap': 'SHAP Value Drift',
    'clustering': 'Clustering Shift',
    'autoencoder': 'Autoencoder Reconstruction',
    'prequential_auc': 'Prequential AUC Degradation',
    'champion_vs_challenger': 'Champion vs Challenger',
}

METHOD_ICONS = {
    'ks_stats': '📐', 'psi': '📊', 'kl_divergence': '🔀',
    'ddm': '📉', 'eddm': '⏱️', 'adwin': '🪟', 'hddm': '🎯',
    'shap': '🧬', 'clustering': '🧩', 'autoencoder': '🧠',
    'prequential_auc': '📈', 'champion_vs_challenger': '🏆',
}

METHOD_COLORS = {
    'ks_stats': '#3B82F6', 'psi': '#8B5CF6', 'kl_divergence': '#06B6D4',
    'ddm': '#6366F1', 'eddm': '#A855F7', 'adwin': '#14B8A6', 'hddm': '#0EA5E9',
    'shap': '#F59E0B', 'clustering': '#EC4899', 'autoencoder': '#EF4444',
    'prequential_auc': '#F97316', 'champion_vs_challenger': '#10B981',
}

METHOD_DESCRIPTIONS = {
    'ks_stats': "Two-sample Kolmogorov–Smirnov test comparing each monitored feature's "
                "(and the model's prediction) distribution now vs. the baseline.",
    'psi': "Population Stability Index — buckets the baseline distribution into deciles "
           "and measures how much current data's bucket proportions have shifted.",
    'kl_divergence': "Kullback–Leibler divergence between binned densities of current vs. "
                     "baseline data for each monitored feature.",
    'ddm': "Gama et al. (2004) — a persistent tracker (carried across weeks, reset only when this "
           "method retrains) that watches the model's running error rate; flags drift when error "
           "rate + std exceeds the lowest-ever value by a multiple of its std.",
    'eddm': "Baena-García et al. (2006) — a persistent tracker (carried across weeks, reset only "
            "when this method retrains) that watches the distance between consecutive prediction "
            "errors; flags drift when that distance shrinks below a fraction of its historical max.",
    'adwin': "Bifet & Gavaldà (2007) — adaptive windowing over the prediction stream; flags a "
             "mean-shift using River's native ADWIN (or a windowed z-test fallback).",
    'shap': "Explains the model's own reasoning via SHAP values, then KS-tests whether each "
            "feature's influence on the model has shifted — even if the raw feature looks stable.",
    'clustering': "K-Means (k=5) fit on the baseline; flags drift when new data sits farther from "
                  "its nearest centroid, or when cluster-assignment proportions shift (PSI-style).",
    'autoencoder': "A small bottleneck MLP is trained to reconstruct baseline data; flags drift when "
                   "current reconstruction error (RMSE) spikes relative to the baseline distribution.",
    'hddm': "Frías-Blanco et al. (2015) — like DDM but with distribution-free Hoeffding bounds "
            "instead of Bernoulli control limits, so it stays sensitive on long-stable models.",
    'prequential_auc': "Monitors the champion's out-of-sample AUC directly against the AUC at its "
                       "adoption; flags drift when the drop clears both a floor and 2 bootstrap SE.",
    'champion_vs_challenger': "Trains a fresh challenger model on the current week and compares it "
                              "head-to-head against the live champion, using out-of-fold challenger "
                              "predictions so the comparison is not biased by in-sample scoring.",
}

# Fallback thresholds — used only if a report predates the 'method_thresholds' field.
DEFAULT_THRESHOLDS = {
    'ks_stats': {'alpha': 0.05, 'min_feature_fraction': 0.6},
    'psi': {'warning': 0.10, 'drift': 0.20, 'min_feature_fraction': 0.6},
    'kl_divergence': {'warning': 0.25, 'drift': 0.50, 'min_feature_fraction': 0.6},
    'ddm': {'warning_level': 2.0, 'drift_level': 3.0},
    'eddm': {'beta_warning': 0.85, 'beta_drift': 0.75, 'min_errors': 100},
    'adwin': {'delta': 0.002},
    'shap': {'alpha': 0.05, 'min_feature_fraction': 0.6},
    'clustering': {'distance_ratio': 1.5, 'cluster_psi': 0.20},
    'autoencoder': {'z_score': 3.0},
    'hddm': {'drift_confidence': 0.001, 'warning_confidence': 0.005},
    'prequential_auc': {'min_drop': 0.02, 'n_sigma': 2.0},
    'champion_vs_challenger': {'auc_degradation': 0.05, 'auc_gap': 0.03},
}

REPORT_JSON_PATH = 'reports/unified_drift_report.json'
REPORT_CSV_PATH = 'reports/unified_drift_report.csv'


# ──────────────────────────────────────────────
# Page configuration (must run once, before any other st.* call)
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Model Drift Monitoring Dashboard",
    page_icon="\U0001f6e1️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .stMetric {
        background-color: #1E222D;
        padding: 12px 14px;
        border-radius: 8px;
        border: 1px solid #2E3440;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.35rem;
        white-space: normal;
        overflow-wrap: break-word;
        line-height: 1.25;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.82rem;
        opacity: 0.85;
    }
    .status-card {
        padding: 15px 20px;
        border-radius: 10px;
        color: white;
        font-weight: bold;
        margin-bottom: 15px;
        font-size: 1.05em;
    }
    .status-ok {
        background: linear-gradient(135deg, #166534, #14532d);
        border: 1px solid #22c55e;
    }
    .status-drift {
        background: linear-gradient(135deg, #991b1b, #7f1d1d);
        border: 1px solid #ef4444;
    }
    .reason-box {
        background-color: #1E222D;
        border-left: 4px solid #F59E0B;
        padding: 10px 16px;
        border-radius: 4px;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────
@st.cache_data
def load_report_data(json_path=REPORT_JSON_PATH, csv_path=REPORT_CSV_PATH):
    if not os.path.exists(json_path):
        return None, None
    with open(json_path, 'r') as f:
        data = json.load(f)
    csv_df = pd.read_csv(csv_path) if os.path.exists(csv_path) else None
    return data, csv_df


# ──────────────────────────────────────────────
# Generic helpers
# ──────────────────────────────────────────────
def _safe_get(d, *keys, default=0.0):
    """Safely traverse nested dicts."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k, default)
        else:
            return default
    return d if d is not None else default


def _get_inference_records(records):
    return [r for r in records if not r.get('is_baseline', False)]


def _get_retrain_weeks(triggers, method):
    return sorted(set(t['week'] for t in triggers if t['method'] == method))


def _method_thresholds(json_data, method):
    """Runtime thresholds the pipeline actually used, falling back to defaults."""
    stored = json_data.get('method_thresholds', {}).get(method)
    return stored if stored else DEFAULT_THRESHOLDS.get(method, {})


def _method_ran(json_data, method):
    """Whether this method has any data in the current report."""
    methods_run = json_data.get('methods_run')
    if methods_run is None:
        return True  # older report predating this field — assume present
    return method in methods_run


def _reason_for_trigger(t):
    """Human-readable reason for a retrain trigger, with backward-compat fallback."""
    if t.get('reason'):
        return t['reason']
    return "Drift detected (reason not recorded — generated by an older pipeline run)."


def _feature_display_map(top_features):
    """Map real engineered feature names to clean, presentation-ready labels,
    ranked by importance (Feature 1 = most important). Charts/tables/exports
    show these instead of raw engineered column names."""
    return {feat: f"Feature {i + 1}" for i, feat in enumerate(top_features)}


def _disp(feat, feat_map):
    """Display label for a single feature."""
    return feat_map.get(feat, feat)


def _prettify_text(text, feat_map):
    """Replace any real feature-name substrings in free text (e.g. a retrain
    reason) with their clean display label. Longer names are matched first
    so one feature name can't clobber a partial match inside another."""
    if not text or not feat_map:
        return text
    names = sorted(feat_map.keys(), key=len, reverse=True)
    if not names:
        return text
    pattern = re.compile('|'.join(re.escape(n) for n in names))
    return pattern.sub(lambda m: feat_map[m.group(0)], text)


def _add_markers(fig, method_name, retrain_triggers, selected_week):
    """Add thin retrain markers and a current-week marker to a figure."""
    rw = _get_retrain_weeks(retrain_triggers, method_name)
    for w in rw:
        fig.add_vline(x=w, line_dash="dot", line_color="#EF4444", line_width=1, opacity=0.5)
    fig.add_vline(x=selected_week, line_dash="solid", line_color="#10B981", line_width=3,
                  annotation_text=f"W{selected_week}")
    if rw:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode='lines', line=dict(color="#EF4444", dash='dot', width=1),
            name='Retrain week', showlegend=True,
        ))


def _retrain_history_table(method_name, retrain_triggers, feat_map=None):
    """Render a retraining history table (with reasons) for a method, if any."""
    mtr = [t for t in retrain_triggers if t['method'] == method_name]
    if not mtr:
        st.info("No retraining events recorded for this method yet.")
        return
    st.markdown("#### 🔁 Retraining History")
    df = pd.DataFrame(mtr)
    df['reason'] = df.apply(lambda r: _prettify_text(_reason_for_trigger(r), feat_map), axis=1)
    cols = ['week', 'retrain_count', 'cumulative_rows', 'new_auc', 'new_f1', 'reason']
    cols = [c for c in cols if c in df.columns]
    st.dataframe(
        df[cols].rename(columns={
            'week': 'Week', 'retrain_count': 'Retrain #', 'cumulative_rows': 'Cumulative Rows',
            'new_auc': 'New AUC', 'new_f1': 'New F1', 'reason': 'Reason',
        }),
        use_container_width=True, hide_index=True,
    )


def _current_week_reason_banner(method_name, retrain_triggers, selected_week, feat_map=None):
    """If this method retrained on the selected week, show why front-and-center."""
    match = [t for t in retrain_triggers if t['method'] == method_name and t['week'] == selected_week]
    if match:
        st.markdown(
            f'<div class="reason-box">🔁 <b>Retrained this week (#{match[0].get("retrain_count","?")})</b> — '
            f'{_prettify_text(_reason_for_trigger(match[0]), feat_map)}</div>',
            unsafe_allow_html=True,
        )


def _not_run_notice(method_name):
    label = METHOD_LABELS.get(method_name, method_name)
    st.warning(
        f"⚠️ **{label}** has not been run yet — no data found in the current report.\n\n"
        f"Run it with:\n```\npython run_drift_analysis.py --method {method_name}\n```"
    )


def _run_commands_expander(json_data):
    methods_run = json_data.get('methods_run', DRIFT_METHODS)
    registry = json_data.get('model_registry', {})
    with st.expander("▶️ How to (re)generate this data"):
        st.markdown(
            "All detectors share a **single model registry**: they start from one baseline "
            "model, and when several detectors flag drift in the same week they all adopt the "
            "*identical* retrained model. A detector that does not flag drift keeps pointing at "
            "the version it last adopted. Distinct models are therefore capped at "
            "`1 + n_weeks`, and any performance difference between detectors is caused purely "
            "by *when* each chose to retrain — not by training randomness."
        )
        st.code("python run_drift_analysis.py --top_k 10 --n_bags 5", language="bash")
        st.markdown("Or restrict to a single detector:")
        st.code("python run_drift_analysis.py --method psi", language="bash")

        if registry:
            c1, c2, c3 = st.columns(3)
            c1.metric("Distinct models", registry.get('distinct_models_trained', '—'))
            c2.metric("Per-detector equivalent", registry.get('naive_per_method_train_calls', '—'))
            ratio = registry.get('training_reduction_ratio')
            c3.metric("Training reduction", f"{ratio:.1f}x" if ratio else '—')

        missing = [m for m in DRIFT_METHODS if m not in methods_run]
        if missing:
            st.warning(f"Not yet run in this report: {', '.join(METHOD_LABELS.get(m, m) for m in missing)}")
        else:
            st.success(f"All {len(DRIFT_METHODS)} methods have data in this report.")


# ──────────────────────────────────────────────
# Recompute helpers — turn stored raw metrics + a user threshold into a
# live drift decision, without needing to rerun the pipeline.
# ──────────────────────────────────────────────
def recompute_ks(week_record, features, alpha):
    fm = week_record.get('feature_metrics', {})
    drifted = [f for f in features if fm.get(f, {}).get('ks_p', 1.0) < alpha]
    pred = week_record.get('method_status', {}).get('ks_stats', {}).get('prediction_level', {})
    pred_drift = pred.get('p_value', 1.0) < alpha if pred else False
    return bool(drifted or pred_drift), drifted, pred_drift


def recompute_psi(week_record, features, warn_thresh, drift_thresh):
    fm = week_record.get('feature_metrics', {})
    drifted = [f for f in features if fm.get(f, {}).get('psi', 0.0) >= drift_thresh]
    warned = [f for f in features if warn_thresh <= fm.get(f, {}).get('psi', 0.0) < drift_thresh]
    return bool(drifted), drifted, warned


def recompute_kl(week_record, features, drift_thresh):
    fm = week_record.get('feature_metrics', {})
    drifted = [f for f in features if fm.get(f, {}).get('kl_div', 0.0) >= drift_thresh]
    return bool(drifted), drifted


def recompute_ddm(ms, warning_level, drift_level):
    p_min, s_min = ms.get('p_min', 0.0), ms.get('s_min', 0.0)
    err = ms.get('mean_error_rate', 0.0)
    drift_thr = p_min + drift_level * s_min
    warn_thr = p_min + warning_level * s_min
    return bool(err > drift_thr), bool(err > warn_thr), drift_thr, warn_thr


def recompute_eddm(ms, beta_warning, beta_drift):
    max_metric = ms.get('max_metric', 0.0)
    metric = ms.get('metric_value', 0.0)
    drift_thr = beta_drift * max_metric
    warn_thr = beta_warning * max_metric
    is_drift = bool(max_metric > 0 and metric < drift_thr)
    is_warn = bool(max_metric > 0 and metric < warn_thr) and not is_drift
    return is_drift, is_warn, drift_thr, warn_thr


def recompute_adwin(ms, delta):
    z_score = ms.get('z_score', 0.0)
    z_thr = float(norm.ppf(1 - delta / 2))
    return bool(z_score > z_thr), z_thr, z_score


def recompute_shap(ms, features, alpha):
    fsd = ms.get('feature_shap_drift', {})
    drifted = [f for f in features if fsd.get(f, {}).get('p_value', 1.0) < alpha]
    return bool(drifted), drifted


def recompute_clustering(ms, distance_thresh, psi_thresh):
    dr = ms.get('distance_ratio', 1.0)
    cp = ms.get('cluster_psi', 0.0)
    is_drift = bool(dr >= distance_thresh or cp >= psi_thresh)
    is_warn = bool(dr >= distance_thresh * 0.8 or cp >= psi_thresh * 0.5) and not is_drift
    return is_drift, is_warn


def recompute_autoencoder(ms, z_thresh):
    z = ms.get('mse_z_score', 0.0)
    is_drift = bool(z > z_thresh)
    is_warn = bool(z > z_thresh * 0.67) and not is_drift
    return is_drift, is_warn


def recompute_cvc(ms, degradation_thresh, gap_thresh):
    deg = ms.get('auc_degradation', 0.0) or 0.0
    gap = ms.get('auc_gap', 0.0)
    is_drift = bool(deg > degradation_thresh or gap > gap_thresh)
    is_warn = bool(deg > degradation_thresh * 0.5 or gap > gap_thresh * 0.5) and not is_drift
    return is_drift, is_warn


# ──────────────────────────────────────────────
# Weekly diagnostic log — shows, week by week, exactly why a method DID or
# DID NOT retrain: the metric value(s), the live threshold, and a plain-
# English explanation. This is the primary answer to "why didn't it fire?".
# ──────────────────────────────────────────────
def _weekly_feature_gate_table(ctx, method_name, get_feature_value, is_feature_drift, metric_label, frac_threshold):
    """Per-week diagnostic table for feature-fraction-gated methods
    (KS/PSI/KL/SHAP): how many/which monitored features drifted, the
    resulting fraction vs. the retrain gate, and whether/why a retrain did
    or didn't happen.

    get_feature_value(week_record, feat) -> float  (per-feature metric, for a 'max' summary column)
    is_feature_drift(week_record, feat) -> bool     (live per-feature drift decision)
    """
    top_features = ctx['top_features']
    retrain_triggers = ctx['retrain_triggers']
    n = max(len(top_features), 1)
    rows = []
    for r in ctx['inference']:
        drifted = [f for f in top_features if is_feature_drift(r, f)]
        frac = len(drifted) / n
        vals = [get_feature_value(r, f) for f in top_features]
        max_val = max(vals) if vals else 0.0
        retrain_count = r.get('method_retrain_info', {}).get(method_name, {}).get('retrain_count', 0)
        retrained = any(t['method'] == method_name and t['week'] == r['week'] for t in retrain_triggers)
        gate_met = frac >= frac_threshold
        if retrained:
            explanation = (f"{len(drifted)}/{len(top_features)} features ({frac:.0%}) drifted — "
                           f"≥{frac_threshold:.0%} gate met → retrained (#{retrain_count}).")
        elif gate_met:
            explanation = (f"{len(drifted)}/{len(top_features)} features ({frac:.0%}) drifted — gate met, "
                           f"but no retrain recorded this run (method may not have been included in it).")
        else:
            explanation = (f"Only {len(drifted)}/{len(top_features)} features ({frac:.0%}) drifted — "
                           f"below the {frac_threshold:.0%} gate, so no retrain.")
        rows.append({
            'Week': r['week'],
            f'Max {metric_label}': round(max_val, 4),
            'Features Drifted': f"{len(drifted)}/{len(top_features)}",
            'Drift %': f"{frac:.0%}",
            'Gate (≥' + f'{frac_threshold:.0%})'.replace("'", ''): '✅ met' if gate_met else '❌ not met',
            'Retrained': f"\U0001f504 #{retrain_count}" if retrained else '—',
            'Explanation': explanation,
        })
    return pd.DataFrame(rows)


def _weekly_metric_table(ctx, method_name, row_builder_fn):
    """Per-week diagnostic table for stream/model-level methods (DDM, EDDM,
    ADWIN, Clustering, Autoencoder, Champion vs Challenger).

    row_builder_fn(week_record) -> dict of columns to show (should include
    an 'Explanation' key); 'Week' and 'Retrained' are added automatically.
    """
    retrain_triggers = ctx['retrain_triggers']
    rows = []
    for r in ctx['inference']:
        retrain_count = r.get('method_retrain_info', {}).get(method_name, {}).get('retrain_count', 0)
        retrained = any(t['method'] == method_name and t['week'] == r['week'] for t in retrain_triggers)
        row = {'Week': r['week']}
        row.update(row_builder_fn(r))
        row['Retrained'] = f"\U0001f504 #{retrain_count}" if retrained else '—'
        rows.append(row)
    return pd.DataFrame(rows)


def _style_diagnostic(df):
    """Softly highlight weeks where a retrain actually happened."""
    def _highlight(row):
        if row.get('Retrained', '—') != '—':
            return ['background-color: #7f1d1d55'] * len(row)
        return [''] * len(row)
    return df.style.apply(_highlight, axis=1)


def _render_weekly_diagnostic(df, caption=None, expanded=False):
    with st.expander("📋 Weekly Diagnostic Log — why did/didn't it retrain each week?", expanded=expanded):
        if caption:
            st.caption(caption)
        st.dataframe(_style_diagnostic(df), use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────
# Shared per-page bootstrap: load data, sidebar week/feature controls
# ──────────────────────────────────────────────
def _bootstrap(need_features=True):
    json_data, csv_df = load_report_data()
    if json_data is None:
        st.error("⚠️ Report not found! Run `python run_drift_analysis.py` first.")
        st.stop()

    records = json_data['weekly_records']
    inference = _get_inference_records(records)
    weeks = [r['week'] for r in inference]
    top_features = json_data.get('top_features', [])
    retrain_triggers = json_data.get('retrain_triggers', [])
    feat_map = _feature_display_map(top_features)

    st.sidebar.markdown("### ⚙️ Controls")
    selected_week = st.sidebar.selectbox(
        "Inference Window (week)", weeks, index=len(weeks) - 1 if weeks else 0, key="global_week",
    )
    selected_features = top_features
    if need_features:
        selected_features = st.sidebar.multiselect(
            "Features to Monitor", options=top_features,
            default=top_features[:min(10, len(top_features))], key="global_features",
            format_func=lambda f: _disp(f, feat_map),
        )
        selected_features = selected_features or top_features
        # Keep rank order (Feature 1, 2, 3, ...) regardless of multiselect click order.
        selected_features = sorted(selected_features, key=lambda f: top_features.index(f))

    week_record = next((r for r in records if r['week'] == selected_week), records[-1])

    return dict(
        json_data=json_data, csv_df=csv_df, records=records, inference=inference,
        weeks=weeks, top_features=top_features, retrain_triggers=retrain_triggers,
        selected_week=selected_week, selected_features=selected_features, week_record=week_record,
        feat_map=feat_map,
    )


def _method_header(method_name, ctx):
    """Common page header: icon+title, description, run-status guard, current-week banner."""
    st.title(f"{METHOD_ICONS.get(method_name, '📈')} {METHOD_LABELS.get(method_name, method_name)}")
    st.caption(METHOD_DESCRIPTIONS.get(method_name, ""))

    if not _method_ran(ctx['json_data'], method_name):
        _not_run_notice(method_name)
        st.stop()

    is_drift = ctx['week_record'].get('drift_method_flags', {}).get(method_name, False)
    banner_cls = "status-drift" if is_drift else "status-ok"
    banner_txt = "🚨 DRIFT DETECTED" if is_drift else "✅ STABLE"
    st.markdown(
        f'<div class="status-card {banner_cls}">{banner_txt} — Week {ctx["selected_week"]} '
        f'(as generated by the pipeline)</div>',
        unsafe_allow_html=True,
    )
    _current_week_reason_banner(method_name, ctx['retrain_triggers'], ctx['selected_week'], ctx.get('feat_map'))


# ══════════════════════════════════════════════
# PAGE: Overview
# ══════════════════════════════════════════════
def overview_page():
    ctx = _bootstrap()
    json_data, records, inference, weeks = ctx['json_data'], ctx['records'], ctx['inference'], ctx['weeks']
    top_features, retrain_triggers = ctx['top_features'], ctx['retrain_triggers']
    selected_week, week_record = ctx['selected_week'], ctx['week_record']
    selected_features = ctx['selected_features']
    feat_map = ctx['feat_map']
    model_info_per_method = json_data.get('model_info_per_method', {})

    st.title("\U0001f6e1️ Fraud Model & Feature Drift Telemetry — Overview")

    _run_commands_expander(json_data)

    # Header: model metadata
    st.markdown("### \U0001f4cb Model Architecture & Baseline Metadata")
    first_meta = next(iter(model_info_per_method.values()), {})
    model_type_short = first_meta.get('model_type', 'LightGBM').replace(' GBDT Classifier', '')

    row1 = st.columns(4)
    row1[0].metric("Methods w/ Data", len(model_info_per_method))
    row1[1].metric("Model Type", model_type_short)
    row1[2].metric("Training Window", "3 Months")
    row1[3].metric("Features Count", first_meta.get('total_features_trained', 0))

    row2 = st.columns(4)
    row2[0].metric("Baseline Samples", f"{first_meta.get('baseline_rows', 0):,}")
    row2[1].metric("Baseline AUC", f"{first_meta.get('baseline_auc', 0):.4f}")
    row2[2].metric("Baseline F1", f"{first_meta.get('baseline_f1', 0):.4f}")

    with st.expander("🏷️ Feature Name Reference (clean labels used in charts → actual engineered column)"):
        st.dataframe(
            pd.DataFrame({
                'Display Label': [_disp(f, feat_map) for f in top_features],
                'Engineered Feature': top_features,
            }),
            use_container_width=True, hide_index=True,
        )

    # Drift status banner
    is_drift = week_record.get('overall_drift_flag', False)
    drifted_methods = [m for m, v in week_record.get('drift_method_flags', {}).items() if v]
    triggered_this_week = [t['method'] for t in retrain_triggers if t['week'] == selected_week]

    if is_drift:
        st.markdown(
            f'<div class="status-card status-drift">\U0001f6a8 DRIFT DETECTED (Week {selected_week}): '
            f'{", ".join(METHOD_LABELS.get(m, m) for m in drifted_methods)}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="status-card status-ok">✅ MODEL STABLE (Week {selected_week}): '
            f'All methods within bounds.</div>',
            unsafe_allow_html=True,
        )

    if triggered_this_week:
        st.info(f"\U0001f504 Retraining triggered this week by: **{', '.join(METHOD_LABELS.get(m, m) for m in triggered_this_week)}**  "
                f"— open a method's page from the sidebar to see why.")

    # Summary table
    st.markdown("### \U0001f4ca All Methods Summary")
    summary_rows = []
    for m in DRIFT_METHODS:
        ran = _method_ran(json_data, m)
        ri = week_record.get('method_retrain_info', {}).get(m, {})
        has_drift = week_record.get('drift_method_flags', {}).get(m, False)
        last_retrain = max((t['week'] for t in retrain_triggers if t['method'] == m), default=None)
        summary_rows.append({
            'Method': METHOD_LABELS.get(m, m),
            'Data Present': '✅' if ran else '❌ not run',
            'Retrains': ri.get('retrain_count', 0),
            'Status': ('\U0001f6a8 DRIFT' if has_drift else '✅ OK') if ran else '—',
            'Last Retrain': f"Week {last_retrain}" if last_retrain else 'None',
            'Model AUC': f"{ri.get('model_auc', 0):.4f}" if ran else '—',
            'Model F1': f"{ri.get('model_f1', 0):.4f}" if ran else '—',
        })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # Feature × Method heatmap for selected week
    st.markdown("### Drift Detection Heatmap")
    flist = selected_features or top_features
    heatmap_data = []
    for feat in flist:
        fm = week_record.get('feature_metrics', {}).get(feat, {})
        heatmap_data.append({
            'Feature': _disp(feat, feat_map),
            'KS': 1 if fm.get('ks_drift', False) else 0,
            'PSI': 1 if fm.get('psi_drift', False) else 0,
            'KL Div': 1 if fm.get('kl_drift', False) else 0,
            'DDM': 1 if week_record.get('drift_method_flags', {}).get('ddm', False) else 0,
            'EDDM': 1 if week_record.get('drift_method_flags', {}).get('eddm', False) else 0,
            'ADWIN': 1 if week_record.get('drift_method_flags', {}).get('adwin', False) else 0,
            'SHAP': 1 if fm.get('shap_drift', False) else 0,
            'Clustering': 1 if week_record.get('drift_method_flags', {}).get('clustering', False) else 0,
            'Autoencoder': 1 if week_record.get('drift_method_flags', {}).get('autoencoder', False) else 0,
            'CvC': 1 if week_record.get('drift_method_flags', {}).get('champion_vs_challenger', False) else 0,
        })

    if heatmap_data:
        fig = px.imshow(
            pd.DataFrame(heatmap_data).set_index('Feature'),
            color_continuous_scale=[[0, "#1E293B"], [1, "#EF4444"]],
            labels=dict(x="Method", y="Feature", color="Alert"),
            title=f"Features × Methods Drift Status (Week {selected_week})",
        )
        fig.update_layout(height=max(350, 40 * len(flist)))
        st.plotly_chart(fig, use_container_width=True)

    # Drift timeline heatmap: weeks × methods
    st.markdown("### Drift Timeline Heatmap")
    timeline_data = []
    for r in inference:
        row = {'Week': r['week']}
        for m in DRIFT_METHODS:
            row[METHOD_LABELS.get(m, m)] = 1 if r.get('drift_method_flags', {}).get(m, False) else 0
        timeline_data.append(row)

    if timeline_data:
        tl_df = pd.DataFrame(timeline_data).set_index('Week')
        fig_tl = px.imshow(
            tl_df.T,
            color_continuous_scale=[[0, "#1E293B"], [0.5, "#F59E0B"], [1, "#EF4444"]],
            labels=dict(x="Week", y="Method", color="Drift"),
            title="Drift Events Over Time (All Methods × All Weeks)",
            aspect="auto",
        )
        fig_tl.update_layout(height=400)
        st.plotly_chart(fig_tl, use_container_width=True)

    # AUC / F1 comparison across methods over time
    st.markdown("### Model Performance Across Methods Over Time")
    perf_col1, perf_col2 = st.columns(2)

    with perf_col1:
        fig_auc = go.Figure()
        for m in DRIFT_METHODS:
            if not _method_ran(json_data, m):
                continue
            # holdout_auc is the model's out-of-sample AUC on that week's data.
            # model_auc is the version's training AUC, which is flat between
            # retrains and so says nothing about how the model was actually
            # performing in the weeks it was serving.
            aucs = [_safe_get(r, 'method_retrain_info', m, 'holdout_auc')
                    or _safe_get(r, 'method_retrain_info', m, 'model_auc') for r in inference]
            fig_auc.add_trace(go.Scatter(
                x=weeks, y=aucs, mode='lines+markers', name=METHOD_LABELS.get(m, m)[:15],
                line=dict(color=METHOD_COLORS.get(m, '#888'), width=2),
                marker=dict(size=4),
            ))
        fig_auc.update_layout(title="Out-of-sample AUC by Method Over Weeks",
                              xaxis_title="Week", yaxis_title="AUC",
                              height=400, hovermode="x unified", legend=dict(font=dict(size=9)))
        st.plotly_chart(fig_auc, use_container_width=True)

    with perf_col2:
        fig_f1 = go.Figure()
        for m in DRIFT_METHODS:
            if not _method_ran(json_data, m):
                continue
            f1s = [_safe_get(r, 'method_retrain_info', m, 'holdout_f1')
                   or _safe_get(r, 'method_retrain_info', m, 'model_f1') for r in inference]
            fig_f1.add_trace(go.Scatter(
                x=weeks, y=f1s, mode='lines+markers', name=METHOD_LABELS.get(m, m)[:15],
                line=dict(color=METHOD_COLORS.get(m, '#888'), width=2),
                marker=dict(size=4),
            ))
        fig_f1.update_layout(title="Out-of-sample F1 by Method Over Weeks",
                             xaxis_title="Week", yaxis_title="F1",
                             height=400, hovermode="x unified", legend=dict(font=dict(size=9)))
        st.plotly_chart(fig_f1, use_container_width=True)

    # Retrain summary
    st.markdown("### \U0001f504 Retraining Activity Summary")
    if retrain_triggers:
        df_t = pd.DataFrame(retrain_triggers)
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Retraining Events", len(df_t))
        c2.metric("Methods That Retrained", df_t['method'].nunique())
        c3.metric("Events This Week", len(triggered_this_week))
        method_counts = df_t['method'].value_counts().reset_index()
        method_counts.columns = ['Method', 'Count']
        method_counts['Method'] = method_counts['Method'].map(lambda m: METHOD_LABELS.get(m, m))
        fig_counts = px.bar(method_counts, x='Method', y='Count', color='Method', text='Count',
                            color_discrete_sequence=px.colors.qualitative.Set2)
        fig_counts.update_layout(showlegend=False, height=350, title="Retraining Count by Method")
        st.plotly_chart(fig_counts, use_container_width=True)
        st.caption("Open **Retrain Events** in the sidebar for the full timeline, reasons, and metadata.")
    else:
        st.info("No retraining triggers recorded during this analysis period.")

    _policy_comparison_section(json_data)


def _policy_comparison_section(json_data):
    """Each detector as a retraining policy, against the controls that matter."""
    pc = json_data.get('policy_comparison')
    if not pc:
        return

    st.markdown("### \U0001f3c1 Retraining Policy Comparison")
    st.markdown(
        "A detector is only useful as a **retraining policy**. Each row is the "
        "out-of-sample performance of the model that policy actually had in force, "
        "week by week. The rows to compare against are `never_retrain` (the floor) "
        "and `always_retrain` (the ceiling)."
    )

    df = pd.DataFrame(pc.get('summary', []))
    if df.empty:
        return

    display = df[['policy', 'n_retrains', 'mean_auc', 'min_auc', 'mean_f1',
                  'random_control_percentile', 'beats_random_control']].copy()
    display.columns = ['Policy', 'Retrains', 'Mean AUC', 'Worst-week AUC', 'Mean F1',
                       'Random-control pct', 'Beats random?']
    display['Policy'] = display['Policy'].map(lambda p: METHOD_LABELS.get(p, p))
    st.dataframe(
        display.style.format({
            'Mean AUC': '{:.4f}', 'Worst-week AUC': '{:.4f}', 'Mean F1': '{:.4f}',
            'Random-control pct': lambda v: '—' if pd.isna(v) else f'{v:.2f}',
        }, na_rep='—'),
        use_container_width=True, hide_index=True,
    )
    st.caption(
        "**Random-control pct** is the detector's percentile against 200 random policies "
        "that retrain the *same number of times* at randomly chosen weeks. This is the "
        "control that matters: more retraining generally helps, so a detector beating "
        "`never_retrain` proves nothing on its own — it has to beat a random policy of "
        "equal cost. A percentile below ~0.95 means the detector's *timing* added nothing "
        "beyond its retraining frequency."
    )

    agreement = pc.get('agreement_jaccard')
    if agreement:
        st.markdown("#### Detector Agreement (Jaccard over flagged weeks)")
        agr_df = pd.DataFrame(agreement)
        agr_df = agr_df.reindex(index=agr_df.columns)
        fig = px.imshow(
            agr_df, color_continuous_scale='Blues', zmin=0, zmax=1, aspect='auto',
            labels=dict(x="Detector", y="Detector", color="Jaccard"),
        )
        fig.update_layout(height=520)
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Detectors that always agree are not independent opinions. High agreement "
            "*across families* (distributional vs. performance-aware) suggests drift visible "
            "in both the feature marginals and the feature→label concept; agreement confined "
            "to the performance-aware detectors suggests pure concept drift."
        )


# ══════════════════════════════════════════════
# PAGE FACTORY — one function per feature-distribution method
# ══════════════════════════════════════════════
def page_ks():
    ctx = _bootstrap()
    _method_header('ks_stats', ctx)
    inference, weeks = ctx['inference'], ctx['weeks']
    flist = ctx['selected_features']
    week_record, selected_week = ctx['week_record'], ctx['selected_week']
    retrain_triggers = ctx['retrain_triggers']

    with st.expander("⚙️ Adjust Threshold (recomputed live from stored p-values)", expanded=True):
        stored = _method_thresholds(ctx['json_data'], 'ks_stats')
        alpha = st.number_input("Alpha (p-value cutoff)", min_value=0.001, max_value=0.5,
                                 value=float(stored.get('alpha', 0.05)), step=0.005, key="ks_alpha")
        st.caption(f"Pipeline default when this data was generated: alpha = {stored.get('alpha', 0.05)}")

    live_drift, live_feats, live_pred = recompute_ks(week_record, flist, alpha)
    c1, c2, c3 = st.columns(3)
    c1.metric("As Run (pipeline)", "🚨 Drift" if week_record.get('drift_method_flags', {}).get('ks_stats') else "✅ OK")
    c2.metric("Live (your alpha)", "🚨 Drift" if live_drift else "✅ OK")
    c3.metric("Drifted Features (live)", len(live_feats))

    ks_trend = [max(_safe_get(r, 'feature_metrics', f, 'ks_stat') for f in flist) if flist else 0 for r in inference]
    fig_ks = go.Figure()
    fig_ks.add_trace(go.Scatter(x=weeks, y=ks_trend, mode='lines+markers', name='Max KS Stat',
                                line=dict(color='#3B82F6', width=2)))
    _add_markers(fig_ks, 'ks_stats', retrain_triggers, selected_week)
    fig_ks.update_layout(title="Max KS Statistic Across Monitored Features", xaxis_title="Week", yaxis_title="KS Stat",
                         height=400, hovermode="x unified")
    st.plotly_chart(fig_ks, use_container_width=True)

    feat_map = ctx['feat_map']
    col_bar, col_table = st.columns([1.2, 1])
    with col_bar:
        vals = [week_record.get('feature_metrics', {}).get(f, {}).get('ks_stat', 0.0) for f in flist]
        drift_status = [week_record.get('feature_metrics', {}).get(f, {}).get('ks_p', 1.0) < alpha for f in flist]
        colors = ['#EF4444' if d else '#3B82F6' for d in drift_status]
        fig_bar = go.Figure(go.Bar(x=[_disp(f, feat_map) for f in flist], y=vals, marker_color=colors))
        fig_bar.update_layout(title=f"Per-Feature KS Stat (Week {selected_week})", height=350,
                              xaxis_title="Feature", yaxis_title="KS Stat")
        st.plotly_chart(fig_bar, use_container_width=True)
    with col_table:
        table_data = []
        for f in flist:
            fm = week_record.get('feature_metrics', {}).get(f, {})
            table_data.append({
                'Feature': _disp(f, feat_map),
                'KS Stat': f"{fm.get('ks_stat', 0):.4f}",
                'p-value': f"{fm.get('ks_p', 1):.4f}",
                'Drift (live)': '\U0001f6a8' if fm.get('ks_p', 1.0) < alpha else '✅',
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)

    pred = week_record.get('method_status', {}).get('ks_stats', {}).get('prediction_level', {})
    if pred:
        st.caption(f"Prediction-distribution KS: stat={pred.get('stat', 0):.4f}, "
                   f"p-value={pred.get('p_value', 1):.4f} "
                   f"({'drift' if live_pred else 'stable'} at alpha={alpha})")

    frac_thresh = stored.get('min_feature_fraction', 0.6)
    df_diag = _weekly_feature_gate_table(
        ctx, 'ks_stats',
        get_feature_value=lambda r, f: r.get('feature_metrics', {}).get(f, {}).get('ks_stat', 0.0),
        is_feature_drift=lambda r, f: r.get('feature_metrics', {}).get(f, {}).get('ks_p', 1.0) < alpha,
        metric_label='KS Stat', frac_threshold=frac_thresh,
    )
    _render_weekly_diagnostic(
        df_diag,
        caption=f"KS only retrains once more than {frac_thresh:.0%} of monitored features individually "
                f"show KS drift (p < {alpha}) — a single noisy feature no longer forces a retrain.",
    )

    _retrain_history_table('ks_stats', retrain_triggers, ctx['feat_map'])


def page_psi():
    ctx = _bootstrap()
    _method_header('psi', ctx)
    inference, weeks = ctx['inference'], ctx['weeks']
    flist = ctx['selected_features']
    week_record, selected_week = ctx['week_record'], ctx['selected_week']
    retrain_triggers = ctx['retrain_triggers']

    with st.expander("⚙️ Adjust Thresholds (recomputed live from stored PSI values)", expanded=True):
        stored = _method_thresholds(ctx['json_data'], 'psi')
        c1, c2 = st.columns(2)
        warn_t = c1.number_input("Warning Threshold", min_value=0.01, max_value=1.0,
                                  value=float(stored.get('warning', 0.10)), step=0.01, key="psi_warn")
        drift_t = c2.number_input("Drift Threshold", min_value=0.02, max_value=1.0,
                                   value=float(stored.get('drift', 0.20)), step=0.01, key="psi_drift")
        st.caption(f"Pipeline defaults: warning={stored.get('warning', 0.10)}, drift={stored.get('drift', 0.20)}")

    live_drift, live_feats, live_warn = recompute_psi(week_record, flist, warn_t, drift_t)
    c1, c2, c3 = st.columns(3)
    c1.metric("As Run (pipeline)", "🚨 Drift" if week_record.get('drift_method_flags', {}).get('psi') else "✅ OK")
    c2.metric("Live (your thresholds)", "🚨 Drift" if live_drift else "✅ OK")
    c3.metric("Drifted Features (live)", len(live_feats))

    psi_trend = [max(_safe_get(r, 'feature_metrics', f, 'psi') for f in flist) if flist else 0 for r in inference]
    fig_psi = go.Figure()
    fig_psi.add_trace(go.Scatter(x=weeks, y=psi_trend, mode='lines+markers', name='Max PSI',
                                 line=dict(color='#8B5CF6', width=2)))
    fig_psi.add_hline(y=warn_t, line_dash="dot", line_color="#F59E0B", annotation_text=f"Warning ({warn_t})")
    fig_psi.add_hline(y=drift_t, line_dash="dash", line_color="#EF4444", annotation_text=f"Drift ({drift_t})")
    fig_psi.add_hrect(y0=0, y1=warn_t, fillcolor="#22c55e", opacity=0.05)
    fig_psi.add_hrect(y0=warn_t, y1=drift_t, fillcolor="#F59E0B", opacity=0.05)
    fig_psi.add_hrect(y0=drift_t, y1=max(psi_trend) * 1.2 if psi_trend else drift_t * 2, fillcolor="#EF4444", opacity=0.05)
    _add_markers(fig_psi, 'psi', retrain_triggers, selected_week)
    fig_psi.update_layout(title="Max PSI Across Monitored Features", xaxis_title="Week", yaxis_title="PSI",
                          height=400, hovermode="x unified")
    st.plotly_chart(fig_psi, use_container_width=True)

    feat_map = ctx['feat_map']
    vals = [week_record.get('feature_metrics', {}).get(f, {}).get('psi', 0.0) for f in flist]
    colors = ['#EF4444' if v >= drift_t else ('#F59E0B' if v >= warn_t else '#8B5CF6') for v in vals]
    fig_bar = go.Figure(go.Bar(x=[_disp(f, feat_map) for f in flist], y=vals, marker_color=colors))
    fig_bar.add_hline(y=drift_t, line_dash="dash", line_color="red", annotation_text=f"Drift ({drift_t})")
    fig_bar.add_hline(y=warn_t, line_dash="dot", line_color="orange", annotation_text=f"Warning ({warn_t})")
    fig_bar.update_layout(title=f"Per-Feature PSI (Week {selected_week})", height=350)
    st.plotly_chart(fig_bar, use_container_width=True)

    frac_thresh = stored.get('min_feature_fraction', 0.6)
    df_diag = _weekly_feature_gate_table(
        ctx, 'psi',
        get_feature_value=lambda r, f: r.get('feature_metrics', {}).get(f, {}).get('psi', 0.0),
        is_feature_drift=lambda r, f: r.get('feature_metrics', {}).get(f, {}).get('psi', 0.0) >= drift_t,
        metric_label='PSI', frac_threshold=frac_thresh,
    )
    _render_weekly_diagnostic(
        df_diag,
        caption=f"PSI only retrains once more than {frac_thresh:.0%} of monitored features individually "
                f"exceed PSI ≥ {drift_t} — this is why, e.g., a week with only 2 drifted features doesn't retrain.",
    )

    _retrain_history_table('psi', retrain_triggers, ctx['feat_map'])


def page_kl():
    ctx = _bootstrap()
    _method_header('kl_divergence', ctx)
    inference, weeks = ctx['inference'], ctx['weeks']
    flist = ctx['selected_features']
    week_record, selected_week = ctx['week_record'], ctx['selected_week']
    retrain_triggers = ctx['retrain_triggers']

    with st.expander("⚙️ Adjust Threshold (recomputed live from stored KL values)", expanded=True):
        stored = _method_thresholds(ctx['json_data'], 'kl_divergence')
        drift_t = st.number_input("Drift Threshold", min_value=0.05, max_value=5.0,
                                   value=float(stored.get('drift', 0.5)), step=0.05, key="kl_drift")
        st.caption(f"Pipeline default: drift = {stored.get('drift', 0.5)}")

    live_drift, live_feats = recompute_kl(week_record, flist, drift_t)
    c1, c2, c3 = st.columns(3)
    c1.metric("As Run (pipeline)", "🚨 Drift" if week_record.get('drift_method_flags', {}).get('kl_divergence') else "✅ OK")
    c2.metric("Live (your threshold)", "🚨 Drift" if live_drift else "✅ OK")
    c3.metric("Drifted Features (live)", len(live_feats))

    kl_trend = [max(_safe_get(r, 'feature_metrics', f, 'kl_div') for f in flist) if flist else 0 for r in inference]
    fig_kl = go.Figure()
    fig_kl.add_trace(go.Scatter(x=weeks, y=kl_trend, mode='lines+markers', name='Max KL Div',
                                line=dict(color='#06B6D4', width=2)))
    fig_kl.add_hline(y=drift_t, line_dash="dash", line_color="#EF4444", annotation_text=f"Drift ({drift_t})")
    _add_markers(fig_kl, 'kl_divergence', retrain_triggers, selected_week)
    fig_kl.update_layout(title="Max KL Divergence Across Monitored Features", xaxis_title="Week", yaxis_title="KL Divergence",
                         height=400, hovermode="x unified")
    st.plotly_chart(fig_kl, use_container_width=True)

    feat_map = ctx['feat_map']
    vals = [week_record.get('feature_metrics', {}).get(f, {}).get('kl_div', 0.0) for f in flist]
    colors = ['#EF4444' if v >= drift_t else '#06B6D4' for v in vals]
    fig_bar = go.Figure(go.Bar(x=[_disp(f, feat_map) for f in flist], y=vals, marker_color=colors))
    fig_bar.add_hline(y=drift_t, line_dash="dash", line_color="red", annotation_text=f"Threshold ({drift_t})")
    fig_bar.update_layout(title=f"Per-Feature KL Divergence (Week {selected_week})", height=350)
    st.plotly_chart(fig_bar, use_container_width=True)

    frac_thresh = stored.get('min_feature_fraction', 0.6)
    df_diag = _weekly_feature_gate_table(
        ctx, 'kl_divergence',
        get_feature_value=lambda r, f: r.get('feature_metrics', {}).get(f, {}).get('kl_div', 0.0),
        is_feature_drift=lambda r, f: r.get('feature_metrics', {}).get(f, {}).get('kl_div', 0.0) >= drift_t,
        metric_label='KL Div', frac_threshold=frac_thresh,
    )
    _render_weekly_diagnostic(
        df_diag,
        caption=f"KL Divergence only retrains once more than {frac_thresh:.0%} of monitored features "
                f"individually exceed KL ≥ {drift_t}.",
    )

    _retrain_history_table('kl_divergence', retrain_triggers, ctx['feat_map'])


def page_ddm():
    ctx = _bootstrap(need_features=False)
    _method_header('ddm', ctx)
    inference, weeks = ctx['inference'], ctx['weeks']
    week_record, selected_week = ctx['week_record'], ctx['selected_week']
    retrain_triggers = ctx['retrain_triggers']

    with st.expander("⚙️ Adjust Thresholds (recomputed live from stored error-rate stats)", expanded=True):
        stored = _method_thresholds(ctx['json_data'], 'ddm')
        c1, c2 = st.columns(2)
        warn_l = c1.number_input("Warning Level (× std)", min_value=0.5, max_value=5.0,
                                  value=float(stored.get('warning_level', 2.0)), step=0.1, key="ddm_warn")
        drift_l = c2.number_input("Drift Level (× std)", min_value=0.5, max_value=6.0,
                                   value=float(stored.get('drift_level', 3.0)), step=0.1, key="ddm_drift")
        st.caption(f"Pipeline defaults: warning_level={stored.get('warning_level', 2.0)}, drift_level={stored.get('drift_level', 3.0)}")

    curr_ddm = week_record.get('method_status', {}).get('ddm', {})
    live_drift, live_warn, live_drift_thr, live_warn_thr = recompute_ddm(curr_ddm, warn_l, drift_l)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Error Rate", f"{curr_ddm.get('mean_error_rate', 0):.4f}")
    c2.metric("p_min / s_min", f"{curr_ddm.get('p_min', 0):.4f} / {curr_ddm.get('s_min', 0):.4f}")
    c3.metric("As Run", "🚨 Drift" if curr_ddm.get('drift_detected', False) else "✅ OK")
    c4.metric("Live", "🚨 Drift" if live_drift else ("⚠️ Warning" if live_warn else "✅ OK"))

    ddm_error_rates, ddm_pmin, ddm_smin = [], [], []
    for r in inference:
        ms = r.get('method_status', {}).get('ddm', {})
        ddm_error_rates.append(ms.get('mean_error_rate', 0.0))
        ddm_pmin.append(ms.get('p_min', 0.0))
        ddm_smin.append(ms.get('s_min', 0.0))
    ddm_drift_thresholds = [p + drift_l * s for p, s in zip(ddm_pmin, ddm_smin)]
    ddm_warning_thresholds = [p + warn_l * s for p, s in zip(ddm_pmin, ddm_smin)]

    fig_ddm = go.Figure()
    fig_ddm.add_trace(go.Scatter(x=weeks, y=ddm_error_rates, mode='lines+markers', name='Error Rate',
                                  line=dict(color='#6366F1', width=2)))
    fig_ddm.add_trace(go.Scatter(x=weeks, y=ddm_drift_thresholds, mode='lines', name='Drift Threshold (live)',
                                  line=dict(color='#EF4444', dash='dash', width=1.5)))
    fig_ddm.add_trace(go.Scatter(x=weeks, y=ddm_warning_thresholds, mode='lines', name='Warning Threshold (live)',
                                  line=dict(color='#F59E0B', dash='dot', width=1.5)))
    _add_markers(fig_ddm, 'ddm', retrain_triggers, selected_week)
    fig_ddm.update_layout(title="DDM Error Rate vs. Thresholds", xaxis_title="Week", yaxis_title="Error Rate",
                          height=400, hovermode="x unified")
    st.plotly_chart(fig_ddm, use_container_width=True)
    st.caption("p_min/s_min (and the threshold they define) now accumulate across weeks — a single "
               "persistent tracker, reset only when DDM actually retrains — rather than restarting "
               "every week. The 'Live' threshold line here still uses each week's own p_min/s_min as "
               "recomputed with your slider above, as an approximation: the pipeline's actual "
               "'As Run' decision is made per-instant while scanning the stream (it can catch a "
               "mid-week spike even if the week's average error rate ends up looking fine), so 'As "
               "Run' and 'Live' can occasionally disagree — that's expected, not a bug.")

    def _ddm_row(r):
        ms = r.get('method_status', {}).get('ddm', {})
        as_run = ms.get('drift_detected', False)
        d, w, thr, warn_thr = recompute_ddm(ms, warn_l, drift_l)
        err = ms.get('mean_error_rate', 0.0)
        if d:
            expl = f"error_rate={err:.4f} exceeded drift_threshold={thr:.4f} → drift."
        elif w:
            expl = f"error_rate={err:.4f} in warning band (> {warn_thr:.4f}, below drift_threshold={thr:.4f})."
        else:
            expl = f"error_rate={err:.4f} below warning_threshold={warn_thr:.4f} — stable."
        if as_run != d:
            expl += (" (Note: 'As Run' used the per-instant stream during that week and can "
                     "differ from this week-average approximation.)")
        return {
            'Error Rate': round(err, 4),
            'Drift Threshold (live)': round(thr, 4),
            'Drift (As Run)': '🚨' if as_run else '✅',
            'Drift (Live)': '🚨' if d else ('⚠️' if w else '✅'),
            'Explanation': expl,
        }

    df_diag = _weekly_metric_table(ctx, 'ddm', _ddm_row)
    _render_weekly_diagnostic(
        df_diag,
        caption="DDM retrains whenever the error rate crosses its own running drift_threshold — "
                "there is no feature-count gate for this method. p_min/s_min persist across weeks "
                "(reset only on retrain), so 'Drift Threshold (live)' generally trends downward/stabilizes "
                "over time rather than jumping around week to week.",
    )

    _retrain_history_table('ddm', retrain_triggers, ctx['feat_map'])


def page_eddm():
    ctx = _bootstrap(need_features=False)
    _method_header('eddm', ctx)
    inference, weeks = ctx['inference'], ctx['weeks']
    week_record, selected_week = ctx['week_record'], ctx['selected_week']
    retrain_triggers = ctx['retrain_triggers']

    with st.expander("⚙️ Adjust Thresholds (recomputed live from stored metric values)", expanded=True):
        stored = _method_thresholds(ctx['json_data'], 'eddm')
        c1, c2 = st.columns(2)
        beta_warn = c1.number_input("Beta Warning", min_value=0.3, max_value=0.999,
                                     value=float(stored.get('beta_warning', 0.85)), step=0.01, key="eddm_bw")
        beta_drift = c2.number_input("Beta Drift", min_value=0.3, max_value=0.999,
                                      value=float(stored.get('beta_drift', 0.75)), step=0.01, key="eddm_bd")
        st.caption(f"Pipeline defaults: beta_warning={stored.get('beta_warning', 0.85)}, "
                   f"beta_drift={stored.get('beta_drift', 0.75)}, "
                   f"min_errors={stored.get('min_errors', 100)} (warm-up before evaluation starts/restarts; "
                   f"fixed at pipeline run time, not adjustable here). The literature defaults (0.90/0.95) "
                   f"turned out to be extremely sensitive to natural noise in error-gap statistics — on "
                   f"stable data with no real drift they retrained ~13/14 weeks in testing; these looser "
                   f"values still catch a genuine 2.5x error-rate spike within 3-4 weeks.")

    curr_eddm = week_record.get('method_status', {}).get('eddm', {})
    live_drift, live_warn, live_drift_thr, live_warn_thr = recompute_eddm(curr_eddm, beta_warn, beta_drift)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Error Rate", f"{curr_eddm.get('mean_error_rate', 0):.4f}")
    c2.metric("Metric (p'+2s')", f"{curr_eddm.get('metric_value', 0):.2f}")
    c3.metric("As Run", "🚨 Drift" if curr_eddm.get('drift_detected', False) else "✅ OK")
    c4.metric("Live", "🚨 Drift" if live_drift else ("⚠️ Warning" if live_warn else "✅ OK"))

    eddm_metrics, eddm_max, eddm_err = [], [], []
    for r in inference:
        ms = r.get('method_status', {}).get('eddm', {})
        eddm_metrics.append(ms.get('metric_value', 0.0))
        eddm_max.append(ms.get('max_metric', 0.0))
        eddm_err.append(ms.get('mean_error_rate', 0.0))
    eddm_drift_thresholds = [beta_drift * m for m in eddm_max]
    eddm_warning_thresholds = [beta_warn * m for m in eddm_max]

    fig_eddm = go.Figure()
    fig_eddm.add_trace(go.Scatter(x=weeks, y=eddm_metrics, mode='lines+markers', name="p' + 2·s'",
                                   line=dict(color='#A855F7', width=2)))
    fig_eddm.add_trace(go.Scatter(x=weeks, y=eddm_drift_thresholds, mode='lines', name='Drift Threshold (live)',
                                   line=dict(color='#EF4444', dash='dash', width=1.5)))
    fig_eddm.add_trace(go.Scatter(x=weeks, y=eddm_warning_thresholds, mode='lines', name='Warning Threshold (live)',
                                   line=dict(color='#F59E0B', dash='dot', width=1.5)))
    _add_markers(fig_eddm, 'eddm', retrain_triggers, selected_week)
    fig_eddm.update_layout(title="EDDM Metric vs. Thresholds", xaxis_title="Week", yaxis_title="Metric Value",
                           height=400, hovermode="x unified")
    st.plotly_chart(fig_eddm, use_container_width=True)

    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(x=weeks, y=eddm_err, mode='lines+markers', name='Error Rate',
                                  line=dict(color='#8B5CF6', width=2)))
    fig_err.update_layout(title="EDDM Error Rate Over Time", xaxis_title="Week", yaxis_title="Error Rate",
                          height=300, hovermode="x unified")
    st.plotly_chart(fig_err, use_container_width=True)

    def _eddm_row(r):
        ms = r.get('method_status', {}).get('eddm', {})
        as_run = ms.get('drift_detected', False)
        d, w, thr, warn_thr = recompute_eddm(ms, beta_warn, beta_drift)
        metric = ms.get('metric_value', 0.0)
        if d:
            expl = f"metric={metric:.2f} fell below drift_threshold={thr:.2f} → drift."
        elif w:
            expl = f"metric={metric:.2f} in warning band (below {warn_thr:.2f}, above drift_threshold={thr:.2f})."
        else:
            expl = f"metric={metric:.2f} above warning_threshold={warn_thr:.2f} — stable."
        if as_run != d:
            expl += (" (Note: 'As Run' reflects whether ANY instant that week dipped below threshold; "
                     "'Live' only checks the metric's value at week's end.)")
        return {
            'Metric (p\'+2s\')': round(metric, 2),
            'Drift Threshold (live)': round(thr, 2),
            'Drift (As Run)': '🚨' if as_run else '✅',
            'Drift (Live)': '🚨' if d else ('⚠️' if w else '✅'),
            'Explanation': expl,
        }

    df_diag = _weekly_metric_table(ctx, 'eddm', _eddm_row)
    _render_weekly_diagnostic(
        df_diag,
        caption="EDDM retrains whenever its inter-error-distance metric falls below β·max_metric_ever_seen "
                "— there is no feature-count gate for this method. max_metric_ever_seen now accumulates "
                "across weeks (reset only on retrain) instead of restarting every week.",
    )

    _retrain_history_table('eddm', retrain_triggers, ctx['feat_map'])


def page_adwin():
    ctx = _bootstrap(need_features=False)
    _method_header('adwin', ctx)
    inference, weeks = ctx['inference'], ctx['weeks']
    week_record, selected_week = ctx['week_record'], ctx['selected_week']
    retrain_triggers = ctx['retrain_triggers']

    with st.expander("⚙️ Adjust Threshold (recomputed live from stored z-scores)", expanded=True):
        stored = _method_thresholds(ctx['json_data'], 'adwin')
        delta = st.number_input("Delta (confidence — lower = more sensitive)", min_value=0.0001, max_value=0.5,
                                 value=float(stored.get('delta', 0.002)), step=0.001, format="%.4f", key="adwin_delta")
        st.caption(f"Pipeline default: delta = {stored.get('delta', 0.002)}. "
                   f"If River's native ADWIN was used, the 'As Run' decision came from its internal state; "
                   f"'Live' always uses the windowed z-test approximation.")

    curr_adwin = week_record.get('method_status', {}).get('adwin', {})
    live_drift, live_z_thr, live_z = recompute_adwin(curr_adwin, delta)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estimation", f"{curr_adwin.get('estimation', 0):.4f}")
    c2.metric("Z-Score", f"{live_z:.2f}")
    c3.metric("As Run", "🚨 Drift" if curr_adwin.get('drift_detected', False) else "✅ OK")
    c4.metric("Live", "🚨 Drift" if live_drift else "✅ OK")

    adwin_estimations, adwin_z, adwin_detected = [], [], []
    for r in inference:
        ms = r.get('method_status', {}).get('adwin', {})
        adwin_estimations.append(ms.get('estimation', 0.0))
        adwin_z.append(ms.get('z_score', 0.0))
        adwin_detected.append(1 if ms.get('drift_detected', False) else 0)

    col_a1, col_a2 = st.columns(2)
    with col_a1:
        fig_est = go.Figure()
        fig_est.add_trace(go.Scatter(x=weeks, y=adwin_estimations, mode='lines+markers', name='Estimation',
                                      line=dict(color='#14B8A6', width=2)))
        _add_markers(fig_est, 'adwin', retrain_triggers, selected_week)
        fig_est.update_layout(title="ADWIN Estimation Over Time", xaxis_title="Week", yaxis_title="Estimation",
                              height=350, hovermode="x unified")
        st.plotly_chart(fig_est, use_container_width=True)

    with col_a2:
        fig_z = go.Figure()
        colors_occ = ['#EF4444' if z > live_z_thr else '#14B8A6' for z in adwin_z]
        fig_z.add_trace(go.Bar(x=weeks, y=adwin_z, marker_color=colors_occ, name='Z-Score'))
        fig_z.add_hline(y=live_z_thr, line_dash="dash", line_color="#EF4444", annotation_text=f"Threshold ({live_z_thr:.2f})")
        fig_z.update_layout(title="ADWIN Z-Score vs. Live Threshold", xaxis_title="Week",
                            yaxis_title="Z-Score", height=350)
        st.plotly_chart(fig_z, use_container_width=True)

    def _adwin_row(r):
        ms = r.get('method_status', {}).get('adwin', {})
        as_run = ms.get('drift_detected', False)
        d, z_thr_row, z = recompute_adwin(ms, delta)
        if d:
            expl = f"z={z:.2f} exceeded threshold z={z_thr_row:.2f} → prediction stream shifted from reference."
        else:
            expl = f"z={z:.2f} below threshold z={z_thr_row:.2f} — predictions still consistent with reference."
        return {
            'Z-Score (vs. reference)': round(z, 2),
            'Z-Threshold (live)': round(z_thr_row, 2),
            'Drift (As Run)': '🚨' if as_run else '✅',
            'Drift (Live)': '🚨' if d else '✅',
            'Explanation': expl,
        }

    df_diag = _weekly_metric_table(ctx, 'adwin', _adwin_row)
    _render_weekly_diagnostic(
        df_diag,
        caption="ADWIN retrains whenever the current week's predictions shift significantly (z-score) "
                "relative to the reference stream — there is no feature-count gate for this method.",
    )

    _retrain_history_table('adwin', retrain_triggers, ctx['feat_map'])


def page_shap():
    ctx = _bootstrap()
    _method_header('shap', ctx)
    inference, weeks = ctx['inference'], ctx['weeks']
    flist = ctx['selected_features']
    week_record, selected_week = ctx['week_record'], ctx['selected_week']
    retrain_triggers = ctx['retrain_triggers']

    with st.expander("⚙️ Adjust Threshold (recomputed live from stored per-feature p-values)", expanded=True):
        stored = _method_thresholds(ctx['json_data'], 'shap')
        alpha = st.number_input("Alpha (p-value cutoff)", min_value=0.001, max_value=0.5,
                                 value=float(stored.get('alpha', 0.05)), step=0.005, key="shap_alpha")
        st.caption(f"Pipeline default: alpha = {stored.get('alpha', 0.05)}")

    shap_status = week_record.get('method_status', {}).get('shap', {})
    live_drift, live_feats = recompute_shap(shap_status, flist, alpha)

    c1, c2, c3 = st.columns(3)
    c1.metric("As Run (pipeline)", "🚨 Drift" if week_record.get('drift_method_flags', {}).get('shap') else "✅ OK")
    c2.metric("Live (your alpha)", "🚨 Drift" if live_drift else "✅ OK")
    c3.metric("Drifted Features (live)", len(live_feats))

    shap_drift_counts = [r.get('method_status', {}).get('shap', {}).get('drifted_features_count', 0) for r in inference]
    fig_shap_trend = go.Figure()
    fig_shap_trend.add_trace(go.Bar(x=weeks, y=shap_drift_counts, name='Drifted Features (as run)',
                                     marker_color='#F59E0B'))
    _add_markers(fig_shap_trend, 'shap', retrain_triggers, selected_week)
    fig_shap_trend.update_layout(title="SHAP: Number of Drifted Features Per Week",
                                 xaxis_title="Week", yaxis_title="Drifted Features Count",
                                 height=380, hovermode="x unified")
    st.plotly_chart(fig_shap_trend, use_container_width=True)

    fsd = shap_status.get('feature_shap_drift', {})
    fm = week_record.get('feature_metrics', {})
    ri = [fm.get(f, {}).get('ref_importance', 0.0) for f in flist]
    ci = [fm.get(f, {}).get('curr_importance', 0.0) for f in flist]
    sh = [fm.get(f, {}).get('shap_shift', 0.0) for f in flist]

    if not (all(v == 0.0 for v in ri) and all(v == 0.0 for v in ci)):
        feat_map = ctx['feat_map']
        df_s = pd.DataFrame({
            'Feature': [_disp(f, feat_map) for f in flist],
            'Baseline': ri, 'Current': ci, 'Shift': sh,
            'p-value': [fsd.get(f, {}).get('p_value', 1.0) for f in flist],
        })
        df_s['Drifted (live)'] = df_s['p-value'] < alpha

        st.markdown("##### Per-Feature SHAP Importance Drift")
        st.dataframe(
            df_s.style.map(
                lambda v: 'background-color: #EF444433' if v else '', subset=['Drifted (live)']
            ).format({
                'Baseline': '{:.4f}', 'Current': '{:.4f}', 'Shift': '{:.4f}', 'p-value': '{:.4f}',
            }),
            use_container_width=True, hide_index=True,
        )

        ca, cb = st.columns(2)
        with ca:
            fig_sb = go.Figure()
            fig_sb.add_trace(go.Bar(y=df_s['Feature'], x=df_s['Baseline'], name='Baseline',
                                    orientation='h', marker_color='#3B82F6'))
            fig_sb.add_trace(go.Bar(y=df_s['Feature'], x=df_s['Current'], name=f'Week {selected_week}',
                                    orientation='h', marker_color='#EF4444'))
            fig_sb.update_layout(barmode='group', height=400, yaxis=dict(autorange="reversed"),
                                 title="Baseline vs Current SHAP Importance")
            st.plotly_chart(fig_sb, use_container_width=True)
        with cb:
            fig_sd = px.bar(df_s, y='Feature', x='Shift', orientation='h', color='Shift',
                            color_continuous_scale=px.colors.diverging.RdYlGn,
                            title="SHAP Importance Delta")
            fig_sd.update_layout(height=400, yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_sd, use_container_width=True)
    else:
        st.warning("SHAP values are all zero — TreeExplainer may have failed. Check pipeline logs.")

    frac_thresh = stored.get('min_feature_fraction', 0.6)
    df_diag = _weekly_feature_gate_table(
        ctx, 'shap',
        get_feature_value=lambda r, f: r.get('method_status', {}).get('shap', {})
            .get('feature_shap_drift', {}).get(f, {}).get('ks_stat', 0.0),
        is_feature_drift=lambda r, f: r.get('method_status', {}).get('shap', {})
            .get('feature_shap_drift', {}).get(f, {}).get('p_value', 1.0) < alpha,
        metric_label='SHAP KS Stat', frac_threshold=frac_thresh,
    )
    _render_weekly_diagnostic(
        df_diag,
        caption=f"SHAP only retrains once more than {frac_thresh:.0%} of monitored features individually "
                f"show SHAP-value distribution drift (p < {alpha}).",
    )

    _retrain_history_table('shap', retrain_triggers, ctx['feat_map'])


def page_clustering():
    ctx = _bootstrap(need_features=False)
    _method_header('clustering', ctx)
    inference, weeks = ctx['inference'], ctx['weeks']
    week_record, selected_week = ctx['week_record'], ctx['selected_week']
    retrain_triggers = ctx['retrain_triggers']

    with st.expander("⚙️ Adjust Thresholds (recomputed live from stored distance/PSI values)", expanded=True):
        stored = _method_thresholds(ctx['json_data'], 'clustering')
        c1, c2 = st.columns(2)
        dist_t = c1.number_input("Distance Ratio Threshold", min_value=1.0, max_value=5.0,
                                  value=float(stored.get('distance_ratio', 1.5)), step=0.1, key="clust_dist")
        psi_t = c2.number_input("Cluster PSI Threshold", min_value=0.02, max_value=1.0,
                                 value=float(stored.get('cluster_psi', 0.20)), step=0.01, key="clust_psi")
        st.caption(f"Pipeline defaults: distance_ratio={stored.get('distance_ratio', 1.5)}, cluster_psi={stored.get('cluster_psi', 0.20)}")

    curr_clust = week_record.get('method_status', {}).get('clustering', {})
    live_drift, live_warn = recompute_clustering(curr_clust, dist_t, psi_t)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Distance Ratio", f"{curr_clust.get('distance_ratio', 1.0):.3f}")
    c2.metric("Cluster PSI", f"{curr_clust.get('cluster_psi', 0.0):.4f}")
    c3.metric("As Run", "🚨 Drift" if curr_clust.get('drift_detected', False) else "✅ OK")
    c4.metric("Live", "🚨 Drift" if live_drift else ("⚠️ Warning" if live_warn else "✅ OK"))

    dist_ratios = [_safe_get(r, 'method_status', 'clustering', 'distance_ratio', default=1.0) for r in inference]
    clust_psis = [_safe_get(r, 'method_status', 'clustering', 'cluster_psi') for r in inference]

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        fig_dr = go.Figure()
        fig_dr.add_trace(go.Scatter(x=weeks, y=dist_ratios, mode='lines+markers', name='Distance Ratio',
                                     line=dict(color='#EC4899', width=2)))
        fig_dr.add_hline(y=dist_t, line_dash="dash", line_color="#EF4444", annotation_text=f"Drift ({dist_t})")
        fig_dr.add_hline(y=1.0, line_dash="dot", line_color="#F59E0B", annotation_text="Baseline (1.0)")
        _add_markers(fig_dr, 'clustering', retrain_triggers, selected_week)
        fig_dr.update_layout(title="Centroid Distance Ratio Over Time", xaxis_title="Week",
                             yaxis_title="Distance Ratio", height=380, hovermode="x unified")
        st.plotly_chart(fig_dr, use_container_width=True)

    with col_c2:
        fig_cp = go.Figure()
        fig_cp.add_trace(go.Scatter(x=weeks, y=clust_psis, mode='lines+markers', name='Cluster PSI',
                                     line=dict(color='#F472B6', width=2)))
        fig_cp.add_hline(y=psi_t, line_dash="dash", line_color="#EF4444", annotation_text=f"Drift ({psi_t})")
        fig_cp.update_layout(title="Cluster Distribution PSI Over Time", xaxis_title="Week",
                             yaxis_title="Cluster PSI", height=380, hovermode="x unified")
        st.plotly_chart(fig_cp, use_container_width=True)

    pca = week_record.get('pca_cluster_data', {})
    rx, ry = pca.get('ref_pca_x', []), pca.get('ref_pca_y', [])
    cx, cy = pca.get('curr_pca_x', []), pca.get('curr_pca_y', [])
    if rx and cx:
        fig_pca = go.Figure()
        fig_pca.add_trace(go.Scatter(x=rx, y=ry, mode='markers', name='Reference',
                                      marker=dict(size=5, color='#3B82F6', opacity=0.4)))
        fig_pca.add_trace(go.Scatter(x=cx, y=cy, mode='markers', name=f'Week {selected_week}',
                                      marker=dict(size=5, color='#EF4444', opacity=0.6)))
        if pca.get('centroids_x') and pca.get('centroids_y'):
            fig_pca.add_trace(go.Scatter(x=pca['centroids_x'], y=pca['centroids_y'], mode='markers',
                                          name='Centroids',
                                          marker=dict(size=14, color='#F59E0B', symbol='diamond')))
        fig_pca.update_layout(title="2D PCA Cluster Projection", xaxis_title="PC1", yaxis_title="PC2", height=450)
        st.plotly_chart(fig_pca, use_container_width=True)

    def _clustering_row(r):
        ms = r.get('method_status', {}).get('clustering', {})
        as_run = ms.get('drift_detected', False)
        d, w = recompute_clustering(ms, dist_t, psi_t)
        dr = ms.get('distance_ratio', 1.0)
        cp = ms.get('cluster_psi', 0.0)
        reasons = []
        if dr >= dist_t:
            reasons.append(f"distance_ratio={dr:.2f} ≥ {dist_t}")
        if cp >= psi_t:
            reasons.append(f"cluster_psi={cp:.4f} ≥ {psi_t}")
        if d:
            expl = " and ".join(reasons) + " → drift."
        else:
            expl = f"distance_ratio={dr:.2f} (< {dist_t}) and cluster_psi={cp:.4f} (< {psi_t}) — stable."
        return {
            'Distance Ratio': round(dr, 3),
            'Cluster PSI': round(cp, 4),
            'Drift (As Run)': '🚨' if as_run else '✅',
            'Drift (Live)': '🚨' if d else ('⚠️' if w else '✅'),
            'Explanation': expl,
        }

    df_diag = _weekly_metric_table(ctx, 'clustering', _clustering_row)
    _render_weekly_diagnostic(
        df_diag,
        caption="Clustering retrains whenever distance_ratio OR cluster_psi crosses its threshold — "
                "there is no feature-count gate for this method.",
    )

    _retrain_history_table('clustering', retrain_triggers, ctx['feat_map'])


def page_autoencoder():
    ctx = _bootstrap(need_features=False)
    _method_header('autoencoder', ctx)
    inference, weeks = ctx['inference'], ctx['weeks']
    week_record, selected_week = ctx['week_record'], ctx['selected_week']
    retrain_triggers = ctx['retrain_triggers']

    with st.expander("⚙️ Adjust Threshold (recomputed live from stored RMSE z-scores)", expanded=True):
        stored = _method_thresholds(ctx['json_data'], 'autoencoder')
        z_t = st.number_input("Z-Score Threshold", min_value=0.5, max_value=10.0,
                               value=float(stored.get('z_score', 3.0)), step=0.1, key="ae_z")
        st.caption(f"Pipeline default: z_score={stored.get('z_score', 3.0)}. Features are standardized "
                   f"before the autoencoder sees them, and the decision is z-score only — a KS-test "
                   f"p-value on reconstruction error was dropped because with large sample sizes it "
                   f"flags 'significant' for almost any nonzero difference, firing on nearly every batch.")

    curr_ae = week_record.get('method_status', {}).get('autoencoder', {})
    live_drift, live_warn = recompute_autoencoder(curr_ae, z_t)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ref RMSE Mean", f"{curr_ae.get('ref_rmse_mean', 0):.4f}")
    c2.metric("Current RMSE Mean", f"{curr_ae.get('curr_rmse_mean', 0):.4f}")
    c3.metric("As Run", "🚨 Drift" if curr_ae.get('drift_detected', False) else "✅ OK")
    c4.metric("Live", "🚨 Drift" if live_drift else ("⚠️ Warning" if live_warn else "✅ OK"))

    z_scores = [_safe_get(r, 'method_status', 'autoencoder', 'mse_z_score') for r in inference]
    rmse_means = [_safe_get(r, 'method_status', 'autoencoder', 'curr_rmse_mean') for r in inference]

    col_ae1, col_ae2 = st.columns(2)
    with col_ae1:
        fig_z = go.Figure()
        fig_z.add_trace(go.Scatter(x=weeks, y=z_scores, mode='lines+markers', name='RMSE Z-Score',
                                    line=dict(color='#EF4444', width=2)))
        fig_z.add_hline(y=z_t, line_dash="dash", line_color="red", annotation_text=f"Drift (z={z_t})")
        _add_markers(fig_z, 'autoencoder', retrain_triggers, selected_week)
        fig_z.update_layout(title="Autoencoder RMSE Z-Score Timeline", xaxis_title="Week",
                            yaxis_title="Z-Score", height=400, hovermode="x unified")
        st.plotly_chart(fig_z, use_container_width=True)

    with col_ae2:
        fig_mse = go.Figure()
        fig_mse.add_trace(go.Scatter(x=weeks, y=rmse_means, mode='lines+markers', name='Current RMSE Mean',
                                      line=dict(color='#F97316', width=2)))
        ref_rmse = curr_ae.get('ref_rmse_mean', 0)
        if ref_rmse > 0:
            fig_mse.add_hline(y=ref_rmse, line_dash="dash", line_color="#3B82F6",
                              annotation_text=f"Ref RMSE ({ref_rmse:.4f})")
        fig_mse.update_layout(title="Reconstruction Error (RMSE Mean) Over Time", xaxis_title="Week",
                              yaxis_title="RMSE Mean", height=400, hovermode="x unified")
        st.plotly_chart(fig_mse, use_container_width=True)

    def _ae_row(r):
        ms = r.get('method_status', {}).get('autoencoder', {})
        as_run = ms.get('drift_detected', False)
        d, w = recompute_autoencoder(ms, z_t)
        z = ms.get('mse_z_score', 0.0)
        if d:
            expl = f"RMSE z-score={z:.2f} exceeded threshold {z_t} → reconstruction error spiked."
        elif w:
            expl = f"RMSE z-score={z:.2f} in warning band (> {z_t * 0.67:.2f}, below {z_t})."
        else:
            expl = f"RMSE z-score={z:.2f} below threshold {z_t} — stable."
        return {
            'RMSE Z-Score': round(z, 2),
            'Z-Threshold (live)': z_t,
            'Drift (As Run)': '🚨' if as_run else '✅',
            'Drift (Live)': '🚨' if d else ('⚠️' if w else '✅'),
            'Explanation': expl,
        }

    df_diag = _weekly_metric_table(ctx, 'autoencoder', _ae_row)
    _render_weekly_diagnostic(
        df_diag,
        caption="Autoencoder retrains whenever the reconstruction-error RMSE z-score crosses its "
                "threshold — there is no feature-count gate for this method.",
    )

    _retrain_history_table('autoencoder', retrain_triggers, ctx['feat_map'])


def page_cvc():
    ctx = _bootstrap(need_features=False)
    _method_header('champion_vs_challenger', ctx)
    inference, weeks = ctx['inference'], ctx['weeks']
    week_record, selected_week = ctx['week_record'], ctx['selected_week']
    retrain_triggers = ctx['retrain_triggers']

    with st.expander("⚙️ Adjust Thresholds (recomputed live from stored AUC gap/degradation)", expanded=True):
        stored = _method_thresholds(ctx['json_data'], 'champion_vs_challenger')
        c1, c2 = st.columns(2)
        deg_t = c1.number_input("AUC Degradation Threshold", min_value=0.005, max_value=0.5,
                                 value=float(stored.get('auc_degradation', 0.05)), step=0.005, key="cvc_deg")
        gap_t = c2.number_input("AUC Gap Threshold", min_value=0.005, max_value=0.5,
                                 value=float(stored.get('auc_gap', 0.03)), step=0.005, key="cvc_gap")
        st.caption(f"Pipeline defaults: auc_degradation={stored.get('auc_degradation', 0.05)}, auc_gap={stored.get('auc_gap', 0.03)}")

    curr_cvc = week_record.get('method_status', {}).get('champion_vs_challenger', {})
    live_drift, live_warn = recompute_cvc(curr_cvc, deg_t, gap_t)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Champion AUC", f"{curr_cvc.get('champion_auc', 0):.4f}")
    c2.metric("Challenger AUC", f"{curr_cvc.get('challenger_auc', 0):.4f}")
    c3.metric("As Run", "🚨 Drift" if curr_cvc.get('drift_detected', False) else "✅ OK")
    c4.metric("Live", "🚨 Drift" if live_drift else ("⚠️ Warning" if live_warn else "✅ OK"))

    champ_aucs = [_safe_get(r, 'method_status', 'champion_vs_challenger', 'champion_auc') for r in inference]
    chall_aucs = [_safe_get(r, 'method_status', 'champion_vs_challenger', 'challenger_auc') for r in inference]
    model_aucs = [_safe_get(r, 'method_retrain_info', 'champion_vs_challenger', 'model_auc') for r in inference]

    fig_cvc = go.Figure()
    fig_cvc.add_trace(go.Scatter(x=weeks, y=champ_aucs, mode='lines+markers', name='Champion AUC',
                                  line=dict(color='#10B981', width=2)))
    fig_cvc.add_trace(go.Scatter(x=weeks, y=chall_aucs, mode='lines+markers', name='Challenger AUC',
                                  line=dict(color='#EF4444', width=2)))
    fig_cvc.add_trace(go.Scatter(x=weeks, y=model_aucs, mode='lines', name='Active Model AUC',
                                  line=dict(color='#F59E0B', width=1.5, dash='dot')))
    _add_markers(fig_cvc, 'champion_vs_challenger', retrain_triggers, selected_week)
    fig_cvc.update_layout(title="Champion vs Challenger AUC Over Time", xaxis_title="Week",
                          yaxis_title="AUC", height=400, hovermode="x unified")
    st.plotly_chart(fig_cvc, use_container_width=True)

    champ_f1s = [_safe_get(r, 'method_status', 'champion_vs_challenger', 'champion_f1') for r in inference]
    chall_f1s = [_safe_get(r, 'method_status', 'champion_vs_challenger', 'challenger_f1') for r in inference]
    fig_f1_cvc = go.Figure()
    fig_f1_cvc.add_trace(go.Scatter(x=weeks, y=champ_f1s, mode='lines+markers', name='Champion F1',
                                     line=dict(color='#10B981', width=2)))
    fig_f1_cvc.add_trace(go.Scatter(x=weeks, y=chall_f1s, mode='lines+markers', name='Challenger F1',
                                     line=dict(color='#EF4444', width=2)))
    fig_f1_cvc.update_layout(title="Champion vs Challenger F1 Over Time", xaxis_title="Week",
                             yaxis_title="F1 Score", height=350, hovermode="x unified")
    st.plotly_chart(fig_f1_cvc, use_container_width=True)

    pred_psi = [_safe_get(r, 'method_status', 'champion_vs_challenger', 'prediction_psi') for r in inference]
    if any(v > 0 for v in pred_psi):
        fig_ppsi = go.Figure()
        fig_ppsi.add_trace(go.Scatter(x=weeks, y=pred_psi, mode='lines+markers', name='Prediction PSI',
                                       line=dict(color='#8B5CF6', width=2)))
        fig_ppsi.update_layout(title="Prediction Distribution PSI (Champion vs Challenger)",
                               xaxis_title="Week", yaxis_title="PSI", height=300)
        st.plotly_chart(fig_ppsi, use_container_width=True)

    def _cvc_row(r):
        ms = r.get('method_status', {}).get('champion_vs_challenger', {})
        as_run = ms.get('drift_detected', False)
        d, w = recompute_cvc(ms, deg_t, gap_t)
        deg = ms.get('auc_degradation', 0.0) or 0.0
        gap = ms.get('auc_gap', 0.0)
        reasons = []
        if deg > deg_t:
            reasons.append(f"degradation={deg:.4f} > {deg_t}")
        if gap > gap_t:
            reasons.append(f"gap={gap:.4f} > {gap_t}")
        if d:
            expl = " and ".join(reasons) + " → challenger notably better / champion has degraded."
        else:
            expl = f"gap={gap:.4f} (≤ {gap_t}) and degradation={deg:.4f} (≤ {deg_t}) — champion still adequate."
        return {
            'AUC Gap': round(gap, 4),
            'AUC Degradation': round(deg, 4),
            'Drift (As Run)': '🚨' if as_run else '✅',
            'Drift (Live)': '🚨' if d else ('⚠️' if w else '✅'),
            'Explanation': expl,
        }

    df_diag = _weekly_metric_table(ctx, 'champion_vs_challenger', _cvc_row)
    _render_weekly_diagnostic(
        df_diag,
        caption="Champion vs Challenger retrains whenever the challenger notably outperforms the "
                "champion, or the champion has degraded from its own baseline — there is no "
                "feature-count gate for this method.",
    )

    _retrain_history_table('champion_vs_challenger', retrain_triggers, ctx['feat_map'])


# ══════════════════════════════════════════════
# PAGE: Retrain Events
# ══════════════════════════════════════════════
def retrain_events_page():
    ctx = _bootstrap(need_features=False)
    json_data = ctx['json_data']
    retrain_triggers = ctx['retrain_triggers']
    week_record, selected_week = ctx['week_record'], ctx['selected_week']
    model_info_per_method = json_data.get('model_info_per_method', {})

    st.title("\U0001f504 Per-Method Model Retraining Triggers")
    st.caption("Each method independently retrains its champion model on cumulative data when it detects drift.")

    if not retrain_triggers:
        st.info("No retraining triggers recorded during this analysis period.")
        return

    feat_map = ctx['feat_map']
    df_triggers = pd.DataFrame(retrain_triggers)
    df_triggers['reason'] = df_triggers.apply(lambda r: _prettify_text(_reason_for_trigger(r), feat_map), axis=1)
    df_triggers['method_label'] = df_triggers['method'].map(lambda m: METHOD_LABELS.get(m, m))

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Retraining Events", len(df_triggers))
    c2.metric("Methods That Retrained", df_triggers['method'].nunique())
    c3.metric("Max Cumulative Rows", f"{df_triggers['cumulative_rows'].max():,}")

    st.markdown("### Retraining Timeline")
    fig_timeline = px.scatter(
        df_triggers, x='week', y='method_label', size='cumulative_rows', color='method_label',
        hover_data=['retrain_count', 'new_auc', 'new_f1', 'cumulative_rows', 'reason'],
        title="Retraining Events Over Weekly Windows",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={'method_label': 'Method'},
    )
    fig_timeline.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
    fig_timeline.update_layout(height=400)
    st.plotly_chart(fig_timeline, use_container_width=True)

    col_rc1, col_rc2 = st.columns(2)
    with col_rc1:
        st.markdown("### Retraining Count by Method")
        method_counts = df_triggers['method_label'].value_counts().reset_index()
        method_counts.columns = ['Method', 'Count']
        fig_counts = px.bar(method_counts, x='Method', y='Count', color='Method', text='Count',
                            color_discrete_sequence=px.colors.qualitative.Set2)
        fig_counts.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_counts, use_container_width=True)

    with col_rc2:
        st.markdown("### Post-Retrain Performance")
        fig_perf = make_subplots(rows=1, cols=2, subplot_titles=["AUC", "F1"])
        fig_perf.add_trace(go.Bar(x=df_triggers['method_label'], y=df_triggers['new_auc'],
                                   name='AUC', marker_color='#10B981'), row=1, col=1)
        fig_perf.add_trace(go.Bar(x=df_triggers['method_label'], y=df_triggers['new_f1'],
                                   name='F1', marker_color='#3B82F6'), row=1, col=2)
        fig_perf.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_perf, use_container_width=True)

    st.markdown("### Current Week Model Status")
    mri = week_record.get('method_retrain_info', {})
    if mri:
        status_rows = []
        for mn, info in mri.items():
            triggered = any(t['method'] == mn and t['week'] == selected_week for t in retrain_triggers)
            status_rows.append({
                'Method': METHOD_LABELS.get(mn, mn),
                'Retrain Count': info.get('retrain_count', 0),
                'Model AUC': f"{info.get('model_auc', 0):.4f}",
                'Model F1': f"{info.get('model_f1', 0):.4f}",
                'Triggered This Week': '✅' if triggered else '❌',
            })
        st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)

    st.markdown("### All Retraining Events (with Reasons)")
    st.dataframe(
        df_triggers[['method_label', 'week', 'retrain_count', 'cumulative_rows', 'new_auc', 'new_f1', 'reason']]
        .rename(columns={
            'method_label': 'Method', 'week': 'Week', 'retrain_count': 'Retrain #',
            'cumulative_rows': 'Cumulative Rows', 'new_auc': 'New AUC', 'new_f1': 'New F1', 'reason': 'Reason',
        }),
        use_container_width=True, hide_index=True,
    )

    if model_info_per_method:
        st.markdown("### Per-Method Model Metadata & Thresholds")
        method_thresholds = json_data.get('method_thresholds', {})
        meta_rows = [{
            'Method': METHOD_LABELS.get(mn, mn),
            'Baseline AUC': f"{meta.get('baseline_auc', 0):.4f}",
            'Baseline F1': f"{meta.get('baseline_f1', 0):.4f}",
            'Retrain Count': meta.get('retrain_count', 0),
            'Baseline Rows': meta.get('baseline_rows', 0),
            'Thresholds (as run)': ", ".join(f"{k}={v}" for k, v in method_thresholds.get(mn, DEFAULT_THRESHOLDS.get(mn, {})).items()),
        } for mn, meta in model_info_per_method.items()]
        st.dataframe(pd.DataFrame(meta_rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# Navigation
# ══════════════════════════════════════════════
def main():
    json_data, _ = load_report_data()
    if json_data is None:
        st.title("\U0001f6e1️ Fraud Model & Feature Drift Telemetry")
        st.error("⚠️ Report not found! Run `python run_drift_analysis.py` first "
                 "(use `--method <name>` to run a single method, or omit it to run all 10).")
        st.stop()

    pages = [st.Page(overview_page, title="Overview", icon="📊", default=True)]
    page_fns = {
        'ks_stats': page_ks, 'psi': page_psi, 'kl_divergence': page_kl,
        'ddm': page_ddm, 'eddm': page_eddm, 'adwin': page_adwin,
        'shap': page_shap, 'clustering': page_clustering,
        'autoencoder': page_autoencoder, 'champion_vs_challenger': page_cvc,
    }
    # Detectors without a bespoke page (hddm, prequential_auc) still appear in
    # the Overview heatmaps, the retrain timeline and the policy comparison.
    for m in METHODS_WITH_PAGES:
        pages.append(st.Page(page_fns[m], title=METHOD_LABELS[m], icon=METHOD_ICONS[m]))
    pages.append(st.Page(retrain_events_page, title="Retrain Events", icon="🔁"))

    nav = st.navigation({"Monitoring": pages})
    nav.run()


if __name__ == "__main__":
    main()
