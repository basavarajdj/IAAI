"""
Generate all RESEARCH_PRESENTATION.md charts from the pipeline's own CSV/JSON
output (``reports/*``), so every figure in the presentation traces back to a
real run rather than to a one-off, unsaved plotting session.

Usage
-----
    python report/generate_visuals.py

Reads from ``reports/`` (relative to the repo root), writes PNGs to
``report/visuals/``. Run this after ``run_drift_analysis.py`` and
``run_rl_experiment.py`` have both completed — it is step 3 of the full
"rerun everything" sequence documented at the top of RESEARCH_PRESENTATION.md.

Feature names are rendered through ``feature_engineering.feature_label`` so a
chart never shows a raw column like ``_var_TransactionAmt__P_emaildomain__ProductCD``
— readability for a non-implementer audience is the reason this indirection
exists at all (see feature_engineering.py's own comment on FEATURE_LABELS).
"""
import os
import sys
import json
import ast

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering import feature_label

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(REPO_ROOT, 'reports')
VISUALS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visuals')
os.makedirs(VISUALS_DIR, exist_ok=True)

# ── Palette (validated categorical/sequential/status set) ──────────────────
BLUE = '#2a78d6'
ORANGE = '#eb6834'
AQUA = '#1baf7a'
YELLOW = '#eda100'
MAGENTA = '#e87ba4'
GREEN = '#008300'
VIOLET = '#4a3aa7'
RED = '#e34948'
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

SURFACE = '#fcfcfb'
INK_PRIMARY = '#0b0b0b'
INK_SECONDARY = '#52514e'
INK_MUTED = '#898781'
GRIDLINE = '#e1e0d9'
BASELINE = '#c3c2b7'
STATUS_GOOD = '#0ca30c'
STATUS_CRITICAL = '#d03b3b'

# Sequential single-hue ramp (blue, light->dark) for magnitude/similarity heatmaps.
SEQUENTIAL_BLUE = matplotlib.colors.LinearSegmentedColormap.from_list(
    'sequential_blue', ['#fcfcfb', '#cde2fb', '#6da7ec', '#256abf', '#0d366b'])

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Segoe UI', 'Arial', 'DejaVu Sans'],
    'axes.edgecolor': BASELINE,
    'axes.labelcolor': INK_SECONDARY,
    'text.color': INK_PRIMARY,
    'xtick.color': INK_MUTED,
    'ytick.color': INK_MUTED,
    'axes.facecolor': SURFACE,
    'figure.facecolor': SURFACE,
    'savefig.facecolor': SURFACE,
    'grid.color': GRIDLINE,
    'figure.dpi': 150,
})


def style_axes(ax, hide_spines=('top', 'right')):
    for s in hide_spines:
        ax.spines[s].set_visible(False)
    for s in ax.spines:
        if s not in hide_spines:
            ax.spines[s].set_color(BASELINE)
    ax.tick_params(colors=INK_MUTED)


def savefig(fig, name):
    path = os.path.join(VISUALS_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  wrote {path}')


def rl(name):
    return pd.read_csv(os.path.join(REPORTS_DIR, name))


_UNIFIED = None


def unified():
    """Lazily load unified_drift_report.json (weekly_records, weeks 1-14)."""
    global _UNIFIED
    if _UNIFIED is None:
        with open(os.path.join(REPORTS_DIR, 'unified_drift_report.json')) as f:
            _UNIFIED = json.load(f)
    return _UNIFIED


def weekly_records():
    return [r for r in unified()['weekly_records'] if not r.get('is_baseline')]


# ── 01: Feature funnel (structural — same raw dataset every run) ───────────
def chart_01_feature_funnel():
    stages = ['Raw columns', 'Trained features', 'Monitored features']
    counts = [432, 113, 20]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    widths = [c / counts[0] for c in counts]
    y = np.arange(len(stages))[::-1]
    min_label_w = 0.30  # bars narrower than this get an outside label instead
    for i, (yi, w, c, stage) in enumerate(zip(y, widths, counts, stages)):
        left = (1 - w) / 2
        ax.barh(yi, w, left=left, height=0.55, color=CATEGORICAL[i], zorder=3)
        label = f'{stage}: {c}'
        if w >= min_label_w:
            ax.text(0.5, yi, label, ha='center', va='center',
                    color='white', fontsize=11, fontweight='bold', zorder=4)
        else:
            ax.text(left + w + 0.02, yi, label, ha='left', va='center',
                    color=INK_PRIMARY, fontsize=11, fontweight='bold', zorder=4)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(stages) - 0.4)
    ax.axis('off')
    ax.set_title('Feature funnel: 432 raw columns → 113 trained → 20 monitored',
                  fontsize=12, color=INK_PRIMARY, pad=14)
    savefig(fig, '01_feature_funnel.png')


# ── 02: Raw vs final feature counts by category ─────────────────────────────
def chart_02_feature_category_breakdown():
    rows = [
        ('Vesta (V1–V339)', 339, 4),
        ('Identity / device', 40, 38),
        ('Counters (C1–C14)', 14, 16),
        ('Timedeltas (D1–D15)', 15, 16),
        ('Match flags (M1–M9)', 9, 9),
        ('Everything else', 15, 31),
    ]
    labels = [r[0] for r in rows]
    raw = [r[1] for r in rows]
    final = [r[2] for r in rows]
    y = np.arange(len(labels))
    h = 0.34
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(y + h / 2 + 0.02, raw, height=h, color=INK_MUTED, label='Raw columns')
    ax.barh(y - h / 2 - 0.02, final, height=h, color=BLUE, label='Final features')
    for yi, v in zip(y + h / 2 + 0.02, raw):
        ax.text(v + 4, yi, str(v), va='center', fontsize=9, color=INK_SECONDARY)
    for yi, v in zip(y - h / 2 - 0.02, final):
        ax.text(v + 4, yi, str(v), va='center', fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel('Column count')
    ax.set_title('Raw vs. final feature counts by category', fontsize=12, pad=12)
    ax.legend(frameon=False, loc='lower right')
    ax.grid(axis='x', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax, hide_spines=('top', 'right', 'left'))
    savefig(fig, '02_feature_category_breakdown.png')


# ── 03: Confirmed alarms per detector ───────────────────────────────────────
def chart_03_confirmed_alarms_per_detector():
    df = rl('method_week_matrix.csv')
    counts = df[df['confirmed_flag'] == True].groupby('method').size()
    order = ['ks_stats', 'psi', 'kl_divergence', 'ddm', 'eddm', 'hddm', 'adwin',
             'shap', 'clustering', 'autoencoder', 'prequential_auc', 'champion_vs_challenger']
    label_map = {
        'ks_stats': 'KS test', 'psi': 'PSI', 'kl_divergence': 'Jensen-Shannon',
        'ddm': 'DDM', 'eddm': 'EDDM', 'hddm': 'HDDM', 'adwin': 'ADWIN',
        'shap': 'SHAP', 'clustering': 'Clustering', 'autoencoder': 'Autoencoder',
        'prequential_auc': 'Prequential AUC', 'champion_vs_challenger': 'Champion vs Challenger',
    }
    vals = [int(counts.get(m, 0)) for m in order]
    labels = [label_map[m] for m in order]
    colors = [RED if v > 0 else BASELINE for v in vals]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(order))[::-1]
    ax.barh(y, vals, color=colors, height=0.6, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.15, yi, str(v), va='center', fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Confirmed alarms (out of 14 weeks)')
    ax.set_title('Confirmed alarms per detector, out of 14 weeks', fontsize=12, pad=12)
    ax.grid(axis='x', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax, hide_spines=('top', 'right', 'left'))
    savefig(fig, '03_confirmed_alarms_per_detector.png')


# ── 04: Weekly trend of the four feature-vote detectors ────────────────────
# KS / Jensen-Shannon / SHAP vote at a 0.30 consensus (6 of 20 features);
# PSI keeps the stricter 0.60 (see run_drift_analysis.METHOD_THRESHOLDS —
# per-method min_feature_fraction). Two different bars, so both are drawn.
FEATURE_VOTE_CONSENSUS = {'ks_stats': 0.30, 'psi': 0.60, 'kl_divergence': 0.30, 'shap': 0.30}


def chart_04_feature_vote_weekly_trend():
    df = rl('method_week_matrix.csv')
    methods = ['ks_stats', 'psi', 'kl_divergence', 'shap']
    label_map = {'ks_stats': 'KS test', 'psi': 'PSI', 'kl_divergence': 'Jensen-Shannon', 'shap': 'SHAP'}
    colors = {'ks_stats': BLUE, 'psi': ORANGE, 'kl_divergence': AQUA, 'shap': VIOLET}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for m in methods:
        sub = df[df['method'] == m].sort_values('week')
        ax.plot(sub['week'], sub['features_crossed_fraction'], marker='o',
                markersize=4, linewidth=2, color=colors[m], label=label_map[m])
    ax.axhline(0.30, color=INK_MUTED, linewidth=1.5, linestyle='--')
    ax.text(0.3, 0.315, 'KS / Jensen-Shannon / SHAP consensus (0.30 — 6 of 20 features)',
            fontsize=8, color=INK_MUTED)
    ax.axhline(0.60, color=INK_MUTED, linewidth=1.5, linestyle=':')
    ax.text(0.3, 0.615, 'PSI consensus (0.60 — 12 of 20 features)', fontsize=8, color=INK_MUTED)
    ax.set_xlabel('Week')
    ax.set_ylabel('Fraction of monitored features crossed')
    ax.set_ylim(0, 0.7)
    ax.set_title('Weekly trend of the four feature-vote detectors vs. their consensus thresholds',
                  fontsize=12, pad=12)
    ax.legend(frameon=False, ncol=4, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    ax.grid(axis='y', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    savefig(fig, '04_feature_vote_weekly_trend.png')


# ── 05: Heatmap of every detector's raw/confirmed state, by week ───────────
def chart_05_method_week_heatmap():
    df = rl('method_week_matrix.csv')
    order = ['ks_stats', 'psi', 'kl_divergence', 'shap', 'clustering', 'autoencoder',
             'ddm', 'eddm', 'hddm', 'adwin', 'prequential_auc', 'champion_vs_challenger']
    label_map = {
        'ks_stats': 'KS test', 'psi': 'PSI', 'kl_divergence': 'Jensen-Shannon',
        'shap': 'SHAP', 'clustering': 'Clustering', 'autoencoder': 'Autoencoder',
        'ddm': 'DDM', 'eddm': 'EDDM', 'hddm': 'HDDM', 'adwin': 'ADWIN',
        'prequential_auc': 'Prequential AUC', 'champion_vs_challenger': 'Champion vs Challenger',
    }
    weeks = sorted(df['week'].unique())
    grid = np.zeros((len(order), len(weeks)))
    for i, m in enumerate(order):
        sub = df[df['method'] == m].set_index('week')
        for j, w in enumerate(weeks):
            if w in sub.index:
                raw = bool(sub.loc[w, 'raw_flag'])
                conf = bool(sub.loc[w, 'confirmed_flag'])
                grid[i, j] = 2 if conf else (1 if raw else 0)

    cmap = matplotlib.colors.ListedColormap([SURFACE, '#f6c9a0', RED])
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.imshow(grid, aspect='auto', cmap=cmap, vmin=0, vmax=2)
    ax.set_xticks(range(len(weeks)))
    ax.set_xticklabels(weeks)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([label_map[m] for m in order])
    ax.set_xlabel('Week')
    for i in range(len(order) + 1):
        ax.axhline(i - 0.5, color=GRIDLINE, linewidth=0.8)
    for j in range(len(weeks) + 1):
        ax.axvline(j - 0.5, color=GRIDLINE, linewidth=0.8)
    handles = [
        mpatches.Patch(color=SURFACE, ec=GRIDLINE, label='No alarm'),
        mpatches.Patch(color='#f6c9a0', label='Raw alarm'),
        mpatches.Patch(color=RED, label='Confirmed alarm'),
    ]
    ax.legend(handles=handles, frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.08))
    ax.set_title("Every detector's raw and confirmed alarm state, by week", fontsize=12, pad=12)
    style_axes(ax, hide_spines=('top', 'right', 'left', 'bottom'))
    savefig(fig, '05_method_week_heatmap.png')


# ── 06: Top individually-drifting features ──────────────────────────────────
def chart_06_top_drifting_features():
    df = rl('method_week_matrix.csv')
    vote_methods = {'ks_stats', 'psi', 'kl_divergence', 'shap'}
    sub = df[df['method'].isin(vote_methods) & df['drifted_feature_names'].notna()]
    counts = {}
    for names in sub['drifted_feature_names']:
        for n in str(names).split(';'):
            n = n.strip()
            if n:
                counts[n] = counts.get(n, 0) + 1
    top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:12]
    raw_names = [t[0] for t in top][::-1]
    vals = [t[1] for t in top][::-1]
    labels = [feature_label(n) for n in raw_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=BLUE, height=0.6, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.3, yi, str(v), va='center', fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Weeks crossing threshold (out of 14), summed across KS / PSI / Jensen-Shannon / SHAP')
    ax.set_title('Top individually-drifting features across all four feature-vote detectors',
                  fontsize=12, pad=12)
    ax.grid(axis='x', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax, hide_spines=('top', 'right', 'left'))
    savefig(fig, '06_top_drifting_features.png')


# ── 07: RL policy comparison ────────────────────────────────────────────────
# All rows come from rl_policy_comparison.csv only — every policy here acts on
# the same neural classifier and the same precomputed model lattice, so bars
# are directly comparable. (The classical-detector numbers under the LightGBM
# model, in policy_comparison.csv, are a different benchmark — Section 3.4 —
# and must not be mixed into this chart.)
def chart_07_rl_policy_comparison():
    rlp = rl('rl_policy_comparison.csv')
    label_map = {
        'rl_agent_greedy': 'RL agent (greedy)',
        'rl_agent_thompson': 'RL agent (Thompson)',
        'always_partial_update': 'Always partial update',
        'always_retrain': 'Always full retrain',
        'fixed_schedule_every_2w': 'Fixed schedule (every 2 weeks)',
        'fixed_schedule_every_4w': 'Fixed schedule (every 4 weeks)',
        'adwin': 'ADWIN',
        'prequential_auc': 'Prequential AUC',
        'eddm': 'EDDM',
        'champion_vs_challenger': 'Champion vs Challenger',
        'never_retrain': 'Never retrain',
        'ks_stats': 'KS test',
        'kl_divergence': 'Jensen-Shannon',
        'psi': 'PSI',
        'shap': 'SHAP',
    }
    rows = [(label_map.get(r['policy'], r['policy']), r['mean_auc']) for _, r in rlp.iterrows()]
    rows = sorted(rows, key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    colors = [GREEN if 'RL agent' in l else (INK_MUTED if l == 'Never retrain' else BLUE) for l in labels]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(labels))
    ax.bar(x, vals, color=colors, width=0.6, zorder=3)
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.001, f'{v:.4f}', ha='center', fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel('Mean AUC')
    ax.set_ylim(min(vals) - 0.01, max(vals) + 0.008)
    ax.set_title('RL agent vs. classical detectors vs. naive baselines', fontsize=12, pad=12)
    ax.grid(axis='y', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    savefig(fig, '07_rl_policy_comparison.png')


# ── 08: Gain decomposition ──────────────────────────────────────────────────
# "Best classical detector" here means the best classical policy *within the
# RL benchmark* (rl_policy_comparison.csv, neural model) — ADWIN in this run —
# not the LightGBM-benchmark winner from Section 3.4 (a different model, a
# different table, not comparable to the RL numbers).
CLASSICAL_POLICIES = {'adwin', 'prequential_auc', 'eddm', 'champion_vs_challenger',
                      'ks_stats', 'kl_divergence', 'psi', 'shap'}


def chart_08_gain_decomposition():
    rlp = rl('rl_policy_comparison.csv').set_index('policy')['mean_auc']
    best_classical_val = rlp[rlp.index.isin(CLASSICAL_POLICIES)].max()
    always_partial = float(rlp.get('always_partial_update', np.nan))
    rl_agent = float(rlp.get('rl_agent_greedy', np.nan))

    action_space_gain = always_partial - best_classical_val
    learning_gain = rl_agent - always_partial
    total_gain = rl_agent - best_classical_val

    fig, ax = plt.subplots(figsize=(8, 5.5))
    labels = ['Best classical\ndetector', '+ cheap adaptation\naction available', '+ learned\npolicy']
    cum = [best_classical_val, always_partial, rl_agent]
    colors = [INK_MUTED, BLUE, GREEN]
    x = np.arange(3)
    ax.bar(x, cum, color=colors, width=0.55, zorder=3)
    for xi, v in zip(x, cum):
        ax.text(xi, v + 0.0006, f'{v:.4f}', ha='center', fontsize=9, color=INK_SECONDARY)
    ax.set_ylim(min(cum) - 0.006, max(cum) + 0.004)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Mean AUC')
    pct_action = 100 * action_space_gain / total_gain if total_gain else 0
    ax.set_title(f'Gain decomposition: action space vs. learned policy\n'
                 f'(action space ≈ {pct_action:.0f}% of the total gain over best classical)',
                 fontsize=11, pad=12)
    ax.grid(axis='y', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    savefig(fig, '08_gain_decomposition.png')


# ── 09: Ablation ────────────────────────────────────────────────────────────
def chart_09_ablation():
    df = rl('rl_ablation.csv')
    label_map = {'full': 'Full agent\n(signals + context)', 'context_only': 'Context only\n(no drift signals)',
                 'signals_only': 'Signals only\n(no model context)'}
    df['label'] = df['observes'].map(lambda o: label_map.get(o, o))
    colors_map = {'full': GREEN, 'context_only': INK_MUTED, 'signals_only': BLUE}
    fig, ax = plt.subplots(figsize=(8, 5.5))
    x = np.arange(len(df))
    ax.bar(x, df['mean_auc'], color=[colors_map.get(o, BLUE) for o in df['observes']], width=0.55, zorder=3)
    for xi, v in zip(x, df['mean_auc']):
        ax.text(xi, v + 0.0006, f'{v:.4f}', ha='center', fontsize=9, color=INK_SECONDARY)
    ax.set_xticks(x)
    ax.set_xticklabels(df['label'], fontsize=9)
    ax.set_ylabel('Mean AUC')
    ax.set_ylim(df['mean_auc'].min() - 0.006, df['mean_auc'].max() + 0.004)
    ax.set_title('Ablation: full agent vs. context-only vs. signals-only', fontsize=12, pad=12)
    ax.grid(axis='y', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    savefig(fig, '09_ablation.png')


# ── 10: Policy reliance ─────────────────────────────────────────────────────
# RL state/context names (rl_env.CONTEXT_NAMES, drift_signals.SIGNAL_NAMES) are
# a different namespace than the pipeline's engineered features, so they get
# their own readable-name table rather than feature_label().
CONTEXT_SIGNALS = {
    'weeks_since_full_retrain', 'weeks_since_partial_update', 'ensemble_alpha',
    'recent_auc_delta', 'recent_f1', 'progress',
}
RL_SIGNAL_LABELS = {
    'progress': 'Progress through replay (which week)',
    'weeks_since_reference': 'Weeks since reference window',
    'weeks_since_full_retrain': 'Weeks since last full retrain',
    'weeks_since_partial_update': 'Weeks since last partial update',
    'ensemble_alpha': 'Current ensemble weight',
    'recent_auc_delta': 'Recent AUC vs. never-retrain baseline',
    'recent_f1': 'Recent F1',
    'ks_drift_fraction': 'KS test: share of features failing',
    'ks_mean_statistic': 'KS test: mean statistic magnitude',
    'psi_drift_fraction': 'PSI: share of features failing',
    'psi_mean_ratio': 'PSI: mean ratio to bootstrap null',
    'js_mean_distance': 'Jensen-Shannon: mean distance',
    'attribution_drift_fraction': 'SHAP: share of features failing',
    'attribution_mean_shift': 'SHAP: mean attribution shift',
    'cluster_distance_ratio': 'Clustering: distance ratio to reference',
    'cluster_psi': 'Clustering: cluster-membership PSI',
    'autoencoder_z_score': 'Autoencoder: reconstruction-error z-score',
}


def chart_10_policy_reliance():
    df = rl('rl_policy_reliance.csv').sort_values('mean_abs_attribution', ascending=True)
    labels = [RL_SIGNAL_LABELS.get(s, feature_label(s)) for s in df['signal']]
    colors = [INK_MUTED if s in CONTEXT_SIGNALS else BLUE for s in df['signal']]

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(df))
    ax.barh(y, df['mean_abs_attribution'], color=colors, height=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Mean |attribution|')
    ax.set_title('Policy reliance: which inputs drive the agent\'s actions', fontsize=12, pad=12)
    handles = [mpatches.Patch(color=INK_MUTED, label='Model-context feature'),
               mpatches.Patch(color=BLUE, label='Drift-detector signal')]
    ax.legend(handles=handles, frameon=False, loc='lower right')
    ax.grid(axis='x', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax, hide_spines=('top', 'right', 'left'))
    savefig(fig, '10_policy_reliance.png')


# ── 11: PSI weekly trend (actual PSI magnitude, not just pass/fail count) ──
def chart_11_psi_weekly_trend():
    weeks, means, maxes = [], [], []
    for r in weekly_records():
        details = r['method_status']['psi']['details']
        vals = [v['psi'] for v in details.values()]
        weeks.append(r['week'])
        means.append(np.mean(vals))
        maxes.append(np.max(vals))

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(weeks, means, marker='o', markersize=4, linewidth=2, color=BLUE,
            label='Mean PSI across 20 monitored features')
    ax.plot(weeks, maxes, marker='o', markersize=4, linewidth=1.5, color=ORANGE,
            linestyle='--', label='Max PSI (single worst feature)')
    ax.axhline(0.10, color=INK_MUTED, linewidth=1, linestyle=':')
    ax.axhline(0.20, color=INK_MUTED, linewidth=1, linestyle=':')
    ax.text(0.3, 0.105, '"Moderate shift" folklore band (0.10)', fontsize=8, color=INK_MUTED)
    ax.text(0.3, 0.205, '"Major shift" folklore band (0.20)', fontsize=8, color=INK_MUTED)
    ax.set_xlabel('Week')
    ax.set_ylabel('PSI')
    ax.set_title('PSI week-over-week trend, against the scorecard folklore bands', fontsize=12, pad=12)
    ax.legend(frameon=False, loc='upper left')
    ax.grid(axis='y', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    savefig(fig, '11_psi_weekly_trend.png')


# ── 12: SHAP feature-importance shift ───────────────────────────────────────
def chart_12_shap_importance_shift():
    shifts = {}
    for r in weekly_records():
        for feat, v in r['method_status']['shap']['feature_shap_drift'].items():
            shifts.setdefault(feat, []).append(abs(v['importance_shift']))
    mean_shift = {f: np.mean(v) for f, v in shifts.items()}
    top = sorted(mean_shift.items(), key=lambda kv: kv[1], reverse=True)[:10][::-1]
    labels = [feature_label(f) for f, _ in top]
    vals = [v for _, v in top]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=VIOLET, height=0.6, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + max(vals) * 0.01, yi, f'{v:.4f}', va='center', fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Mean |SHAP importance shift| (current-week vs. reference), averaged over 14 weeks')
    ax.set_title('SHAP: which features\' influence on the model shifts the most', fontsize=12, pad=12)
    ax.grid(axis='x', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax, hide_spines=('top', 'right', 'left'))
    savefig(fig, '12_shap_importance_shift.png')


# ── 13: Clustering — distance ratio & cluster PSI vs. threshold ────────────
def chart_13_clustering_trend():
    weeks, dist_frac, psi_frac = [], [], []
    for r in weekly_records():
        c = r['method_status']['clustering']
        weeks.append(r['week'])
        dist_frac.append(c['distance_ratio'] / 1.5)      # distance_threshold=1.5
        psi_frac.append(c['cluster_psi'] / 0.2)           # psi_threshold=0.2

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(weeks, dist_frac, marker='o', markersize=4, linewidth=2, color=BLUE,
            label='Centroid distance ratio / threshold (1.5)')
    ax.plot(weeks, psi_frac, marker='o', markersize=4, linewidth=2, color=ORANGE,
            label='Cluster-assignment PSI / threshold (0.2)')
    ax.axhline(1.0, color=RED, linewidth=1.5, linestyle='--')
    ax.text(0.3, 1.03, 'Drift threshold (either series crossing 1.0)', fontsize=8, color=RED)
    ax.set_xlabel('Week')
    ax.set_ylabel('Fraction of each metric\'s own drift threshold')
    ax.set_title('Clustering (k=5 fixed): both drift signals vs. their thresholds', fontsize=12, pad=12)
    ax.legend(frameon=False, loc='upper left')
    ax.grid(axis='y', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    savefig(fig, '13_clustering_trend.png')


# ── 14: DDM & HDDM — shared error rate vs. each tracker's own boundary ─────
# DDM and HDDM watch the identical error stream (mean_error_rate is the same
# number in both method_status blocks each week), so it is plotted once. Each
# tracker's own end-of-week boundary is shown as context, not as a literal
# per-week trigger test: both are *stateful, per-sample* trackers (a sample-
# by-sample cumulative comparison, continuing across weeks) — the boundary
# line is where that state stood when the week ended, which is why a line
# dipping under the boundary here does not retroactively mean the tracker
# fired; the exact per-week fire count is `drift_occurrences`, reported
# in the method table instead of reconstructed visually. EDDM tracks a
# different quantity (inter-error distance, not an error rate) and is not
# on this axis — see its own table.
def chart_14_error_rate_trend():
    weeks, err, ddm_bound, hddm_bound = [], [], [], []
    for r in weekly_records():
        ms = r['method_status']
        weeks.append(r['week'])
        err.append(ms['ddm']['mean_error_rate'])
        ddm_bound.append(ms['ddm']['drift_threshold'])
        hddm_bound.append(ms['hddm']['reference_mean'] + ms['hddm']['hoeffding_bound'])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(weeks, err, marker='o', markersize=4, linewidth=2.5, color=INK_PRIMARY, label='Observed weekly error rate')
    ax.plot(weeks, ddm_bound, marker='o', markersize=4, linewidth=1.5, color=BLUE,
            linestyle='--', label="DDM's boundary (end of week)")
    ax.plot(weeks, hddm_bound, marker='o', markersize=4, linewidth=1.5, color=AQUA,
            linestyle='--', label="HDDM's boundary (end of week)")
    ax.set_xlabel('Week')
    ax.set_ylabel('Error rate')
    ax.set_title('DDM & HDDM: the shared error-rate stream stays well under both boundaries',
                 fontsize=12, pad=12)
    ax.set_ylim(0.26, 0.42)
    ax.legend(frameon=False, loc='lower right')
    ax.grid(axis='y', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    savefig(fig, '14_error_rate_trend.png')


# ── 15: ADWIN z-score trend ─────────────────────────────────────────────────
def chart_15_adwin_zscore_trend():
    weeks, z, thr = [], [], []
    for r in weekly_records():
        a = r['method_status']['adwin']
        weeks.append(r['week'])
        z.append(a['z_score'])
        thr.append(a['z_threshold'])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(weeks, z, marker='o', markersize=4, linewidth=2, color=BLUE, label='ADWIN z-score')
    ax.plot(weeks, thr, color=RED, linewidth=1.5, linestyle='--', label=f'Drift threshold ({thr[0]:.2f})')
    ax.set_xlabel('Week')
    ax.set_ylabel('z-score')
    ax.set_title('ADWIN: mean-shift z-score vs. its formal threshold', fontsize=12, pad=12)
    ax.legend(frameon=False, loc='upper left')
    ax.grid(axis='y', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    savefig(fig, '15_adwin_zscore_trend.png')


# ── 16: Prequential AUC trend ───────────────────────────────────────────────
def chart_16_prequential_auc_trend():
    weeks, curr, ref = [], [], []
    for r in weekly_records():
        p = r['method_status']['prequential_auc']
        weeks.append(r['week'])
        curr.append(p['current_auc'])
        ref.append(p['reference_auc'])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(weeks, curr, marker='o', markersize=4, linewidth=2, color=BLUE, label='Current-week AUC')
    ax.plot(weeks, ref, marker='o', markersize=4, linewidth=2, color=INK_MUTED,
            linestyle='--', label='Reference-window AUC (resets on retrain)')
    ax.set_xlabel('Week')
    ax.set_ylabel('AUC')
    ax.set_title('Prequential AUC: current performance vs. its own reference', fontsize=12, pad=12)
    ax.legend(frameon=False, loc='lower left')
    ax.grid(axis='y', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    savefig(fig, '16_prequential_auc_trend.png')


# ── 17: Champion vs Challenger — performance comparison ─────────────────────
def chart_17_champion_challenger():
    weeks, champ, chall = [], [], []
    for r in weekly_records():
        c = r['method_status']['champion_vs_challenger']
        weeks.append(r['week'])
        champ.append(c['champion_auc'])
        chall.append(c['challenger_auc'])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(weeks, champ, marker='o', markersize=4, linewidth=2, color=INK_MUTED, label='Champion (deployed model)')
    ax.plot(weeks, chall, marker='o', markersize=4, linewidth=2, color=GREEN, label='Challenger (shadow retrain, out-of-fold)')
    ax.set_xlabel('Week')
    ax.set_ylabel('AUC')
    ax.set_title('Champion vs. Challenger: would retraining actually help this week?', fontsize=12, pad=12)
    ax.legend(frameon=False, loc='lower left')
    ax.grid(axis='y', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    savefig(fig, '17_champion_challenger.png')


# ── 18: Autoencoder reconstruction-error z-score trend ─────────────────────
def chart_18_autoencoder_zscore_trend():
    weeks, z = [], []
    for r in weekly_records():
        z.append(r['method_status']['autoencoder']['mse_z_score'])
        weeks.append(r['week'])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(weeks, z, marker='o', markersize=4, linewidth=2, color=BLUE, label='Reconstruction-error z-score')
    ax.axhline(3.0, color=RED, linewidth=1.5, linestyle='--', label='Drift threshold (3.0)')
    ax.set_xlabel('Week')
    ax.set_ylabel('z-score')
    ax.set_ylim(0, 3.5)
    ax.set_title('Autoencoder: reconstruction-error z-score vs. threshold', fontsize=12, pad=12)
    ax.legend(frameon=False, loc='upper left')
    ax.grid(axis='y', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    savefig(fig, '18_autoencoder_zscore_trend.png')


# ── 19: Clustering — cluster composition, reference vs. every test week ────
def chart_19_cluster_composition():
    weeks_data = weekly_records()
    n_clusters = weeks_data[0]['method_status']['clustering']['n_clusters']
    ref_counts = weeks_data[0]['method_status']['clustering']['ref_cluster_counts']
    ref_n = weeks_data[0]['method_status']['clustering']['ref_n_rows']

    labels = ['Reference\n(training)'] + [f'Week {r["week"]}' for r in weeks_data]
    all_counts = [ref_counts] + [r['method_status']['clustering']['curr_cluster_counts'] for r in weeks_data]
    all_n = [ref_n] + [r['method_status']['clustering']['curr_n_rows'] for r in weeks_data]
    shares = np.array([[c / n for c in counts] for counts, n in zip(all_counts, all_n)])  # (15, k)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))
    cluster_colors = CATEGORICAL[:n_clusters]
    for k in range(n_clusters):
        vals = shares[:, k]
        ax.bar(x, vals, bottom=bottom, color=cluster_colors[k], width=0.65,
               label=f'Cluster {k}', zorder=3,
               edgecolor=SURFACE, linewidth=1)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Share of records')
    ax.set_ylim(0, 1)
    ax.set_title(f'Cluster composition (k={n_clusters}, fit once on {ref_n:,} reference rows): '
                 f'reference vs. every test week', fontsize=12, pad=12)
    ax.legend(frameon=False, ncol=n_clusters, loc='upper center', bbox_to_anchor=(0.5, -0.20))
    ax.axvline(0.5, color=INK_MUTED, linewidth=1, linestyle=':')
    style_axes(ax, hide_spines=('top', 'right', 'left'))
    savefig(fig, '19_cluster_composition.png')


# ── 20: Clustering — which features drive the distance-ratio shift ─────────
def chart_20_clustering_feature_contributions():
    freq, shift_sum = {}, {}
    for r in weekly_records():
        for item in r['method_status']['clustering'].get('top_contributing_features', []):
            f = item['feature']
            freq[f] = freq.get(f, 0) + 1
            shift_sum[f] = shift_sum.get(f, 0.0) + item['shift']

    top = sorted(freq.items(), key=lambda kv: (kv[1], shift_sum[kv[0]]), reverse=True)[:10][::-1]
    labels = [feature_label(f) for f, _ in top]
    vals = [c for _, c in top]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=AQUA, height=0.6, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.15, yi, str(v), va='center', fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Weeks in the top-5 distance-driving features (out of 14)')
    ax.set_title('Clustering: which features drive the centroid-distance shift most often',
                 fontsize=11, pad=12)
    ax.grid(axis='x', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax, hide_spines=('top', 'right', 'left'))
    savefig(fig, '20_clustering_feature_contributions.png')


# ── 21: Autoencoder — which features drive reconstruction error ────────────
def chart_21_autoencoder_feature_contributions():
    freq, shift_sum = {}, {}
    for r in weekly_records():
        for item in r['method_status']['autoencoder'].get('top_contributing_features', []):
            f = item['feature']
            freq[f] = freq.get(f, 0) + 1
            shift_sum[f] = shift_sum.get(f, 0.0) + item['shift']

    top = sorted(freq.items(), key=lambda kv: (kv[1], shift_sum[kv[0]]), reverse=True)[:10][::-1]
    labels = [feature_label(f) for f, _ in top]
    vals = [c for _, c in top]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=MAGENTA, height=0.6, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 0.15, yi, str(v), va='center', fontsize=9, color=INK_SECONDARY)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Weeks in the top-5 reconstruction-error-driving features (out of 14)')
    ax.set_title('Autoencoder: which features drive reconstruction error most often',
                 fontsize=11, pad=12)
    ax.grid(axis='x', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax, hide_spines=('top', 'right', 'left'))
    savefig(fig, '21_autoencoder_feature_contributions.png')


# ── 22: Per-method top drifting features (small multiples) ─────────────────
def chart_22_per_method_top_features():
    df = rl('method_week_matrix.csv')
    methods = ['ks_stats', 'psi', 'kl_divergence', 'shap']
    label_map = {'ks_stats': 'KS test', 'psi': 'PSI', 'kl_divergence': 'Jensen-Shannon', 'shap': 'SHAP'}
    colors = {'ks_stats': BLUE, 'psi': ORANGE, 'kl_divergence': AQUA, 'shap': VIOLET}

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for ax, m in zip(axes.flat, methods):
        sub = df[(df['method'] == m) & df['drifted_feature_names'].notna()]
        counts = {}
        for names in sub['drifted_feature_names']:
            for n in str(names).split(';'):
                n = n.strip()
                if n:
                    counts[n] = counts.get(n, 0) + 1
        top = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:6][::-1]
        labels = [feature_label(f) for f, _ in top]
        vals = [v for _, v in top]
        y = np.arange(len(labels))
        ax.barh(y, vals, color=colors[m], height=0.6, zorder=3)
        for yi, v in zip(y, vals):
            ax.text(v + 0.2, yi, str(v), va='center', fontsize=8, color=INK_SECONDARY)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Weeks crossed (out of 14)', fontsize=9)
        ax.set_title(label_map[m], fontsize=11, pad=8)
        ax.grid(axis='x', linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        style_axes(ax, hide_spines=('top', 'right', 'left'))

    fig.suptitle('Top individually-drifting features, by detector', fontsize=13, y=1.00)
    fig.tight_layout()
    savefig(fig, '22_per_method_top_features.png')


# ── 23: Feature-level agreement — do methods implicate the same features? ──
# For each of the 6 feature-aware methods, build a 20-dim vector: how many of
# the 14 weeks each monitored feature was "implicated" by that method (an
# individual KS/PSI/Jensen-Shannon/SHAP flag, or a top-5 distance/reconstruction
# contributor for Clustering/Autoencoder). Cosine similarity between two
# methods' vectors then answers "do these two methods point at the same
# features" as a single number — independent of whether the methods agree on
# which *weeks* had drift (chart 24 answers that, separately).
FEATURE_AWARE_METHODS = ['ks_stats', 'psi', 'kl_divergence', 'shap', 'clustering', 'autoencoder']
FEATURE_METHOD_LABELS = {
    'ks_stats': 'KS test', 'psi': 'PSI', 'kl_divergence': 'Jensen-Shannon',
    'shap': 'SHAP', 'clustering': 'Clustering', 'autoencoder': 'Autoencoder',
}


def _feature_implication_vectors():
    d = unified()
    top_features = d['top_features']
    idx = {f: i for i, f in enumerate(top_features)}
    n = len(top_features)
    vote_methods = ['ks_stats', 'psi', 'kl_divergence', 'shap']
    vecs = {m: np.zeros(n) for m in FEATURE_AWARE_METHODS}

    matrix = rl('method_week_matrix.csv')
    for _, row in matrix.iterrows():
        m = row['method']
        if m in vote_methods and pd.notna(row['drifted_feature_names']):
            for feat in str(row['drifted_feature_names']).split(';'):
                feat = feat.strip()
                if feat in idx:
                    vecs[m][idx[feat]] += 1

    for r in weekly_records():
        for m in ['clustering', 'autoencoder']:
            for item in r['method_status'][m].get('top_contributing_features', []):
                f = item['feature']
                if f in idx:
                    vecs[m][idx[f]] += 1

    return vecs, top_features


def _cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def chart_23_feature_agreement():
    vecs, _ = _feature_implication_vectors()
    methods = FEATURE_AWARE_METHODS
    labels = [FEATURE_METHOD_LABELS[m] for m in methods]
    n = len(methods)
    sim = np.zeros((n, n))
    for i, m1 in enumerate(methods):
        for j, m2 in enumerate(methods):
            sim[i, j] = _cosine(vecs[m1], vecs[m2])

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(sim, cmap=SEQUENTIAL_BLUE, vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            v = sim[i, j]
            txt_color = 'white' if v > 0.6 else INK_PRIMARY
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=10, color=txt_color)
    for i in range(n + 1):
        ax.axhline(i - 0.5, color=SURFACE, linewidth=2)
        ax.axvline(i - 0.5, color=SURFACE, linewidth=2)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Cosine similarity of feature-implication vectors', color=INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_MUTED)
    ax.set_title('Do these detectors implicate the same features?\n'
                 '(1.0 = identical feature-drift pattern, 0.0 = unrelated)', fontsize=12, pad=12)
    style_axes(ax, hide_spines=('top', 'right', 'left', 'bottom'))
    savefig(fig, '23_feature_agreement.png')


# ── 24: Week-level agreement — did detectors fire in the same weeks? ───────
# Different question from chart 23: this is pairwise Jaccard similarity over
# *which weeks each detector raised a raw alarm*, already computed by the
# pipeline (reports/detector_agreement.csv) — reused here, not recomputed.
def chart_24_week_agreement():
    df = rl('detector_agreement.csv').set_index(rl('detector_agreement.csv').columns[0])
    label_map = {
        'ks_stats': 'KS test', 'psi': 'PSI', 'kl_divergence': 'Jensen-Shannon',
        'ddm': 'DDM', 'eddm': 'EDDM', 'adwin': 'ADWIN', 'hddm': 'HDDM',
        'shap': 'SHAP', 'clustering': 'Clustering', 'autoencoder': 'Autoencoder',
        'prequential_auc': 'Prequential AUC', 'champion_vs_challenger': 'Champion vs Challenger',
    }
    methods = list(df.columns)
    labels = [label_map.get(m, m) for m in methods]
    mat = df.loc[methods, methods].to_numpy(dtype=float)
    mat_masked = np.ma.masked_invalid(mat)

    fig, ax = plt.subplots(figsize=(10, 9))
    cmap = SEQUENTIAL_BLUE.copy()
    cmap.set_bad(GRIDLINE)
    im = ax.imshow(mat_masked, cmap=cmap, vmin=0, vmax=1)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(methods)):
        for j in range(len(methods)):
            v = mat[i, j]
            if np.isnan(v):
                continue
            txt_color = 'white' if v > 0.6 else INK_PRIMARY
            ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=8, color=txt_color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Jaccard similarity of raw-alarm weeks', color=INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_MUTED)
    ax.set_title('Did these detectors raise raw alarms in the same weeks?\n'
                 '(grey = at least one detector never raised a raw alarm — undefined overlap)',
                 fontsize=12, pad=12)
    style_axes(ax, hide_spines=('top', 'right', 'left', 'bottom'))
    savefig(fig, '24_week_agreement.png')


# ── 25: Would retraining actually have helped? Champion vs Challenger, signed ──
def chart_25_retraining_would_help():
    weeks, gaps = [], []
    for r in weekly_records():
        c = r['method_status']['champion_vs_challenger']
        weeks.append(r['week'])
        gaps.append(c['auc_gap'])

    colors = [GREEN if g > 0 else RED for g in gaps]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(weeks, gaps, color=colors, width=0.6, zorder=3)
    ax.axhline(0, color=INK_MUTED, linewidth=1)
    ax.axhline(0.03, color=INK_MUTED, linewidth=1, linestyle='--')
    ax.text(0.3, 0.032, 'Gap threshold (0.03)', fontsize=8, color=INK_MUTED)
    n_help = sum(1 for g in gaps if g > 0)
    n_hurt = sum(1 for g in gaps if g < 0)
    ax.set_xlabel('Week')
    ax.set_ylabel('AUC gap (challenger − champion)')
    ax.set_title(f'Would retraining actually have helped this week? '
                 f'{n_help} of 14 weeks yes, {n_hurt} of 14 no',
                 fontsize=12, pad=12)
    handles = [mpatches.Patch(color=GREEN, label='Retraining would have helped'),
               mpatches.Patch(color=RED, label='Retraining would have hurt')]
    ax.legend(handles=handles, frameon=False, loc='lower left')
    ax.grid(axis='y', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    style_axes(ax)
    savefig(fig, '25_retraining_would_help.png')


if __name__ == '__main__':
    print(f'Reading reports from: {REPORTS_DIR}')
    print(f'Writing visuals to:   {VISUALS_DIR}')
    chart_01_feature_funnel()
    chart_02_feature_category_breakdown()
    chart_03_confirmed_alarms_per_detector()
    chart_04_feature_vote_weekly_trend()
    chart_05_method_week_heatmap()
    chart_06_top_drifting_features()
    chart_07_rl_policy_comparison()
    chart_08_gain_decomposition()
    chart_09_ablation()
    chart_10_policy_reliance()
    chart_11_psi_weekly_trend()
    chart_12_shap_importance_shift()
    chart_13_clustering_trend()
    chart_14_error_rate_trend()
    chart_15_adwin_zscore_trend()
    chart_16_prequential_auc_trend()
    chart_17_champion_challenger()
    chart_18_autoencoder_zscore_trend()
    chart_19_cluster_composition()
    chart_20_clustering_feature_contributions()
    chart_21_autoencoder_feature_contributions()
    chart_22_per_method_top_features()
    chart_23_feature_agreement()
    chart_24_week_agreement()
    chart_25_retraining_would_help()
    print('Done.')
