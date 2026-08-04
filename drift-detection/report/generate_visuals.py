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
    ax.axhline(0.60, color=INK_MUTED, linewidth=1.5, linestyle='--')
    ax.text(0.3, 0.615, 'Consensus threshold (0.60)', fontsize=9, color=INK_MUTED)
    ax.set_xlabel('Week')
    ax.set_ylabel('Fraction of monitored features crossed')
    ax.set_ylim(0, 0.7)
    ax.set_title('Weekly trend of the four feature-vote detectors vs. consensus threshold',
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
CLASSICAL_POLICIES = {'adwin', 'prequential_auc', 'eddm', 'champion_vs_challenger'}


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
    print('Done.')
