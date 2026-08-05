"""
One-off generator for the Section 3.3 method-by-method markdown tables.
Not part of the regular reproduction pipeline (generate_visuals.py is) —
this just prints markdown to stdout, which gets pasted into
RESEARCH_PRESENTATION.md by hand. Kept in the repo so the exact
provenance of those tables is inspectable/rerunnable, not because it's
meant to be run on every rebuild.
"""
import csv
import json
import os

REPORTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'reports')


def load_json_field(method, field):
    with open(os.path.join(REPORTS, 'unified_drift_report.json')) as f:
        d = json.load(f)
    out = {}
    for r in d['weekly_records']:
        if r.get('is_baseline'):
            continue
        out[r['week']] = r['method_status'][method][field]
    return out


def load_matrix():
    rows = {}
    with open(os.path.join(REPORTS, 'method_week_matrix.csv')) as f:
        for row in csv.DictReader(f):
            rows.setdefault(row['method'], {})[int(row['week'])] = row
    return rows


def fmt(v, nd=4):
    if v in (None, ''):
        return '—'
    return f'{float(v):.{nd}f}'


FEATURE_VOTE_CONSENSUS = {'ks_stats': 0.30, 'psi': 0.60, 'kl_divergence': 0.30, 'shap': 0.30}


def feature_vote_table(m, rows):
    consensus = FEATURE_VOTE_CONSENSUS[m]
    lines = [f'| Week | Features crossed | Fraction | vs. {consensus:.2f} consensus | Raw alarm | Confirmed |',
             '|---|---|---|---|---|---|']
    for w in range(1, 15):
        r = rows[m][w]
        frac = float(r['features_crossed_fraction'])
        lines.append(
            f"| {w} | {int(float(r['features_crossed']))}/{int(float(r['features_total']))} "
            f"| {frac:.2f} | {frac - consensus:+.2f} | {'Yes' if r['raw_flag']=='True' else 'No'} "
            f"| {'**Yes**' if r['confirmed_flag']=='True' else 'No'} |"
        )
    return '\n'.join(lines)


def metric_table(m, rows, metric_label, threshold_label, nd=4):
    lines = [f'| Week | {metric_label} | {threshold_label} | Raw alarm | Confirmed | Model version after this week |',
             '|---|---|---|---|---|---|']
    for w in range(1, 15):
        r = rows[m][w]
        lines.append(
            f"| {w} | {fmt(r['key_metric_value'], nd)} | {fmt(r['key_metric_threshold'], nd)} "
            f"| {'Yes' if r['raw_flag']=='True' else 'No'} "
            f"| {'**Yes**' if r['confirmed_flag']=='True' else 'No'} | v{r['model_version']} |"
        )
    return '\n'.join(lines)


if __name__ == '__main__':
    rows = load_matrix()

    print('=== KS TEST ===')
    print(feature_vote_table('ks_stats', rows))
    print()

    print('=== PSI ===')
    print(feature_vote_table('psi', rows))
    print()

    print('=== JENSEN-SHANNON ===')
    print(feature_vote_table('kl_divergence', rows))
    print()

    print('=== SHAP ===')
    print(feature_vote_table('shap', rows))
    print()

    print('=== DDM ===')
    print(metric_table('ddm', rows, 'Mean error rate', 'DDM boundary (p_min + 3 x s_min)'))
    print()

    print('=== EDDM ===')
    print(metric_table('eddm', rows, "Inter-error metric (p'+2s')", 'Drift boundary (0.90 x running max)'))
    print('> EDDM alarms when the metric value *drops below* its boundary (errors bunching closer together), the only method here where "below" means worse.')
    print()

    print('=== HDDM ===')
    print(metric_table('hddm', rows, 'Mean error rate', 'Boundary (best-window mean + Hoeffding bound)'))
    print()

    print('=== ADWIN ===')
    print(metric_table('adwin', rows, 'z-score', 'z-threshold', nd=3))
    print()

    print('=== PREQUENTIAL AUC ===')
    print(metric_table('prequential_auc', rows, 'AUC drop vs. reference', 'Effective drop threshold'))
    print()

    print('=== CHAMPION VS CHALLENGER ===')
    degr = load_json_field('champion_vs_challenger', 'auc_degradation')
    lines = ['| Week | AUC gap (challenger - champion) | Gap threshold (0.03) '
             '| AUC degradation from baseline | Degradation threshold (0.05) | Raw alarm | Confirmed | Model version after this week |',
             '|---|---|---|---|---|---|---|---|']
    for w in range(1, 15):
        r = rows['champion_vs_challenger'][w]
        lines.append(
            f"| {w} | {fmt(r['key_metric_value'])} | {fmt(r['key_metric_threshold'])} "
            f"| {degr[w]:.4f} | 0.0500 "
            f"| {'Yes' if r['raw_flag']=='True' else 'No'} "
            f"| {'**Yes**' if r['confirmed_flag']=='True' else 'No'} | v{r['model_version']} |"
        )
    print('\n'.join(lines))
    print('> Fires when EITHER the gap clears 0.03 (and its own bootstrap SE) OR degradation from baseline clears 0.05 — the two right-hand columns are two independent triggers, not one combined score.')
    print()

    print('=== CLUSTERING ===')
    cpsi = load_json_field('clustering', 'cluster_psi')
    lines = ['| Week | Centroid distance ratio | Distance threshold (1.5) '
             '| Cluster-assignment PSI | PSI threshold (0.2) | Raw alarm | Confirmed | Model version after this week |',
             '|---|---|---|---|---|---|---|---|']
    for w in range(1, 15):
        r = rows['clustering'][w]
        lines.append(
            f"| {w} | {fmt(r['key_metric_value'], 3)} | 1.500 "
            f"| {cpsi[w]:.3f} | 0.200 "
            f"| {'Yes' if r['raw_flag']=='True' else 'No'} "
            f"| {'**Yes**' if r['confirmed_flag']=='True' else 'No'} | v{r['model_version']} |"
        )
    print('\n'.join(lines))
    print('> Fires when EITHER series crosses its own threshold — two independent triggers sharing one fixed k=5 K-Means model, fit once on the reference window.')
    print()

    print('=== AUTOENCODER ===')
    print(metric_table('autoencoder', rows, 'Reconstruction-error z-score', 'Threshold (3.0)', nd=3))
