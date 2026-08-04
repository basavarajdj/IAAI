"""
RL Drift-Adaptation Experiment
==============================

Runs the whole comparison:

    1. Load data, freeze the feature representation, pick the monitoring set.
    2. Build the model lattice — every reachable (full, partial, alpha) state.
    3. Precompute the drift-signal matrix (the agent's observations).
    4. Train the PPO agent on repeated replays of the stream.
    5. Evaluate the agent against every classical detector policy, plus
       never-retrain, always-retrain, and frequency-matched random controls.
    6. Emit explanations: per-method profiles, the agent's decision trace, and
       which signals its policy actually relies on.

The comparison is apples-to-apples by construction: every policy — classical or
learned — selects from the same precomputed lattice, so differences come from
*decisions*, not from training randomness.

    python run_rl_experiment.py --data_dir ./dataset
"""

import argparse
import json
import logging
import os

import numpy as np
import pandas as pd

from data_processing import load_data
from feature_engineering import FeatureEngineer, add_causal_sequence_features, label_features
from feature_selection import select_monitoring_features
from run_drift_analysis import preprocess_and_align, THREE_MONTHS_SECONDS, WEEK_SECONDS
from model_lattice import ModelLattice, ALPHAS
from drift_signals import DriftSignalMatrix
from rl_env import (DriftAdaptationEnv, ACTION_NAMES, ACTION_COSTS, REWARD_SCALE,
                    DO_NOTHING, FULL_RETRAIN, PARTIAL_UPDATE, STATE_DIM)
from drift_signals import N_SIGNALS
from rl_agent import PPOAgent
from explain import explain_episode, policy_summary, method_profile_frame

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('logs/rl_experiment.log'), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════
# Data
# ══════════════════════════════════════════════
def load_windows(data_dir, top_k=20, n_bags=3):
    """Baseline window + weekly windows, in the frozen feature space."""
    logger.info("Loading and preparing data...")
    tr_id, tr_trn, _, _ = load_data(data_dir)
    raw = pd.merge(tr_trn, tr_id, on='TransactionID', how='left').sort_values('TransactionDT')
    raw = add_causal_sequence_features(raw)

    start = raw['TransactionDT'].min()
    cutoff = start + THREE_MONTHS_SECONDS

    fe = FeatureEngineer()
    processed = fe.fit_transform(raw[raw['TransactionDT'] < cutoff])
    schema = fe.feature_schema
    ref_X = preprocess_and_align(processed.drop('isFraud', axis=1), schema).reset_index(drop=True)
    ref_y = processed['isFraud'].reset_index(drop=True)

    weekly_X, weekly_y = {}, {}
    t, week = cutoff, 1
    end = raw['TransactionDT'].max()
    while t < end:
        chunk = raw[(raw['TransactionDT'] >= t) & (raw['TransactionDT'] < t + WEEK_SECONDS)]
        if len(chunk):
            p = fe.transform(chunk)
            weekly_X[week] = preprocess_and_align(
                p.drop('isFraud', axis=1), schema).reset_index(drop=True)
            weekly_y[week] = p['isFraud'].reset_index(drop=True)
            week += 1
        t += WEEK_SECONDS

    logger.info(f"Baseline {len(ref_X):,} rows; {len(weekly_X)} weekly windows.")

    from model_training import train_model
    seed_model = train_model(ref_X, ref_y)
    features, _, report = select_monitoring_features(
        ref_X, ref_y, train_fn=lambda X, y, seed: train_model(X, y, seed=seed),
        reference_model=seed_model, top_k=top_k, n_bags=n_bags)
    logger.info(f"Monitoring set (stability {report['stability_index']:.3f}):")
    for raw, label in zip(features, label_features(features)):
        logger.info(f"    {raw:<45} {label}")

    return ref_X, ref_y, weekly_X, weekly_y, features


# ══════════════════════════════════════════════
# Baseline policies over the same lattice
# ══════════════════════════════════════════════
def evaluate_fixed(env, retrain_weeks, name):
    """Run a policy that full-retrains on a fixed set of weeks."""
    target = set(retrain_weeks)
    result = env.run_policy(lambda s: FULL_RETRAIN if env.weeks[env.t] in target else DO_NOTHING)
    result['policy'] = name
    return result


def evaluate_agent(env, agent, mode='greedy'):
    result = env.run_policy(lambda s: agent.act(s, mode=mode)[0])
    result['policy'] = f'rl_agent_{mode}'
    return result


def random_control(env, n_full, n_partial=0, n_samples=200, seed=42):
    """Budget-matched random policies — the control that decides the question.

    Matched on *every* adaptation action, not just full retrains. An agent that
    took 3 retrains and 10 partial updates must be compared against random
    policies that also take 3 and 10, or the comparison rewards it for spending
    more rather than for spending it at better moments.
    """
    total = n_full + n_partial
    if total == 0:
        return None

    rng = np.random.default_rng(seed)
    scores = []
    for _ in range(n_samples):
        weeks = rng.choice(env.weeks, size=min(total, len(env.weeks)), replace=False)
        full_weeks = set(weeks[:n_full])
        partial_weeks = set(weeks[n_full:])

        def choose(_state):
            w = env.weeks[env.t]
            if w in full_weeks:
                return FULL_RETRAIN
            return PARTIAL_UPDATE if w in partial_weeks else DO_NOTHING

        scores.append(env.run_policy(choose)['mean_auc'])
    return np.array(scores)


# ══════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════
def main(data_dir='./dataset', output_dir='./reports', top_k=20, n_bags=3,
         n_updates=150, episodes_per_update=8, seed=42):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)

    ref_X, ref_y, weekly_X, weekly_y, features = load_windows(data_dir, top_k, n_bags)

    logger.info("=" * 60)
    logger.info("Building the model lattice (every reachable adaptation state)")
    logger.info("=" * 60)
    lattice = ModelLattice(ref_X, ref_y, weekly_X, weekly_y, seed=seed).build()

    logger.info("=" * 60)
    logger.info("Precomputing drift signals")
    logger.info("=" * 60)
    signals = DriftSignalMatrix(ref_X, ref_y, weekly_X, weekly_y,
                                features, lattice, seed=seed).build()

    env = DriftAdaptationEnv(lattice, signals)

    logger.info("=" * 60)
    logger.info(f"Training PPO agent ({n_updates} updates x {episodes_per_update} episodes)")
    logger.info("=" * 60)
    agent = PPOAgent(seed=seed)
    train_history = agent.train(env, n_updates=n_updates,
                                episodes_per_update=episodes_per_update)
    agent.save('models/rl_drift_agent.pt')

    ablations = _run_ablations(lattice, signals, n_updates, episodes_per_update, seed)

    # ── Evaluation ──
    logger.info("=" * 60)
    logger.info("Evaluating policies")
    logger.info("=" * 60)

    results = []

    # Classical detectors, replayed from the existing drift report if present.
    detector_weeks = _detector_retrain_weeks(output_dir)
    for name, weeks in detector_weeks.items():
        results.append(evaluate_fixed(env, weeks, name))

    results.append(evaluate_fixed(env, [], 'never_retrain'))
    results.append(evaluate_fixed(env, env.weeks, 'always_retrain'))

    # Naive policies over the expanded action space. These are the controls that
    # decide whether the agent learned anything, or whether the benefit comes
    # simply from *having* a cheap action available. Without them, an agent that
    # blindly fine-tunes every week would look like a success.
    for name, action in [('always_partial_update', PARTIAL_UPDATE)]:
        r = env.run_policy(lambda s: action)
        r['policy'] = name
        results.append(r)

    for period in (2, 4):
        r = env.run_policy(
            lambda s, p=period: FULL_RETRAIN if env.weeks[env.t] % p == 0 else DO_NOTHING)
        r['policy'] = f'fixed_schedule_every_{period}w'
        results.append(r)
    results.append(evaluate_agent(env, agent, 'greedy'))
    results.append(evaluate_agent(env, agent, 'thompson'))

    summary = pd.DataFrame([{
        'policy': r['policy'],
        'mean_auc': r['mean_auc'],
        'min_auc': r['min_auc'],
        'mean_f1': r['mean_f1'],
        'total_reward': r['total_reward'],
        'full_retrains': r['n_full_retrains'],
        'partial_updates': r['n_partial_updates'],
        'hedges': r['n_hedges'],
    } for r in results]).sort_values('mean_auc', ascending=False)

    # Random control, matched on the full adaptation budget each policy spent.
    for row in summary.itertuples():
        control = random_control(env, row.full_retrains, row.partial_updates, seed=seed)
        if control is not None:
            summary.loc[summary['policy'] == row.policy, 'random_control_pct'] = \
                float(np.mean(row.mean_auc >= control))

    ablation_df = pd.DataFrame(ablations.values())
    ablation_df.to_csv(os.path.join(output_dir, 'rl_ablation.csv'), index=False)
    logger.info("\nAblation — what the agent is allowed to observe:\n"
                + ablation_df.to_string(index=False))

    logger.info("\n" + summary.to_string(index=False))
    summary.to_csv(os.path.join(output_dir, 'rl_policy_comparison.csv'), index=False)

    # ── Explanations ──
    trace = explain_episode(agent, env)
    trace.to_csv(os.path.join(output_dir, 'rl_decision_trace.csv'), index=False)
    logger.info("\nAgent decision trace:\n" + trace[
        ['week', 'action', 'confidence', 'reward', 'top_drivers']].to_string(index=False))

    reliance = policy_summary(agent, env)
    reliance.to_csv(os.path.join(output_dir, 'rl_policy_reliance.csv'), index=False)
    logger.info("\nSignals the policy relies on:\n" + reliance.head(8).to_string(index=False))

    method_profile_frame().to_csv(
        os.path.join(output_dir, 'method_profiles.csv'), index=False)

    forgetting = _forgetting_table(lattice)
    forgetting.to_csv(os.path.join(output_dir, 'forgetting_analysis.csv'), index=False)
    logger.info("\nCatastrophic forgetting after partial updates:\n"
                + forgetting.head(10).to_string(index=False))

    with open(os.path.join(output_dir, 'rl_experiment.json'), 'w') as f:
        json.dump({
            'monitoring_features': features,
            'monitoring_feature_labels': label_features(features),
            'ablation': ablations,
            'n_models_in_lattice': lattice.n_models_trained,
            'alphas': list(ALPHAS),
            'action_costs': {ACTION_NAMES[k]: v for k, v in ACTION_COSTS.items()},
            'reward_scale': REWARD_SCALE,
            'training_history': train_history[-20:],
            'policy_summary': summary.to_dict(orient='records'),
            'decision_trace': trace.to_dict(orient='records'),
        }, f, indent=2, default=str)

    logger.info(f"\nReports written to {output_dir}/")
    return agent, env, summary


def _run_ablations(lattice, signals, n_updates, episodes_per_update, seed):
    """Does the agent actually use the drift signals, or just a calendar?

    This is the experiment that decides whether "combine the detectors" is a
    real contribution. Three agents, identical in every way except what they are
    allowed to observe:

      full          drift signals + model context
      context_only  staleness, recent performance, position in the replay —
                    everything EXCEPT the detectors. A strong result here would
                    mean the detectors add nothing and a fixed schedule suffices.
      signals_only  detector outputs with no model context. Isolates how much
                    of the policy is drift-driven.

    If `full` does not beat `context_only`, the honest conclusion is that the
    detector ensemble contributed nothing and the agent learned a schedule.
    """
    masks = {
        'full': np.ones(STATE_DIM, dtype=np.float32),
        'context_only': np.concatenate([np.zeros(N_SIGNALS), np.ones(STATE_DIM - N_SIGNALS)]),
        'signals_only': np.concatenate([np.ones(N_SIGNALS), np.zeros(STATE_DIM - N_SIGNALS)]),
    }

    results = {}
    for name, mask in masks.items():
        logger.info(f"  ablation '{name}' ...")
        env = DriftAdaptationEnv(lattice, signals, state_mask=mask)
        agent = PPOAgent(seed=seed)
        agent.train(env, n_updates=n_updates, episodes_per_update=episodes_per_update,
                    log_every=10_000)
        outcome = env.run_policy(lambda s: agent.act(s, mode='greedy')[0])
        results[name] = {
            'observes': name,
            'mean_auc': outcome['mean_auc'],
            'min_auc': outcome['min_auc'],
            'total_reward': outcome['total_reward'],
            'full_retrains': outcome['n_full_retrains'],
            'partial_updates': outcome['n_partial_updates'],
            'hedges': outcome['n_hedges'],
        }
        logger.info(f"    {name}: AUC {outcome['mean_auc']:.4f}, "
                    f"reward {outcome['total_reward']:.2f}")
    return results


def _detector_retrain_weeks(output_dir):
    """Reuse the classical detectors' decisions from the earlier drift run."""
    path = os.path.join(output_dir, 'unified_drift_report.json')
    if not os.path.exists(path):
        logger.warning("No unified_drift_report.json — skipping classical detector policies.")
        return {}
    with open(path) as f:
        report = json.load(f)
    weeks = {}
    for trigger in report.get('retrain_triggers', []):
        weeks.setdefault(trigger['method'], set()).add(trigger['week'])
    return {m: sorted(w) for m, w in weeks.items()}


def _forgetting_table(lattice):
    """How much each partial update cost on the broader distribution."""
    rows = []
    for (full_w, partial_w) in lattice.preds:
        if partial_w is None:
            continue
        for week in lattice.weeks:
            if week <= partial_w:
                continue
            cost, best_alpha = lattice.forgetting_cost(full_w, partial_w, week)
            if cost > 0:
                rows.append({'full_retrain_week': full_w, 'partial_update_week': partial_w,
                             'evaluated_week': week, 'auc_lost_by_full_trust': round(cost, 5),
                             'best_ensemble_alpha': best_alpha})
    df = pd.DataFrame(rows)
    return df.sort_values('auc_lost_by_full_trust', ascending=False) if len(df) else df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RL-based drift adaptation experiment')
    parser.add_argument('--data_dir', default='./dataset')
    parser.add_argument('--output_dir', default='./reports')
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--n_bags', type=int, default=3)
    parser.add_argument('--n_updates', type=int, default=150)
    parser.add_argument('--episodes_per_update', type=int, default=8)
    args = parser.parse_args()

    main(data_dir=args.data_dir, output_dir=args.output_dir, top_k=args.top_k,
         n_bags=args.n_bags, n_updates=args.n_updates,
         episodes_per_update=args.episodes_per_update)
