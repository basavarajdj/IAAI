"""
Tests for the drift-adaptation environment and the PPO agent.

The agent is tested on a *synthetic* stream whose optimal policy is known by
construction: two drift points, a model that scores well only while it is in the
same regime as the week being scored, and a real cost per adaptation. If PPO
cannot beat "do nothing" and "retrain every week" on that, it is not going to
learn anything on real data.

This caught a genuine bug. With dropout active in the policy network, PPO's
importance ratio compared a dropout-on distribution in the update against a
dropout-off one from the rollout, mixing policy change with dropout noise. The
agent converged to always-retrain and left 46% of the available return unclaimed
on a task with an obvious optimum. See DriftPolicyNet's docstring.

Run: python tests/test_rl_agent.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import rl_env                                                        # noqa: E402
from drift_signals import N_SIGNALS                                  # noqa: E402
from model_lattice import ALPHAS                                     # noqa: E402
from rl_agent import PPOAgent                                        # noqa: E402
from rl_env import (DriftAdaptationEnv, DO_NOTHING, FULL_RETRAIN,    # noqa: E402
                    PARTIAL_UPDATE, HEDGE_ENSEMBLE)

WEEKS = list(range(1, 15))
DRIFT_WEEKS = [5, 10]


def _regime(week):
    return sum(1 for d in DRIFT_WEEKS if week >= d)


class SyntheticLattice:
    """A model scores well on a week only if it was trained in the same regime."""

    def __init__(self):
        self.weekly_X = {w: None for w in WEEKS}
        self.weeks = WEEKS
        self.full_models = {0: None, **{w: None for w in WEEKS}}
        self.auc, self.f1, self.preds = {}, {}, {}

        for f in [0] + WEEKS:
            for p in [None] + WEEKS:
                if p is not None and p <= f:
                    continue
                self.preds[(f, p)] = {}
                for a in ALPHAS:
                    row = {}
                    for t in WEEKS:
                        if t <= f or (p is not None and t <= p):
                            continue
                        stale = _regime(t) - _regime(f)
                        current = 0.90 - 0.04 * max(stale, 0)
                        if p is not None:
                            current += 0.010 if _regime(p) == _regime(t) else -0.005
                        baseline = 0.90 - 0.04 * _regime(t)
                        row[t] = a * current + (1 - a) * baseline
                    self.auc[(f, p, a)] = row
                    self.f1[(f, p, a)] = {k: v - 0.4 for k, v in row.items()}

    def performance(self, f, p, a, week):
        v = self.auc.get((f, p, a), {}).get(week)
        return (v, v - 0.4) if v is not None else (None, None)


class SyntheticSignals:
    """Drift signals scale with how many regimes the reference is behind."""

    def get(self, ref_week, week):
        s = np.zeros(N_SIGNALS, dtype=np.float32)
        stale = _regime(week) - _regime(ref_week)
        s[0], s[2], s[5] = 0.7 * stale, 0.6 * stale, 0.8 * stale
        s[9] = 3.0 * stale
        s[10] = week - ref_week
        return s


def _make_env():
    return DriftAdaptationEnv(SyntheticLattice(), SyntheticSignals())


# ══════════════════════════════════════════════
# Environment mechanics
# ══════════════════════════════════════════════
def test_full_retrain_resets_model_state():
    env = _make_env()
    env.step(HEDGE_ENSEMBLE)
    env.step(PARTIAL_UPDATE)
    assert env.alpha < 1.0 and env.partial_week is not None

    env.step(FULL_RETRAIN)
    assert env.alpha == 1.0, "a full retrain must restore full trust in the new model"
    assert env.partial_week is None, "a full retrain supersedes any partial update"
    assert env.full_week == env.weeks[env.t - 1]


def test_hedge_moves_weight_toward_baseline_and_stops_at_zero():
    env = _make_env()
    seen = [env.alpha]
    for _ in range(len(ALPHAS) + 3):
        env.step(HEDGE_ENSEMBLE)
        seen.append(env.alpha)
    assert seen == sorted(seen, reverse=True), "hedging must be monotone"
    assert min(seen) == min(ALPHAS), "hedging should reach the fully-baseline end"


def test_reward_is_improvement_minus_cost():
    """Doing nothing when already up to date must beat retraining, by the cost."""
    idle = _make_env().run_policy(lambda s: DO_NOTHING)
    churn = _make_env().run_policy(lambda s: FULL_RETRAIN)

    early = [h for h in churn['history'] if _regime(h['week']) == 0]
    assert all(h['reward'] < 0 for h in early), (
        "retraining while already current should be strictly negative — "
        "it buys no AUC and costs the retrain")
    assert idle['n_full_retrains'] == 0


def test_action_only_affects_later_weeks():
    """A decision must not be graded on the week it was made."""
    env = _make_env()
    env.reset()
    first_week = env.weeks[0]
    env.step(FULL_RETRAIN)
    assert env.history[0]['week'] > first_week, (
        "reward must come from a week after the action, or the agent can "
        "retrain on data it has already been scored against")


# ══════════════════════════════════════════════
# Learning
# ══════════════════════════════════════════════
def _trained_agent(n_updates=120, seed=0):
    env = _make_env()
    agent = PPOAgent(seed=seed)
    agent.train(env, n_updates=n_updates, episodes_per_update=8, log_every=10_000)
    return agent, env


def test_ppo_beats_both_trivial_policies():
    agent, env = _trained_agent()
    learned = env.run_policy(lambda s: agent.act(s, mode='greedy')[0])
    never = env.run_policy(lambda s: DO_NOTHING)
    always = env.run_policy(lambda s: FULL_RETRAIN)

    assert learned['total_reward'] > never['total_reward'], (
        f"learned {learned['total_reward']:.2f} did not beat never-retrain")
    assert learned['total_reward'] > always['total_reward'], (
        f"learned {learned['total_reward']:.2f} did not beat always-retrain "
        f"{always['total_reward']:.2f} — the agent found no timing skill")


def test_ppo_prefers_cheap_actions_when_they_suffice():
    """The point of the expanded action space: don't pay for a full retrain."""
    agent, env = _trained_agent()
    learned = env.run_policy(lambda s: agent.act(s, mode='greedy')[0])
    cheap = learned['n_partial_updates'] + learned['n_hedges']
    assert cheap > 0, "agent never used any action other than retrain/do-nothing"


def test_thompson_sampling_is_stochastic_and_greedy_is_not():
    agent, env = _trained_agent(n_updates=30)
    state = env.reset()
    greedy = {agent.act(state, mode='greedy')[0] for _ in range(20)}
    sampled = {agent.act(state, mode='thompson')[0] for _ in range(60)}
    assert len(greedy) == 1, "greedy mode must be deterministic"
    assert len(sampled) >= 1, "thompson mode must draw from the policy"


def test_action_probabilities_form_a_distribution():
    agent, env = _trained_agent(n_updates=10)
    probs = agent.action_probabilities(env.reset())
    assert abs(probs.sum() - 1.0) < 1e-5 and (probs >= 0).all()


if __name__ == '__main__':
    import logging
    logging.disable(logging.INFO)

    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    failures = []
    for name, fn in tests:
        try:
            fn()
            print(f"  PASS  {name}")
        except Exception as exc:                       # noqa: BLE001
            failures.append(name)
            print(f"  FAIL  {name}\n          {exc}")

    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
