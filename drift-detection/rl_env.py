"""
Drift Adaptation as a Sequential Decision Problem
=================================================

Why RL rather than a better threshold
-------------------------------------
Every detector in this repo answers a one-shot question: "has drift occurred?"
The question an operator actually faces is different and sequential: *given
everything I have observed, and the model I currently have, what should I do
this week?* Those differ in three ways a threshold cannot express:

1. **The answer depends on the current model, not just the data.** The same
   drift signal warrants a retrain if the model is six months stale and nothing
   if it was retrained last week. A detector has no notion of its own model's
   age.
2. **The choice is not binary.** Between "do nothing" and "retrain from
   scratch" sit a cheap fine-tune and a costless ensemble re-weighting. A
   detector cannot recommend the middle.
3. **Actions have delayed, compounding consequences.** Retraining during a
   turbulent week permanently folds that turbulence into a cumulative training
   set — the cost lands weeks later. This is precisely what Section 7.3 of
   PAPER.md measured: the two most trigger-happy detectors underperformed
   *random* policies of equal cost, because they fired at bad moments. Credit
   assignment across time is what RL is for.

Formulation
-----------
    state   s_t = [ drift signals vs. the current reference | model context ]
    action  a_t in {do nothing, partial update, full retrain, hedge ensemble}
    reward  r_t = performance gained over never-retraining, minus action cost

An episode is one pass over the weekly replay. Because every reachable model was
enumerated in advance (model_lattice.py), stepping the environment is a table
lookup — so PPO can run thousands of episodes over a 14-week dataset that
otherwise provides exactly one trajectory.

Action semantics
----------------
    0  DO_NOTHING       keep the current model and ensemble weight
    1  PARTIAL_UPDATE   fine-tune the last full model on the recent window;
                        cheap, fast, but risks overfitting to that window
    2  FULL_RETRAIN     retrain on all data so far; resets the reference,
                        the ensemble weight, and the staleness counters
    3  HEDGE_ENSEMBLE   shift weight from the current model toward the stable
                        baseline; costs nothing, trains nothing, and is the
                        recovery move when a partial update went badly

Weight only moves *toward* the baseline between full retrains, and a full
retrain resets it to 1.0. That keeps the state small and the semantics honest:
hedging is a response to declining trust in the current model, and regaining
trust requires actually rebuilding it.
"""

import numpy as np

from drift_signals import N_SIGNALS
from model_lattice import ALPHAS

DO_NOTHING, PARTIAL_UPDATE, FULL_RETRAIN, HEDGE_ENSEMBLE = 0, 1, 2, 3
ACTION_NAMES = ['do_nothing', 'partial_update', 'full_retrain', 'hedge_ensemble']
N_ACTIONS = len(ACTION_NAMES)

# Costs in AUC points — the performance a manager should be willing to give up
# for the operational convenience of not doing this. A full retrain on the whole
# history is the expensive one; hedging is free because it trains nothing.
ACTION_COSTS = {
    DO_NOTHING: 0.0000,
    PARTIAL_UPDATE: 0.0010,
    FULL_RETRAIN: 0.0040,
    HEDGE_ENSEMBLE: 0.0000,
}

REWARD_SCALE = 100.0     # AUC deltas are ~1e-2; scale so rewards are O(1)

CONTEXT_NAMES = [
    'weeks_since_full_retrain',
    'weeks_since_partial_update',
    'ensemble_alpha',
    'recent_auc_delta',        # last week's AUC minus the never-retrain reference
    'recent_f1',
    'progress',                # position through the episode, in [0, 1]
]

STATE_DIM = N_SIGNALS + len(CONTEXT_NAMES)


def state_names():
    """Input layout of the policy network — the explainer reads this."""
    from drift_signals import SIGNAL_NAMES
    return list(SIGNAL_NAMES) + list(CONTEXT_NAMES)


class DriftAdaptationEnv:
    """Weekly replay as an episodic MDP, backed by the precomputed lattice."""

    def __init__(self, lattice, signals, baseline_auc_by_week=None, state_mask=None):
        """
        state_mask: optional boolean array over the observation vector. False
            entries are zeroed before the agent sees them. This exists for the
            ablation that matters most — an agent given only model context
            (staleness, recent performance, position in the year) and no drift
            signals can still learn a *calendar* policy. Comparing it against
            the full agent is the only way to show the drift signals are
            contributing something a schedule could not.
        """
        self.lattice = lattice
        self.signals = signals
        self.weeks = sorted(lattice.weekly_X)
        self.state_names = state_names()
        self.state_mask = None if state_mask is None else np.asarray(state_mask, dtype=np.float32)

        # Never-retrain performance is the zero point for reward: it is what you
        # get for free, so only improvement over it should be rewarded.
        self.baseline_auc = baseline_auc_by_week or {
            w: lattice.auc[(0, None, 1.0)].get(w, 0.5) for w in self.weeks
        }
        self.reset()

    # ── episode lifecycle ────────────────────────────────────────
    def reset(self):
        self.t = 0
        self.full_week = 0          # last full retrain (0 = the baseline model)
        self.partial_week = None    # last partial update since that retrain
        self.alpha = 1.0            # ensemble weight on the current model
        self.last_auc_delta = 0.0
        self.last_f1 = 0.0
        self.history = []
        return self._observe()

    def _observe(self):
        week = self.weeks[self.t]
        drift = self.signals.get(self.full_week, week)
        context = np.array([
            (week - self.full_week) / 10.0,
            (week - self.partial_week) / 10.0 if self.partial_week else 1.0,
            self.alpha,
            self.last_auc_delta * REWARD_SCALE,
            self.last_f1,
            self.t / max(len(self.weeks) - 1, 1),
        ], dtype=np.float32)
        observation = np.concatenate([drift, context])
        return observation if self.state_mask is None else observation * self.state_mask

    def _performance(self, week):
        """AUC/F1 of the current model state on `week`, with graceful fallback."""
        auc, f1 = self.lattice.performance(self.full_week, self.partial_week, self.alpha, week)
        if auc is None:                       # state not valid yet for this week
            auc, f1 = self.lattice.performance(self.full_week, None, self.alpha, week)
        if auc is None:
            auc, f1 = self.baseline_auc.get(week, 0.5), 0.0
        return auc, f1

    def _apply(self, action, week):
        if action == FULL_RETRAIN:
            self.full_week = week
            self.partial_week = None
            self.alpha = 1.0
        elif action == PARTIAL_UPDATE:
            self.partial_week = week
        elif action == HEDGE_ENSEMBLE:
            lower = [a for a in ALPHAS if a < self.alpha]
            self.alpha = max(lower) if lower else self.alpha

    def step(self, action):
        """Act on the current week, then observe the following week's outcome.

        The ordering matters and mirrors reality: a decision made in week t can
        only affect weeks after t. Rewarding the action with week t's own score
        would let the agent 'retrain' on a week it has already been graded on.
        """
        acted_on = self.weeks[self.t]
        self._apply(action, acted_on)

        self.t += 1
        if self.t >= len(self.weeks):
            return self._zeros(), 0.0, True, {}

        week = self.weeks[self.t]
        auc, f1 = self._performance(week)
        auc_delta = auc - self.baseline_auc.get(week, 0.5)
        reward = REWARD_SCALE * (auc_delta - ACTION_COSTS[action])

        self.last_auc_delta, self.last_f1 = auc_delta, f1
        self.history.append({
            'week': week, 'action': action, 'action_name': ACTION_NAMES[action],
            'auc': auc, 'f1': f1, 'auc_delta': auc_delta, 'reward': reward,
            'full_week': self.full_week, 'partial_week': self.partial_week, 'alpha': self.alpha,
        })
        return self._observe(), reward, False, {'auc': auc, 'f1': f1}

    def _zeros(self):
        return np.zeros(STATE_DIM, dtype=np.float32)

    # ── evaluation helper ────────────────────────────────────────
    def run_policy(self, choose_action):
        """Play one deterministic episode with `choose_action(state) -> int`."""
        state = self.reset()
        total = 0.0
        while True:
            action = choose_action(state)
            state, reward, done, _ = self.step(action)
            total += reward
            if done:
                break
        aucs = [h['auc'] for h in self.history]
        return {
            'total_reward': total,
            'mean_auc': float(np.mean(aucs)) if aucs else 0.0,
            'min_auc': float(np.min(aucs)) if aucs else 0.0,
            'mean_f1': float(np.mean([h['f1'] for h in self.history])) if self.history else 0.0,
            'n_full_retrains': sum(h['action'] == FULL_RETRAIN for h in self.history),
            'n_partial_updates': sum(h['action'] == PARTIAL_UPDATE for h in self.history),
            'n_hedges': sum(h['action'] == HEDGE_ENSEMBLE for h in self.history),
            'history': list(self.history),
        }
