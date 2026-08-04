"""
Explainability for the detectors and for the agent
==================================================

Two audiences, two kinds of explanation:

1. **Per-method profiles** — what each classical detector is structurally good
   at, what it is blind to, and what its measured behaviour on this dataset was.
   These are stated as properties of the statistic, not opinions, so they hold
   beyond this dataset.

2. **Agent decision traces** — for each week, what the agent did, how confident
   it was, and which input signals drove the decision. A policy that cannot
   explain itself is not deployable for a decision as expensive as retraining.

The agent explanation uses gradient x input on the policy head, which for a
categorical policy answers: "which observation components, at this state, pushed
the logit of the chosen action up?" That is the right question — not global
feature importance, but *why this action, here*.
"""

import numpy as np
import pandas as pd
import torch

from rl_env import ACTION_NAMES, state_names

# ══════════════════════════════════════════════
# 1. What each classical detector is for
# ══════════════════════════════════════════════
METHOD_PROFILES = {
    'ks_stats': {
        'family': 'distributional',
        'observes': 'marginal distribution of each monitored feature',
        'needs_labels': False,
        'best_at': 'abrupt covariate shift in a continuous feature — a new '
                   'acquiring bank, a changed upstream default value',
        'blind_to': 'concept drift. If the same transactions start being '
                    'fraudulent at a different rate, no feature marginal moves '
                    'and KS sees nothing.',
        'failure_mode': 'The p-value is a function of sample size. At 10^5 vs '
                        '10^4 rows the critical statistic is ~0.015, so a 1.5% '
                        'shift is "significant". Unqualified, it fires every '
                        'window; that is why it flagged 14/14 weeks originally.',
        'fix_applied': 'Effect-size floor (D >= 0.10), FDR correction across '
                       'features, bounded sample sizes.',
    },
    'psi': {
        'family': 'distributional',
        'observes': 'binned marginal distribution per feature',
        'needs_labels': False,
        'best_at': 'gradual, monotone population shifts; long industry track '
                   'record and an interpretable magnitude',
        'blind_to': 'concept drift; also insensitive to changes within a bin',
        'failure_mode': 'The 0.10/0.20 bands are scorecard folklore at an '
                        'unstated sample size. E[PSI] under the null scales as '
                        '(B-1)/n, so a 100-row window expects ~0.09 from noise '
                        'alone — nearly the "moderate" band.',
        'fix_applied': 'Bootstrap null calibrated to the current window size.',
    },
    'kl_divergence': {
        'family': 'distributional',
        'observes': 'binned densities per feature',
        'needs_labels': False,
        'best_at': 'detecting mass appearing where the reference had none',
        'blind_to': 'concept drift',
        'failure_mode': 'Unbounded and asymmetric, and undefined wherever the '
                        'current window has support the reference lacks. The '
                        'epsilon-smoothing fix makes its absolute value depend '
                        'on the arbitrary epsilon, so one threshold cannot mean '
                        'the same thing for two features.',
        'fix_applied': 'Jensen-Shannon distance as the decision statistic: '
                       'symmetric, finite, bounded in [0, 1].',
    },
    'ddm': {
        'family': 'performance',
        'observes': 'binary error stream',
        'needs_labels': True,
        'best_at': 'abrupt degradation on balanced problems; cheap and online',
        'blind_to': 'anything the majority class hides. At 3.5% prevalence a '
                    'total collapse in recall moves the raw error rate by 3.5 '
                    'points — inside its own noise band.',
        'failure_mode': 'Its control limits assume Bernoulli variance shrinking '
                        'as 1/n, so it grows *harder* to trigger the longer a '
                        'model has been stable.',
        'fix_applied': 'Class-balanced error stream (subsampled, not '
                       'reweighted, so it stays Bernoulli).',
    },
    'eddm': {
        'family': 'performance',
        'observes': 'distance between consecutive errors',
        'needs_labels': True,
        'best_at': 'gradual degradation — errors bunching up before the rate '
                   'itself moves much. The earliest warning of the classical set.',
        'blind_to': 'abrupt shifts that do not change error spacing statistics',
        'failure_mode': 'The paper defaults (beta 0.90/0.95) are extremely '
                        'noise-sensitive; they retrained ~13/14 weeks on a '
                        'stable stream.',
        'fix_applied': 'beta 0.75/0.85 with a longer warm-up.',
    },
    'adwin': {
        'family': 'performance-adjacent',
        'observes': 'the prediction stream (no labels needed)',
        'needs_labels': False,
        'best_at': 'detecting a mean shift in scores with a formal guarantee, '
                   'and adapting its own window size automatically',
        'blind_to': 'degradation that preserves the score distribution — the '
                    'model can rank worse while its score histogram is unchanged',
        'failure_mode': 'Fires on score-distribution turbulence, which is not '
                        'the same as model staleness. On this dataset its '
                        'timing was worse than random at equal cost.',
        'fix_applied': 'Compared against the reference stream rather than an '
                       'early/late split of the current window.',
    },
    'hddm': {
        'family': 'performance',
        'observes': 'bounded error stream via Hoeffding bounds',
        'needs_labels': True,
        'best_at': 'staying sensitive on long-stable models, where DDM stops '
                   'being able to trigger; distribution-free',
        'blind_to': 'same imbalance blindness as DDM if fed a raw 0/1 stream',
        'failure_mode': 'Conservative by design — the price of the guarantee.',
        'fix_applied': 'Fed the class-balanced stream.',
    },
    'shap': {
        'family': 'attribution',
        'observes': 'distribution of per-feature attributions',
        'needs_labels': False,
        'best_at': 'catching a change in what the model *relies on*, even when '
                   'every raw feature marginal is stable — the one label-free '
                   'signal with any purchase on concept drift',
        'blind_to': 'drift that changes the label mapping without changing '
                    'model attributions (it reads the model, not the truth)',
        'failure_mode': 'Expensive, and prefix-sampling a time-ordered window '
                        'samples only its earliest days, making seasonality '
                        'look like permanent drift.',
        'fix_applied': 'Random sampling, FDR correction, effect-size floor; on '
                       'a neural model, gradient x input replaces tree SHAP.',
    },
    'clustering': {
        'family': 'representation',
        'observes': 'joint geometry via K-Means centroid distances',
        'needs_labels': False,
        'best_at': 'multivariate shifts no per-feature test can see — a new '
                   'correlation structure with unchanged marginals',
        'blind_to': 'concept drift; sensitive to k and to feature scaling',
        'failure_mode': 'Without standardisation the largest-scale feature '
                        'dominates every Euclidean distance and the method '
                        'silently measures that one feature.',
        'fix_applied': 'Features standardised on the reference window.',
    },
    'autoencoder': {
        'family': 'representation',
        'observes': 'reconstruction error of the joint distribution',
        'needs_labels': False,
        'best_at': 'novel regions of feature space — genuinely new behaviour '
                   'rather than a shift in existing behaviour',
        'blind_to': 'concept drift; needs enough data to train reliably',
        'failure_mode': 'A KS test on reconstruction errors flags almost any '
                        'batch at realistic sample sizes.',
        'fix_applied': 'Decision on the RMSE z-score (an effect size) alone.',
    },
    'prequential_auc': {
        'family': 'performance',
        'observes': 'windowed out-of-sample ranking quality',
        'needs_labels': True,
        'best_at': 'measuring the thing that actually matters, directly, with '
                   'no proxy and no second model to train',
        'blind_to': 'nothing in principle — but it is strictly reactive. It '
                    'cannot fire until the damage is already in the metric.',
        'failure_mode': 'Noisy on low-fraud weeks; and being reactive, it tends '
                        'to fire *during* turbulence, which is the worst moment '
                        'to freeze a cumulative training set.',
        'fix_applied': 'Bootstrap SE gate plus an absolute floor.',
    },
    'champion_vs_challenger': {
        'family': 'shadow model',
        'observes': 'the realisable benefit of retraining right now',
        'needs_labels': True,
        'best_at': 'answering the decision question directly — not "did drift '
                   'happen" but "would retraining help"',
        'blind_to': 'nothing structural; the limit is cost',
        'failure_mode': 'Scoring the challenger in-sample inflates it by more '
                        'than the trigger threshold, so it fires every week on '
                        'its own overfitting. And it doubles training cost.',
        'fix_applied': 'Out-of-fold challenger predictions and a bootstrap SE '
                       'on the gap.',
    },
}


def method_profile_frame():
    rows = [{'method': m, **{k: v for k, v in p.items()}} for m, p in METHOD_PROFILES.items()]
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════
# 2. Why the agent acted
# ══════════════════════════════════════════════
def action_attributions(agent, state):
    """Gradient x input on the chosen action's logit.

    Positive value = this observation component pushed the agent toward the
    action it selected. Answers "why this action, in this state" rather than
    global importance.
    """
    agent.net.eval()
    x = agent._to_tensor(state).unsqueeze(0).requires_grad_(True)
    logits, _ = agent.net(x)
    action = int(torch.argmax(logits, dim=-1).item())
    logits[0, action].backward()
    return action, (x.grad * x).detach().cpu().numpy()[0]


def explain_episode(agent, env, top_k=4):
    """Week-by-week decision trace with the drivers of each decision."""
    names = state_names()
    rows = []

    state = env.reset()
    while True:
        probs = agent.action_probabilities(state)
        action, attributions = action_attributions(agent, state)

        order = np.argsort(-np.abs(attributions))[:top_k]
        drivers = '; '.join(
            f"{names[i]}={state[i]:.3f} ({'+' if attributions[i] > 0 else '-'}{abs(attributions[i]):.2f})"
            for i in order
        )

        week = env.weeks[env.t]
        next_state, reward, done, _ = env.step(action)
        rows.append({
            'week': week,
            'action': ACTION_NAMES[action],
            'confidence': float(probs[action]),
            'action_probs': {ACTION_NAMES[i]: round(float(p), 3) for i, p in enumerate(probs)},
            'reward': round(reward, 3),
            'top_drivers': drivers,
        })
        state = next_state
        if done:
            break

    return pd.DataFrame(rows)


def policy_summary(agent, env):
    """Which signals the policy relies on overall, averaged across states."""
    names = state_names()
    totals = np.zeros(len(names))
    counts = 0

    state = env.reset()
    while True:
        _, attributions = action_attributions(agent, state)
        totals += np.abs(attributions)
        counts += 1
        state, _, done, _ = env.step(agent.act(state, mode='greedy')[0])
        if done:
            break

    return (pd.DataFrame({'signal': names, 'mean_abs_attribution': totals / max(counts, 1)})
            .sort_values('mean_abs_attribution', ascending=False)
            .reset_index(drop=True))
