"""
Model Lattice — precomputed adaptation outcomes
===============================================

The problem this solves
-----------------------
PPO needs thousands of episodes. A 14-week replay is *one* trajectory, and
training a model inside the RL loop would mean tens of thousands of fits.

The way out is that the reachable model space is small and enumerable. Under
this action set, the model in force at any time is fully determined by three
numbers:

    (last_full_retrain_week, last_partial_update_week, ensemble_weight)

So we enumerate every reachable model once, cache what each scores on every
future week, and the RL environment becomes a lookup table. Episodes then cost
nothing and PPO can run properly.

Design choices that keep the space small — and why they are defensible
---------------------------------------------------------------------
**Partial updates are not chained.** A partial update always fine-tunes from the
*last full-retrain model*, on a recent window, rather than from the previous
partial. Chaining would make the model depend on the entire action history
(4^14 possibilities) and destroy the Markov property the RL agent relies on.
It also compounds catastrophic forgetting: each fine-tune on recent-only data
pulls further from the original distribution, and the drift is unrecoverable
without a full retrain. Re-deriving from the last full model bounds that damage
by construction, which is the honest engineering choice regardless of the RL.

**The ensemble blends the current model with the baseline model**, with weight
`alpha` on the current one. This is the cheap hedge: if a partial update has
overfitted to a noisy month, shifting weight back toward the stable baseline
recovers most of the loss without any training. It is also the mechanism that
makes forgetting *measurable* — the gap between alpha=1 and the best alpha tells
you how much the recent update cost on the broader distribution.
"""

import logging

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from neural_model import NeuralFraudModel

logger = logging.getLogger(__name__)

ALPHAS = (0.0, 0.25, 0.50, 0.75, 1.0)   # weight on the current (adapted) model
PARTIAL_WINDOW = 4                       # weeks of recent data a partial update sees


class ModelLattice:
    """Enumerates every reachable model and caches its weekly performance."""

    def __init__(self, ref_X, ref_y, weekly_X, weekly_y,
                 partial_window=PARTIAL_WINDOW, partial_epochs=5, seed=42):
        self.ref_X, self.ref_y = ref_X, ref_y
        self.weekly_X, self.weekly_y = weekly_X, weekly_y
        self.weeks = sorted(weekly_X)
        self.partial_window = partial_window
        self.partial_epochs = partial_epochs
        self.seed = seed

        self.full_models = {}       # full_week -> NeuralFraudModel
        self.preds = {}             # (full_week, partial_week|None) -> {week: probs}
        self.auc = {}               # (full_week, partial_week|None, alpha) -> {week: auc}
        self.f1 = {}
        self.n_models_trained = 0

    # ── data assembly ────────────────────────────────────────────
    def _cumulative(self, through_week):
        """Baseline window plus every monitored week up to and including `through_week`."""
        Xs, ys = [self.ref_X], [self.ref_y]
        for w in self.weeks:
            if w <= through_week:
                Xs.append(self.weekly_X[w])
                ys.append(self.weekly_y[w])
        return np.vstack([np.asarray(x, dtype=np.float32) for x in Xs]), np.concatenate(ys)

    def _recent(self, through_week):
        """The last `partial_window` weeks ending at `through_week`."""
        lo = through_week - self.partial_window + 1
        Xs, ys = [], []
        for w in self.weeks:
            if lo <= w <= through_week:
                Xs.append(self.weekly_X[w])
                ys.append(self.weekly_y[w])
        if not Xs:
            return None, None
        return np.vstack([np.asarray(x, dtype=np.float32) for x in Xs]), np.concatenate(ys)

    # ── construction ─────────────────────────────────────────────
    def build(self):
        n_features = np.asarray(self.ref_X).shape[1]

        logger.info(f"Training {len(self.weeks) + 1} full-retrain models...")
        for f in [0] + self.weeks:
            X, y = self._cumulative(f)
            model = NeuralFraudModel(n_features, seed=self.seed).fit(X, y)
            self.full_models[f] = model
            self.n_models_trained += 1
            self._score(model, key=(f, None), valid_from=f)

        logger.info("Deriving partial-update models (fine-tune last full model on recent data)...")
        for f in self.full_models:
            for p in self.weeks:
                if p <= f:
                    continue
                Xr, yr = self._recent(p)
                if Xr is None:
                    continue
                child = self.full_models[f].partial_fit(Xr, yr, epochs=self.partial_epochs)
                self.n_models_trained += 1
                self._score(child, key=(f, p), valid_from=p)

        logger.info(f"Lattice built: {self.n_models_trained} models, "
                    f"{len(self.auc)} (model, alpha) states cached.")
        return self

    def _score(self, model, key, valid_from):
        """Cache predictions and blended AUC/F1 for every week after `valid_from`."""
        probs = {w: model.predict(self.weekly_X[w]) for w in self.weeks if w > valid_from}
        self.preds[key] = probs

        base = self.preds.get((0, None), {})
        threshold = model.decision_threshold

        for alpha in ALPHAS:
            auc_row, f1_row = {}, {}
            for w, p_cur in probs.items():
                p_base = base.get(w)
                blended = p_cur if p_base is None else alpha * p_cur + (1 - alpha) * p_base
                y = np.asarray(self.weekly_y[w])
                if len(np.unique(y)) > 1:
                    auc_row[w] = float(roc_auc_score(y, blended))
                f1_row[w] = float(f1_score(y, (blended >= threshold).astype(int), zero_division=0))
            self.auc[(key[0], key[1], alpha)] = auc_row
            self.f1[(key[0], key[1], alpha)] = f1_row

    # ── lookup ───────────────────────────────────────────────────
    def performance(self, full_week, partial_week, alpha, week):
        """(auc, f1) of this model state on `week`; None if not yet available."""
        auc = self.auc.get((full_week, partial_week, alpha), {}).get(week)
        f1 = self.f1.get((full_week, partial_week, alpha), {}).get(week)
        return (auc, f1) if auc is not None else (None, None)

    def forgetting_cost(self, full_week, partial_week, week):
        """AUC lost by trusting the partial update fully instead of hedging.

        `best_alpha_auc - alpha_1_auc`. A positive value means the fine-tune
        overfitted the recent window and blending back toward the baseline would
        have scored better — i.e. a direct, per-week measurement of catastrophic
        forgetting rather than an assumption about it.
        """
        scores = [(a, self.auc.get((full_week, partial_week, a), {}).get(week)) for a in ALPHAS]
        scores = [(a, s) for a, s in scores if s is not None]
        if not scores:
            return 0.0, 1.0
        best_alpha, best = max(scores, key=lambda t: t[1])
        full_trust = dict(scores).get(1.0, best)
        return float(best - full_trust), float(best_alpha)
