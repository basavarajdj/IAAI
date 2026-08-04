"""
Neural Fraud Classifier
=======================

A small PyTorch MLP replacing the LightGBM booster.

Why a neural net here
---------------------
This is not a claim that an MLP beats gradient boosting on tabular fraud — it
usually does not. The reason to switch is that **a GBDT cannot be partially
updated**. Adding trees to an existing booster is not the same operation as
adapting it, and there is no principled "fine-tune on the last month" for a
fixed forest.

That matters because the drift-adaptation agent (see rl_env.py) has an action
space of {do nothing, partial update, full retrain, adjust ensemble}. With a
GBDT, "partial update" does not exist and the action space collapses to
{retrain, don't}. A differentiable model makes the middle ground real: you can
take a few gradient steps on recent data for a fraction of the cost of a full
retrain, and — crucially — you can then *measure* what that cheap update cost
you in forgetting.

Design notes
------------
* **Frozen standardisation.** The scaler is fitted once, on the data the model
  was first trained on, and carried through every partial update. This mirrors
  the frozen-encoder rule in feature_engineering.py: adaptation changes weights,
  not the representation, so two model versions stay comparable.
* **Class imbalance** is handled with a positive-class weight in the loss rather
  than resampling, so every row is still seen.
* **Temporal validation split** (last 20% by row order, which is time order) for
  early stopping and threshold calibration — a random split leaks same-day rows
  across the boundary.
* **`partial_fit` returns a new model** rather than mutating in place. The
  lattice in model_lattice.py needs to branch from a parent model repeatedly,
  which in-place updates would make impossible.
"""

import copy
import logging

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, roc_auc_score

logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)


class TabularNet(nn.Module):
    """Three-layer MLP. Deliberately small — the point is adaptability, not capacity."""

    def __init__(self, n_features, hidden=(256, 128, 64), dropout=0.2):
        super().__init__()
        layers, dim = [], n_features
        for h in hidden:
            layers += [nn.Linear(dim, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            dim = h
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)          # logits


class NeuralFraudModel:
    """Trainable, fine-tunable fraud classifier with a calibrated threshold."""

    def __init__(self, n_features, hidden=(256, 128, 64), dropout=0.2,
                 lr=1e-3, batch_size=1024, max_epochs=30, patience=5, seed=42):
        torch.manual_seed(seed)
        self.n_features = n_features
        self.net = TabularNet(n_features, hidden, dropout).to(DEVICE)
        self.lr = lr
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed

        self.mean_ = None
        self.std_ = None
        self.decision_threshold = 0.5

    # ── internals ────────────────────────────────────────────────
    def _scale(self, X):
        arr = np.asarray(X, dtype=np.float32)
        return (arr - self.mean_) / self.std_

    def _tensor(self, X):
        return torch.from_numpy(self._scale(X)).to(DEVICE)

    def _run_epoch(self, X, y, optimizer, loss_fn):
        self.net.train()
        order = torch.randperm(len(X), device=DEVICE)
        total = 0.0
        for i in range(0, len(X), self.batch_size):
            idx = order[i:i + self.batch_size]
            if len(idx) < 2:                     # BatchNorm needs >1 row
                continue
            optimizer.zero_grad()
            loss = loss_fn(self.net(X[idx]), y[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        return total / max(len(X), 1)

    def _pos_weight(self, y):
        n_pos = float(y.sum())
        n_neg = float(len(y) - n_pos)
        return torch.tensor([n_neg / max(n_pos, 1.0)], device=DEVICE)

    # ── public API ───────────────────────────────────────────────
    def fit(self, X, y, verbose=False):
        """Full training run, with a temporal holdout for early stopping."""
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0

        split = int(len(X) * 0.8)
        Xtr, Xva = self._tensor(X[:split]), self._tensor(X[split:])
        ytr = torch.from_numpy(y[:split]).to(DEVICE)
        yva_np = y[split:]

        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight(y[:split]))

        best_auc, best_state, waited = -np.inf, None, 0
        for epoch in range(self.max_epochs):
            self._run_epoch(Xtr, ytr, optimizer, loss_fn)
            auc = self._auc(yva_np, self._predict_tensor(Xva))
            if auc > best_auc + 1e-5:
                best_auc, best_state, waited = auc, copy.deepcopy(self.net.state_dict()), 0
            else:
                waited += 1
                if waited >= self.patience:
                    break
            if verbose:
                logger.info(f"    epoch {epoch:2d}  val AUC {auc:.4f}")

        if best_state is not None:
            self.net.load_state_dict(best_state)

        self._calibrate_threshold(yva_np, self._predict_tensor(Xva))
        logger.info(f"  trained on {len(X):,} rows — val AUC {best_auc:.4f} "
                    f"@ threshold {self.decision_threshold:.2f}")
        return self

    def partial_fit(self, X, y, epochs=5, lr=None):
        """Fine-tune a COPY of this model on recent data.

        Returns a new model, leaving `self` untouched, so a parent model can be
        branched from more than once. The scaler is inherited rather than refit:
        re-standardising on a small recent window would shift the input
        distribution the existing weights were trained against, which is a
        representation change disguised as an update.
        """
        child = self.clone()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if len(X) < 32 or len(np.unique(y)) < 2:
            return child

        Xt = child._tensor(X)
        yt = torch.from_numpy(y).to(DEVICE)
        optimizer = torch.optim.Adam(child.net.parameters(), lr=lr or self.lr * 0.1)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=child._pos_weight(y))

        for _ in range(epochs):
            child._run_epoch(Xt, yt, optimizer, loss_fn)

        child._calibrate_threshold(y, child._predict_tensor(Xt))
        return child

    def clone(self):
        child = NeuralFraudModel.__new__(NeuralFraudModel)
        child.__dict__.update(self.__dict__)
        child.net = copy.deepcopy(self.net)
        return child

    def predict(self, X):
        """Fraud probability per row."""
        return self._predict_tensor(self._tensor(X))

    def gradient_attributions(self, X, max_rows=500):
        """Per-feature attribution via gradient x input.

        The differentiable-model analogue of the SHAP detector. Real SHAP on a
        neural net needs KernelExplainer (far too slow to run for every
        reference/window pair) or DeepExplainer (still costly); gradient x input
        is a standard, cheap attribution that captures the same thing the SHAP
        drift detector actually uses — *whether the model's reliance on a
        feature has shifted* — rather than an exact Shapley decomposition.
        """
        arr = np.asarray(X, dtype=np.float32)[:max_rows]
        x = self._tensor(arr).requires_grad_(True)
        self.net.eval()
        self.net(x).sum().backward()
        return (x.grad * x).detach().cpu().numpy()

    # ── evaluation helpers ───────────────────────────────────────
    def _predict_tensor(self, Xt):
        self.net.eval()
        with torch.no_grad():
            return torch.sigmoid(self.net(Xt)).cpu().numpy()

    @staticmethod
    def _auc(y, probs):
        return float(roc_auc_score(y, probs)) if len(np.unique(y)) > 1 else 0.5

    def _calibrate_threshold(self, y, probs):
        if len(np.unique(y)) < 2:
            return
        scores = [f1_score(y, (probs >= t).astype(int), zero_division=0) for t in THRESHOLD_GRID]
        self.decision_threshold = float(THRESHOLD_GRID[int(np.argmax(scores))])

    def evaluate(self, X, y):
        """(auc, f1) at the calibrated threshold."""
        probs = self.predict(X)
        y = np.asarray(y)
        f1 = f1_score(y, (probs >= self.decision_threshold).astype(int), zero_division=0)
        return self._auc(y, probs), float(f1), probs
