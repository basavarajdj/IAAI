"""
Model Training Module — LightGBM GBDT Classifier
Provides train, evaluate, save/load, and optional confusion matrix plotting utilities
for fraud detection models.
"""

import os
import pickle
import logging
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Default LightGBM hyper-parameters
# ──────────────────────────────────────────────
DEFAULT_PARAMS = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'auc',
    'n_jobs': -1,
    'learning_rate': 0.01,
    'num_leaves': 64,
    'max_depth': -1,
    'tree_learner': 'serial',
    'colsample_bytree': 0.7,
    'subsample_freq': 1,
    'subsample': 0.7,
    'n_estimators': 500,
    'min_data_in_leaf': 20,
    'verbose': -1,
    'seed': 42,
    'predict_disable_shape_check': True,
    'is_unbalance': True,
}

# Early stopping must be loose enough for the learning rate. At lr=0.01 the
# first few dozen boosting rounds barely move validation AUC, so a patience of
# 5 rounds terminates training at iteration 1 — the model that produced the
# original reports was a single tree, which is why its F1 was exactly 0.000 on
# every window. Patience is scaled to the learning rate instead.
EARLY_STOPPING_ROUNDS = 100

# Decision threshold search grid for converting probabilities to labels.
THRESHOLD_GRID = np.linspace(0.05, 0.95, 91)


# ──────────────────────────────────────────────
# Custom F1 eval metric for LightGBM callbacks
# ──────────────────────────────────────────────
def _lgb_f1_score(preds, data):
    """LightGBM custom evaluation function that computes F1 score."""
    labels = data.get_label()
    preds_binary = (preds > 0.5).astype(int)
    return 'f1', f1_score(labels, preds_binary), True


# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
def train_model(X_train, y_train, X_valid=None, y_valid=None, params=None, seed=None):
    """Train a LightGBM binary classifier with optional validation split.

    If validation data is not supplied, a stratified 80/20 split is created
    automatically from the provided training set.

    Args:
        X_train: Training features (DataFrame or array-like).
        y_train: Training labels.
        X_valid: Optional validation features.
        y_valid: Optional validation labels.
        params: Optional dict of LightGBM parameters (defaults to DEFAULT_PARAMS).

    Returns:
        Trained LightGBM Booster.
    """
    logger.info(f"Training LightGBM model on {len(X_train)} samples...")

    if params is None:
        params = DEFAULT_PARAMS.copy()
    if seed is not None:
        params = {**params, 'seed': seed, 'bagging_seed': seed, 'feature_fraction_seed': seed}

    # Temporal validation split if none provided.
    #
    # A random stratified split is the wrong control for a time-series problem:
    # it places rows from the same day (often the same card, via the UID-derived
    # features) on both sides of the split, so validation AUC is optimistic and
    # early stopping picks a model tuned to a leak. The caller passes rows in
    # chronological order, so the last 20% is a genuine forward holdout.
    if X_valid is None or y_valid is None:
        split = int(len(X_train) * 0.8)
        X_train, X_valid = X_train.iloc[:split], X_train.iloc[split:]
        y_train, y_valid = y_train.iloc[:split], y_train.iloc[split:]
        logger.info(
            f"Temporal split: {len(X_train)} train / {len(X_valid)} validation (last 20% by time)."
        )
        if y_valid.nunique() < 2 or y_train.nunique() < 2:
            # A degenerate tail (e.g. a tiny final window with no fraud) makes
            # early stopping meaningless; fall back to a stratified split.
            X_all = pd.concat([X_train, X_valid])
            y_all = pd.concat([y_train, y_valid])
            X_train, X_valid, y_train, y_valid = train_test_split(
                X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
            )
            logger.info("Temporal tail lacked both classes — used a stratified split instead.")

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid)

    clf = lgb.train(
        params,
        dtrain,
        valid_sets=[dtrain, dvalid],
        valid_names=['train', 'valid'],
        callbacks=[
            lgb.log_evaluation(period=1000),
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
        ],
        feval=_lgb_f1_score,
    )

    # Calibrate the decision threshold on the validation split and store it on
    # the booster. A fixed 0.5 cut on a 3.5%-prevalence ranking model puts
    # essentially no mass above the line, giving F1 = 0 — and, more damagingly
    # for this study, an error stream that is constant at the prevalence rate,
    # which is what DDM/EDDM/HDDM are supposed to be monitoring.
    threshold = tune_decision_threshold(clf, X_valid, y_valid)
    # LightGBM's Booster has no attr API (unlike XGBoost), but it is a plain
    # Python object and the attribute survives pickling, which is how the
    # registry persists versions.
    clf.decision_threshold = threshold

    logger.info(f"Model training complete ({clf.num_trees()} trees).")
    auc_val, f1_val, _ = evaluate(clf, X_valid, y_valid)
    logger.info(f"Validation — AUC: {auc_val:.4f}, F1: {f1_val:.4f} @ threshold {threshold:.3f}")
    return clf


def tune_decision_threshold(model, X_valid, y_valid, grid=None):
    """Pick the probability cut-off that maximises validation F1.

    Returns 0.5 when the validation set has a single class (nothing to tune).
    """
    probs = model.predict(X_valid, predict_disable_shape_check=True)
    y = np.asarray(y_valid)
    if len(np.unique(y)) < 2:
        return 0.5

    grid = THRESHOLD_GRID if grid is None else grid
    scores = [f1_score(y, (probs >= t).astype(int), zero_division=0) for t in grid]
    best = float(grid[int(np.argmax(scores))])
    logger.info(f"Tuned decision threshold: {best:.3f} (validation F1 {max(scores):.4f})")
    return best


def get_decision_threshold(model, default=0.5):
    """Read the calibrated threshold stored on a booster, if present."""
    return float(getattr(model, 'decision_threshold', default))


# ──────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────
def evaluate(model, X, y, threshold=None):
    """Compute AUC and F1 for a LightGBM model on the given data.

    ``threshold`` defaults to the value calibrated during training rather than
    a hard-coded 0.5, so F1 reflects the operating point the model is actually
    deployed at.

    Returns:
        Tuple of (auc, f1, raw_probabilities).
    """
    probs = model.predict(X, predict_disable_shape_check=True)
    if threshold is None:
        threshold = get_decision_threshold(model)

    auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else 0.5
    preds = (probs >= threshold).astype(int)
    f1 = f1_score(y, preds, zero_division=0)

    logger.info(f"Evaluation — AUC: {auc:.4f}, F1: {f1:.4f} @ {threshold:.3f}")
    return auc, f1, probs


# ──────────────────────────────────────────────
# Persistence
# ──────────────────────────────────────────────
def save_model(model, path='fraud_model.pkl'):
    """Pickle-serialize the model to disk."""
    with open(path, 'wb') as f:
        pickle.dump(model, f)
    logger.info(f"Model saved to {path}")


def load_model(path='fraud_model.pkl'):
    """Load a pickled model from disk. Returns None if not found."""
    if os.path.exists(path):
        logger.info(f"Loading model from {path}...")
        with open(path, 'rb') as f:
            return pickle.load(f)
    logger.warning(f"Model file {path} not found.")
    return None


# ──────────────────────────────────────────────
# Visualization (standalone utility — not called during training)
# ──────────────────────────────────────────────
def plot_confusion_matrix(y_true, y_pred, save_path='confusion_matrix.png'):
    """Plot and save a confusion matrix heatmap."""
    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='YlOrRd',
        xticklabels=['Legit', 'Fraud'],
        yticklabels=['Legit', 'Fraud'],
    )
    plt.title('LightGBM Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {save_path}")
