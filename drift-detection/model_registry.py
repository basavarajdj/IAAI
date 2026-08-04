"""
Shared Model Registry
=====================

The problem this replaces
-------------------------
The original pipeline gave each of the N drift detectors its own private
champion model. That meant:

    * N models trained at the baseline, on **identical data with identical
      hyper-parameters** — N independent fits of the same thing, differing only
      by LightGBM's internal randomness.
    * When k detectors flagged drift in the same week, k separate models were
      trained, again on identical cumulative data.

Over a 14-week replay with 10 detectors this can reach 10 + 10x14 = 150 fits,
of which at most 15 are distinct. Beyond the wasted compute, it makes the
comparison between detectors *unsound*: two detectors that retrain in exactly
the same week end up with different models, so any difference in their
downstream AUC is partly seed noise rather than a consequence of their
retraining policy.

The model this implements
-------------------------
A model version is identified by **the data it was trained on**, which in a
cumulative-window replay is fully determined by the week boundary:

    version 0   → baseline window (first 90 days)
    version w   → all data from the start through week w

The registry trains at most one model per week, no matter how many detectors
requested it, and hands the *same* Booster object to all of them. Each detector
holds only a pointer — ``method_versions[detector] = version_id``. A detector
that does not flag drift in week w simply keeps pointing at whatever version it
last adopted, which is the "use the old version of the model" requirement.

The upper bound on distinct models is therefore ``1 + n_weeks`` (one baseline
plus at most one per week), independent of how many detectors are being
compared — 15 for a 14-week replay, versus 150 before.

Because every detector that retrains in week w now shares byte-identical
weights, differences in their measured performance are attributable purely to
*when* they chose to retrain. That is the comparison the study is trying to
make.
"""

import logging
import os
import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

BASELINE_VERSION = 0


@dataclass
class ModelVersion:
    """One trained model plus the reference state captured at training time."""

    version_id: int
    week: int
    model: Any
    n_train_rows: int
    train_auc: float
    train_f1: float
    train_predictions: np.ndarray
    # Which detectors adopted this version, and when
    adopted_by: List[str] = field(default_factory=list)
    # Detector-specific artefacts fitted against this version's training data
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return 'v0 (baseline)' if self.version_id == BASELINE_VERSION else f'v{self.version_id} (week {self.week})'

    def summary(self) -> dict:
        return {
            'version_id': self.version_id,
            'week': self.week,
            'label': self.label,
            'n_train_rows': int(self.n_train_rows),
            'train_auc': float(self.train_auc),
            'train_f1': float(self.train_f1),
            'adopted_by': list(self.adopted_by),
        }


class ModelRegistry:
    """Content-addressed store of model versions, keyed by training week.

    Args:
        train_fn: ``(X, y) -> Booster``.
        evaluate_fn: ``(model, X, y) -> (auc, f1, probs)``.
    """

    def __init__(self, train_fn, evaluate_fn, methods, models_dir='models'):
        self.train_fn = train_fn
        self.evaluate_fn = evaluate_fn
        self.methods = list(methods)
        self.models_dir = models_dir

        self.versions: Dict[int, ModelVersion] = {}
        self.method_versions: Dict[str, int] = {}
        # week -> version_id, so a second detector asking for the same week
        # is served from cache rather than triggering another fit
        self._week_index: Dict[int, int] = {}
        self._next_version_id = 0
        self.train_calls = 0
        self.cache_hits = 0
        self.adoption_log: List[dict] = []

    # ── construction ─────────────────────────────────────────────
    def register_baseline(self, X, y):
        """Train the single shared baseline model and point every method at it."""
        logger.info(f"Training the shared baseline model on {len(X)} rows "
                    f"(one fit for all {len(self.methods)} detectors)...")
        model = self.train_fn(X, y)
        self.train_calls += 1
        auc, f1, probs = self.evaluate_fn(model, X, y)

        version = ModelVersion(
            version_id=BASELINE_VERSION,
            week=0,
            model=model,
            n_train_rows=len(X),
            train_auc=float(auc),
            train_f1=float(f1),
            train_predictions=np.asarray(probs),
            adopted_by=list(self.methods),
        )
        self.versions[BASELINE_VERSION] = version
        self._week_index[0] = BASELINE_VERSION
        self._next_version_id = 1
        self.method_versions = {m: BASELINE_VERSION for m in self.methods}

        logger.info(f"  baseline {version.label} — AUC {auc:.4f}, F1 {f1:.4f}")
        return version

    # ── the core operation ───────────────────────────────────────
    def get_or_train(self, week, X, y):
        """Return the model version for ``week``, training it at most once.

        The first detector to request week w pays for the fit; every other
        detector that drifted in the same week is served the identical object
        from cache.
        """
        if week in self._week_index:
            self.cache_hits += 1
            version = self.versions[self._week_index[week]]
            logger.info(f"  reusing {version.label} — already trained this week "
                        f"(cache hit #{self.cache_hits}; no second fit)")
            return version, False

        version_id = self._next_version_id
        logger.info(f"  training new model v{version_id} on {len(X)} cumulative rows "
                    f"(week {week})...")
        model = self.train_fn(X, y)
        self.train_calls += 1
        auc, f1, probs = self.evaluate_fn(model, X, y)

        version = ModelVersion(
            version_id=version_id,
            week=week,
            model=model,
            n_train_rows=len(X),
            train_auc=float(auc),
            train_f1=float(f1),
            train_predictions=np.asarray(probs),
        )
        self.versions[version_id] = version
        self._week_index[week] = version_id
        self._next_version_id += 1

        logger.info(f"  {version.label} — AUC {auc:.4f}, F1 {f1:.4f}")
        return version, True

    def adopt(self, method, version, week, reason=None):
        """Point ``method`` at ``version``. Methods not calling this keep theirs."""
        previous = self.method_versions.get(method)
        self.method_versions[method] = version.version_id
        if method not in version.adopted_by:
            version.adopted_by.append(method)
        self.adoption_log.append({
            'method': method,
            'week': week,
            'from_version': previous,
            'to_version': version.version_id,
            'reason': reason,
        })
        logger.info(f"  [{method}] adopted {version.label} (was v{previous})")

    # ── lookups ──────────────────────────────────────────────────
    def version_for(self, method) -> ModelVersion:
        return self.versions[self.method_versions[method]]

    def model_for(self, method):
        return self.version_for(method).model

    def retrain_count(self, method) -> int:
        """How many times this method adopted a *new* version after baseline."""
        return sum(1 for a in self.adoption_log if a['method'] == method)

    def distinct_models(self) -> int:
        return len(self.versions)

    def sharing_summary(self) -> dict:
        """Evidence of the compute saved, for reporting."""
        naive = len(self.methods) + sum(
            len(v.adopted_by) for vid, v in self.versions.items() if vid != BASELINE_VERSION
        )
        return {
            'distinct_models_trained': self.distinct_models(),
            'actual_train_calls': self.train_calls,
            'cache_hits': self.cache_hits,
            'naive_per_method_train_calls': naive,
            'training_reduction_ratio': float(naive / max(self.train_calls, 1)),
            'versions': [v.summary() for v in self.versions.values()],
            'method_versions': dict(self.method_versions),
            'adoption_log': list(self.adoption_log),
        }

    # ── persistence ──────────────────────────────────────────────
    def save_all(self):
        """Persist every distinct version — at most ``1 + n_weeks`` files."""
        os.makedirs(self.models_dir, exist_ok=True)
        paths = {}
        for vid, version in self.versions.items():
            path = os.path.join(self.models_dir, f'model_v{vid}_week{version.week}.pkl')
            with open(path, 'wb') as f:
                pickle.dump(version.model, f)
            paths[vid] = path
        logger.info(f"Persisted {len(paths)} distinct model versions to {self.models_dir}/")
        return paths
