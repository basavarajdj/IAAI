# From 432 Raw Columns to a Trained Model: Feature Engineering and Modelling in Detail

This is a companion to [PAPER.md](PAPER.md) and
[DRIFT_ANALYSIS_EXPLAINED.md](DRIFT_ANALYSIS_EXPLAINED.md), zoomed in on two
things those documents only summarise: **exactly** how the raw IEEE-CIS
columns become the 113 features the model trains on, and **exactly** how the
model is trained, calibrated, and (for the RL agent) made partially-updatable.
Every number below is read off the actual pipeline
([feature_engineering.py](../feature_engineering.py),
[model_training.py](../model_training.py), [neural_model.py](../neural_model.py)),
not estimated.

---

## 1. Where the columns start

| Source | File | Columns | Notes |
|---|---|---|---|
| `train_transaction.csv` | 590,540 rows | 394 | includes `TransactionID`, `isFraud` |
| `train_identity.csv` | 144,233 rows | 41 | includes `TransactionID`; joined on it, left join (most transactions have no identity row) |
| **Merged** | `pd.merge(..., on='TransactionID', how='left')` | **434** | `TransactionID` (key) + `isFraud` (target) + **432 predictor columns** (392 from transaction, 40 from identity) |

Those 432 predictor columns fall into a few large, differently-shaped blocks
that the engineering pipeline treats very differently:

| Block | Raw columns | What they are |
|---|---|---|
| `V1`–`V339` (Vesta) | 339 | Vesta's own engineered features, largely opaque, highly collinear |
| `C1`–`C14` (counters) | 14 | Counts of things associated with the card/address (exact semantics undisclosed) |
| `D1`–`D15` (timedeltas) | 15 | Days since some prior event per entity |
| `M1`–`M9` (match flags) | 9 | Whether name/address on the card matches the billing/shipping info |
| `id_01`–`id_38`, `DeviceType`, `DeviceInfo` | 40 | Device and network fingerprint, from the identity table |
| Everything else | 15 | `TransactionAmt`, `ProductCD`, `card1`–`card6`, `addr1`, `addr2`, `dist1`, `dist2`, `P_emaildomain`, `R_emaildomain`, `TransactionDT` |

339 of the 432 raw columns — 78% — are the Vesta block alone. That single
fact drives a lot of what the engineering pipeline does: with 339 opaque,
correlated float columns, the right move is aggressive compression, not
per-column treatment.

---

## 2. The engineering pipeline, stage by stage

All of this lives in `FeatureEngineer._build()`
([feature_engineering.py:299](../feature_engineering.py#L299)), which runs the
same eleven stages whether it is *fitting* (on the 90-day baseline) or
*transforming* (replaying frozen state on a later window) — see §3 for why
that distinction is the most important design decision in this file.

### 2.1 Distance (`_distance_features`)
`dist1` and `dist2` (billing-to-shipping distance, two variants with
different missingness patterns) collapse to one log-scaled feature,
`_log_dist_1_2 = log1p(dist1 if present else dist2)`.

### 2.2 Date / time (`_date_features`)
`TransactionDT` (a seconds offset) becomes:
- `_days` — day index, used internally by the D-column and redundancy stages, then dropped from the final matrix (see §2.11)
- `_hour_cos`, `_hour_sin` — cyclic encoding of hour-of-day, so 23:00 and 00:00 are numerically adjacent
- `_weekday__hour` — a combined weekday × hour-of-day categorical

The cyclic period is **frozen at fit time** (`hour_period = 23.0`, not
recomputed per window) — deriving it from `.max()` on each window would
silently rescale every hour encoding on a quiet week with no late-night
transactions.

### 2.3 Combination keys (`_combination_features`)
Two categorical crosses: `_P_emaildomain__ProductCD` (email domain × product
code) and `_card3__card5` (card issuer country × card sub-type). These exist
because the individual columns are weak but their interaction is informative,
and because `_P_emaildomain__ProductCD` becomes a grouping key for §2.6.

### 2.4 Identity / causal sequence (`add_causal_sequence_features`, called once globally)
An approximate entity key is built — not a real card ID (none exists in the
data) but a proxy: `_uid2 = card1 __ addr1 __ (day_index - D1) __ P_emaildomain`.
Grouped by `_uid2`, over the **entire chronologically-sorted frame** (not
per-window, so the lag horizon isn't truncated to 7 days):
- `_day_lag_uid2` — seconds since this entity's previous transaction
- `_amount_lag_pct_uid2` — |percent change| in amount vs. this entity's previous transaction
- `_uid2_seq_index` — this transaction's position in the entity's history (a scale-free tenure proxy)

Each of these only ever looks backward in time, so despite being an
"engineered" feature it is not leakage — the prior transaction for a card is
genuinely available at inference time.

### 2.5 Amount (`_amount_features`)
`TransactionAmt` produces `_amount_decimal` (the cents portion, e.g. 49.99 →
990), `_amount_decimal_len` (how many decimal places were written — a
laundering/structuring signal), and `_log_amount`.

### 2.6 Aggregates (`_aggregate_features`) — the one place a map is *learned*
Group-level statistics of `TransactionAmt` per `_P_emaildomain__ProductCD`:
max, mean, variance. The map is learned once on the reference window
(`fit=True`) and **replayed** on later windows; a group unseen at fit time
falls back to the reference's *global* aggregate rather than 0, so it doesn't
sit on a different scale from every mapped row. Two "was this ever seen at
fit time" indicators (`_unseen__uid2`, `_unseen__P_emaildomain__ProductCD`)
are emitted explicitly rather than left as a silent fill artifact — both are
later dropped by the redundancy filter (§2.11) because almost every row is
"seen" in this particular dataset, but the mechanism generalises to datasets
where that isn't true.

`_uid2` is deliberately **not** used as a grouping key here (see the
docstring at [feature_engineering.py:184](../feature_engineering.py#L184)): it
is close to a unique entity id, so an aggregate keyed on it degenerates to a
single fallback value for almost every row of any future window — a null
experiment (Section 3.1 of PAPER.md) measured this reaching KS D = 0.61 with
zero real drift.

### 2.7 Count / frequency encoding (`_count_encoding`)
For each of `C1`–`C14` plus `_P_emaildomain__ProductCD`: the **relative**
frequency of each value, learned once on the reference window
(`value_counts / len(df)`) and replayed. Two corrections were necessary
before this was safe to use as a drift-monitored feature (both documented
in-line and covered in PAPER.md §3.1):
- **Relative, not raw, counts** — a raw count is proportional to the window's
  row count, so a category with a perfectly stable *rate* looks ~13× rarer
  in a 7-day window than in the 90-day baseline.
- **Frozen, not per-window** — even a per-window *relative* frequency is
  still resolution-limited by 1/n; an earlier version emitting a live
  `_freq_now_` feature reached KS D = 0.88 under the null experiment with no
  real drift present. It was removed; only the frozen `_freq_ref_*` variant
  remains.

`_all_na` (count of missing fields in the row) is also computed here.

### 2.8 C-columns → pattern features (`_c_features`)
The 14 raw `C1`–`C14` columns are **replaced** (not kept) by three derived
features — `_ccols_nonzero` (how many are non-zero), `_ccols_sum`, and
`_ccols_0_bin` (a string encoding *which* of the 14 are zero, e.g.
`"01001..."` — the pattern, not just the count, since different zero-patterns
can mean different things). The raw columns are dropped after this stage.

### 2.9 D-columns (`_d_features`)
The 15 raw `D1`–`D15` are kept, with only missing values filled to 0.
`_dcol_na` (count missing) and `_dcols_na_bin` (missingness pattern, same
idea as `_ccols_0_bin`) are added alongside.

**A bug found and fixed while writing this document.** An earlier version of
this method computed `D_i ← D_i.fillna(0) - _days`, apparently intending to
turn each "days since some event" column into a delta relative to the
current row's absolute day index. This was wrong, and empirically so: raw
D-columns are already relative and roughly *stationary* across the six-month
replay (D2's raw weekly mean stays in the 160–190 range throughout), while
`_days` — the row's absolute day offset from the start of the dataset —
grows *linearly*, from ~40 in the reference window to ~180 by week 14.
Subtracting a monotonically growing quantity from a stationary one
manufactures a monotonic trend out of nothing: the transformed `D2`'s weekly
mean fell in a straight line from +40 at the reference window to −74 by week
14, and every distributional detector correctly identified it as "drifting"
in **14 of 14 weeks**, for the entire replay, on both `D2` and `D15` once
they entered the top-20 monitoring set.

This is not covariate shift and it is not seasonality — it is the same
"monotone-in-time proxy" failure mode that `TransactionDT`/`TransactionID`
are excluded from the matrix for entirely (§2.13, `DROP_COLS`), just
introduced here by a transformation on an otherwise-legitimate column rather
than by leaving a raw timestamp in. It was found by comparing the raw and
transformed weekly means side by side (the same diagnostic that would catch
it in any future column: if a "relative" feature's mean moves in lockstep
with elapsed dataset time, the relativity isn't real). The fix is simply to
drop the subtraction — see PAPER.md §2.1.1 for the measured before/after
effect on the drift replay.

### 2.10 M-columns → missingness pattern (`_m_features`)
`_mcols_na_bin` — the missingness pattern across `M1`–`M9`, as one string
column. The raw M-columns are *also* kept (factorized in §2.12), so this adds
one derived feature on top of the 9 raw ones.

### 2.11 V-columns → PCA compression (`_v_features`)
The 339-column Vesta block is compressed to **4 features**:
`_vcols_dec0`, `_vcols_dec1` (2 principal components — `n_v_components=2`,
fit with a `MinMaxScaler` + `PCA` learned once and reused), `_vcols_na`
(count missing), `_vcols_sum` (sum over a fixed subset of the V-columns,
excluding seven that were found to behave differently — `V144, V145, V150,
V151, V159, V160, V307` — chosen once at fit time). The raw `V1`–`V339` are
dropped entirely after this. The PCA rotation is **frozen**: refitting it per
window would let a component's sign or orientation flip arbitrarily between
two windows in which nothing in the data actually changed.

### 2.12 Redundancy filter (`_redundancy_filter`)
Any column that is ≥98% one value **in the reference window** is dropped —
decided once, at fit time, and the same columns are dropped from every later
window (deciding this per-window would make the aligned schema itself drift,
silently zero-padding differently week to week). On the current run, those 8
are:

```
addr2, D13, M1, id_04, id_27, _unseen__uid2, _unseen__P_emaildomain__ProductCD, _freq_ref_C3
```

`D13` is a new arrival to this list — it only became visibly near-constant
once §2.9's `-_days` bug was fixed. The buggy subtraction had been adding a
large, ever-changing offset to an otherwise near-constant column, which
incidentally gave it enough apparent variance to survive the redundancy
filter. Fixing the bug exposed its true (near-constant) distribution, and the
filter now correctly drops it.

### 2.13 Drop leakage / identifier columns (`DROP_COLS`)
Five columns are removed from the model matrix entirely at this point:
`_days`, `TransactionDT`, `TransactionID`, `_uid1`, `_uid2`.

`TransactionDT`/`TransactionID` are monotone in time — left in, a KS test
against them reports D = 1.0 every single window regardless of whether
anything drifted, and a tree model can split directly on "when did this
happen" rather than learning anything behavioural. Their legitimate signal
(hour of day, day of week, transaction ordering) is already carried by
`_hour_cos`/`_hour_sin`/`_weekday__hour`/`_uid2_seq_index`.

`_uid1`/`_uid2` are near-unique entity keys — a frozen ordinal code maps
almost every row of a future window to the "unseen" sentinel, so the raw
identity column is informative in-sample (flattering the baseline model) and
then contributes nothing at inference. Entity *behaviour* is carried instead
by the causal sequence features from §2.4, which remain comparable across
windows because they're relative (a lag, a percent change, a position in
history) rather than an identity itself.

### 2.14 Factorize remaining categoricals (`_factorize`)
Every column that is still non-numeric at this point (`ProductCD`, `card4`,
`card6`, `P_emaildomain`, `R_emaildomain`, `DeviceType`, `DeviceInfo`,
`_weekday__hour`, `_card3__card5`, `_ccols_0_bin`, `_dcols_na_bin`,
`_mcols_na_bin`, ...) gets a **frozen** ordinal code: `pd.factorize` is run
once at fit time, and the resulting `{category: code}` map is replayed on
every later window. A category never seen at fit time maps to a reserved
sentinel (`-1`), not a collision with an existing code. This is the single
fix with the largest measured effect in the whole pipeline: refitting
`factorize` independently per window (numbering categories by order of first
appearance *within that window*) alone accounted for the majority of the
35.1% → 0.9% false-alarm-rate drop in the null experiment (PAPER.md §3.1).

> **Implementation note:** this stage checks `is_numeric_dtype`, not
> `dtype == object`. Under pandas 3.x, string columns carry a dedicated
> `str` dtype rather than `object` — an object-only check silently skips
> every categorical column, leaving it to be factorized later, per-window,
> which is exactly the artifact this stage exists to prevent. This was a real
> bug caught only because the null-experiment check kept failing.

---

## 3. Why *fit once, replay* instead of *refit every window*

Six of the fourteen stages above learn something from the data they're
handed: the aggregate maps (§2.6), the frequency maps (§2.7), the PCA
rotation (§2.11), the redundancy decision (§2.12), and the factorize maps
(§2.14). `FeatureEngineer` is a class specifically so all of these can be
**learned once**, on the reference window (`fit_transform`), and **replayed
unchanged** on every later window (`transform`) — the model's representation
never moves, only its weights do when a retrain happens, and that is a
prerequisite for a drift detector's output to mean "the data changed" rather
than "the encoder changed."

Refitting these per window (the original design) produces drift signal that
is an artifact of the encoder, not of the world:
1. `pd.factorize` numbers categories by order of first appearance — the same
   email domain can be code 3 in one window and code 17 in the next with
   nothing real having changed.
2. Raw counts and per-window relative frequencies both scale with window
   size (§2.7).
3. PCA components are sign- and rotation-arbitrary under a refit (§2.11).
4. A per-window redundancy filter changes the schema itself, so the aligned
   matrix is zero-padded differently from week to week.

PAPER.md §3.1 quantifies the effect of removing this: a null experiment
(splitting one window into two random halves, where no drift can exist by
construction) alarmed on 35.1% of features under the original per-window
design and 0.9% under the frozen one.

---

## 4. From 432 raw columns to 113 trained features

| Block | Raw columns | Final features | What happened |
|---|---:|---:|---|
| Vesta (V) | 339 | 4 | PCA to 2 components + na-count + fixed-subset sum; all 339 raw columns dropped |
| Identity / device (`id_*`, Device*) | 40 | 38 | kept mostly as-is (factorized); `id_04`, `id_27` dropped as near-constant |
| Timedelta (D) | 15 | 16 | 14 columns kept as-is (fillna only), +2 derived (na-count, na-pattern); `D13` dropped as near-constant once the `-_days` artifact stopped masking it (see §2.9) |
| Counter (C) | 14 | 16 | 14 raw dropped; replaced by 3 pattern features + 13 frequency encodings (`C3`'s frequency encoding dropped as near-constant) |
| Match flag (M) | 8 | 8 | `M1`–`M3`, `M5`–`M9` (the pipeline's `MCOLS` list excludes `M4`, see below); `M1` dropped as near-constant, +1 derived na-pattern nets back to 8 |
| Card / entity | 6 | 7 | `card1`–`card6` kept, +1 combination (`_card3__card5`) |
| Amount | 1 | 4 | `TransactionAmt` kept, +3 derived (decimal, decimal-length, log) |
| Product / email | 3 | 4 | `ProductCD`, `P_emaildomain`, `R_emaildomain` kept, +1 combination |
| Distance | 2 | 3 | `dist1`, `dist2` kept as columns *and* combined into `_log_dist_1_2`; the redundancy filter didn't remove the originals on this run |
| Address | 2 | 1 | `addr1` kept, `addr2` dropped as near-constant |
| Temporal | 0 | 3 | derived entirely: `_hour_cos`, `_hour_sin`, `_weekday__hour` |
| Causal sequence | 0 | 3 | derived entirely: lag, amount-change, sequence-index |
| Frequency encoding (cross-feature) | 0 | 1 | `_freq_ref__P_emaildomain__ProductCD` |
| Other | 2 | 5 | `TransactionDT` dropped; `M4` kept as-is (see below); +3 aggregate + `_all_na` |
| **Total** | **432** | **113** | |

**A quirk worth flagging:** the pipeline's `MCOLS` constant
([feature_engineering.py:67](../feature_engineering.py#L67)) is
`['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']` — it skips `M4`. `M4` is
still present in the raw data and still reaches the final feature set (it
survives factorization as an ordinary passthrough column), so no information
is lost, but it is *not* included in the `_mcols_na_bin` missingness pattern
or counted in any M-block aggregate. Whether this was deliberate (M4 behaves
differently from the other match flags) or an oversight isn't documented in
the code; it doesn't affect correctness, only the completeness of the
missingness-pattern feature.

The 8 near-constant columns dropped by the redundancy filter (≥98% one
value): `addr2`, `D13`, `M1`, `id_04`, `id_27`, `_unseen__uid2`,
`_unseen__P_emaildomain__ProductCD`, `_freq_ref_C3`.

**A note on an inconsistency in earlier drafts:** PAPER.md §1.2 previously
stated "149 features after a near-constant-column filter." The actual,
current run of `FeatureEngineer` on this dataset produces **113** features
(114 before the `D13` near-constancy was exposed by the §2.9 fix) — the 149
figure was stale (from an earlier iteration of the pipeline, before later
fixes such as the `_uid2` exclusion from aggregates/count-encoding and the
D-column handling were finalised) and has been corrected in PAPER.md to match
this document.

### The full list of 113, by group

<details>
<summary>Click to expand the complete feature list</summary>

**Vesta (4):** `_vcols_dec0`, `_vcols_dec1`, `_vcols_na`, `_vcols_sum`

**Counter / C-block (16):** `_ccols_nonzero`, `_ccols_sum`, `_ccols_0_bin`,
`_freq_ref_C1`, `_freq_ref_C2`, `_freq_ref_C4`, `_freq_ref_C5`,
`_freq_ref_C6`, `_freq_ref_C7`, `_freq_ref_C8`, `_freq_ref_C9`,
`_freq_ref_C10`, `_freq_ref_C11`, `_freq_ref_C12`, `_freq_ref_C13`,
`_freq_ref_C14`

**Timedelta / D-block (16):** `D1`–`D12`, `D14`, `D15` (`D13` dropped as
near-constant), `_dcol_na`, `_dcols_na_bin`

**Match flags (8):** `M2`, `M3`, `M5`, `M6`, `M7`, `M8`, `M9`, `_mcols_na_bin`
— plus `M4`, which is in the schema but outside the pipeline's `MCOLS` list
(counted under "Other" below; see the funnel-table note)

**Identity / device (38):** `id_01`, `id_02`, `id_03`, `id_05`–`id_26`
(minus `id_04`, `id_27`), `id_28`–`id_38`, `DeviceType`, `DeviceInfo`

**Card / entity (7):** `card1`, `card2`, `card3`, `card4`, `card5`, `card6`,
`_card3__card5`

**Amount (4):** `TransactionAmt`, `_amount_decimal`, `_amount_decimal_len`,
`_log_amount`

**Product / email (4):** `ProductCD`, `P_emaildomain`, `R_emaildomain`,
`_P_emaildomain__ProductCD`

**Distance (3):** `dist1`, `dist2`, `_log_dist_1_2`

**Address (1):** `addr1`

**Temporal (3):** `_hour_cos`, `_hour_sin`, `_weekday__hour`

**Causal sequence (3):** `_day_lag_uid2`, `_amount_lag_pct_uid2`,
`_uid2_seq_index`

**Cross-feature frequency (1):** `_freq_ref__P_emaildomain__ProductCD`

**Aggregates + misc (5):** `_max_TransactionAmt__P_emaildomain__ProductCD`,
`_mean_TransactionAmt__P_emaildomain__ProductCD`,
`_var_TransactionAmt__P_emaildomain__ProductCD`, `_all_na`, `M4`

</details>

Human-readable labels for all of these (e.g. `_freq_ref_C5` →
*"How common this value of 'Address/card counter 5' was in training"*) are
generated by `feature_label()` in feature_engineering.py, so drift reports
read by risk/ops teams don't require reading the pipeline source.

---

## 5. From 113 trained features to the monitoring set

The model **trains on all 113**. Monitoring all 113 for drift would be both
expensive (SHAP attribution and per-feature distributional tests for every
one) and statistically self-defeating (many of the 113 are near-duplicates of
each other, so a "drift" vote would double-count correlated signal). A
smaller, deliberately-chosen **20-feature monitoring set** is selected
separately — covered in full detail, including why the set size was
increased from 10 to 20, how the consensus-vote threshold was checked against
a null calibration, and a feature-by-feature audit of which of the 20's
persistent "drift" signals are real vs. artifacts, in
[FEATURE_SELECTION_PROCESS.md](FEATURE_SELECTION_PROCESS.md).

---

## 6. Modelling, in detail

Two classifiers appear in this project, trained the same way conceptually but
for different reasons — see PAPER.md §1.3 for why both exist.

### 6.1 The classical model — LightGBM GBDT

Used by all twelve classical detectors ([drift_engine.py](../drift_engine.py))
via the shared [model_registry.py](../model_registry.py).

**Hyperparameters** ([model_training.py:26](../model_training.py#L26)):

| Parameter | Value | Why |
|---|---|---|
| `objective` | `binary` | fraud is a binary label |
| `boosting_type` | `gbdt` | standard gradient-boosted trees |
| `learning_rate` | 0.01 | slow, stable learning — see the early-stopping note below |
| `num_leaves` | 64 | tree complexity |
| `colsample_bytree` / `subsample` | 0.7 / 0.7 | row/column subsampling for regularisation |
| `n_estimators` | 500 (upper bound; early stopping usually halts earlier) | |
| `min_data_in_leaf` | 20 | |
| `is_unbalance` | `True` | LightGBM's built-in class-weighting for the 3.5% prevalence |

**Validation split is temporal, not random** — the last 20% of rows by time
order become the validation set. A random stratified split would place
same-day (often same-entity) rows on both sides of the split, via the
`_uid2`-derived features, giving early stopping a leaked, optimistic AUC to
tune against.

**Early stopping patience is 100 rounds**, not the more typical 5–10. This
number has history: at `learning_rate=0.01`, a patience of 5 rounds halts
training at **iteration 1** — the very first tree is already "5 rounds
without improvement" because 0.01 moves validation AUC too slowly for a
5-round window to register progress. The model that produced the earliest
version of this project's results was, unknowingly, a single decision tree,
with **F1 = 0.0000 on every evaluation window** as a direct consequence (see
PAPER.md §3.3 for the full story, including how this silently disabled
DDM/EDDM/HDDM, all three of which monitor the model's error stream and got a
constant to look at).

**Decision threshold is calibrated, not fixed at 0.5.** After training,
`tune_decision_threshold()` sweeps a 91-point grid from 0.05 to 0.95 and picks
the F1-maximising cut on the validation split. At 3.5% prevalence, a fixed
0.5 threshold on a well-calibrated ranking model puts essentially no
probability mass above the line — the same F1 = 0 failure mode as the
early-stopping bug, from an unrelated cause. The tuned threshold is stored as
a plain Python attribute on the LightGBM `Booster` object
(`clf.decision_threshold = threshold`) — LightGBM's Booster has no `set_attr`
API (unlike XGBoost's), but a Python object attribute survives pickling,
which is how `ModelRegistry` persists it across versions.

**Custom F1 eval callback.** `_lgb_f1_score` is registered as an additional
`feval` purely for the training log (`lgb.log_evaluation`) — the actual early
stopping and threshold selection use AUC and the tuned-threshold F1
respectively, not this callback's fixed 0.5 cut.

### 6.2 The neural model — for the RL agent

Used only by the reinforcement-learning experiment
([run_rl_experiment.py](../run_rl_experiment.py)), via
[neural_model.py](../neural_model.py) and [model_lattice.py](../model_lattice.py).

**Why not reuse LightGBM here:** a GBDT cannot be partially updated. Adding
trees to an existing booster is a different operation from adapting it, and
there is no principled "fine-tune on the last month" for a fixed forest. With
a GBDT, the agent's action space collapses from four choices (do nothing,
partial update, full retrain, hedge ensemble) to two (do nothing, full
retrain) — the entire question the RL formulation is built to answer
(*how* to adapt, not just *whether*) disappears. A differentiable model makes
the middle ground real, and — just as importantly — makes its cost
*measurable* (PAPER.md §6.5, catastrophic forgetting).

**Architecture** (`TabularNet`): a 3-layer MLP, `256 → 128 → 64 → 1`, each
hidden layer followed by BatchNorm, ReLU, and Dropout(0.2). Deliberately
small — the point of this model is adaptability, not raw capacity.

**Training** (`NeuralFraudModel.fit`):
- Adam optimiser, `lr=1e-3`, batch size 1024, up to 30 epochs with patience 5 on validation AUC (temporal 80/20 split, same reasoning as §6.1).
- Loss is `BCEWithLogitsLoss` with a **positive-class weight** (`n_neg / n_pos`) rather than resampling — every row is still seen once per epoch, unlike undersampling, and unlike oversampling no row is duplicated.
- The best-AUC epoch's weights are kept (`copy.deepcopy` of the state dict), not simply the last epoch's.
- After training, the same 91-point threshold sweep as §6.1 calibrates the decision cut.

**Frozen standardisation.** `mean_`/`std_` are computed once, on the data the
model was *first* trained on, and inherited unchanged by every partial
update. This mirrors the frozen-encoder principle of §3: re-standardising on
a small recent window would shift the input distribution the existing
weights were trained against — a representation change disguised as a weight
update, which would make forgetting unmeasurable (you couldn't tell whether a
performance drop came from the new weights or from the rescaled inputs).

**`partial_fit` returns a new model, not a mutation.** `child = self.clone()`
deep-copies the network before fine-tuning it for a few epochs (default 5) at
a **tenth** of the base learning rate on the supplied window. `self` is left
untouched. This is required by `model_lattice.py`, which needs to branch a
partial update from the same parent model more than once (the lattice
enumerates every reachable `(full_retrain_week, partial_update_week,
ensemble_α)` combination) — an in-place update would make that impossible,
since the second branch would start from the first branch's already-modified
weights instead of the shared parent.

**Gradient attributions**, not SHAP. `gradient_attributions()` computes
gradient × input per feature — the standard, cheap attribution method, used
as the differentiable-model analogue of the SHAP drift detector
(`SHAPDriftDetector` in [drift_engine.py](../drift_engine.py) uses
`shap.TreeExplainer`, which requires a tree model). Real SHAP on a neural net
would need `KernelExplainer` (too slow to run per reference/window pair,
across hundreds of pairs in the drift-signal matrix) or `DeepExplainer`
(still costly). Gradient × input captures the same thing the SHAP-drift
comparison actually depends on — whether the model's *reliance* on a feature
has shifted — without an exact Shapley decomposition.

### 6.3 Shared model registry (both classifiers)

`ModelRegistry` ([model_registry.py](../model_registry.py)) trains **at most
one model per week** and hands the identical fitted model to every detector
that flagged drift that week, rather than each of the twelve detectors
training its own private model. `get_or_train(week, X, y)` checks whether a
version for that week already exists before training; `adopt(method,
version, week, reason)` points a detector at a version without retraining.
This caps the number of distinct models at `1 + n_weeks` regardless of how
many of the twelve detectors are running, and — just as importantly for the
drift comparison itself — guarantees that two detectors which happen to
retrain in the same week are being compared against *bit-identical* models,
not two independently-seeded fits of the same data.

### 6.4 The model lattice (RL only)

Under the RL action set, the model in force at any point is fully determined
by `(last_full_retrain_week, last_partial_update_week, ensemble_alpha)` — a
small, enumerable space. `ModelLattice.build()`
([model_lattice.py](../model_lattice.py)) trains every reachable full model and
every reachable partial-update branch **once**, up front, and caches each
one's AUC/F1 against every future week. PPO episodes then become table
lookups instead of tens of thousands of individual model fits — see PAPER.md
§5.1 for the full argument, including why partial updates are deliberately
*not* chained (each one re-derives from the last **full** model, never from
the previous partial), which both keeps the lattice small and bounds
catastrophic forgetting by construction.

---

## 7. Summary funnel

```
434 raw merged columns (incl. TransactionID key, isFraud target)
  → 432 raw predictor columns
      → 339 Vesta columns compressed to 4
      → 14 Counter columns replaced/expanded to 16
      → 15 Timedelta columns kept as-is (fillna only), +2 derived → 16
      → 8 Match-flag columns kept, +1 derived → 8 (net; M4 handled separately, see §4)
      → 40 Identity/device columns kept mostly as-is → 38
      → 16 remaining raw columns (card/amount/product/distance/address/other)
        kept, transformed, or combined, plus temporal/causal-sequence/
        cross-feature-frequency features derived from scratch → 31
      → 5 leakage/identifier columns dropped (TransactionDT, TransactionID, _uid1, _uid2, _days)
      → 8 near-constant columns dropped (redundancy filter, incl. D13 — see §2.9)
  = 113 trained features
      → bagged-importance + SHAP + monitorability + redundancy/family-diversity selection
      = 20 monitored features (see FEATURE_SELECTION_PROCESS.md)
```
