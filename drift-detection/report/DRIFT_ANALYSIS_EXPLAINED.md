# Drift Analysis — Explained Simply

Plain-language walkthrough of what this system does, how each drift method
works, and — importantly — which parts of the earlier design were producing
misleading results and why.

For the research framing (motivation, related work, experimental protocol,
threats to validity) see [PAPER.md](PAPER.md).

---

## 1. The Big Idea

A fraud model is trained once on historical data. Fraud tactics, customer
behaviour and merchant mixes then change — this is **drift**. If nobody
notices, the model quietly gets worse.

This pipeline replays history to study the problem:

1. Train a model on the **first 90 days** (the baseline).
2. Replay the rest **one week at a time**, like a movie of the model's life in
   production.
3. Each week, run **12 drift-detection methods**, each asking "has the data or
   the model's behaviour changed enough to worry about?"
4. If a method's alarm **persists**, that method retrains — and methods that
   did *not* alarm keep using the older model.
5. Log everything for the dashboard, then compare the 12 resulting **retraining
   policies** against never-retrain, always-retrain, and random controls.

The point is not "detect drift". The point is: **if you let method X decide when
to retrain, do you end up with a better model than if you'd retrained at random
the same number of times?**

---

## 2. One Model, Many Pointers

This is the structural change from the earlier design.

**Before:** each method owned a private model. 12 baseline fits of *identical*
data, then a separate fit per method per drifting week — approaching 180 fits
where at most 15 are distinct.

Worse than the waste: if two methods retrained in the same week they got
*different* models (different random seeds), so any later performance gap
between them was partly seed noise rather than a consequence of their policy.

**Now:** a model version is identified by **the data it was trained on**, which
under cumulative retraining is just the week boundary:

```
v0  = baseline            (first 90 days)
v3  = everything through week 3
v7  = everything through week 7
...
```

The registry ([model_registry.py](../model_registry.py)) trains **at most one
model per week**, however many methods asked for it, and hands the identical
model to all of them. Each method holds only a pointer:

```
method_versions = {
    'psi'             : 0,   ← never drifted, still on the baseline
    'ddm'             : 3,   ← drifted in week 3
    'prequential_auc' : 7,   ← drifted in weeks 3 and 7, now on v7
    ...
}
```

**Maximum distinct models = 1 + number of weeks** (15 for a 14-week replay),
no matter how many detectors you compare.

Because methods that retrain in the same week now share byte-identical weights,
performance differences between them are caused purely by *when* they retrained.

### What is shared, what is not

| Thing | Shared per version? | Why |
|---|---|---|
| Model weights | Yes | Same training data ⇒ same model |
| Reference sample (20k rows) | Yes | The reference *is* that version's training data |
| Reference predictions | Yes | Determined by model + reference |
| SHAP / K-Means / autoencoder state | Yes | Fitted against the version's reference |
| Weekly prediction vectors | Yes | Same version ⇒ same predictions |
| DDM/EDDM/HDDM/AUC trackers | **No** | Each method's decision history is its own |
| Persistence-gate history | **No** | Ditto |
| Version pointer | **No** | This *is* the policy being compared |

---

## 3. The Model

There are **two** classifiers in this repo, used by two different experiments:

| Experiment | Model | Why |
|---|---|---|
| Classical detector comparison (§5–8.2) | LightGBM GBDT | Strongest tabular baseline; retrain-or-don't is the only action available |
| RL adaptation agent (§8.3) | PyTorch MLP ([neural_model.py](../neural_model.py)) | **A GBDT cannot be partially updated** — without a differentiable model the agent's action space collapses back to retrain/don't |

Everything in this section describes the LightGBM baseline. The neural model
shares the same frozen features, temporal split, and calibrated threshold, and
scores slightly lower on its own (val AUC ≈ 0.88 vs ≈ 0.885) — which is expected
for an MLP on tabular data. We use it for the RL work because *adaptability*
matters more there than a fraction of a point of AUC.

- **Algorithm**: LightGBM GBDT binary classifier, target `isFraud`
- `learning_rate = 0.01`, `num_leaves = 64`, `n_estimators = 500`,
  `subsample = colsample_bytree = 0.7`, `min_data_in_leaf = 20`,
  `is_unbalance = True`

Three things were fixed here, and they matter more than they look:

**Early stopping patience: 5 → 100 rounds.** At `learning_rate = 0.01` the first
few dozen boosting rounds barely move validation AUC, so a patience of 5
terminated training at **iteration 1**. Every model in the previous reports was
a *single tree*.

**Validation split: random-stratified → temporal (last 20%).** A random split
puts same-day, often same-card rows on both sides. Validation AUC was
optimistic and early stopping was tuning against a leak. The data arrives in
chronological order, so the last 20% is a genuine forward holdout.

**Decision threshold: fixed 0.5 → calibrated on validation.** At 3.5% fraud
prevalence, essentially no probability mass sits above 0.5 — so the model
predicted "not fraud" for *every* transaction and **F1 was exactly 0.0000 in
every window of the previous reports.**

That last one had a knock-on effect that invalidated three detectors: DDM, EDDM
and HDDM all monitor a **binary error stream**. If the model always predicts the
negative class, the error stream *is* the label vector — its rate is the
prevalence, ~3.5%, constant, containing no information about the model at all.
DDM's "drift" in 14 of 14 weeks was fluctuation in the weekly fraud rate.

After the fix: validation F1 ≈ 0.47 at a threshold of ≈ 0.40. The tuned
threshold is stored on the model and used for every downstream error stream and
metric.

---

## 4. Data & Features

- IEEE-CIS fraud data (`train_transaction.csv` + `train_identity.csv`, joined on
  `TransactionID`), sorted by `TransactionDT`. 590,540 rows, 3.50% fraud.
- Feature engineering produces **149 features** (distance, temporal, entity/UID,
  sequence/velocity, amount, aggregates, frequency, C/D/M summaries, V-block PCA).

### 4.1 The encoders are frozen — and this is the single most important fix

Feature engineering contains **learned encoders**: ordinal encoding of
categoricals, frequency encoding, group aggregates, the MinMax scaler, the PCA
rotation, and the near-constant-column filter. All of them estimate something
from the data they're given.

The old code called one stateless `apply_feature_engineering(df)` separately on
the baseline **and on every weekly window**. That silently manufactures drift:

- **Ordinal codes depend on order of appearance.** `pd.factorize` numbers
  categories by first appearance. An email domain that is code `3` in the
  baseline can be `17` next week with nothing having changed in the world. Every
  distribution test then reports a huge shift, forever.
- **Raw count encodings scale with window length.** Baseline = 90 days, window =
  7 days. A category appearing at a perfectly constant *rate* has a count ~13x
  smaller in the weekly window. KS, PSI and KL all call that drift. It's
  arithmetic.
- **PCA components are sign-arbitrary.** Refitting can flip a component,
  inverting the feature.
- **A per-window redundancy filter changes the schema**, so the aligned matrix
  gets zero-padded differently each week.

`FeatureEngineer` ([feature_engineering.py](../feature_engineering.py)) now fits
every encoder **once** on the baseline (`fit_transform`) and replays them
unchanged (`transform`). Count encoding became *relative frequency* rather than
raw counts, which removes the window-length dependence entirely.

> **Rule: freeze the representation, version only the model.**
> Retraining updates model weights, never the feature space.

A category the baseline never saw maps to a reserved `UNSEEN` code — so genuine
new-category drift becomes an *observable signal* instead of silently deforming
the features.

**Sequence features are computed globally.** A card's "time since previous
transaction" computed inside a 7-day window can only look back 7 days, versus 90
in the baseline — another manufactured shift. These are computed once over the
full sorted frame. Still strictly causal (each row only sees earlier rows): in
production, a card's previous transaction really is available.

### 4.2 Choosing what to monitor

Picking features to **monitor** is a different problem from picking features to
**train on**. The old approach — top-10 by a single LightGBM gain ranking — has
three problems:

1. **Unstable.** One stochastic fit. A different seed gives a different top-10,
   so "fraction of monitored features that drifted" inherits that arbitrariness.
2. **Redundant.** Gain splits arbitrarily among correlated features, and
   correlated features drift *together*. Ten features representing three real
   signals give a "6 of 10 drifted" vote that is not 6 independent pieces of
   evidence.
3. **Unmonitorable features.** A three-valued column can carry real gain but KS
   is dominated by ties and PSI bucketing collapses.

[feature_selection.py](../feature_selection.py) does:

1. **Bagged importance** — 5 bootstrap fits, different seeds; rank by mean
   reciprocal rank; record each feature's *selection frequency*.
2. **SHAP corroboration** — global mean |SHAP|, which (unlike gain) isn't biased
   toward high-cardinality features.
3. **Monitorability filter** — drop <10 distinct values or ~zero variance.
4. **Redundancy pruning** — walk top-down, skip anything with |Spearman ρ| ≥ 0.90
   against an already-selected feature.

It reports **Nogueira's stability index** (reproducibility of the selection) and
the max pairwise ρ among the selected set (evidence the votes are ~independent).

---

## 5. The 12 Methods

Grouped by **what each one looks at**, which determines its cost and its blind
spots.

| # | Method | Looks at | Needs labels? |
|---|---|---|---|
| 1 | KS test | Feature distributions | No |
| 2 | PSI | Feature distributions (binned) | No |
| 3 | Jensen-Shannon | Feature distributions (binned) | No |
| 4 | DDM | Error stream | Yes |
| 5 | EDDM | Gaps between errors | Yes |
| 6 | HDDM | Error stream (Hoeffding bound) | Yes |
| 7 | ADWIN | Prediction stream | No |
| 8 | SHAP drift | Explanation distributions | No |
| 9 | Clustering | Joint geometry | No |
| 10 | Autoencoder | Reconstruction error | No |
| 11 | Prequential AUC | Ranking quality | Yes |
| 12 | Champion vs Challenger | Value of retraining | Yes |

### Feature-distribution methods

**1. KS test** — two-sample Kolmogorov-Smirnov per monitored feature.

*The problem:* the p-value asks "could this have arisen by chance?" With a
90-day reference (~10⁵ rows) vs a 7-day window (~10⁴), the critical statistic at
α = 0.05 is

```
D_crit = 1.358 × √((n+m)/(n·m)) ≈ 0.015
```

A shift of 1.5% of probability mass — operationally meaningless — is
"statistically significant". **The old pipeline flagged KS drift in 14 of 14
weeks.** It was measuring sample size.

*The fix:* require significance **and** a real effect — `D ≥ 0.10` (the CDFs
must separate by 10 percentage points somewhere). D is itself a bounded,
interpretable effect size. Both windows are capped at 20,000 rows so the
p-value keeps some discriminating power.

*Also:* **Benjamini-Hochberg FDR correction** across the monitored set. Testing
10 features at α = 0.05 with no correction gives a 40% chance of at least one
false positive in a *stable* week; over 14 weeks that's a certainty. Bonferroni
would be too conservative for correlated features; BH controls the expected
*proportion* of false discoveries, which matches a decision rule that is itself
a proportion.

*Verified:* 100k vs 10k samples from an identical Gaussian → D = 0.007, no
drift. A genuine 0.5σ shift → D = 0.202, drift.

**2. PSI** — Population Stability Index over 10 percentile buckets.

*The problem:* the famous bands (<0.10 stable, 0.10–0.20 moderate, >0.20 drift)
are credit-scorecard folklore calibrated to an unstated sample size. PSI's null
distribution is asymptotically `χ²(B−1)/n`, so its expected value under **no
drift** scales as `(B−1)/n`. A 1,000-row window expects PSI ≈ 0.009 from noise;
a 100-row window expects ≈ 0.09 — nearly the "moderate" band. Comparing against
a fixed constant confuses drift with window size.

*The fix:* a **bootstrap null calibrated to this window**. Resample `n_curr` rows
from the reference 100 times, compute PSI each time, and require the observed
PSI to beat both the 0.20 threshold *and* the null's 99th percentile.

*Verified:* at n = 300 with no drift, PSI = 0.020 vs null p99 = 0.087 → correctly
suppressed. A real 1.2σ shift → PSI = 1.35 vs null 0.006 → correctly flagged.

**3. Jensen-Shannon distance** (replaces raw KL as the decision statistic).

*The problem:* KL divergence is unbounded, asymmetric, and undefined wherever
the current window has mass the reference didn't — routine for rare categories.
The standard fix, ε-smoothing, makes KL's absolute value depend on the arbitrary
ε, so a fixed `KL ≥ 0.5` threshold means different things for different features.

*The fix:* JS distance is symmetric, always finite, and bounded in [0, 1], so one
threshold is comparable across features of any scale. **Per-feature rule:
`JS ≥ 0.10`.** Raw KL is still reported for continuity with old reports.

**Method-level rule for all three:** flag only when **≥ 60%** of monitored
features drift individually. With the redundancy-pruned monitoring set those
votes are approximately independent, so the fraction is a real consensus rather
than one signal counted several times.

### Error-stream methods

All three feed on a **class-balanced error stream** — see below for why.

**4. DDM** (Gama et al., 2004) — tracks running error rate `p` and std `s`,
remembering the lowest `p + s` ever seen. Drift when `p + s > p_min + 3·s_min`;
warning at `2·s_min`. Needs 30 observations to start.

**5. EDDM** (Baena-García et al., 2006) — tracks the *distance between*
consecutive errors (are errors bunching up?). Monitors `mean + 2·std` of
inter-error distances via Welford's algorithm, remembering the max ever seen.
Drift when the current value falls below `0.75 × max_metric`.

> The original 0.90/0.95 β values are extremely sensitive to noise — on a stable
> error rate with no real drift they retrained ~13/14 weeks. 0.75/0.85 with a
> longer warm-up brings that to ~1/14 while still catching a genuine 2.5x
> error-rate spike within 3–4 weeks.

**6. HDDM** (Frías-Blanco et al., 2015) — **new.** DDM's control limits assume a
Bernoulli stream whose variance shrinks as 1/n, making it progressively *harder*
to trigger the longer a model has been stable — exactly the regime where drift
is most likely. HDDM uses a distribution-free **Hoeffding bound** instead, which
only requires the stream be bounded in [0, 1].

> **DDM/EDDM/HDDM trackers are persistent.** They're created once per method and
> fed successive weekly batches, so `p_min`/`s_min`/`max_metric` accumulate over
> the tracker's lifetime as the algorithms intend. An earlier version rebuilt
> them every week, which reset the `min_instances` gate constantly — and since
> fraud models are accurate, an early lucky streak of zero errors collapses
> `p_min`/`s_min` to 0 and makes the next error look like "drift", every week.
> They're reset **only** when that method actually retrains, because the model
> they were watching no longer exists.

#### Why the error stream is class-balanced

At 3.5% prevalence, a 0/1 error stream is ~96% determined by how the model treats
*legitimate* transactions. A model can lose **all** of its fraud-catching ability
and move the raw error rate by only 3.5 percentage points — well inside the noise
band DDM's `p_min + 3·s_min` rule tolerates. Fed that stream, these detectors are
effectively monitoring the majority class.

So `build_error_stream(..., mode='balanced')` builds the error stream over a
**class-balanced subsample**: every fraud case, plus an equal number of randomly
drawn legitimate ones, kept in time order. The stream's mean is then the
balanced error rate `0.5·(FNR + FPR)`.

Note it balances by **subsampling, not reweighting**. Instance weights of
`1/0.035 ≈ 29` aren't bounded by 1, and DDM's variance term and HDDM's Hoeffding
bound both need a bounded stream; clipping the weights back to 1 undoes the
rebalancing completely. (The first implementation made exactly that mistake — a
unit test caught it.)

*Verified:* under a total recall collapse, the 0/1 stream's mean moves by 0.036
while the balanced stream's moves by 0.500 — a **14x** amplification of exactly
the failure you most need to catch.

**7. ADWIN** (Bifet & Gavaldà, 2007) — compares the current week's *prediction*
stream against that method's own reference stream (baseline predictions, or the
predictions at its last retrain). Uses `river`'s native ADWIN when installed
(`delta = 0.002`); otherwise a two-sample z-test with the equivalent threshold
(z ≈ 3.09).

> An earlier version only split the current week in half and compared the halves,
> which could only catch a shift *within* one week and was structurally blind to
> drift relative to the baseline — it essentially never fired.

### Structural / explanation methods

**8. SHAP drift** — computes SHAP values for reference and current data, then
per-feature KS on the *SHAP value distributions* rather than the raw features.
This catches cases where a feature's **influence on the model** shifted even
though its raw distribution looks stable.

Fixes: **random sampling** instead of `.iloc[:200]` (taking the first 200 rows of
a time-ordered window samples only the earliest days of the baseline, so any
weekly/seasonal structure registers as "SHAP drift" forever); **BH correction**;
and the same **effect-size gate** (`D ≥ 0.10`).

**9. Clustering drift** — K-Means (k=5) on the monitored features of the
reference. Tracks (a) **distance ratio**: mean distance of current points to
their nearest centroid ÷ the same on reference; (b) **cluster PSI**: shift in how
points distribute across the 5 clusters. Drift when `ratio ≥ 1.5` OR `PSI ≥ 0.2`.

> **Features are standardized first.** K-Means uses Euclidean distance, so
> without scaling whichever feature has the largest raw scale (a dollar amount vs
> a small count) dominates every distance — clusters split along that one axis,
> and drift in the smaller-scale features is invisible. Verified with a synthetic
> test: unscaled, a 5σ shift in a small-scale feature barely moved
> `distance_ratio`; scaled, it correctly crossed the threshold.

**10. Autoencoder drift** — small bottleneck MLP trained to reconstruct the
monitored reference features. On new data, reconstruction **RMSE** is compared to
baseline via z-score: `(curr_rmse_mean − ref_rmse_mean) / ref_rmse_std`. Drift
when `z > 3.0`. Features standardized, for the same reason as clustering.

> An earlier version also OR'd in a KS test on the reconstruction-error
> distributions. Removed: with thousands of rows a KS test flags "significant"
> for almost any nonzero difference, so it fired on nearly every batch. The KS
> stat is still reported, just not used to decide.

### Performance methods

**11. Prequential AUC** — **new.** Every error-stream detector is a *proxy* for
what an operator actually cares about: has ranking quality dropped? Under 3.5%
prevalence those proxies are dominated by the majority class. This measures the
thing directly — windowed out-of-sample AUC vs the AUC at the incumbent model's
adoption — and declares drift only when the drop exceeds **both** an absolute
floor (0.02) **and** two bootstrap standard errors. Unlike champion-vs-challenger
it needs no second model.

**12. Champion vs Challenger** — trains a challenger on the current window and
compares it head-to-head with the incumbent.

*The problem:* the obvious implementation is not a fair comparison. The champion
is scored strictly out-of-sample; the challenger is scored on the rows it was
just fitted to. A 500-tree LightGBM memorises a 10k-row window substantially, so
its in-sample AUC is inflated by **more than the 0.03 gap threshold**. The
detector fires on the challenger's overfitting rather than the champion's
staleness — and fires *every* week, drifting or not, because the bias is
constant.

*The fix:* the challenger is evaluated on **out-of-fold predictions** (2-fold
stratified), putting both models out-of-sample. The AUC gap must also exceed its
own **bootstrap standard error** — a gap of 0.04 on a week with 200 fraud cases
has an SE of comparable size, and acting on it is reading noise. Since one fold
model is reused for the weekly feature-importance snapshot, this costs no extra
fits.

Drift when `auc_degradation > 0.05` (champion fell from baseline) OR the OOF gap
is significant.

---

## 6. Persistence: alarms must repeat

Retraining is expensive and resets every downstream reference statistic. A
single-window trigger converts a detector's per-window false-positive rate
**directly** into a retraining rate — which is how the old pipeline reached "DDM
retrained in 14 of 14 weeks".

Every method's raw flag now passes through a **k-of-n persistence gate**
(default **k = n = 2**: must fire two weeks running). For roughly independent
windows this cuts the false-alarm probability from `p` to about `C(n,k)·pᵏ` — a
20% weekly false-positive rate becomes ~4% — while delaying a genuine detection
by at most `k−1` windows.

Both **raw** and **confirmed** flags are stored, so the gate's cost is auditable
rather than buried in threshold tuning.

---

## 7. Retraining Logic

When a method's **confirmed** flag is true:

1. All data from the start through the current week is assembled — a
   **cumulative** window (built by concatenating already-transformed weekly
   frames, not by re-running feature engineering).
2. The registry trains a model **for that week** — or reuses it if another
   method already triggered the same week.
3. Every method that drifted this week adopts **that same version**.
4. Detector artefacts (SHAP explainer, K-Means, autoencoder) are re-fitted once
   per *version*, not per method.
5. Each adopting method's tracker and persistence gate are **reset**, because
   their accumulated history refers to a model that no longer exists.

Methods that did not confirm drift do nothing and keep their existing pointer.

---

## 8. Comparing Retraining Policies

A detector is only interesting as a **retraining policy**. The final phase
compares them properly.

Under cumulative retraining the entire reachable model space is the lattice
`{v0, v1, …, v_nweeks}` — one version per possible retraining week, at most 15.
So the pipeline **materialises the whole lattice once** (it's the registry's
bound anyway) and precomputes a (version × week) out-of-sample AUC matrix,
filling only cells where the version hadn't already trained on that week.

After that, **any** policy is a table lookup: a policy is just a map from week to
version id. That's what makes these controls affordable:

| Policy | What it is |
|---|---|
| **never_retrain** | Baseline `v0` forever. The floor. |
| **always_retrain** | Retrain every week. Practical ceiling and cost upper bound. |
| **each detector** | Its actual realised retraining weeks. |
| **random, frequency-matched** | 200 random policies with the *same number* of retrains as the detector. |

The random control is the one that matters. A detector that retrains 6 times is
only interesting if it beats a policy that retrains 6 times **at random** — more
retraining generally helps, so without this control, apparent detector skill may
be entirely explained by retraining frequency. Each detector gets a percentile
against its own frequency-matched ensemble.

We also report **pairwise Jaccard agreement** over the weeks each detector
flagged. Twelve detectors that always agree are not twelve independent opinions.

---

## 8.1 What actually happened on the full dataset

This section reflects two corrections made after the numbers below were first
produced: the monitoring set grew from 10 to 20 features (with a fix for
redundant near-duplicate features voting as if they were independent), and a
transformation bug that manufactured a fake trend in two "days since" columns
was found and fixed. Both are covered in [FEATURE_SELECTION_PROCESS.md](FEATURE_SELECTION_PROCESS.md)
and [FEATURE_ENGINEERING_AND_MODELING.md](FEATURE_ENGINEERING_AND_MODELING.md).
Confirmed (persistence-gated) alarms over the 14-week replay, after both fixes:

| Method | Weeks it fired | Count |
|---|---|---|
| Prequential AUC | 2, 5, 7, 9, 11, 13 | 6 |
| ADWIN | 2, 4, 8, 10, 12 | 5 |
| EDDM | 2, 10, 12, 14 | 4 |
| Champion vs Challenger | 2, 8 | 2 |
| KS, PSI, Jensen-Shannon, DDM, HDDM, SHAP, Clustering, Autoencoder | never | 0 |

**KS went from 14/14 weeks to 0/14. DDM went from 14/14 to 0/14.** Neither was
silenced by loosening a threshold — KS had been reporting sample size, and DDM
had been watching a constant error stream from a model that never predicted
fraud. Remove the artifacts and the alarms disappear with them. **SHAP and
Clustering, which each fired once at the old 10-feature setting, are silent
at 20** — SHAP's one signal turned out to lean heavily on over-representing
one family of correlated features; Clustering's week-12 signal is still the
largest reading in its own table by a wide margin, but no longer repeats into
an adjacent week under the wider feature set, so the persistence gate no
longer confirms it.

**The headline finding is unchanged:** every method that looks only at
*feature distributions* stayed silent, while the methods that watch the
*model's behaviour* fired repeatedly. That means the drift on this dataset is
**concept drift, not covariate shift** — the transactions look the same, but
what makes one fraudulent changed.

The practical consequence is uncomfortable: **label-free monitoring would have
caught nothing here.** Monitoring feature distributions is the most common
production setup precisely because it needs no labels — and on this problem it
would have shown a flat dashboard for six months while the model decayed. Only
methods needing labels (or, for ADWIN, the prediction stream) noticed.

Week 2 is the one week where every behaviour-based method agrees.

### The full week-by-method ledger

The table above is a summary. The pipeline also writes a row per (week,
method) — 168 rows for 14 weeks × 12 methods — to
`reports/method_week_matrix.csv`, recording for every one: which model version
it was pointed at, whether it raised a *raw* flag that week, whether that flag
survived the persistence gate, and — for the four methods that vote across the
monitored feature set (KS, PSI, Jensen-Shannon, SHAP) — exactly how many of the
20 monitored features individually crossed that method's threshold, and which
ones by name. This is what lets you ask "why didn't PSI fire in week 7" and
get an exact answer back, rather than a re-run.

Three things fall out of that ledger that the summary table hides:

- **KS, PSI and Jensen-Shannon never get close to firing.** Across all 14
  weeks their feature-crossing fraction sits between 0.15 and 0.50 — well
  under the 0.6 needed for consensus. This is the concrete form of "no
  covariate shift": it isn't that these methods missed drift by a little,
  it's that the monitored features individually just aren't moving in
  concert.
- **Three specific features drive most of that baseline fraction, every
  week.** `_mcols_na_bin` and the two Vesta PCA components individually cross
  their own threshold in 14 of 14 weeks each. Investigated in detail in
  FEATURE_SELECTION_PROCESS.md — the pattern is a one-time *step* between the
  90-day reference window and week 1 that then stays flat, not a progressive
  ramp, most plausibly because the reference window (which spans the
  dataset's holiday-season start) is genuinely different in composition from
  any later, non-holiday week.
- **The persistence gate is filtering real noise, not adding pointless
  delay.** ADWIN and Prequential AUC raise a *raw* flag in 12 of the 14 weeks
  each — almost every week — but only 5 and 6 of those, respectively, survive
  confirmation. Without the gate, these two alone would retrain almost every
  week, which is exactly the "retraining constantly barely helps" result from
  §8.2 below, arrived at from the opposite direction.

## 8.2 Which retraining policy actually won

Mean out-of-sample AUC of whichever model each policy had in force, week by
week. "Random control" is the policy's rank against 200 random policies that
retrain **the same number of times** at randomly chosen weeks.

| Policy | Retrains | Mean AUC | Random control | Beat random? |
|---|---|---|---|---|
| **Champion vs Challenger** | 2 | **0.8819** | 0.810 | no |
| EDDM | 3 | 0.8776 | 0.215 | no |
| Prequential AUC | 6 | 0.8760 | 0.200 | no |
| ADWIN | 5 | 0.8733 | 0.055 | no |
| *always retrain* | 13 | 0.8726 | — | — |
| *never retrain* (= KS, PSI, JS, DDM, HDDM, SHAP, Clustering, Autoencoder) | 0 | 0.8725 | — | — |

Three things fall out of this, sharper now than before the D-column fix.

**Retraining every single week now buys statistically nothing.** Going from 0
retrains to 13 improves mean AUC by **+0.0001** — indistinguishable from
noise. Champion vs Challenger's *two* retrains improve it by **+0.0094**. The
mechanism is the same as before: retraining at week *w* trains on everything
up to *w*, so retraining during a noisy stretch permanently folds that noise
into the model.

**Every detector that retrains more than twice is worse than random.** ADWIN
(5 retrains) lands at the **5.5th percentile** of random policies with the
same budget; Prequential AUC (6 retrains) at the **20th**; EDDM, now
retraining 3 times, at the **21.5th**. All three beat never-retraining, which
is exactly why the random control exists — without it, this would look like
three success stories instead of three near-misses.

**No detector clears a 95th-percentile bar this time.** Before the D-column
fix, EDDM had cleared it at the 96th percentile with 2 retrains — that result
depended on which weeks a since-fixed feature-engineering bug made look
turbulent to EDDM's error stream. The best detector now, Champion vs
Challenger, reaches the 81st percentile: real, positive, but not one we'd
call a clean win.

The honest summary is *not* "Champion vs Challenger is the best drift
detector" — it's one dataset and 14 weeks, and the identity of the "best"
detector already changed once when we fixed a bug. It's: **most of these
detectors show no timing skill here, and the busier ones are actively
harmful for what they cost.**

### Registry cost

The replay trained **11** distinct models (baseline + one per drifting week)
where the old per-method design would have trained **~29**. A few more were
trained afterwards purely so the random controls could be evaluated, giving
**15 = 1 + 14 weeks** — exactly the predicted cap. Week 2 shows the sharing
working: every behaviour-based detector flagged drift, and all of them got
the same model.

---

## 8.3 The RL agent: learning *when* and *how* to adapt

Section 8.2 leaves an awkward result: retraining frequency barely matters,
timing matters a lot, and most detectors are bad at timing. You cannot fix that
by moving a threshold, because the problem is not the threshold — it is that a
detector answers the wrong question.

A detector asks **"has drift occurred?"** The operator asks **"given everything
I've seen, and the model I currently have, what should I do this week?"** Those
differ in three ways:

1. **The right answer depends on your model, not just the data.** The same
   signal justifies a retrain if the model is six months stale and nothing if it
   was rebuilt last week. No detector knows its own model's age.
2. **The choice isn't binary.** Between "do nothing" and "rebuild from scratch"
   sit a cheap fine-tune and a free ensemble re-weighting.
3. **Actions have delayed consequences.** Retraining during a turbulent week
   permanently folds that turbulence into a cumulative training set, and the
   cost shows up weeks later.

So we frame it as a decision problem and learn the policy.

### What the agent sees, does, and is paid

**State** — every detector's *continuous* output, plus context about the model:

| Block | Signals |
|---|---|
| Distributional | KS drift fraction, KS mean statistic, PSI drift fraction, PSI-to-null ratio, JS distance |
| Attribution | Attribution drift fraction, mean attribution shift |
| Representation | Cluster distance ratio, cluster PSI, autoencoder z-score |
| Model context | Weeks since full retrain, weeks since partial update, ensemble weight, recent AUC vs baseline, recent F1, progress through the replay |

Note the signals are **numbers, not flags**. "PSI is 0.19" and "PSI is 0.02" are
both "no drift" to a threshold rule and obviously different to a learner.
Thresholding at the detector throws away exactly what the agent needs.

**Actions** — four, not two:

| Action | What it does | Cost |
|---|---|---|
| do nothing | keep the current model | free |
| partial update | fine-tune the last full model on the recent 4 weeks | cheap |
| full retrain | rebuild on all data so far | expensive |
| hedge ensemble | shift weight from the current model toward the stable baseline | free |

**Reward** — how much better than never-retraining, minus what the action cost.
A decision made in week *t* is graded on week *t+1* onward; grading it on its own
week would let the agent retrain on data it's already been scored against.

### Why the model had to change from LightGBM to a neural net

**A gradient-boosted forest cannot be partially updated.** Adding trees isn't
adapting, and there's no principled "fine-tune on last month" for a fixed
forest. With LightGBM the action space collapses back to retrain/don't and the
whole question of *how* to adapt disappears.

A neural net makes the middle ground real — and makes its cost measurable. That
matters for catastrophic forgetting: fine-tuning on recent-only data pulls the
model away from the broader distribution, and the ensemble hedge is how you
recover. Because we can compare "trust the fine-tune fully" against "blend it
back toward the baseline", **forgetting becomes a number we measure, not a
hazard we assume**.

### How PPO is trainable on 14 weeks

PPO needs thousands of episodes; we have *one* 14-week trajectory. Training a
model inside the RL loop would mean tens of thousands of fits.

The trick: under this action set, the model in force is fully determined by
three numbers — `(last full retrain week, last partial update week, ensemble
weight)`. That space is small and enumerable. So we build **every reachable
model once**, cache what each scores on every future week, and the environment
becomes a lookup table. Episodes then cost nothing.

Two rules keep it small, and both are good engineering regardless of the RL:

- **Partial updates aren't chained** — each fine-tunes from the last *full*
  model, not from the previous partial. Chaining would make the model depend on
  the whole action history, and would compound forgetting with no way back.
- **The ensemble is two-component** — current model blended with the baseline.

### What happened

Every policy below acts on the same neural model and the same precomputed
lattice, so the differences are decisions, not luck. This run uses the same
20-feature monitoring set and the D-column fix described in §8.1.

| Policy | Mean AUC | Worst week | Full retrains | Partial updates |
|---|---|---|---|---|
| **RL agent** | **0.8831** | **0.8528** | 1 | 12 |
| *always partial update (no learning)* | 0.8820 | 0.8528 | 0 | 13 |
| always retrain | 0.8711 | 0.8361 | 13 | 0 |
| ADWIN (best classical) | 0.8696 | 0.8424 | 5 | 0 |
| every-2-weeks schedule | 0.8703 | 0.8424 | 6 | 0 |
| Prequential AUC | 0.8664 | 0.8424 | 6 | 0 |
| never retrain | 0.8523 | 0.8214 | 0 | 0 |

The agent wins — best mean AUC and a worst week matching the best naive
policy, which for fraud matters more than the mean (it's the week your losses
spike).

**But look at the second row.** "Fine-tune every week" involves no learning, no
drift signals, and no decisions, and it gets 0.8820. So the gain breaks down as:

| Where the improvement came from | AUC |
|---|---|
| Having a cheap partial-update action at all | **+0.0124** over the best classical detector |
| The learned policy on top of that | +0.0011 |

**About 92% of the win is the action space, not the RL and not the detector
fusion** — an even bigger share than we measured before fixing the D-column
bug and widening the feature set. That's worth saying plainly, because "RL
beats every classical detector by 0.0135 AUC" is technically true and would
give the wrong impression on its own.

### Did combining the detectors actually help? We first said no — that was wrong.

We trained three identical agents that differ only in what they're allowed to
see:

| Agent sees | Mean AUC | Policy it found |
|---|---|---|
| Everything | 0.8831 | 1 retrain + 12 partial |
| Model context only (no detectors) | 0.8820 | 0 retrain + 13 partial |
| Detectors only (no model context) | 0.8831 | 1 retrain + 12 partial |

**"Model context only" landed on exactly the naive fine-tune-every-week
policy — it gained nothing from having context features but no drift
signals. "Detectors only" landed on exactly the full agent's policy** — the
drift signals alone were enough to find the better policy, with no model
context at all.

**This is the opposite of what an earlier run found.** Before we fixed the
D-column bug (§8.1) and widened the monitoring set from 10 to 20 features,
this same ablation showed the reverse: model context alone reproduced almost
all of the full agent's performance, and dropping the drift signals cost
almost nothing. We had reported that as a negative result for feeding
detector outputs to a learned controller. **We no longer believe that
conclusion** — it was measured on a feature pipeline that had a real bug in
it. The corrected version says the drift signals were doing real work the
whole time; the earlier measurement just couldn't see it. We think the
correction is more important than either individual result: a single
feature-engineering bug was enough to flip the answer to "do the detectors
matter," on only 14 data points. Treat both versions as directionally
suggestive, not as a settled fact about detector fusion in general.

Which inputs drive the agent's decisions now:

| Input | Reliance |
|---|---|
| `progress` (how far through the year we are) | **0.470** |
| `weeks_since_reference` | 0.171 |
| `recent_auc_delta` | 0.163 |
| `weeks_since_full_retrain` | 0.139 |
| `ks_mean_statistic` | 0.133 |
| `js_mean_distance` | 0.112 |

The calendar (`progress`) is still the single biggest driver, but far less
dominant than before (it was 4× the next input; now under 3×), and **two
drift signals now sit in the top six inputs** rather than being crowded out
entirely.

### Catastrophic forgetting: measured, not assumed

Because we can compare "trust the fine-tune fully" against "blend it back
toward the baseline", forgetting becomes a number:

- **139** cases evaluated, and hedging would have helped in **all of them**
- **0.0089** max AUC lost by fully trusting a partial update (was 0.0182
  before the fixes — forgetting measures even milder now)
- **0.0020** mean loss where positive
- the best hedge is usually gentle — α = 0.75 in 119 of 139 cases

So forgetting is real but mild here, which is a direct result of the design
choice that partial updates re-derive from the last *full* model instead of
chaining. Chaining would let it compound.

Worth noting: the trained agent **still never used the hedge action**, even
though it's free. That remains a missed opportunity, though a smaller one now
that forgetting itself measures milder than before.

---

## 9. Threshold Summary

| # | Method | Per-feature / stream rule | Method-level rule |
|---|---|---|---|
| 1 | KS | BH-adjusted p < 0.05 **AND** D ≥ 0.10 | ≥60% of monitored features |
| 2 | PSI | PSI ≥ 0.20 **AND** > bootstrap null p99 | ≥60% of monitored features |
| 3 | Jensen-Shannon | JS distance ≥ 0.10 | ≥60% of monitored features |
| 4 | DDM | p+s > p_min + 3·s_min (balanced stream) | stream-level |
| 5 | EDDM | metric < 0.75 × max_metric (balanced stream) | stream-level |
| 6 | HDDM | mean − ref_mean > Hoeffding bound (δ=0.001) | stream-level |
| 7 | ADWIN | change point vs reference stream, δ = 0.002 | stream-level |
| 8 | SHAP | BH-adjusted p < 0.05 **AND** D ≥ 0.10 | ≥60% of monitored features |
| 9 | Clustering | distance_ratio ≥ 1.5 OR cluster_psi ≥ 0.20 | — |
| 10 | Autoencoder | RMSE z-score > 3.0 | — |
| 11 | Prequential AUC | drop > 0.02 **AND** > 2 bootstrap SE | — |
| 12 | Champion vs Challenger | degradation > 0.05 OR OOF gap > max(0.03, SE) | — |

**All twelve** then pass through the 2-of-2 persistence gate before retraining.

DDM/EDDM/HDDM/ADWIN monitor the model's *stream*, not individual features, so
the "≥60% of features" rule doesn't apply to them.

---

## 10. Outputs

| File | Contents |
|---|---|
| `reports/unified_drift_report.json` | Full per-week, per-method detail; every raw metric, so thresholds can be re-applied without rerunning |
| `reports/unified_drift_report.csv` | Flat weekly summary + which version each method had in force |
| `reports/method_week_matrix.csv` | One row per (week, method): threshold, raw/confirmed flag, features crossed |
| `reports/policy_comparison.csv` | Detectors vs never/always/random controls |
| `reports/detector_agreement.csv` | Pairwise Jaccard over flagged weeks |
| `reports/version_week_auc_matrix.csv` | The (version × week) performance matrix |
| `reports/feature_selection_diagnostics.csv` | Per-feature stability, redundancy, monitorability |
| `models/model_v*.pkl` | At most `1 + n_weeks` distinct models |
| `reports/rl_policy_comparison.csv` | RL agent vs every classical policy and the controls |
| `reports/rl_ablation.csv` | Full agent vs context-only vs signals-only |
| `reports/rl_decision_trace.csv` | Week-by-week: action, confidence, and what drove it |
| `reports/rl_policy_reliance.csv` | Which signals the learned policy leans on |
| `reports/forgetting_analysis.csv` | AUC lost by trusting a partial update fully |
| `reports/method_profiles.csv` | Each detector's strengths, blind spots, failure mode |
| `models/rl_drift_agent.pt` | Trained PPO policy |

Run with:

```
python tests/test_drift_engine.py                                  # detector behaviour
python tests/test_rl_agent.py                                      # RL env + PPO learning
python validate_monitor.py --data_dir ./dataset --compare_legacy   # calibration check
python validate_monitor.py --data_dir ./dataset --calibrate_consensus  # consensus-threshold null check
python run_drift_analysis.py --top_k 20 --n_bags 5                 # classical replay
python run_rl_experiment.py --data_dir ./dataset --top_k 20        # RL agent experiment
streamlit run dashboard.py                                         # inspect
```

`tests/test_drift_engine.py` checks each detector on synthetic data where the
answer is known: every detector gets a **null case** it must stay quiet on and a
**signal case** it must fire on. A detector that never fires and one that always
fires are both useless, and only the pair tells them apart.

---

## 11. Summary of What Changed and Why

| # | Problem | Affected | Fix |
|---|---|---|---|
| 1 | Encoders refit on every window | All distributional methods | Freeze encoders; fit once on reference |
| 2 | Frozen encoder silently inert (pandas `str` vs `object`) | All distributional methods | Test "not numeric"; warn on any downstream re-encode |
| 3 | Per-window relative frequency (floor 1/n) | All distributional methods | Removed; keep only the frozen map |
| 4 | Entity aggregates on a near-unique id | All distributional methods | Removed; keep causal sequence features + unseen-entity flag |
| 5 | `TransactionID`/`TransactionDT`/raw UIDs in the model | Everything | Dropped — monotone in time, so KS = 1.0 every window |
| 6 | Significance saturates at large n | KS, SHAP | Effect-size floor (D ≥ 0.10) + capped samples |
| 7 | 10 simultaneous tests, uncorrected | KS, SHAP | Benjamini-Hochberg FDR |
| 8 | PSI thresholds ignore window size | PSI | Bootstrap null per window |
| 9 | KL unbounded / ε-dependent | KL | Jensen-Shannon distance |
| 10 | Model trained 1 tree; F1 = 0.0000 | DDM, EDDM, HDDM | Early stopping 5→100, temporal split, calibrated threshold |
| 11 | 0/1 error dominated by majority class | DDM, EDDM, HDDM | Class-balanced subsampled stream |
| 12 | Challenger scored in-sample | Champion vs Challenger | Out-of-fold predictions + bootstrap SE |
| 13 | Single-window triggers | All | 2-of-2 persistence gate |
| 14 | Per-method models confound seeds | Experimental design | Shared model registry |
| 15 | Unstable, redundant monitoring set | Feature selection | Bagged importance + redundancy pruning |
| 16 | No baseline for comparison | Evaluation | Never/always/random-matched policy controls |
| 17 | Detectors answer "did drift occur?", not "what should I do?" | All | RL controller over the detector ensemble (§8.3) |
| 18 | Only two possible responses to drift | Adaptation | Neural model enables partial update + ensemble hedge |
| 19 | Raw column names unreadable in reports | Reporting | `feature_label()` gives every feature a plain-English name |
| 20 | Dropout in the PPO policy net broke the importance ratio | RL agent | Dropout off by default; covered by a test |

### How we know items 1–3 mattered: the null experiment

Claiming "refitting encoders manufactures drift" is easy. Measuring it is
harder, because comparing the baseline to a later week mixes the artifact
together with *real* drift.

So: take **one** window and split it into two random halves — one large (playing
the reference), one small (playing a monitored week). Two random halves of the
same data cannot differ systematically, so **every alarm is an artifact**.

Run it yourself:

```
python validate_monitor.py --data_dir ./dataset --compare_legacy
```

Both arms use **identical feature definitions**. The only difference is whether
encoders are fitted once on the reference half or refitted on each half — so the
gap is attributable to refitting alone. Over 114 testable features:

| Encoder fitting | KS (p-value only) | KS (+ effect size) | PSI ≥ 0.20 | PSI (calibrated) |
|---|---|---|---|---|
| Refit per window (original) | 35.1% | 24.6% | 14.0% | 14.0% |
| Fitted once, frozen (current) | **0.9%** | **0.0%** | **0.0%** | **0.0%** |

The corrected pipeline sits at or below its nominal 5% significance level on
data with no drift. The original alarms on more than **one feature in three**.

Three things worth noting:

- **The statistical fixes alone don't rescue it.** The effect-size gate helps
  (35.1% → 24.6%) but leaves it an order of magnitude off. The representation
  had to be fixed first — a well-specified test on a badly-specified feature is
  still a bad monitor.
- **Several of these bugs were introduced while fixing the previous one**, and
  the null experiment is what caught them. The per-window relative frequency hit
  KS D = 0.88 on data with no drift. The entity aggregates hit D = 0.61 —
  exactly the rate of entities not seen in the reference.
- **The frozen encoder was silently doing nothing at first.** It checked
  `dtype == 'object'`, but pandas 3.x gives string columns a `str` dtype, so
  every categorical fell through to a per-window `factorize` further downstream.
  The code looked correct and was completely inert. `preprocess_and_align` now
  logs a loud warning if anything reaches it un-encoded.

This check is cheap and worth running before trusting any drift study.
