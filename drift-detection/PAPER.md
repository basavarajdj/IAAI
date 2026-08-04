# Retraining Policies Under Concept Drift: A Controlled Comparison of Twelve Detectors on Transaction Fraud

**Research paper narrative and experimental protocol**
Companion to the implementation in this repository.

---

## Abstract (draft)

Production fraud-detection models are retrained on schedules that are rarely
justified empirically. The literature offers a large menu of drift detectors —
distributional tests, error-stream monitors, explanation-space monitors,
representation-space monitors, and shadow-model comparisons — but studies that
compare them typically evaluate *detection* on synthetic streams rather than
evaluating the *retraining policy* each detector induces on a real,
imbalanced, temporally structured problem.

We conduct such a comparison on the IEEE-CIS transaction fraud dataset, replayed
as a weekly stream over a six-month horizon. Twelve detectors each control an
independent retraining policy over a **shared model registry**: all detectors
begin from one baseline model, and when several detectors demand retraining in
the same window they receive the *identical* refreshed model. This removes
seed-level confounding, so any difference in downstream performance is
attributable solely to *when* a detector chose to retrain, and reduces the
number of distinct models from O(detectors x windows) to at most
`1 + n_windows`.

Our central empirical finding is negative and, we argue, generalisable: **much
of the detector "signal" in the original configuration of this pipeline was an
artifact of the measurement apparatus rather than of the data.** We quantify
this with a null experiment — splitting a single window into two random halves,
between which no drift can exist by construction. Holding feature definitions
fixed and varying only whether encoders are refitted per window, the original
design reports drift in **35.1% of features** where none exists; the corrected
pipeline reports **0.9%**, at or below the nominal significance level.

We identify and correct artifacts in three groups: **representation** (encoder
refitting, and features that are structurally non-transferable across windows),
**statistics** (sample-size-driven significance, uncorrected multiple testing,
uncalibrated PSI thresholds, an unbounded divergence), and **evaluation** (an
in-sample shadow-model comparison, and an error stream that was constant by
construction because the classifier — trained to a single tree by a
misconfigured early-stopping rule and thresholded at 0.5 under 3.5% prevalence —
never predicted the positive class at all).

With the measurement corrected, the policy comparison inverts a common
assumption. Retraining in *every* one of the 14 windows improves mean
out-of-sample AUC by only **+0.0025** over never retraining, while EDDM's two
well-timed retrains improve it by **+0.0149** — six times the gain at 15% of the
cost. Frequency is nearly worthless on this stream; timing is decisive.

The frequency-matched random control, which the shared registry makes affordable,
is what exposes this. Judged on AUC alone, the two most active detectors look
like successes; against random policies of identical cost they sit at the **3.5th
and 11.5th percentiles** — their timing is worse than choosing weeks by coin
flip. Only one of twelve detectors (EDDM, 96th percentile) demonstrates genuine
timing skill. We also find that all feature-marginal detectors remain silent
throughout while the behaviour-based detectors carry the entire signal,
indicating concept drift rather than covariate shift — and implying that
label-free monitoring, the most common production configuration, would have
detected nothing at all here.

---

## 1. Introduction

### 1.1 The operational question

A deployed fraud classifier faces a non-stationary world: fraud tactics adapt,
merchant mixes change, and the population of legitimate customers shifts.
"Retrain weekly" and "retrain when someone complains" are the two policies most
commonly observed in practice. Drift detection promises a principled middle
ground — retrain when, and only when, evidence says the model has gone stale.

Whether that promise is realised depends on a question the drift literature
mostly does not ask: *what retraining schedule does a given detector actually
produce on real data, and is the resulting model better than the one you would
have had under a naive policy?* Detection accuracy on a synthetic stream with a
known change point is a proxy for this, and a loose one. It says nothing about
false-alarm cost, nothing about detectors whose alarms are correlated with each
other, and nothing about the interaction between detection and the severe class
imbalance that characterises fraud.

### 1.2 Contributions

1. **A shared-registry experimental design** that makes detector comparison
   causally clean and computationally tractable (Section 5).
2. **A null-experiment protocol for validating a drift monitor**, and the
   identification and correction of the artifact classes it exposes
   (Sections 3.2 and 4). Each is a mistake that is easy to make, hard to see,
   and — we suspect — common. The protocol itself is the transferable part: a
   monitor that reports drift between two random halves of the same window is
   miscalibrated, and this can be checked before any drift study is run.
3. **A stability-aware, redundancy-pruned feature selection procedure** for
   choosing *what to monitor*, which is a different problem from choosing what
   to train on (Section 3).
4. **Two detector additions** motivated by the imbalance structure of fraud:
   a class-balanced error stream for the classical error-rate monitors, and a
   direct prequential-AUC degradation monitor (Section 4.5).
5. **A persistence gate** relating per-window false-positive rate to retraining
   rate, with an explicit delay/precision trade-off (Section 4.7).

### 1.3 The thesis

The paper's argument is that **drift detection is a measurement problem before
it is a modelling problem.** Every detector in our suite computes a statistic
comparing a reference window to a current window. If any part of the pipeline
between raw data and that statistic is itself refit per window, the statistic
measures the refitting. We found this to be the dominant effect. We suspect
this is under-reported because a pipeline exhibiting it *looks* like it is
working — it produces alarms, the alarms produce retraining, and the retrained
models perform acceptably. The failure is silent.

---

## 2. Data and Stream Construction

### 2.1 Source

IEEE-CIS Fraud Detection: 590,540 transactions with 394 columns, joined to an
identity table on `TransactionID` (left join; identity coverage is partial).
Positive-class prevalence is **3.50%**. Time is encoded as `TransactionDT`, a
seconds-since-epoch offset spanning approximately six months.

The severe imbalance is not incidental to this study — it is the mechanism
behind two of the five artifacts we identify (Sections 4.5 and 4.6). Any drift
study on fraud, churn, or failure prediction inherits the same exposure.

### 2.2 Stream protocol

- Sort by `TransactionDT`.
- **Baseline / reference window:** first 90 days. Used to fit the feature
  encoders, select the monitoring set, and train model version `v0`.
- **Monitored windows:** consecutive 7-day windows over the remainder, giving
  ~14 weekly evaluation points.
- **Retraining data:** cumulative — a retrain at week *w* uses all data from
  the start of the stream through week *w*. This matches operational practice
  (nobody discards history) and means the training distribution is itself
  slowly drifting toward the present.

We deliberately do **not** hold out a future test period. The evaluation
quantity is each detector's out-of-sample performance on the *next* window it
has not yet trained on, accumulated over the replay — a prequential protocol,
which is the correct evaluation mode for a streaming policy.

### 2.3 Label availability

We adopt the standard simplifying assumption that labels for window *w* are
available when window *w* is evaluated. In production, fraud labels arrive with
a lag of weeks (chargeback cycles). This assumption favours the
performance-aware detectors (DDM, EDDM, HDDM, prequential AUC,
champion-vs-challenger) relative to the label-free distributional detectors
(KS, PSI, JS, clustering, autoencoder, SHAP), and any real deployment must
discount their results accordingly. We flag this as the single largest threat
to external validity and revisit it in Section 8.

---

## 3. Feature Engineering and the Monitoring Set

### 3.1 Engineered features

The representation follows established practice for this dataset:

| Group | Construction | Rationale |
|---|---|---|
| Distance | `log1p` of `dist1`, falling back to `dist2` | Heavy-tailed; log-scale is better behaved |
| Temporal | Day index, cyclic sin/cos hour encoding, weekday x hour | Fraud has strong diel structure |
| Entity (UID) | `(day - D1)` + email domain -> `_uid1`; card1 + addr1 + `_uid1` -> `_uid2` | Proxy identity for a card-holder, the key aggregation unit |
| Sequence | Inter-transaction gap, absolute percentage amount change, within-entity sequence index | Velocity signals: fraud arrives in bursts |
| Amount | Log amount, decimal part, decimal length | Round-number and cents-pattern effects |
| Aggregates | Per-`_uid2` max of C-columns; per-entity amount max/mean/var | Entity-level behavioural envelope |
| Frequency | Relative frequency of C-columns and entity keys | Popularity/rarity |
| C / D / M blocks | Non-zero counts, sums, NA-pattern bit-strings, day-relative D-columns | Compact summaries of large opaque blocks |
| V block | MinMax scaling then 2-component PCA, plus NA count and sum | 339 columns of unlabelled engineered features |

After a near-constant-column filter, the representation is **149 features**.

### 3.2 The representation must be frozen (critical)

This is the paper's first and most consequential methodological point.

Eight of the constructions above are **learned encoders**: they estimate
something from the data they are given. Ordinal encoding of categoricals,
frequency encoding, group aggregates, the MinMax scaler, the PCA rotation, and
the redundancy filter all fall in this class. The original implementation
called a single stateless `apply_feature_engineering(df)` on the baseline
window and, independently, on every weekly window.

Consider what that does to a drift statistic:

- **Ordinal encoding is assignment-order dependent.** `pd.factorize` numbers
  categories by order of first appearance. An email domain encoded as `3` in the
  baseline may be `17` next week with nothing having changed in the world. Every
  distributional test on that column then reports a large shift, forever.
- **Raw count encodings scale with window length.** The baseline covers 90 days;
  each monitored window covers 7. A category appearing at a perfectly constant
  *rate* has a raw count roughly 13x smaller in the weekly window. KS, PSI and
  KL all register this as drift. It is arithmetic.
- **PCA components are sign- and rotation-arbitrary.** Refitting per window can
  flip a component, inverting the feature.
- **A per-window redundancy filter changes the schema itself**, so the aligned
  design matrix is zero-padded differently from week to week.

All four are artifacts of the *encoder*, not the data-generating process, and
all four inflate every distributional detector simultaneously. We therefore
adopt the rule:

> **Freeze the representation; version only the model.**
> Encoders are fitted once on the reference window and replayed unchanged.
> Retraining updates model weights, never the feature space.

This has a second, structural benefit: it is what makes the shared model
registry coherent. Model versions trained at different weeks live in the same
feature space, so they are directly comparable and interchangeable.

A subtlety: freezing the encoders means genuine *representation* drift (a new
category the baseline never saw) does not silently deform the features — it
maps to a reserved `UNSEEN` code and becomes an observable, monitorable signal
in its own right. This is a feature, not a limitation.

### 3.2.1 Validating the fix: the null experiment

Asserting that refitting manufactures drift is easy; measuring it is not,
because a baseline-vs-week comparison confounds the artifact with genuine
drift. We therefore use a **null experiment**: take a single window, split it
into two random halves (one large, one small, preserving the size asymmetry of
the real setting), and run the full monitoring stack on the pair. No drift can
exist between two random halves of the same data, so **every alarm is an
artifact**.

This is, we suggest, the minimum calibration check any drift monitor should pass
before it is trusted, and it is cheap.

Running it exposed two things we had not anticipated, and one of them was a bug
we had just introduced:

**(a) A per-window relative frequency is still window-size dependent.** Our
first correction replaced raw count encoding with *relative* frequency, which
removes the gross scaling problem. But we also emitted the *current window's*
own relative frequency as a feature. Its resolution floor is 1/n: a category
seen once in a 2k-row window encodes as 5e-4, while the same category seen once
in a 24k-row window encodes as 4e-5 — a 10x difference with nothing having
changed. Under the null these features reached **KS D = 0.88**, the largest
false signal in the entire feature set. They were removed.

**(b) Entity-keyed aggregates are not transferable across windows.** Features of
the form "max of C1 over this card's history", keyed on the near-unique `_uid2`,
are computed in-sample during fitting but are undefined for the majority of any
future window, because most entities are new. No fill value repairs this: any
choice creates a point mass, and the resulting KS distance from the reference
equals the unseen-entity rate — **D = 0.61** here — regardless of whether
anything drifted. These features also flatter the baseline model, which sees
them computed in-sample, and then silently degenerate to a constant at
inference. We removed the `_uid2`-keyed aggregates and frequency encodings,
retaining the *causal sequence* features (inter-transaction gap, amount change,
sequence index) which capture the same entity behaviour and, being computed over
the full frame, remain comparable across windows. The "new entity" signal is
retained as an explicit indicator column, where it can be monitored on purpose.

**(c) Identifier and time columns had to be removed.** `TransactionDT` and
`TransactionID` both increase monotonically with time. Left in the design
matrix they are perfect proxies for "when did this happen": the model can split
on them, and because every monitored window occupies a disjoint range from the
reference, every distributional test reports KS D = 1.0 forever regardless of
drift. Their legitimate content — hour of day, day of week, ordering — is
already carried by the cyclic time features.

We also found that our frozen ordinal encoder was **silently inert**: it tested
`dtype == 'object'`, but pandas 3.x assigns string columns a dedicated `str`
dtype, so every categorical fell through to a downstream per-window
`factorize`. The frozen encoding was not actually in force. The alignment step
now raises a warning rather than quietly re-encoding, so the failure cannot
recur unobserved.

**Result.** Both arms use **identical feature definitions**; the only difference
is whether encoders are fitted once on the reference half or refitted
independently on each half. This isolates the refitting effect from the choice
of features. Over 114 testable features:

| Encoder fitting | KS (p-value only) | KS (+effect size) | PSI >= 0.20 | PSI (calibrated null) |
|---|---|---|---|---|
| Refit per window (original design) | 35.1% | 24.6% | 14.0% | 14.0% |
| Fitted once, frozen (corrected) | **0.9%** | **0.0%** | **0.0%** | **0.0%** |

The corrected pipeline is at or below its nominal significance level on data
containing no drift. The original is not remotely calibrated: more than one
feature in three alarms on pure noise.

We stress that the second row is *not* achieved by the statistical corrections
of Section 4 alone. The effect-size gate helps the refitting arm (35.1% ->
24.6%) but leaves it an order of magnitude away from calibrated. **The
representation had to be fixed first**: a well-specified test on a
badly-specified feature is still a bad monitor.

### 3.3 Causal sequence features computed globally

The per-entity lag features are window-length-sensitive in a subtler way:
computed inside a 7-day window, a card's "time since previous transaction" can
only look back 7 days, whereas in the 90-day baseline it looks back 90. The lag
distribution is therefore truncated differently per window — another
manufactured shift.

We compute these once over the fully sorted frame. They remain strictly causal
(each row references only earlier rows), so this is not leakage: in production
a card's previous transaction is genuinely available at scoring time.

### 3.4 Selecting what to monitor

Choosing the features to *monitor* is not the same problem as choosing the
features to *train on*. The original pipeline took the top-10 by a single
LightGBM gain ranking. Three objections:

1. **Instability.** Gain comes from one stochastic fit. A different seed yields
   a materially different top-10, so the headline statistic "fraction of
   monitored features that drifted" inherits that arbitrariness.
2. **Redundancy.** Gain splits arbitrarily among correlated features, and
   correlated features drift *together*. Ten features representing three
   underlying signals give a "6 of 10 drifted" vote that is not six independent
   pieces of evidence. This inflates apparent consensus and understates its
   variance — a serious problem for any rule of the form "flag if >60% of
   monitored features drift."
3. **Monitorability.** A near-constant or three-valued feature may carry real
   gain but is untestable: KS is dominated by ties and PSI bucketing collapses.

Our procedure (`feature_selection.py`):

- **Stage 1 — Bagged importance.** Fit *B* = 5 models on bootstrap resamples
  with distinct seeds. Aggregate by mean reciprocal rank; record each feature's
  *selection frequency* (fraction of bags placing it in the top-*M*).
- **Stage 2 — SHAP corroboration.** Global mean |SHAP| from the reference
  model, as a split-count-independent second view. Gain is biased toward
  high-cardinality features; SHAP attribution is not, so disagreement is
  diagnostic.
- **Stage 3 — Monitorability filter.** Drop features with fewer than 10
  distinct values or effectively zero reference variance.
- **Stage 4 — Greedy redundancy pruning.** Walk the ranking top-down, skipping
  any candidate with |Spearman rho| >= 0.90 against an already-selected feature.

We report **Nogueira's stability index** for the bagged selection as the
reproducibility statistic, and the maximum pairwise rho among the selected set as
evidence that the consensus vote aggregates approximately independent tests.
On the pilot run these were 0.84 and 0.73 respectively, with 39 features
rejected as unmonitorable and 1 as redundant.

---

## 4. Drift Detection: Twelve Methods and Five Corrections

We organise the detector suite by **what each one observes**, which is the axis
that determines its label requirements, its cost, and its failure modes.

| # | Detector | Observes | Needs labels? | Family |
|---|---|---|---|---|
| 1 | KS test | Feature marginals | No | Distributional |
| 2 | PSI | Feature marginals (binned) | No | Distributional |
| 3 | Jensen-Shannon | Feature marginals (binned) | No | Distributional |
| 4 | DDM | Error stream | Yes | Performance |
| 5 | EDDM | Inter-error distances | Yes | Performance |
| 6 | HDDM | Error stream (Hoeffding) | Yes | Performance |
| 7 | ADWIN | Prediction stream | No | Performance-adjacent |
| 8 | SHAP drift | Attribution distributions | No | Explanation-space |
| 9 | Clustering | Joint geometry (K-Means) | No | Representation |
| 10 | Autoencoder | Reconstruction error | No | Representation |
| 11 | Prequential AUC | Ranking quality | Yes | Performance |
| 12 | Champion vs Challenger | Value of retraining | Yes | Shadow model |

Sections 4.1-4.7 present the five corrections, each attached to the detectors
it affects.

### 4.1 Correction 1 — Significance is not evidence at scale (KS)

The two-sample KS p-value answers "could this difference have arisen by
chance?" With a 90-day reference (~10^5 rows) against a 7-day window (~10^4),
the critical statistic at alpha = 0.05 is

    D_crit = 1.358 * sqrt((n + m) / (n * m)) ~= 0.015

A shift of 1.5% of probability mass — operationally meaningless — is
"statistically significant." The original pipeline reported KS drift in **14 of
14 weeks**. It was measuring sample size.

This is not a KS-specific pathology. *Any* monitoring rule built on unqualified
null-hypothesis significance testing over large windows degenerates identically.
It is, in our view, the single most likely defect in a deployed drift monitor.

**Correction.** Require both significance *and* a practical effect size:
D >= 0.10, i.e. the empirical CDFs must separate by at least 10 percentage
points somewhere. D is itself a bounded, interpretable effect size, so this is
a statement about magnitude rather than about *n*. We additionally cap both
windows at 20,000 rows so the p-value retains discriminating power instead of
saturating at zero; subsampling leaves the point estimate of D unbiased and
only widens its variance, which is the honest trade.

*Validation.* On 100k vs 10k samples from an identical Gaussian, the corrected
rule reports D = 0.007 and no drift; on a genuine 0.5-sigma mean shift it reports
D = 0.202 and drift.

### 4.2 Correction 2 — Multiple testing across the monitored set

Monitoring *K* features means running *K* simultaneous hypothesis tests every
window. At alpha = 0.05 with K = 10 and no correction, the probability of at least
one false positive in a *stable* week is 1 - 0.95^10 = 40%; over 14 weeks it is
a near-certainty. Bonferroni controls this but is far too conservative when
features are correlated — and monitored features usually are.

**Correction.** Benjamini-Hochberg FDR control across the monitored set, for
both the KS and SHAP detectors. BH controls the expected *proportion* of false
discoveries, which is the correct target given that the decision rule is itself
a proportion ("what fraction of features drifted?"). The redundancy pruning of
Section 3.4 additionally moves the tests closer to the independence BH assumes.

### 4.3 Correction 3 — PSI thresholds are sample-size dependent

The canonical PSI bands (<0.10 stable, 0.10-0.20 moderate, >0.20 drift) are
credit-scorecard folklore calibrated to an unstated sample size. PSI has a known
null distribution: asymptotically chi2(B-1)/n for B buckets, so its expected
value under *no drift* scales as (B-1)/n. A 1,000-row window expects PSI ~= 0.009
from noise alone; a 100-row window expects ~= 0.09, essentially the "moderate
shift" band. Comparing a weekly window against a fixed constant conflates drift
with window size.

**Correction.** A bootstrap null calibrated to the current window: resample
`n_curr` rows from the reference `B` = 100 times and compute PSI of each
resample against the reference. Declare drift only when observed PSI exceeds
both the folklore threshold *and* the null's 99th percentile.

*Validation.* At n = 300 with no drift, observed PSI = 0.020 against a null 99th
percentile of 0.087 — correctly suppressed, where the fixed 0.10 warning band
would have been approached. A genuine 1.2-sigma shift yields PSI = 1.35 against a
null of 0.006 — correctly flagged.

### 4.4 Correction 4 — KL divergence is the wrong statistic

KL divergence is unbounded, asymmetric, and undefined wherever the current
window places mass the reference did not — which for any rare-category feature
happens routinely. The standard remedy, epsilon-smoothing, makes the divergence's
*absolute value* a function of the arbitrary epsilon, so a fixed threshold
(KL >= 0.5) means different things for different features.

**Correction.** Use the **Jensen-Shannon distance** as the decision statistic:
symmetric, always finite, and bounded in [0, 1] after the square root, so one
threshold is comparable across features of any scale or entropy. Raw KL is still
reported for continuity with prior results.

### 4.5 Correction 5 — The error stream was constant by construction

This is the most damaging defect we found, and the one with the widest
implications for the fraud-detection literature specifically.

The baseline model in the original pipeline had **F1 = 0.0000 on every window**.
Two compounding causes:

1. **Early stopping patience of 5 rounds at a learning rate of 0.01.** At that
   learning rate the first few dozen boosting rounds barely move validation AUC,
   so training terminated at **iteration 1**. The "model" was a single tree.
2. **A fixed 0.5 decision threshold on a 3.5%-prevalence ranking model.**
   Essentially no probability mass sits above 0.5, so the classifier predicted
   "not fraud" for every transaction.

The consequence for the study is severe. DDM, EDDM and HDDM all monitor a
*binary error stream*. If the model always predicts the negative class, the
error stream is identically the label vector — its rate is the prevalence,
constant at 3.5%, containing no information about the model at all. DDM's
alarms in 14 of 14 weeks were fluctuations in the weekly fraud rate, not model
degradation.

**Correction (a): the model.** Early-stopping patience raised to 100 rounds,
scaled to the learning rate; validation split changed from random-stratified to
a **temporal** last-20% holdout (a random split places same-day, often
same-card rows on both sides, so early stopping was tuning against a leak); and
the decision threshold **calibrated on validation** to maximise F1, stored on
the model and used consistently for every downstream error stream and metric.
Validation F1 moves from 0.000 to ~0.47 at a threshold of ~0.40.

**Correction (b): the stream.** Even with a working classifier, a 0/1 error
stream at 3.5% prevalence is ~96% determined by the treatment of legitimate
transactions. A model can lose *all* of its fraud-catching ability while the raw
error rate moves by only 3.5 percentage points — inside the noise band DDM's
`p_min + 3*s_min` rule tolerates.

We therefore feed the classical monitors a **class-balanced error stream**:
every positive, plus an equal number of randomly drawn negatives, kept in
temporal order. The stream's mean is then the balanced error rate
0.5*(FNR + FPR).

An important implementation point: we balance by *subsampling*, not by
*reweighting*. Instance weights of 1/0.035 ~= 29 are not bounded by 1, and DDM's
Bernoulli variance term and HDDM's Hoeffding bound both require a bounded
stream; clipping the weights back to 1 undoes the rebalancing entirely. Our
first implementation made exactly this mistake and a unit test caught it.
Subsampling preserves a genuine Bernoulli stream in {0, 1}.

*Validation.* Under a total recall collapse (a model that ranks no fraud above
any legitimate transaction), the 0/1 stream's mean moves by 0.036 while the
balanced stream's moves by 0.500 — a 14x amplification of exactly the failure an
operator most needs to detect.

### 4.6 Correction 6 — The shadow-model comparison was in-sample

The natural champion-vs-challenger implementation — train a challenger on the
current window, score both on the current window — is not a fair comparison.
The champion is evaluated strictly out-of-sample; the challenger is scored on
the rows it was just fitted to. A 500-tree gradient-boosted model memorises a
10k-row window substantially, so its in-sample AUC is inflated by more than the
0.03 gap threshold. The detector fires on the challenger's overfitting rather
than on the champion's staleness — and fires *every* week, drifting or not,
because the bias is constant.

**Correction.** The challenger is evaluated via **out-of-fold predictions**
(2-fold stratified), placing both models on out-of-sample footing. We
additionally require the AUC gap to exceed its own **bootstrap standard error**:
a gap of 0.04 on a week containing 200 fraud cases has an SE of comparable size,
and declaring drift from it is reading noise. Because one of the OOF fold models
is reused for the weekly feature-importance snapshot, the corrected version
costs no more fits than the biased one.

### 4.7 New detectors, and the persistence gate

**Prequential AUC (new).** Every error-stream detector is a proxy for the
quantity an operator actually cares about — has ranking quality dropped? Under
3.5% prevalence those proxies are dominated by the majority class. We monitor
windowed out-of-sample AUC directly against the AUC at the incumbent model's
adoption, declaring drift only when the drop exceeds both an absolute floor
(0.02) and two bootstrap standard errors. Unlike champion-vs-challenger, this
requires no second model.

**HDDM (new).** DDM's control limits assume a Bernoulli stream whose variance
shrinks as 1/n, making it progressively *harder* to trigger the longer a model
has been stable — precisely the regime where drift is most likely. HDDM
(Frias-Blanco et al., 2015) replaces the normal-approximation limits with a
distribution-free Hoeffding bound requiring only boundedness in [0, 1].

**Persistence gating (all detectors).** Retraining is expensive and resets every
downstream reference statistic. A single-window trigger therefore converts a
detector's per-window false-positive rate *directly* into a retraining rate —
which is how the original pipeline reached "DDM retrained in 14 of 14 weeks."

We apply the standard control-chart remedy: require the detector to fire in
**k of the last n** windows. For approximately independent windows this reduces
the false-alarm probability from p to roughly C(n,k) * p^k, at the cost of
delaying a genuine detection by at most (k-1) windows. With our default
k = n = 2, a 20% weekly false-positive rate becomes ~4%, and a real persistent
drift is acted on one week later. We report both raw and confirmed flags so the
gate's cost is auditable.

---

## 5. Experimental Design: The Shared Model Registry

### 5.1 The confound in per-detector models

The original design gave each detector its own private champion. This produces
two problems.

The **statistical** problem is the serious one. Two detectors that retrain in
exactly the same week end up with *different* models, because each fit has its
own bagging and feature-sampling randomness. Any difference in their subsequent
performance is then partly seed noise rather than a consequence of their
retraining policy — which is the only thing the study is trying to measure.

The **computational** problem compounds it: N models at baseline (N independent
fits of the identical thing) plus one fit per detector per drifting week.
For 12 detectors over 14 weeks this approaches 12 + 12x14 = 180 fits, of which
at most 15 are distinct.

### 5.2 Design

A model version is identified by **the data it was trained on**, which in a
cumulative-window replay is fully determined by the week boundary:

    version 0  ->  baseline window (first 90 days)
    version w  ->  all data from the start through week w

The registry trains **at most one model per week**, regardless of how many
detectors requested it, and hands the identical booster to all of them. Each
detector holds only a pointer, `method_versions[detector] = version_id`. A
detector that does not confirm drift in week *w* keeps its existing pointer —
it continues using the older model, which is the intended semantics.

**Upper bound on distinct models: `1 + n_weeks`** — 15 for a 14-week replay,
independent of detector count.

Because every detector retraining in week *w* now shares byte-identical
weights, differences in measured performance are attributable purely to *when*
each chose to retrain. That is the causal contrast the study requires.

### 5.3 What is shared and what is not

The registry shares the expensive, detector-independent objects and keeps the
detector-specific ones separate:

| Object | Shared? | Why |
|---|---|---|
| Model weights | **Yes**, per version | Identical training data implies an identical model |
| Reference sample (20k rows) | **Yes**, per version | The reference *is* the version's training data |
| Reference predictions | **Yes**, per version | Determined by model + reference |
| SHAP / K-Means / autoencoder artefacts | **Yes**, per version | Fitted against the version's reference; per-detector fitting would produce spuriously different references for detectors looking at the same data |
| Weekly prediction vectors | **Yes**, per version | Detectors on the same version score identically |
| DDM / EDDM / HDDM / prequential trackers | **No** | Each detector's decision history is its own |
| Persistence gate state | **No** | Ditto |
| Version pointer | **No** | This *is* the policy being compared |

Reference windows are subsampled to 20,000 rows. This resolves a KS statistic
to ~0.01 — far finer than any threshold we apply — while bounding memory so all
15 versions can be held simultaneously.

### 5.4 Detector state after adoption

When a detector adopts a new version, its accumulated history refers to a model
that no longer exists. We therefore reset its tracker and persistence gate on
adoption, and re-anchor the prequential AUC detector's reference. This is a
deliberate, logged event — in contrast to the *implicit* per-window resetting
that an earlier version of this pipeline performed by constructing fresh
trackers every week, which destroyed the cross-window accumulation DDM and EDDM
depend on.

---

## 6. Evaluation Protocol

For each detector we report:

**Policy characteristics**
- Retraining count and the specific weeks chosen.
- Raw vs. persistence-confirmed alarm rate (isolating the gate's effect).
- Pairwise alarm agreement between detectors (Jaccard over flagged weeks) —
  detectors that always agree are not 12 independent opinions.

**Downstream model quality** (the outcome that matters)
- Prequential AUC and F1 on each window, under the model the detector's policy
  had in force at that time.
- Mean and worst-case window AUC over the replay.
- **Area between the policy's AUC curve and the always-retrain upper bound** —
  the performance forgone by retraining less often.

**Cost**
- Distinct models trained under the policy.
- Cost-normalised quality: AUC gained per retraining event.

**Reference policies** (essential, and absent from the original design)
- *Never retrain* — baseline `v0` throughout. The floor.
- *Always retrain* — retrain every window. The practical ceiling, and the cost
  upper bound.
- *Random retraining* matched to each detector's retraining count. **This is the
  critical control.** A detector that retrains 6 times is only interesting if it
  beats a policy that retrains 6 times at random. Without this comparison,
  apparent detector performance may be entirely explained by retraining
  frequency — and we expect several detectors will not clear this bar.

### 6.1 The version lattice makes all of this free

Evaluating 200 random control policies per detector by actually retraining
would be tens of thousands of fits. It is instead a table lookup, because of a
property of cumulative retraining: **the reachable model space is exactly the
lattice `{v0, v1, ..., v_nweeks}`**, one version per possible retraining week.

We therefore materialise the lattice once — at most `1 + n_weeks` = 15 models,
which is the registry's bound anyway — and precompute the
(version x week) out-of-sample performance matrix, populating only cells where
the version had not already trained on that week. A policy is then just a map
from week to version id, and its prequential performance is read straight off
the matrix.

This is what turns the random control from an aspiration into a routine part of
every run (`policy_evaluation.py`). It also means the *same* 15 models support
every policy we or a reader might wish to test after the fact, without
retraining anything.

### 6.2 Detector agreement

We report the pairwise Jaccard overlap of the weeks each detector flagged.
Twelve detectors that always agree are not twelve independent opinions, and a
consensus rule built on them would be badly overconfident. The interesting case
is high agreement *across* families — distributional and performance-aware
detectors firing together indicates drift visible in both the marginals and the
concept, whereas performance-only agreement indicates pure concept drift.

---

## 7. Results

### 7.1 Established (measured, reproducible)

**Monitor calibration.** Holding feature definitions fixed and varying only
encoder fitting, the original design alarms on 35.1% of features between two
random halves of the same window; the corrected pipeline alarms on 0.9%
(Section 3.2.1, reproducible via `validate_monitor.py`).

**The classifier never predicted the positive class.** Every model in the
original reports had F1 = 0.0000 across all 14 windows, from a 5-round
early-stopping patience at learning rate 0.01 (training halted at iteration 1 —
a single tree) combined with a fixed 0.5 threshold at 3.5% prevalence. The
error-stream detectors were therefore monitoring a constant equal to the weekly
fraud rate. After correction the model trains to ~180 trees with validation
F1 ≈ 0.47 at a calibrated threshold of ≈ 0.40.

**Detector properties.** Each correction is covered by a paired null/signal test
(`tests/test_drift_engine.py`, 19 tests). Notably, a total recall collapse moves
the 0/1 error stream by 0.036 and the class-balanced stream by 0.500.

**Original alarm rates**, for reference: KS flagged 14/14 weeks (sample size),
DDM flagged 14/14 (constant error stream), champion-vs-challenger fired
continuously (in-sample challenger bias). All three were nominally "the pipeline
working."

### 7.2 Detector behaviour after correction

Over the 14-week replay of the full dataset, the confirmed (persistence-gated)
alarms were:

| Detector | Weeks confirmed | Count |
|---|---|---|
| Prequential AUC | 2, 5, 7, 9, 11, 13 | 6 |
| ADWIN | 2, 4, 8, 10, 12 | 5 |
| EDDM | 2, 9 | 2 |
| Champion vs Challenger | 2, 8 | 2 |
| SHAP | 2 | 1 |
| Clustering | 12 | 1 |
| KS, PSI, Jensen-Shannon, DDM, HDDM, Autoencoder | — | 0 |

Two things are worth drawing out.

**The saturated detectors are now silent.** KS fell from 14/14 weeks to 0/14,
and DDM from 14/14 to 0/14. Neither was "fixed" by weakening a threshold: KS was
reporting sample size, and DDM was watching a constant error stream produced by
a classifier that never predicted the positive class. Removing both artifacts
removed the alarms entirely.

**The drift here is concept drift, not covariate shift.** Every
feature-marginal detector (KS, PSI, Jensen-Shannon) is quiet, while the
detectors that observe the model's *behaviour* — prequential AUC, ADWIN on the
prediction stream, EDDM, champion-vs-challenger — carry essentially all the
signal. The distribution of transactions is stable; the relationship between
those transactions and whether they are fraudulent is not.

This has a direct operational implication, and it is the practical payoff of
the whole exercise: **on this problem, label-free monitoring would have detected
nothing.** A team monitoring only feature distributions — the most common
production setup, because it needs no labels — would have seen a flat dashboard
for six months while the model decayed. Only detectors requiring labels (or, in
ADWIN's case, the prediction stream) fired. Section 2.3's label-latency caveat
therefore bites hardest exactly where it matters most.

The single week on which detectors from *both* families agree is week 2 (five
detectors, including SHAP), which is the one point in the replay where the
change appears to reach the feature space as well.

### 7.3 Retraining policies: frequency buys almost nothing, timing buys everything

Mean out-of-sample AUC of the model each policy actually had in force, over the
14-week replay. `random_control_percentile` is the policy's rank against 200
random policies that retrain the *same number of times* at randomly chosen weeks.

| Policy | Retrains | Mean AUC | Worst week | Random control | Clears 0.95? |
|---|---|---|---|---|---|
| **EDDM** | 2 | **0.8844** | 0.8553 | **0.960** | **yes** |
| Champion vs Challenger | 2 | 0.8838 | 0.8553 | 0.945 | no (marginal) |
| SHAP | 1 | 0.8790 | 0.8553 | 0.755 | no |
| Prequential AUC | 6 | 0.8731 | 0.8553 | 0.115 | no |
| *always_retrain* | 13 | 0.8720 | 0.8553 | — | — |
| ADWIN | 5 | 0.8717 | 0.8553 | 0.035 | no |
| Clustering | 1 | 0.8698 | 0.8515 | 0.195 | no |
| *never_retrain* (= KS, PSI, JS, DDM, HDDM, Autoencoder) | 0 | 0.8695 | 0.8515 | — | — |

Three findings, in descending order of how much they surprised us.

**(1) Retraining every week is nearly worthless here.** Going from zero retrains
to thirteen — the maximum possible, and the policy most production teams
approximate — improves mean AUC by **+0.0025**. EDDM's two retrains improve it
by **+0.0149**, six times as much at 15% of the cost. The usual assumption that
more retraining is safer-but-wasteful does not hold: on this stream, *when* you
retrain dominates *how often*, and the always-retrain policy is beaten by two
well-chosen updates.

The likely mechanism is the cumulative window. Retraining at week *w* trains on
everything through *w*, so a retrain during a noisy stretch dilutes the model
with data from that stretch and cannot be undone; the next retrain inherits it.
Frequent retraining therefore samples good and bad moments indiscriminately and
regresses toward the mean, while a policy that fires twice, at the right times,
captures the shift without the noise. This also predicts that sliding-window or
exponentially-weighted retraining would behave differently — a comparison we
flag as the most valuable follow-up (Section 8).

**(2) The two most active detectors are worse than random at the same budget.**
ADWIN (5 retrains) sits at the **3.5th percentile** of its frequency-matched
random ensemble; prequential AUC (6 retrains) at the **11.5th**. Both beat
`never_retrain` on raw AUC, and a study lacking the random control would have
reported them as successes. They are not: at their own cost, choosing weeks
uniformly at random would have done better roughly 90-96% of the time. Their
timing is not merely uninformative, it is anti-correlated with the useful
moments — consistent with detectors that fire *during* turbulence, which is
precisely when the cumulative window is worst to freeze.

This is the result that justifies the control. Without it, the ranking by mean
AUC alone would have put prequential AUC above always-retrain and called it a
win.

**(3) Only one detector clears the bar, and barely.** EDDM at the 96th
percentile is the sole detector demonstrating genuine timing skill;
champion-vs-challenger at 94.5% is marginal and we would not claim it on a
single dataset with 200 control samples. Six of the twelve detectors never fired
at all and are therefore identical to `never_retrain`.

We resist concluding that "EDDM is the best drift detector." With 14 windows and
one dataset, the honest reading is narrower: **most detectors in common use
showed no timing skill on this problem, one showed some, and two were actively
harmful relative to their own cost.**

### 7.4 Registry cost

The replay trained **11** distinct models (baseline + one per drifting week)
against **29** under the per-detector design — the 17 adoption events plus 12
independent baselines. Four further versions were materialised afterwards purely
so the random controls could be evaluated without training, bringing the stored
total to **15 = 1 + n_weeks**, the predicted bound. Cache reuse is what makes
week 2 cheap: five detectors confirmed drift there and all five received the
same model.

### 7.5 Full detail

| Quantity | Source |
|---|---|
| Per-detector retraining weeks, version in force per week | `unified_drift_report.csv` |
| Raw vs. persistence-confirmed alarms | `drift_method_flags_raw` vs `drift_method_flags` |
| Policy comparison and random controls | `policy_comparison.csv` |
| Pairwise detector agreement | `detector_agreement.csv` |
| (version x week) performance matrix | `version_week_auc_matrix.csv` |

---

## 8. Threats to Validity

- **Label latency.** Section 2.3. Immediate labels materially advantage the
  performance-aware detectors. A lag-aware replay is the most important
  follow-up.
- **Single dataset.** Conclusions about *which* detector wins are
  dataset-specific. The conclusions about *artifacts* (Section 4) are not —
  they are properties of the statistics involved.
- **Frozen representation.** Deliberate (Section 3.2), but it means we measure
  drift relative to a fixed feature space. A pipeline that periodically refits
  encoders faces a genuinely harder version of this problem.
- **Cumulative retraining only, and this now looks load-bearing.** Section 7.3's
  central finding — that frequent retraining barely helps and that the two
  busiest detectors underperform random — has a plausible mechanism specific to
  the cumulative window: a retrain permanently folds the current window into the
  training set, so retraining during turbulence is unrecoverable. Sliding-window
  or exponentially-weighted retraining would not share that property and might
  reverse the ranking entirely. **This is the single most important follow-up**,
  and until it is run, the frequency result should be read as a property of
  cumulative retraining rather than of retraining in general.
- **Control-ensemble resolution.** Random-control percentiles come from 200
  samples over C(14, k) possible policies. At k = 2 that is 91 distinct policies,
  so the ensemble is near-exhaustive; at k = 5 or 6 it is a sparse sample of
  thousands. The 0.945 vs 0.960 distinction between champion-vs-challenger and
  EDDM is within that noise and we do not lean on it.
- **Fourteen decision points.** Every policy result rests on 14 binary choices.
  The direction of the findings is clear, but the precise percentiles are not
  robust to a different seed or a shifted window boundary.
- **Threshold sensitivity.** Every detector has thresholds. We report all raw
  metrics per window so decisions can be recomputed without rerunning, but a
  systematic sensitivity sweep is future work.
- **Weekly granularity.** Fixed by the imbalance: a 7-day window contains only a
  few hundred fraud cases, which is already near the floor for a stable AUC
  estimate. Finer granularity would trade detection latency against estimator
  variance.

---

## 9. Related Work

- **Error-rate monitors.** DDM (Gama et al., 2004), EDDM (Baena-Garcia et al.,
  2006), HDDM (Frias-Blanco et al., 2015), ADWIN (Bifet & Gavalda, 2007). Our
  contribution here is not a new detector but the observation that their
  standard input — the 0/1 error stream — is close to uninformative under
  severe imbalance, and a concrete remedy.
- **Performance-aware detection.** The survey in `papers/` ("From concept drift
  to model degradation") frames the same distinction we use to organise
  Section 4. Our prequential-AUC detector is the direct, threshold-free
  instantiation of that framing.
- **Distributional monitoring.** PSI's scorecard lineage, KS-based monitors, and
  the MMD-based tests explored in this repository's `archives/`. Our
  contribution is the calibration argument: these statistics are compared
  against constants that are only valid at an unstated sample size.
- **Explanation-space monitoring.** SHAP-based drift detection. Our correction
  is procedural — random rather than prefix sampling, and FDR control.

---

## 10. Reproduction

```
# 1. Detector behaviour on synthetic data with known ground truth
python tests/test_drift_engine.py

# 2. Calibration: does the monitor alarm on real data with no drift?
python validate_monitor.py --data_dir ./dataset --compare_legacy

# 3. Full replay: 12 detectors, shared registry, policy comparison
python run_drift_analysis.py --top_k 10 --n_bags 5

streamlit run dashboard.py
```

Steps 1 and 2 are the reproducible form of the claims in Sections 3.2.1 and 4.
Each detector test pairs a null case with a signal case, because a detector that
never fires and one that always fires are both useless and only the pair
distinguishes them.

| Module | Role |
|---|---|
| `data_processing.py` | Load, merge, memory reduction |
| `feature_engineering.py` | `FeatureEngineer` — frozen fit/transform encoders |
| `feature_selection.py` | Stability-aware, redundancy-pruned monitoring set |
| `model_training.py` | LightGBM, temporal split, threshold calibration |
| `model_registry.py` | Shared version store, per-detector pointers |
| `drift_engine.py` | Twelve detectors, corrected statistics |
| `policy_evaluation.py` | Version lattice, policy comparison, random controls |
| `validate_monitor.py` | Null-experiment calibration check (Section 3.2.1) |
| `tests/test_drift_engine.py` | Detector behaviour tests: null case + signal case per detector |
| `run_drift_analysis.py` | Stream orchestration and reporting |
| `dashboard.py` | Streamlit inspection of the report |

Outputs:

| File | Contents |
|---|---|
| `reports/unified_drift_report.json` | Full per-window detail, every raw metric |
| `reports/unified_drift_report.csv` | Flat weekly summary + version in force |
| `reports/policy_comparison.csv` | Detectors vs. reference vs. random controls |
| `reports/detector_agreement.csv` | Pairwise Jaccard over flagged weeks |
| `reports/version_week_auc_matrix.csv` | The (version x week) performance matrix |
| `reports/feature_selection_diagnostics.csv` | Per-feature stability/redundancy |
| `models/model_v*.pkl` | At most `1 + n_weeks` distinct models |

---

## Appendix A — Paper Section Map

| Paper section | Content | Repository source |
|---|---|---|
| 1. Introduction | Operational question, contributions | This document |
| 2. Data | IEEE-CIS, stream protocol, imbalance | `data_processing.py` |
| 3. Representation | Features, freezing argument, monitoring-set selection | `feature_engineering.py`, `feature_selection.py` |
| 4. Detection | Twelve detectors, five corrections | `drift_engine.py` |
| 5. Design | Shared registry, sharing table | `model_registry.py` |
| 6. Protocol | Metrics, reference policies | `run_drift_analysis.py` |
| 7. Results | Policy tables, agreement, cost curves | `reports/` |
| 8. Threats | Label latency foremost | This document |

## Appendix B — Correction Summary

Grouped by where the defect lives. The "evidence" column cites the measurement,
not an argument.

**Representation** (validated by the null experiment, Section 3.2.1)

| # | Artifact | Correction | Evidence |
|---|---|---|---|
| 1 | Encoders refit per window | Freeze; fit once on the reference | Null FP 35.1% -> 0.9% |
| 2 | Frozen encoder silently inert (pandas `str` vs `object` dtype) | Test "not numeric"; warn loudly on any downstream re-encode | Categoricals were being factorised per window regardless |
| 3 | Per-window relative frequency (floor 1/n) | Removed; keep only the frozen map | Reached KS D = 0.88 under the null |
| 4 | Entity-keyed aggregates on a near-unique id | Removed; keep causal sequence features + explicit unseen-entity indicator | Reached KS D = 0.61 = the unseen-entity rate |
| 5 | Identifier / time columns in the design matrix | Drop `TransactionID`, `TransactionDT`, raw UIDs | Monotone in time: KS D = 1.0 every window by construction |

**Statistics**

| # | Artifact | Affects | Correction | Evidence |
|---|---|---|---|---|
| 4 | Significance saturates at large *n* | KS, SHAP | Effect-size floor (D >= 0.10) + bounded samples | D = 0.007 (null) vs 0.202 (real 0.5σ shift) |
| 5 | Uncorrected multiple testing | KS, SHAP | Benjamini-Hochberg FDR | 40% -> ~5% per-week false-positive probability |
| 6 | PSI thresholds ignore window size | PSI | Bootstrap null calibrated per window | PSI 0.020 vs null p99 0.087 at n = 300 |
| 7 | KL unbounded / epsilon-dependent | KL | Jensen-Shannon distance | Bounded [0,1], symmetric |

**Evaluation**

| # | Artifact | Affects | Correction | Evidence |
|---|---|---|---|---|
| 8 | Error stream constant (model never predicted fraud) | DDM, EDDM, HDDM | Early stopping 5 -> 100, temporal split, calibrated threshold | F1 0.000 -> 0.47; model was 1 tree, now ~180 |
| 9 | 0/1 error dominated by majority class | DDM, EDDM, HDDM | Class-balanced *subsampled* stream (not reweighted) | Recall collapse moves signal 0.036 -> 0.500 |
| 10 | Challenger scored in-sample | Champion vs Challenger | Out-of-fold predictions + bootstrap SE on the gap | Section 4.6 |
| 11 | Single-window triggers | All | k-of-n persistence gate | p -> ~C(n,k)·p^k |
| 12 | Per-detector models confound seeds | Experimental design | Shared model registry | Section 5 |

Three of these (2, 3, and the class-balancing entry below) were defects we
introduced *ourselves while fixing something else*, and every one was caught by
a check rather than by inspection — the null experiment and a unit test. We
record this because it is the practical argument for the paper's thesis: these
failures are invisible in code review. Item 2 is the sharpest example — the
frozen-encoder correction was written, reviewed, and documented while having no
effect whatsoever, because of a dtype predicate that was correct in pandas 2.x
and silently wrong in 3.x.
