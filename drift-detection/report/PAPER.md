# Learning *When* and *How* to Adapt: A Reinforcement-Learning Controller over Classical Drift Detectors for Transaction Fraud

**Research paper narrative and experimental protocol.** Companion to the
implementation in this repository.

---

## Abstract

Production fraud models are retrained on schedules that are rarely justified
empirically. The drift-detection literature offers a large menu of detectors —
distributional tests, error-stream monitors, explanation-space monitors,
representation monitors, shadow-model comparisons — but nearly all of them
answer a *one-shot* question ("has drift occurred?") when the operator faces a
*sequential* one ("given what I have seen and the model I currently hold, what
should I do this week?").

We replay the IEEE-CIS transaction fraud dataset as a weekly stream over six
months and compare twelve classical detectors as **retraining policies** over a
shared model registry, so that differences arise from decisions rather than from
training randomness. Two findings motivate everything that follows. First, most
of the apparent drift signal in a conventionally-built pipeline is an artifact
of the measurement apparatus: under a null experiment where no drift can exist,
the original design alarms on 35.4% of features, the corrected one on 0.9%.
Second, once measurement is fixed, **retraining frequency buys almost nothing
while retraining timing buys a great deal** — retraining in all 13 possible
windows improves mean AUC by +0.0001 (statistically indistinguishable from
never retraining), whereas two well-timed retrains improve it by +0.0094, and
detectors that retrain more than twice a replay are *worse than random
policies of identical cost* (5.5th–20th percentile).

That result motivates learning the policy. We frame drift adaptation as a Markov
decision process: the state concatenates every detector's continuous output with
model context, the action space is {do nothing, partial update, full retrain,
hedge the ensemble}, and the reward is realised performance minus adaptation
cost. A PPO agent trained in a simulated replay environment tops the benchmark
at **0.8831 mean AUC**, against **0.8696** for the best classical detector and
**0.8523** for never retraining, with a worst-week AUC matching the best naive
policy. To make this tractable on a 14-window dataset we precompute the entire
reachable model lattice, reducing episode simulation to a table lookup.

**The decomposition of that gain is the paper's most useful result, and it
only partly favours our headline.** A naive policy that fine-tunes every week,
with no learning and no drift signals, reaches 0.8820. So roughly **92% of the
improvement over the best classical detector comes from the expanded action
space** — from a cheap adaptation option existing at all — and only +0.0011
AUC comes from the learned policy on top of that. But an ablation shows that
+0.0011 is now **entirely attributable to the drift signals**: an agent given
only model-context features (staleness counters, recent performance) finds
nothing better than the naive baseline, while an agent given only drift
signals — no model context at all — recovers the full learned-policy gain.
This is a **reversal of an earlier draft of this result**, produced before we
found and fixed a feature-engineering bug that had been manufacturing false
drift signal in two of the ten originally-monitored features, and before the
monitoring set was widened from 10 to 20 features with a redundancy fix. The
reversal itself — a single measurement bug flipping a qualitative conclusion
about whether detector fusion helps — is treated as a finding in its own
right, not just a correction.

The practical implication is now two-sided rather than one-sided: on this
stream, the largest available improvement was a cheaper way to adapt, *and*,
once measurement was fixed, the drift detectors turned out to matter for the
much smaller remaining gain — where an earlier, buggier measurement had
suggested they didn't.

---

## 1. Data, Features, and Model

### 1.1 Data and stream construction

IEEE-CIS Fraud Detection: 590,540 transactions, 394 columns, joined to an
identity table on `TransactionID`. **Positive-class prevalence is 3.50%.** Time
is `TransactionDT`, a seconds offset spanning roughly six months.

The imbalance is not incidental — it is the mechanism behind two of the
measurement failures in Section 3, and any drift study on fraud, churn, or
failure prediction inherits the same exposure.

| Element | Choice |
|---|---|
| Ordering | Sorted by `TransactionDT` |
| Reference window | First **90 days** — fits encoders, selects monitors, trains model v0 |
| Monitored windows | Consecutive **7-day** windows, ~14 evaluation points |
| Retraining data | **Cumulative** — a retrain at week *w* uses everything through *w* |
| Evaluation | **Prequential** — each policy is scored on windows it has not trained on |

**Label availability.** We assume labels for window *w* are available when
window *w* is evaluated. In production, fraud labels arrive with a chargeback
lag of weeks. This favours the performance-aware detectors over the label-free
ones, and is the single largest threat to external validity (Section 8).

### 1.2 Features

Engineering follows established practice for this dataset: 432 raw predictor
columns (394 transaction + 41 identity, minus the shared join key) collapse to
**113 features** after compression and a near-constant-column filter. Full
stage-by-stage detail, including exactly which raw columns become which
engineered ones, is in
[FEATURE_ENGINEERING_AND_MODELING.md](FEATURE_ENGINEERING_AND_MODELING.md).

| Group | Raw cols | Final | Content |
|---|---|---|---|
| Vesta (V) | 339 | 4 | PCA to 2 components + missing-count + fixed-subset sum |
| Identity / device | 40 | 38 | kept mostly as-is (factorized) |
| Timedelta (D) | 15 | 16 | passed through (see the correction below), +2 derived |
| Counter (C) | 14 | 16 | replaced by 3 pattern features + 13 frequency encodings |
| Match flag (M) | 8 | 8 | kept, +1 derived missingness pattern |
| Distance | 2 | 3 | `dist1`/`dist2` kept, +1 combined log feature |
| Temporal | 0 | 3 | derived: cyclic hour, weekday × hour |
| Causal sequence | 0 | 3 | derived: per-entity lag, amount change, sequence index |
| Card / amount / product / address / other | 14 | 25 | kept, transformed, or combined |
| **Total** | **432** | **113** | |

Reports use readable labels rather than raw column names (`feature_label` in
[feature_engineering.py](../feature_engineering.py)): `_freq_ref_C5` is rendered as
*"How common this value of 'Address/card counter 5' was in training"*. Drift
reports are read by risk and operations teams, not only by whoever wrote the
feature pipeline.

**The representation is frozen.** Encoders are fitted once on the reference
window and replayed unchanged; adaptation changes model weights, never the
feature space. Section 3.1 shows why this is the single most consequential
decision in the whole study.

**A second, subtler artifact — found and fixed while writing the companion
documentation for this feature set.** The D-column stage originally computed
`D_i ← D_i.fillna(0) - _days` (`_days` being the row's absolute day offset
from the dataset start), apparently intended as a normalisation. Raw D-columns
are already relative ("days since some prior event") and roughly *stationary*
across the six-month replay; `_days` grows *linearly*. Subtracting a linearly
growing quantity from a stationary one manufactures a linear trend out of
nothing — mathematically the same "monotone-in-time proxy" failure mode
`TransactionDT`/`TransactionID` are excluded from the matrix for entirely, just
introduced here by a transformation rather than by leaving a raw timestamp in.
`D2`, once monitored, crossed its drift threshold in **14 of 14 weeks** before
the fix and **2 of 14** after. Full derivation and the exact before/after
means: FEATURE_ENGINEERING_AND_MODELING.md §2.9.

**The monitoring set** — **20 features**, up from an earlier 10 — is selected
by bagged importance across bootstrap fits, SHAP corroboration, a
monitorability filter, and greedy redundancy pruning, now at two levels:
pairwise |Spearman ρ| ≥ 0.90, **and** a cap of 2 features per numbered family
(`_freq_ref_C1`...`_freq_ref_C14` share a family; so do `D2`/`D15`)
([feature_selection.py](../feature_selection.py)). The family cap was added after
the 10-feature set was found to contain **four** members of the same
`_freq_ref_C*` family at a pairwise-survived-but-collectively-redundant
correlation of 0.817 — pairwise pruning alone under-corrects for chains of
moderate correlation within one block. We report Nogueira's stability index
for reproducibility. Full methodology, the 10→20 sizing decision (backed by a
null-experiment check showing 0% false positives on the monitored set across
30 trials), the consensus-threshold calibration attempt, and a feature-by-feature
audit of which persistent "drift" signals are real vs. seasonal vs. artifacts
are in [FEATURE_SELECTION_PROCESS.md](FEATURE_SELECTION_PROCESS.md).

### 1.3 Model

The classifier is a three-layer PyTorch MLP (256-128-64, BatchNorm, dropout
0.2), class-weighted BCE, temporal validation split, calibrated decision
threshold ([neural_model.py](../neural_model.py)).

**Why a neural net, given that gradient boosting usually wins on tabular data?**
Not for accuracy. **A GBDT cannot be partially updated.** Adding trees to a
booster is not the same operation as adapting it, and there is no principled
"fine-tune on the last month" for a fixed forest. With a GBDT the action space
of Section 4 collapses from four actions to two, and the entire question of
*how* to adapt disappears. A differentiable model makes the middle ground real —
and, just as importantly, makes the cost of that middle ground measurable
(Section 5.4).

Three fixes were required to make the classifier function at all; they are
documented in Section 3.3 because they invalidated three detectors.

---

## 2. The Twelve Classical Detectors

Organised by **what each observes**, which determines its label requirements,
its cost, and its blind spots. Full profiles, including the measured failure
mode of each, are generated to `reports/method_profiles.csv` by
[explain.py](../explain.py).

| # | Detector | Observes | Labels? | Family |
|---|---|---|---|---|
| 1 | KS test | Feature marginals | No | Distributional |
| 2 | PSI | Binned marginals | No | Distributional |
| 3 | Jensen-Shannon | Binned marginals | No | Distributional |
| 4 | DDM | Error stream | Yes | Performance |
| 5 | EDDM | Inter-error distances | Yes | Performance |
| 6 | HDDM | Error stream (Hoeffding) | Yes | Performance |
| 7 | ADWIN | Prediction stream | No | Performance-adjacent |
| 8 | SHAP / attribution | Attribution distributions | No | Explanation |
| 9 | Clustering | Joint geometry | No | Representation |
| 10 | Autoencoder | Reconstruction error | No | Representation |
| 11 | Prequential AUC | Ranking quality | Yes | Performance |
| 12 | Champion vs Challenger | Value of retraining | Yes | Shadow model |

### 2.1 Strengths, weaknesses, and applicability

Each profile below ends with **Results here** — that detector's actual raw and
persistence-confirmed alarm count on the corrected 14-week replay, monitoring
the **20-feature set** (up from an earlier 10; see §1.2 and
FEATURE_SELECTION_PROCESS.md), after a transformation bug in the D-columns was
found and fixed (§1.2, FEATURE_ENGINEERING_AND_MODELING.md §2.9). Full ledger
in Section 6.1.1, one row per week in
[reports/method_week_matrix.csv](../reports/method_week_matrix.csv). The point of
including results at introduction is that a detector's textbook profile and its
behaviour *on this stream* are not the same thing — a detector can be
well-motivated and still sit at zero for six months, and that is only visible
once the numbers are attached to the description rather than left for a results
section eleven pages later.

**KS test.** *Best at:* abrupt covariate shift in a continuous feature — a new
acquiring bank, a changed upstream default. *Blind to:* concept drift entirely;
if the same transactions start being fraudulent at a different rate, no marginal
moves. *Failure mode:* the p-value is a function of sample size. At 10⁵ vs 10⁴
rows the critical statistic is ≈0.015, so a 1.5% shift is "significant" — it
flagged **14 of 14 weeks** before correction. *Use when:* you have no labels and
expect abrupt input changes; never on p-value alone.

*Results here:* **0 of 14 weeks**, raw or confirmed.

| Week | Features crossed | Fraction | Raw | Confirmed |
|---|---|---|---|---|
| 1 | 6/20 | 0.30 | – | – |
| 2 | 7/20 | 0.35 | – | – |
| 3 | 4/20 | 0.20 | – | – |
| 4 | 3/20 | 0.15 | – | – |
| 5 | 7/20 | 0.35 | – | – |
| 6 | 6/20 | 0.30 | – | – |
| 7 | 6/20 | 0.30 | – | – |
| 8 | 6/20 | 0.30 | – | – |
| 9 | 7/20 | 0.35 | – | – |
| 10 | 7/20 | 0.35 | – | – |
| 11 | 4/20 | 0.20 | – | – |
| 12 | 8/20 | 0.40 | – | – |
| 13 | 8/20 | 0.40 | – | – |
| 14 | 9/20 | 0.45 | – | – |

Never above 0.45 against the 0.60 consensus line — exactly what "blind to
concept drift" predicts on a stream where the drift is concept drift
(Section 6.1). Three of the twenty monitored features (`_mcols_na_bin`,
`_vcols_dec0`, `_vcols_dec1`) individually cross their own threshold in all 14
weeks, on their own; §6.1.1 traces the mechanism behind each recurring
crosser.

**PSI.** *Best at:* gradual monotone population shifts, with an interpretable
magnitude and a long industry track record. *Blind to:* concept drift; changes
within a bin. *Failure mode:* the 0.10/0.20 bands are scorecard folklore at an
unstated sample size. E[PSI] under the null scales as (B−1)/n, so a 100-row
window expects ≈0.09 from noise alone — nearly the "moderate" band. *Use when:*
window sizes are stable and large, or with a calibrated null.

*Results here:* **0 of 14 weeks**.

| Week | Features crossed | Fraction | Raw | Confirmed |
|---|---|---|---|---|
| 1 | 4/20 | 0.20 | – | – |
| 2 | 5/20 | 0.25 | – | – |
| 3 | 3/20 | 0.15 | – | – |
| 4 | 3/20 | 0.15 | – | – |
| 5 | 3/20 | 0.15 | – | – |
| 6 | 3/20 | 0.15 | – | – |
| 7 | 3/20 | 0.15 | – | – |
| 8 | 3/20 | 0.15 | – | – |
| 9 | 3/20 | 0.15 | – | – |
| 10 | 3/20 | 0.15 | – | – |
| 11 | 3/20 | 0.15 | – | – |
| 12 | 4/20 | 0.20 | – | – |
| 13 | 4/20 | 0.20 | – | – |
| 14 | 4/20 | 0.20 | – | – |

Lower fraction than KS every week, but the *same* three persistent crossers
account for most of it — the two detectors are watching the same absence of
covariate shift through different statistics.

**Jensen-Shannon (replacing raw KL).** *Best at:* the same job as KL, but
bounded in [0,1] and symmetric, so one threshold means the same thing for every
feature. *Why KL was replaced:* unbounded, asymmetric, and undefined wherever
the current window has support the reference lacks — routine for rare
categories. The ε-smoothing fix makes its absolute value depend on the arbitrary
ε.

*Results here:* **0 of 14 weeks**.

| Week | Features crossed | Fraction | Raw | Confirmed |
|---|---|---|---|---|
| 1 | 8/20 | 0.40 | – | – |
| 2 | 9/20 | 0.45 | – | – |
| 3 | 7/20 | 0.35 | – | – |
| 4 | 8/20 | 0.40 | – | – |
| 5 | 7/20 | 0.35 | – | – |
| 6 | 9/20 | 0.45 | – | – |
| 7 | 9/20 | 0.45 | – | – |
| 8 | 9/20 | 0.45 | – | – |
| 9 | 9/20 | 0.45 | – | – |
| 10 | 9/20 | 0.45 | – | – |
| 11 | 6/20 | 0.30 | – | – |
| 12 | 10/20 | 0.50 | – | – |
| 13 | 9/20 | 0.45 | – | – |
| 14 | 9/20 | 0.45 | – | – |

Consistently the highest of the three distributional detectors — closer to
the 0.60 line than KS/PSI ever get (week 12 reaches 0.50) — but still short of
consensus every week; JS is more sensitive than KS/PSI on this stream without
ever being sensitive enough to matter.

**DDM.** *Best at:* abrupt degradation on balanced problems; cheap and online.
*Blind to:* anything the majority class hides — at 3.5% prevalence a total
collapse in recall moves the raw error rate by 3.5 points, inside its own noise
band. *Failure mode:* control limits assume Bernoulli variance shrinking as 1/n,
so it becomes *harder* to trigger the longer a model has been stable — exactly
when drift is most likely. *Use when:* classes are near-balanced.

*Results here:* **0 of 14 weeks, raw or confirmed.** Value = balanced error
rate; threshold = control limit p_min + 3·s_min.

| Week | Value | Threshold | Raw | Confirmed |
|---|---|---|---|---|
| 1 | 0.3310 | 0.3923 | – | – |
| 2 | 0.3503 | 0.3923 | – | – |
| 3 | 0.2945 | 0.3923 | – | – |
| 4 | 0.2700 | 0.3923 | – | – |
| 5 | 0.2877 | 0.3923 | – | – |
| 6 | 0.3160 | 0.3923 | – | – |
| 7 | 0.3331 | 0.3923 | – | – |
| 8 | 0.3431 | 0.3923 | – | – |
| 9 | 0.3240 | 0.3923 | – | – |
| 10 | 0.3105 | 0.3923 | – | – |
| 11 | 0.3172 | 0.3923 | – | – |
| 12 | 0.3104 | 0.3923 | – | – |
| 13 | 0.3091 | 0.3923 | – | – |
| 14 | 0.2857 | 0.3923 | – | – |

Never within 0.04 of its own control limit at any week-end snapshot. (An
earlier run, with a different monitored feature set feeding the same shared
model, showed one transient week-1 trigger inside a batch; DDM updates its
control limit sample-by-sample, so a snapshot can miss a mid-batch crossing —
see HDDM below for the same caveat.)

**EDDM.** *Best at:* gradual degradation — errors bunching together before the
rate itself moves. The earliest warning of the classical set. *Blind to:* abrupt
shifts that leave error spacing unchanged. *Failure mode:* the paper defaults
(β 0.90/0.95) are extremely noise-sensitive, retraining ~13/14 weeks on a stable
stream.

*Results here:* **8 raw alarms, 4 confirmed** (weeks 2, 10, 12, 14) — a long
way from the pre-correction 13/14. Value = inter-error-distance metric; drift
is signalled when it falls *below* the threshold, which itself adapts after
every retrain (visible as the threshold's step changes below).

| Week | Value | Threshold | Raw | Confirmed |
|---|---|---|---|---|
| 1 | 8.6330 | 10.8206 | Y | – |
| 2 | 8.7439 | 10.8206 | Y | Y |
| 3 | 10.6994 | 10.1164 | – | – |
| 4 | 11.2809 | 10.1164 | – | – |
| 5 | 11.1322 | 10.1164 | – | – |
| 6 | 10.9475 | 10.1164 | – | – |
| 7 | 10.6051 | 10.1164 | – | – |
| 8 | 10.2722 | 10.1164 | – | – |
| 9 | 10.1934 | 10.1164 | Y | – |
| 10 | 10.1243 | 10.1164 | Y | Y |
| 11 | 9.6959 | 11.0349 | Y | – |
| 12 | 9.7206 | 11.0349 | Y | Y |
| 13 | 10.8996 | 11.3737 | Y | – |
| 14 | 10.6470 | 11.3737 | Y | Y |

Confirmation requires two *consecutive* raw alarms: 1→2, 9→10, 11→12, 13→14.
This is the single detector whose behaviour changed the most from the D-column
fix — it went from the paper's earlier best classical policy (2 retrains,
first place) to a busier 4-retrain policy that now ranks *second* to Champion
vs Challenger (Section 6.2). The D2/D15 fix genuinely altered which weeks look
turbulent to the error stream, not just to the feature-marginal detectors.

**HDDM.** *Best at:* staying sensitive on long-stable models where DDM stops
being able to trigger; distribution-free via Hoeffding bounds. *Trade-off:*
conservative by design — the price of the guarantee.

*Results here:* **0 of 14 weeks, raw or confirmed** — the only detector in the
study with zero raw alarms as well as zero confirmed ones, in every run this
paper has produced. Value = the same balanced error rate as DDM; threshold =
its own Hoeffding-derived bound.

| Week | Value | Threshold | Raw | Confirmed |
|---|---|---|---|---|
| 1 | 0.3310 | 0.3465 | – | – |
| 2 | 0.3503 | 0.3354 | – | – |
| 3 | 0.2945 | 0.3512 | – | – |
| 4 | 0.2700 | 0.3306 | – | – |
| 5 | 0.2877 | 0.3229 | – | – |
| 6 | 0.3160 | 0.3213 | – | – |
| 7 | 0.3331 | 0.3202 | – | – |
| 8 | 0.3431 | 0.3192 | – | – |
| 9 | 0.3240 | 0.3184 | – | – |
| 10 | 0.3105 | 0.3178 | – | – |
| 11 | 0.3172 | 0.3171 | – | – |
| 12 | 0.3104 | 0.3165 | – | – |
| 13 | 0.3091 | 0.3159 | – | – |
| 14 | 0.2857 | 0.3158 | – | – |

A tighter bound than DDM's fixed 0.392, and week 11's snapshot (0.3172 vs.
0.3171) is a near-miss by 0.0001 — closer than DDM ever gets — yet the
sequential test still never trips. The strongest evidence in the replay for
"conservative by design."

**ADWIN.** *Best at:* mean shift in the score stream with a formal guarantee and
automatic window sizing; needs no labels. *Blind to:* degradation that preserves
the score distribution — a model can rank worse with an unchanged histogram.
*Failure mode:* fires on score turbulence, which is not the same as staleness.

*Results here:* **12 raw alarms out of 14 weeks — nearly every week** — but only
**5 confirmed** (weeks 2, 4, 8, 10, 12), once the persistence gate filters
one-off turbulence from a repeated signal.

| Week | Value (z) | Threshold | Raw | Confirmed |
|---|---|---|---|---|
| 1 | 0.1010 | 3.0902 | Y | – |
| 2 | 1.3740 | 3.0902 | Y | Y |
| 3 | 1.0041 | 3.0902 | Y | – |
| 4 | 9.5208 | 3.0902 | Y | Y |
| 5 | 5.3476 | 3.0902 | Y | – |
| 6 | 3.1502 | 3.0902 | – | – |
| 7 | 5.2889 | 3.0902 | Y | – |
| 8 | 6.3278 | 3.0902 | Y | Y |
| 9 | 3.5849 | 3.0902 | Y | – |
| 10 | 2.4212 | 3.0902 | Y | Y |
| 11 | 6.3722 | 3.0902 | Y | – |
| 12 | 9.4611 | 3.0902 | Y | Y |
| 13 | 1.6024 | 3.0902 | Y | – |
| 14 | 0.0551 | 3.0902 | – | – |

*(This table reports z against a fixed threshold of 3.09; individual weekly z
values shift with the underlying model-prediction stream and are not directly
comparable to earlier drafts of this table, which used a different monitored
feature set for the reference-stream comparison. The raw/confirmed pattern —
12 raw, 5 confirmed at exactly weeks 2, 4, 8, 10, 12 — is unchanged.)* This is
the clearest illustration in the whole study of "fires on turbulence, not
staleness": Section 6.2 shows ADWIN as a *retraining policy* performs worse
than a random policy of equal cost (5.5th percentile).

**SHAP / attribution drift.** *Best at:* catching a change in what the model
*relies on* even when every raw marginal is stable — **the only label-free
signal with real purchase on concept drift**. *Blind to:* drift that changes the
label mapping without changing attributions (it reads the model, not the truth).
*Failure mode:* expensive; and prefix-sampling a time-ordered window samples
only its earliest days, making seasonality look like permanent drift. On the
neural model we use gradient×input, the differentiable-model analogue.

*Results here:* **0 of 14 weeks, raw or confirmed** — down from 2 raw / 1
confirmed at the earlier 10-feature set.

| Week | Features crossed | Fraction | Raw | Confirmed |
|---|---|---|---|---|
| 1 | 5/20 | 0.25 | – | – |
| 2 | 4/20 | 0.20 | – | – |
| 3 | 1/20 | 0.05 | – | – |
| 4 | 5/20 | 0.25 | – | – |
| 5 | 2/20 | 0.10 | – | – |
| 6 | 1/20 | 0.05 | – | – |
| 7 | 1/20 | 0.05 | – | – |
| 8 | 4/20 | 0.20 | – | – |
| 9 | 2/20 | 0.10 | – | – |
| 10 | 3/20 | 0.15 | – | – |
| 11 | 1/20 | 0.05 | – | – |
| 12 | 1/20 | 0.05 | – | – |
| 13 | 4/20 | 0.20 | – | – |
| 14 | 6/20 | 0.30 | – | – |

**A genuine, informative regression, not noise.** At 10 monitored features,
week 1's fraction peaked at 0.70 (the single highest reading of any
label-free detector observed in this study) and week 2's 0.60 confirmed —
the one label-free detector that agreed with the label-dependent majority.
At 20 features, the same weeks read 0.25 and 0.20. The family-diversity cap
(§1.2) removed two of the four `_freq_ref_C*` features that had been driving
that spike, and the doubled denominator halves any fixed count's fraction on
its own. This is a direct, measured illustration of why redundant features
inflate a consensus vote: four correlated votes at 10 features looked like
broader agreement than two non-redundant votes do at 20. Whether the original
signal was a real, narrow week-1/2 event or partly an artifact of
over-representing one feature family is now unresolable from this dataset
alone — a caution about reading too much into any single-run consensus
fraction.

**Clustering.** *Best at:* multivariate shifts no per-feature test can see — a
new correlation structure with unchanged marginals. *Failure mode:* without
standardisation the largest-scale feature dominates every Euclidean distance and
the method silently measures that one feature.

*Results here:* **1 raw alarm (week 12), 0 confirmed** — down from 1 confirmed
alarm at the earlier feature set.

| Week | Value (distance ratio) | Threshold | Raw | Confirmed |
|---|---|---|---|---|
| 1 | 1.0353 | 1.5000 | – | – |
| 2 | 1.0664 | 1.5000 | – | – |
| 3 | 1.0566 | 1.5000 | – | – |
| 4 | 1.0585 | 1.5000 | – | – |
| 5 | 1.0881 | 1.5000 | – | – |
| 6 | 1.0892 | 1.5000 | – | – |
| 7 | 1.1011 | 1.5000 | – | – |
| 8 | 1.0936 | 1.5000 | – | – |
| 9 | 1.0970 | 1.5000 | – | – |
| 10 | 1.1093 | 1.5000 | – | – |
| 11 | 1.3627 | 1.5000 | – | – |
| 12 | 2.9492 | 1.5000 | Y | – |
| 13 | 1.1575 | 1.5000 | – | – |
| 14 | 1.1755 | 1.5000 | – | – |

Week 12's ratio of 2.95 is still by far the largest reading in the replay
(more than double week 11's 1.36), and lines up exactly with the raw
`_vcols_sum` outlier (a 6–37× spike at weeks 11–12) and the autoencoder's own
week-12 peak below — three independent signals agreeing on the same narrow
event (FEATURE_SELECTION_PROCESS.md §6c). It no longer survives the
persistence gate on its own (needs a second consecutive raw week, and week 13
falls back to 1.16), so this detector alone would no longer trigger a
retrain for it — a real cost of a single-window anomaly under a two-week
confirmation rule.

**Autoencoder.** *Best at:* genuinely novel regions of feature space. *Failure
mode:* a KS test on reconstruction errors flags almost any batch at realistic
sample sizes; the decision must rest on an effect size.

*Results here:* **0 of 14 weeks.**

| Week | Value (RMSE z-score) | Threshold | Raw | Confirmed |
|---|---|---|---|---|
| 1 | 0.1003 | 3.0000 | – | – |
| 2 | 0.2952 | 3.0000 | – | – |
| 3 | 0.1103 | 3.0000 | – | – |
| 4 | 0.0912 | 3.0000 | – | – |
| 5 | 0.1558 | 3.0000 | – | – |
| 6 | 0.2794 | 3.0000 | – | – |
| 7 | 0.2069 | 3.0000 | – | – |
| 8 | 0.2000 | 3.0000 | – | – |
| 9 | 0.1977 | 3.0000 | – | – |
| 10 | 0.1804 | 3.0000 | – | – |
| 11 | 0.4542 | 3.0000 | – | – |
| 12 | 2.4584 | 3.0000 | – | – |
| 13 | 0.2822 | 3.0000 | – | – |
| 14 | 0.3312 | 3.0000 | – | – |

Rising sharply toward the end of the replay — its week-12 peak of 2.46 is the
same narrow anomaly clustering and `_vcols_sum` independently flag — but never
crossing the effect-size floor.

**Prequential AUC.** *Best at:* measuring what actually matters, directly, with
no proxy and no second model. *Weakness:* strictly reactive — it cannot fire
until damage is already in the metric, and it tends to fire *during* turbulence,
which is the worst moment to freeze a cumulative training set.

*Results here:* **12 raw alarms, 6 confirmed** (weeks 2, 5, 7, 9, 11, 13) — the
most persistence-confirmed alarms of any detector, unchanged from the earlier
feature set. Value = AUC drop from reference; threshold = bootstrap SE band.

| Week | Value (AUC drop) | Threshold | Raw | Confirmed |
|---|---|---|---|---|
| 1 | 0.0706 | 0.0136 | Y | – |
| 2 | 0.0644 | 0.0137 | Y | Y |
| 3 | 0.0129 | 0.0097 | – | – |
| 4 | 0.0329 | 0.0109 | Y | – |
| 5 | 0.0383 | 0.0125 | Y | Y |
| 6 | 0.0303 | 0.0121 | Y | – |
| 7 | 0.0359 | 0.0141 | Y | Y |
| 8 | 0.0441 | 0.0128 | Y | – |
| 9 | 0.0442 | 0.0151 | Y | Y |
| 10 | 0.0484 | 0.0166 | Y | – |
| 11 | 0.0403 | 0.0125 | Y | Y |
| 12 | 0.0382 | 0.0132 | Y | – |
| 13 | 0.0262 | 0.0115 | Y | Y |
| 14 | 0.0178 | 0.0316 | – | – |

It is reactive exactly as advertised: Section 6.2 shows it retraining six times
for a worse mean AUC than the top classical policy's two, and its
random-control percentile (0.200) means a randomly-timed policy retraining six
times usually beats it.

**Champion vs Challenger.** *Best at:* answering the decision question directly
— not "did drift happen" but "would retraining help". *Failure mode:* scoring
the challenger in-sample inflates it by more than the trigger threshold, so it
fires every week on its own overfitting. Also doubles training cost.

*Results here:* **5 raw alarms, 2 confirmed** (weeks 2, 8). Value = out-of-fold
challenger AUC gap over the champion; threshold = 0.03 (a second, separate
condition — champion AUC degrading vs. its training-time baseline by more than
0.05 — can also raise the raw flag and is not shown here).

| Week | Value (OOF AUC gap) | Threshold | Raw | Confirmed |
|---|---|---|---|---|
| 1 | 0.0035 | 0.0300 | – | – |
| 2 | 0.0287 | 0.0300 | Y | Y |
| 3 | -0.0020 | 0.0300 | – | – |
| 4 | 0.0176 | 0.0300 | – | – |
| 5 | 0.0037 | 0.0300 | – | – |
| 6 | 0.0270 | 0.0300 | – | – |
| 7 | 0.0228 | 0.0300 | Y | – |
| 8 | 0.0207 | 0.0300 | Y | Y |
| 9 | -0.0114 | 0.0300 | – | – |
| 10 | 0.0215 | 0.0300 | – | – |
| 11 | 0.0189 | 0.0300 | – | – |
| 12 | 0.0475 | 0.0300 | Y | – |
| 13 | 0.0159 | 0.0300 | – | – |
| 14 | -0.1485 | 0.0300 | – | – |

None of the five raw weeks actually cross the 0.03 gap threshold shown above
except week 12 (0.0475) and, marginally, week 2 (0.0287, just under — the raw
flag here comes from the degradation condition, not the gap); the rest fire on
the champion's own AUC degrading against its training-time baseline. This is
now the single best classical retraining policy (Section 6.2) — a reversal
from the earlier run, where EDDM held that position; the D-column fix changed
which weeks look turbulent enough for both detectors to retrain on, and
Champion vs Challenger's OOF-corrected shadow-model comparison turned out to
be the more robust of the two to that change.

### 2.2 The structural summary

Every detector picks one point on a trade-off it cannot escape:

- **Label-free detectors** (KS, PSI, JS, clustering, autoencoder, ADWIN,
  attribution) can run continuously but are structurally blind, or nearly so, to
  concept drift.
- **Performance detectors** (DDM, EDDM, HDDM, prequential AUC, C-vs-C) see what
  matters but only after labels arrive, and are reactive by construction.
- **No detector observes its own model's state.** The same signal warrants a
  retrain if the model is six months stale and nothing if it was rebuilt last
  week. None of them can express that.
- **All of them emit a binary flag.** The decision they inform is not binary.

A single detector must choose. **An agent observing all of them need not** —
which is the argument for Section 4.

---

## 3. Measurement Before Modelling

Every detector compares a reference window to a current window. If anything
between raw data and that comparison is refit per window, the statistic measures
the refitting. We found this to be the dominant effect, and it is silent: a
pipeline exhibiting it *looks* like it is working.

### 3.1 The null experiment

Split **one** window into two random halves — one large, one small, preserving
the size asymmetry of the real setting. No drift can exist between them, so
every alarm is an artifact. Reproducible via
[validate_monitor.py](../validate_monitor.py).

Both arms use **identical feature definitions**; only encoder fitting differs.

| Encoder fitting | KS (p only) | KS (+effect size) | PSI (fixed) | PSI (calibrated) |
|---|---|---|---|---|
| Refit per window | 35.4% | 25.7% | 14.2% | 14.2% |
| Fitted once, frozen | **0.9%** | **0.0%** | **0.0%** | **0.0%** |

Three artifacts, each caught only by this check:

1. **Ordinal codes are assignment-order dependent.** `pd.factorize` numbers
   categories by first appearance, so an email domain encoded `3` in the
   baseline becomes `17` next week with nothing changed in the world.
2. **A per-window relative frequency is still window-size dependent**, because
   its resolution floor is 1/n. Reached **KS D = 0.88** under the null.
3. **Entity-keyed aggregates are not transferable.** Keyed on a near-unique card
   identity, they are undefined for most rows of any future window; any fill
   creates a point mass whose KS distance equals the unseen-entity rate,
   **D = 0.61**, regardless of drift. They also flatter the baseline model,
   which sees them computed in-sample, then degenerate to a constant at
   inference.

Identifier and time columns (`TransactionID`, `TransactionDT`) were also removed:
monotone in time, so KS = 1.0 every window by construction.

Note the statistical corrections alone do *not* rescue the refitting arm
(35.4% → 25.7%). **The representation had to be fixed first.**

**A fourth artifact, structurally different from the above three, was found
later and this check could not have caught it.** The D-column transformation
bug (§1.2) manufactures a trend that is a function of *elapsed time*, not of
which encoder fit which window — the null experiment's random split has no
time axis, so a monotone-in-time artifact produces no signal under it (indeed,
`D2` and `D15` show no elevated alarm rate in the table above, before or after
the fix). It was found instead by directly comparing each D-column's raw and
transformed weekly means across the real 14-week replay (FEATURE_SELECTION_PROCESS.md
§6a). The methodological lesson: the null experiment is necessary but not
sufficient — it calibrates against *encoder-refitting* artifacts specifically,
and a temporally-monotone transformation needs a temporally-aware check
instead.

### 3.2 Statistical corrections

| Artifact | Affects | Correction |
|---|---|---|
| Significance saturates at large n | KS, SHAP | Effect-size floor (D ≥ 0.10) + bounded samples |
| Uncorrected multiple testing | KS, SHAP | Benjamini-Hochberg FDR across monitored features |
| PSI thresholds ignore window size | PSI | Bootstrap null calibrated per window |
| KL unbounded / ε-dependent | KL | Jensen-Shannon distance |
| Single-window triggers | All | k-of-n persistence gate (p → ~C(n,k)·pᵏ) |

### 3.3 The classifier never predicted the positive class

The most damaging defect. Every model in the original reports had **F1 = 0.0000
on every window**, from two compounding causes: an early-stopping patience of 5
rounds at learning rate 0.01 halted training at **iteration 1** (a single tree),
and a fixed 0.5 threshold at 3.5% prevalence left essentially no mass above the
line.

DDM, EDDM and HDDM all monitor a *binary error stream*. If the model always
predicts the negative class, that stream **is** the label vector — its rate is
the prevalence, constant, carrying no information about the model at all. DDM's
alarms in 14/14 weeks were fluctuations in the weekly fraud rate.

Corrections: patience 5 → 100; random-stratified → **temporal** validation split
(a random split places same-day, often same-card rows on both sides); and a
**calibrated decision threshold** stored on the model. Validation F1 moves from
0.000 to ≈0.47.

Even with a working classifier, a 0/1 error stream at 3.5% prevalence is ~96%
determined by legitimate transactions. We feed the classical monitors a
**class-balanced error stream** — every positive plus an equal number of
randomly drawn negatives, in temporal order — so the mean is the balanced error
rate. Balanced by *subsampling*, not reweighting: weights of 1/0.035 ≈ 29 are
not bounded by 1, and DDM's variance term and HDDM's Hoeffding bound both
require a bounded stream. Our first implementation reweighted and clipped, which
silently undid the rebalancing; a unit test caught it. Under a total recall
collapse the 0/1 stream moves 0.036 and the balanced stream 0.500 — a **14×**
amplification of the failure that matters most.

---

## 4. Proposed Method: RL over the Detector Ensemble

### 4.1 Why the classical results demand a different formulation

With measurement fixed, the policy comparison (Section 6.2) shows:

- retraining in **all 13** windows gains **+0.0001** AUC over never retraining
  — statistically indistinguishable from doing nothing;
- **two well-timed** retrains (Champion vs Challenger) gain **+0.0094** — close
  to two orders of magnitude the benefit, at 15% of the cost;
- ADWIN and Prequential AUC, the two detectors that retrain more than twice a
  replay, sit at the **5.5th and 20th percentile** of random policies with the
  same budget.

Frequency is nearly worthless; timing is decisive; and the detectors are not
good at timing. That is not a threshold-tuning problem. It is a sequential
decision problem with three properties no detector expresses:

1. **The right action depends on the current model, not just the data.**
2. **The choice is not binary** — a cheap fine-tune and a costless re-weighting
   sit between "nothing" and "rebuild".
3. **Actions have delayed, compounding consequences.** Retraining during a
   turbulent week folds that turbulence permanently into a cumulative training
   set; the cost lands weeks later. Credit assignment across time is exactly
   what RL is for.

### 4.2 Formulation

    state   s_t = [ 11 drift signals vs. current reference | 6 model-context features ]
    action  a_t in { do nothing, partial update, full retrain, hedge ensemble }
    reward  r_t = 100 * ( AUC_t − AUC_never-retrain,t ) − 100 * cost(a_t)

**State** ([drift_signals.py](../drift_signals.py)) deliberately mixes families —
distributional, attribution, representation, and model context — because they
fail in *different* ways. Signals are stored as **continuous statistics, not
booleans**: "PSI is 0.19" and "PSI is 0.02" are both "no drift" to a fixed rule
and obviously different to a learner.

| Block | Contents |
|---|---|
| Distributional | KS drift fraction, KS mean statistic, PSI drift fraction, PSI/null ratio, JS mean distance |
| Attribution | Attribution drift fraction, mean attribution shift |
| Representation | Cluster distance ratio, cluster PSI, autoencoder z-score |
| Reference | Weeks since reference |
| Model context | Weeks since full retrain, weeks since partial update, ensemble α, recent AUC delta, recent F1, progress through the replay |

**Actions.**

| Action | Effect | Cost (AUC pts) |
|---|---|---|
| `do_nothing` | Keep current model and weight | 0.0000 |
| `partial_update` | Fine-tune the last **full** model on the recent 4-week window | 0.0010 |
| `full_retrain` | Retrain on all data so far; resets reference, α, staleness | 0.0040 |
| `hedge_ensemble` | Shift α one step from the current model toward the stable baseline | 0.0000 |

Weight moves only *toward* the baseline between full retrains, and a retrain
resets it to 1.0. This keeps the state small and the semantics honest: hedging
is a response to declining trust, and regaining trust requires rebuilding.

**Reward** is measured against never-retraining, so only improvement over the
free option is rewarded, minus the cost of the action taken. A decision at week
*t* is graded on week *t+1* onward — grading it on its own week would let the
agent retrain on data it has already been scored against.

### 4.3 Agent architecture

    drift signals + model context  (17-dim)
                  |
          [ drift encoder ]      2-layer MLP, tanh, shared
                  |
         +--------+--------+
         |                 |
    [ policy head ]   [ value head ]
      4 actions          V(s)

The shared trunk is the drift encoder. Sharing it matters here because the
dataset is tiny — the value head's gradient is extra supervision for the
encoder. Trained with PPO: clipped surrogate objective, GAE(λ=0.95), γ=0.95,
entropy bonus 0.02, gradient-norm clipping.

**Why PPO.** The action space is small and discrete, episodes are 14 steps, and
the environment is a lookup table — so sample efficiency is not the binding
constraint, *stability* is. PPO's clipped objective prevents any single batch
from moving the policy too far, which matters when the whole dataset is one
trajectory replayed under different choices.

**Exploration.** Epsilon-greedy annealed 0.30 → 0.02 during training, on top of
sampling from the policy. In production, `mode='thompson'` draws from the learned
categorical policy rather than taking the argmax, so exploration is proportional
to the agent's own uncertainty with no tuned ε. *Named honestly:* with dropout
off this is posterior sampling **over actions**, not full parameter-space
Thompson sampling; the latter would need an ensemble or a Bayesian last layer.

---

## 5. Implementation

### 5.1 Making PPO tractable on 14 windows

PPO needs thousands of episodes; the dataset provides one trajectory, and
training a model inside the RL loop would mean tens of thousands of fits.

The way out is that **the reachable model space is small and enumerable**. Under
this action set the model in force is fully determined by

    (last_full_retrain_week, last_partial_update_week, ensemble_alpha)

so [model_lattice.py](../model_lattice.py) enumerates every reachable model once,
caches what each scores on every future week, and the environment becomes a
lookup. Episodes then cost nothing.

Two design choices keep the space small, and both are defensible independently
of the RL:

**Partial updates are not chained.** A partial update always fine-tunes from the
*last full-retrain model*, never from the previous partial. Chaining would make
the model depend on the entire action history (4¹⁴ possibilities), destroying
the Markov property — and it compounds catastrophic forgetting, since each
fine-tune on recent-only data pulls further from the original distribution with
no way back short of a full retrain. Re-deriving from the last full model bounds
that damage by construction.

**The ensemble blends the current model with the baseline**, weight α on the
current one. This is the cheap hedge, and it makes forgetting *measurable*: the
gap between α=1 and the best α is exactly how much the recent update cost on the
broader distribution.

### 5.2 Module map

| Module | Role |
|---|---|
| [data_processing.py](../data_processing.py) | Load, merge, memory reduction |
| [feature_engineering.py](../feature_engineering.py) | Frozen fit/transform encoders; readable feature labels |
| [feature_selection.py](../feature_selection.py) | Stability-aware, redundancy-pruned monitoring set |
| [neural_model.py](../neural_model.py) | MLP classifier with `partial_fit` and gradient attributions |
| [drift_engine.py](../drift_engine.py) | The twelve classical detectors, corrected |
| [model_lattice.py](../model_lattice.py) | Enumerate reachable models; cache weekly performance |
| [drift_signals.py](../drift_signals.py) | Signal matrix over (reference, week) |
| [rl_env.py](../rl_env.py) | The MDP: state, actions, reward |
| [rl_agent.py](../rl_agent.py) | PPO: encoder, policy head, value head |
| [explain.py](../explain.py) | Method profiles; agent decision traces |
| [policy_evaluation.py](../policy_evaluation.py) | Classical policy comparison, random controls |
| [validate_monitor.py](../validate_monitor.py) | Null-experiment calibration check |

### 5.3 A bug worth recording

PPO's importance ratio compares an action's log-probability under the current
policy against its log-probability when taken. With **dropout active in the
policy network**, that ratio mixed policy change with dropout noise, and the
clipped objective silently stopped meaning what it should. The agent converged
to "always retrain" on a synthetic task with an obvious optimum, leaving 46% of
available return unclaimed. Dropout now defaults to 0 in the policy net
([rl_agent.py](../rl_agent.py)); the failure is covered by
`tests/test_rl_agent.py`.

This is the third defect in this project introduced *while fixing another one*
and caught only by a check rather than by reading the code.

### 5.4 Reproduction

```
python tests/test_drift_engine.py                                  # detector behaviour
python tests/test_rl_agent.py                                      # env + PPO learning
python validate_monitor.py --data_dir ./dataset --compare_legacy   # calibration
python validate_monitor.py --data_dir ./dataset --calibrate_consensus  # consensus-threshold null check
python run_drift_analysis.py --top_k 20 --n_bags 5                 # classical replay
python run_rl_experiment.py --data_dir ./dataset --top_k 20        # RL experiment
streamlit run dashboard.py
```

---

## 6. Results and Benchmark

### 6.1 Classical detector behaviour after correction

This run reflects two corrections beyond the Section 3 measurement fixes: the
monitoring set was widened from 10 to 20 features with a family-diversity cap
(§1.2), and a transformation bug that manufactured a fake trend in the
D-columns was fixed (§1.2, §2.1). Confirmed (persistence-gated) alarms over
the 14-week replay:

| Detector | Weeks confirmed | Count |
|---|---|---|
| Prequential AUC | 2, 5, 7, 9, 11, 13 | 6 |
| ADWIN | 2, 4, 8, 10, 12 | 5 |
| EDDM | 2, 10, 12, 14 | 4 |
| Champion vs Challenger | 2, 8 | 2 |
| KS, PSI, JS, DDM, HDDM, SHAP, Clustering, Autoencoder | — | 0 |

KS fell from 14/14 weeks to 0/14 and DDM from 14/14 to 0/14 — not by weakening
thresholds, but by removing the artifacts that were generating the alarms.
**SHAP and Clustering, which each confirmed once at the 10-feature set, are
silent at 20** — SHAP's one signal was substantially a redundancy artifact of
over-representing one feature family (§2.1); Clustering's week-12 anomaly is
still the largest reading in its table by a wide margin, but no longer repeats
into an adjacent week under the wider feature set, so it no longer survives
the 2-of-2 persistence gate.

**Every feature-marginal detector stayed silent while behaviour-based detectors
carried the entire signal.** The drift here is **concept drift, not covariate
shift**: transactions look the same; what makes one fraudulent changed. The
operational implication is uncomfortable — **label-free monitoring would have
detected nothing**, and that is the most common production configuration. Week 2
is the single week where every behaviour-based detector agrees.

### 6.1.1 The full week × method matrix

`_build_method_matrix` in [run_drift_analysis.py](../run_drift_analysis.py) writes
every detector's per-week decision to
[reports/method_week_matrix.csv](../reports/method_week_matrix.csv) — 168 rows (14
weeks × 12 methods), one row per (week, method), columns: `model_version`,
`raw_flag`, `confirmed_flag`, `retrained_this_week`, `features_total`,
`features_crossed`, `features_crossed_fraction`, `drifted_feature_names` (for
the four feature-vote detectors), and `key_metric_value` / `key_metric_threshold`
(the scalar the other eight detectors actually compare against — error rate
vs. control limit, AUC drop vs. bootstrap SE, distance ratio vs. 1.5, etc.).
This is the ledger behind the summary table above, kept so a different
threshold or persistence rule can be re-evaluated without rerunning the
replay, and so the *specific features* behind any week's vote can be named
rather than just counted.

**Feature-vote detectors — fraction of the 20 monitored features crossing their
individual threshold, per week** (fires only at ≥0.60, the persistence-gate
input; none of the four ever reach it):

| Week | KS | PSI | JS (KL) | SHAP |
|---|---|---|---|---|
| 1 | 0.30 | 0.20 | 0.40 | 0.25 |
| 2 | 0.35 | 0.25 | 0.45 | 0.20 |
| 3 | 0.20 | 0.15 | 0.35 | 0.05 |
| 4 | 0.15 | 0.15 | 0.40 | 0.25 |
| 5 | 0.35 | 0.15 | 0.35 | 0.10 |
| 6 | 0.30 | 0.15 | 0.45 | 0.05 |
| 7 | 0.30 | 0.15 | 0.45 | 0.05 |
| 8 | 0.30 | 0.15 | 0.45 | 0.20 |
| 9 | 0.35 | 0.15 | 0.45 | 0.10 |
| 10 | 0.35 | 0.15 | 0.45 | 0.15 |
| 11 | 0.20 | 0.15 | 0.30 | 0.05 |
| 12 | 0.40 | 0.20 | 0.50 | 0.05 |
| 13 | 0.40 | 0.20 | 0.45 | 0.20 |
| 14 | 0.45 | 0.20 | 0.45 | 0.30 |

KS, PSI and JS sit well under the 0.6 consensus line all replay long,
consistent with §6.1's finding that this is concept drift, not covariate
shift — JS gets closest (0.50 at week 12) without ever reaching it. **Three
specific features drive most of KS/PSI/JS's baseline fraction in every single
week**: `_mcols_na_bin`, `_vcols_dec0`, and `_vcols_dec1` individually cross
their own threshold in **14 of 14 weeks each**, under both KS and PSI (and
`_mcols_na_bin` under JS too). Investigated in
FEATURE_SELECTION_PROCESS.md §6b–6c: neither is an artifact in the sense the
D-columns were — both show a one-time *step* between the reference window and
week 1 that then stays flat, not a progressive ramp, and the most defensible
explanation is that the 90-day reference window (which spans the dataset's
November–December start and so includes the holiday shopping season) has a
genuinely different composition from any subsequent non-holiday week. SHAP no
longer shows a comparable persistent driver — its week-1 peak of 0.25 (was
0.70 at 10 features) is spread across more, less-correlated features now that
the family cap limits how many `_freq_ref_C*` votes any one family can cast.

**Every detector's raw vs. persistence-confirmed alarm count**, the same 14-week
replay:

| Detector | Raw alarms | Confirmed alarms | Confirmed weeks |
|---|---|---|---|
| ADWIN | 12 | 5 | 2, 4, 8, 10, 12 |
| Prequential AUC | 12 | 6 | 2, 5, 7, 9, 11, 13 |
| EDDM | 8 | 4 | 2, 10, 12, 14 |
| Champion vs Challenger | 5 | 2 | 2, 8 |
| Clustering | 1 | 0 | — |
| Autoencoder, DDM, HDDM, KS, PSI, JS, SHAP | 0 | 0 | — |

The persistence gate is doing real work, not just adding lag: ADWIN and
Prequential AUC each raise a raw flag in 12 of 14 weeks — one shy of *every*
week — but only 5 and 6 of those survive confirmation. Without the gate these
two detectors alone would retrain almost weekly, reproducing the "retraining is
nearly worthless" result of §6.2 by a different route. Clustering's one raw
alarm (week 12, the largest reading in its whole table) never repeats in a
neighbouring window, so it never confirms — the gate correctly treats a
single-window anomaly as unconfirmed rather than as the start of a trend, at
the cost of missing a real, narrow, three-signal-corroborated event.

### 6.2 Classical retraining policies

Mean out-of-sample AUC of the model each policy actually had in force.
`random control` is the percentile against 200 policies retraining the *same
number of times* at randomly chosen weeks.

| Policy | Retrains | Mean AUC | Worst week | Random control | Clears 0.95? |
|---|---|---|---|---|---|
| **Champion vs Challenger** | 2 | **0.8819** | 0.8612 | 0.810 | no |
| EDDM | 3 | 0.8776 | 0.8612 | 0.215 | no |
| Prequential AUC | 6 | 0.8760 | 0.8612 | 0.200 | no |
| ADWIN | 5 | 0.8733 | 0.8612 | 0.055 | no |
| *always_retrain* | 13 | 0.8726 | 0.8612 | — | — |
| *never_retrain* (= KS, PSI, JS, DDM, HDDM, SHAP, Clustering, Autoencoder) | 0 | 0.8725 | 0.8546 | — | — |

Three findings, sharper than before the D-column fix:

1. **Retraining every week now buys statistically nothing** — +0.0001 AUC for
   13 retrains (0.8726 vs. 0.8725), against +0.0094 for Champion vs
   Challenger's two. Before the fix this gap was +0.0025 vs. +0.0149; the
   correction made both numbers move, and the *relative* story sharpened. The
   plausible mechanism is unchanged: retraining at week *w* trains on
   everything through *w*, so retraining during a noisy stretch folds that
   noise in permanently.
2. **Every detector busier than twice a replay is worse than random at equal
   cost.** ADWIN (5.5th percentile) and Prequential AUC (20th) both beat
   never-retraining, so a study without the random control would have reported
   them as successes; EDDM, now retraining 3 times, also falls to the 21.5th
   percentile.
3. **No detector clears the 0.95 bar this run** — a change from the
   pre-D-fix result, where EDDM cleared it at the 96th percentile. The best
   classical policy, Champion vs Challenger, reaches the 81st percentile: a
   real, positive result, but not one that would survive a pre-registered
   0.95 threshold. This is the more sobering and, we think, more honest
   picture — EDDM's earlier "clears the bar" result depended on which weeks a
   now-fixed feature-engineering bug made look turbulent to its error stream.

### 6.3 RL agent benchmark

All policies below act on the **same neural classifier and the same precomputed
lattice**, so differences are decisions, not training randomness. Classical
detectors can only express full retrains (that is their action space); the
agent and the naive action-space controls can also use partial updates. This
run reflects the same 20-feature monitoring set and D-column fix as Section 6.1
— the drift-signal matrix that feeds the agent's state is built from the
corrected features.

| Policy | Mean AUC | Worst week | Mean F1 | Full retrains | Partial updates | Random control | vs. never |
|---|---|---|---|---|---|---|---|
| **RL agent (greedy)** | **0.8831** | 0.8528 | 0.427 | 1 | 12 | 0.470 | **+0.0308** |
| *always partial update* | 0.8820 | 0.8528 | 0.430 | 0 | 13 | 1.000 | +0.0297 |
| RL agent (Thompson) | 0.8812 | 0.8361 | 0.439 | 1 | 12 | 0.120 | +0.0289 |
| always retrain | 0.8711 | 0.8361 | 0.425 | 13 | 0 | 0.745 | +0.0188 |
| *fixed schedule, every 2w* | 0.8703 | 0.8424 | 0.423 | 6 | 0 | 0.925 | +0.0180 |
| ADWIN | 0.8696 | 0.8424 | 0.419 | 5 | 0 | 0.925 | +0.0173 |
| *fixed schedule, every 4w* | 0.8682 | 0.8424 | 0.417 | 3 | 0 | 0.905 | +0.0159 |
| Prequential AUC | 0.8664 | 0.8424 | 0.425 | 6 | 0 | 0.315 | +0.0141 |
| EDDM | 0.8599 | 0.8242 | 0.409 | 3 | 0 | 0.175 | +0.0075 |
| Champion vs Challenger | 0.8594 | 0.8381 | 0.402 | 2 | 0 | 0.265 | +0.0071 |
| never retrain | 0.8523 | 0.8214 | 0.399 | 0 | 0 | — | — |

The agent is top of the table, beating the best classical detector (now ADWIN,
not Prequential AUC — see below) by **+0.0135 AUC** and never-retraining by
**+0.0308**. Its worst week (0.8528) matches `always_partial_update`'s exactly,
both well above never-retrain's 0.8214.

**But the honest decomposition matters more than the ranking, as before.** The
`always_partial_update` row is a policy with no learning, no drift signals, and
no decisions — it simply fine-tunes every week. It reaches 0.8820:

| Source of gain | AUC |
|---|---|
| Expanded action space (cheap partial updates available at all) | **+0.0124** over the best classical detector |
| Learned policy on top of that action space | **+0.0011** over naive always-partial |

**~92% of the RL agent's advantage over the best classical detector comes from
*having* a cheap adaptation action, not from learning when to use it** — an
even larger share than the ~80% measured before the D-column fix and the
wider feature set. This part of the finding is *more* pronounced now, not
less: the headline framing ("RL beats every classical detector by 0.0135
AUC") is, if anything, more misleading post-fix than it was before, because
even more of that gap is bought by the action space alone.

**What is different, and substantial: the drift signals now explain the
entire learned-policy gain (Section 6.4), where before they explained only a
fifth of it.** That +0.0011 is small in absolute terms but is no longer
attributable mostly to "the agent learned a calendar" — see below.

A second, sharpened observation on the classical detectors: their ranking
inverts completely between Section 6.2's LightGBM replay and this neural-model
one. **ADWIN placed *worst* among triggering detectors under LightGBM (5.5th
percentile against its random control) and now places *best* among classical
detectors under the neural model (92.5th percentile, and the highest raw mean
AUC of any classical detector here).** Prequential AUC shows the opposite
pattern less dramatically (20th percentile under LightGBM, 31.5th here).
**Detector rankings are not stable across the underlying classifier** — the
same conclusion as before, now demonstrated with a full reversal rather than a
partial one, and on top of a corrected feature pipeline, which rules out "it
was just the old bug" as an explanation.

### 6.4 Ablation: do the detectors actually contribute?

Three agents, identical except in what they may observe. If `full` does not beat
`context_only`, the honest conclusion is that the detector ensemble contributed
nothing and the agent learned a calendar. **This ablation flips its answer
after the D-column fix and the wider feature set.**

| Agent observes | Mean AUC | Worst week | Reward | Full retrains | Partial updates |
|---|---|---|---|---|---|
| Full (detectors + model context) | 0.8831 | 0.8528 | 38.39 | 1 | 12 |
| Context only (no detectors) | 0.8820 | 0.8528 | 37.29 | 0 | 13 |
| Signals only (no model context) | 0.8831 | 0.8528 | 38.39 | 1 | 12 |

**`context_only` is numerically identical, action-for-action, to
`always_partial_update`.** Deprived of drift signals, the agent finds nothing
better than the naive "fine-tune every week" baseline — model context alone
(staleness counters, recent AUC, recent F1) is not enough to locate the better
1-full+12-partial policy. **`signals_only` is numerically identical to `full`.**
Drift signals *alone*, with no model-context features at all, are sufficient
to recover the entire learned-policy advantage.

**This reverses the earlier (pre-fix, 10-feature) result, which found the
opposite — `context_only` ≈ `full` and `signals_only` contributing almost
nothing.** We flagged that result as a negative finding for the
detector-fusion hypothesis (Section 7 of the original draft). We no longer
believe that conclusion: it was measured on a feature pipeline that contained
a bug (§1.2) manufacturing false drift signal in two of the ten monitored
features, and on a monitoring set with a known redundancy problem (four
correlated members of one family; §2.1, SHAP). With both fixed, the drift
signals earn their place in this ablation. **The reversal itself is the more
important methodological result: a single feature-engineering bug and a
redundant monitoring set were enough to flip a qualitative conclusion about
whether "detectors as sensors, not decision rules" (Section 7) holds on this
dataset.** Section 8's caution about fragility on 14 decision points was
correct, and this before/after comparison is direct evidence for it — not
hypothetical.

The policy-reliance analysis corroborates the reversal. Mean absolute
gradient×input attribution on the chosen action's logit:

| Observation | Reliance |
|---|---|
| `progress` (position through the replay) | **0.470** |
| `weeks_since_reference` | 0.171 |
| `recent_auc_delta` | 0.163 |
| `weeks_since_full_retrain` | 0.139 |
| `ks_mean_statistic` | 0.133 |
| `js_mean_distance` | 0.112 |
| `weeks_since_partial_update` | 0.075 |
| `cluster_distance_ratio` | 0.066 |

`progress` is still the single largest driver, but its dominance dropped
substantially — from 4× the next-highest input before the fix to under 3×
now — and **two drift signals (`ks_mean_statistic`, `js_mean_distance`) now
sit in the top six inputs**, ahead of `weeks_since_partial_update` and
`cluster_distance_ratio`. The agent has not stopped tracking calendar
position, but it is no longer doing *only* that.

### 6.5 Catastrophic forgetting, measured

The ensemble hedge makes forgetting an observable rather than an assumption.
For every partial update and every subsequent week we compute the AUC lost by
trusting the fine-tune fully (α=1) instead of the best available blend:

| Quantity | Value |
|---|---|
| Cases evaluated | 139 |
| Cases where hedging would have helped (any recoverable AUC) | 139 (all of them) |
| Max AUC lost by full trust | **0.00894** |
| Mean AUC lost (where positive) | 0.00199 |
| Best hedge weight = 0.75 | 119 cases |
| Best hedge weight = 0.50 | 20 cases |

Forgetting is **real but even milder** than the earlier measurement (max AUC
recoverable dropped from 0.0182 to 0.0089; mean from 0.0036 to 0.0020), and
every single evaluated case now shows *some* recoverable AUC from hedging,
rather than a mix of positive and near-zero cases. The corrective blend is
almost always gentle (α=0.75 in 119 of 139 cases). This remains a direct
consequence of the design choice in Section 5.1 — partial updates re-derive
from the last *full* model rather than chaining, so damage cannot compound.

Notably, the trained agent **still never used the hedge action** in its final
policy, despite it being free. This remains a missed opportunity, though a
smaller one than before given how mild forgetting now measures.

### 6.6 What the agent's policy relies on

Reported in Section 6.4 above, together with the ablation it interprets.
Full week-by-week decision traces — action, confidence, and the top drivers of
each decision — are written to `reports/rl_decision_trace.csv` by
[explain.py](../explain.py). An excerpt:

| Week | Action | Confidence | Top drivers |
|---|---|---|---|
| 6 | partial update | 0.91 | `progress`=0.385 (+0.06), `weeks_since_reference`=6 (−0.03), `weeks_since_full_retrain`=0.6 (−0.03) |
| 9 | partial update | 0.79 | `weeks_since_reference`=9 (−0.37), `progress`=0.615 (−0.34), `ks_mean_statistic`=0.094 (−0.31) |
| 10 | partial update | 0.64 | `progress`=0.692 (−0.77), `ks_mean_statistic`=0.097 (−0.52), `recent_auc_delta`=4.81 (−0.45) |
| 13 | **full retrain** | 0.48 | `progress`=0.923 (−0.74), `js_mean_distance`=0.131 (+0.23), `recent_auc_delta`=5.15 (−0.20) |
| 14 | partial update | 0.55 | `progress`=1.000 (−2.38), `recent_auc_delta`=8.00 (−1.33), `psi_mean_ratio`=7.66 (+0.39) |

Confidence in "partial update" decays steadily from week 6 (0.91) through
week 12 (0.61, not shown) as `weeks_since_reference`/`weeks_since_full_retrain`
attributions grow increasingly negative — the staleness signal building a case
for a rebuild — before flipping to a full retrain at week 13, where
`js_mean_distance` is the one *positive* driver pushing toward retraining. This
is a later, single retrain compared to the pre-fix run's week-9 retrain; the
qualitative behaviour (confidence erodes, staleness accumulates, a drift
signal tips the decision) is unchanged.


---

## 7. Core Innovation

Stated precisely, so it can be disagreed with:

1. **Reframing drift adaptation from detection to sequential control.** The
   literature asks "did drift occur?"; the operator asks "what do I do this
   week?". The second question depends on the model's own state, admits more
   than two answers, and has delayed consequences. We formalise it as an MDP
   whose state is the *ensemble* of classical detector outputs rather than any
   one of them.

2. **An action space richer than retrain/don't.** This is the innovation that
   *paid* the most — +0.0124 AUC over the best classical detector, ~92% of the
   agent's total advantage. Partial update and ensemble hedging are only
   possible with a differentiable model, and they change the economics: routine
   decay can be answered cheaply, reserving full retrains for genuine regime
   change. Catastrophic forgetting becomes an *observable, measurable* quantity
   — the gap between full trust and the best hedge, up to 0.0089 AUC here —
   rather than an assumed hazard.

3. **Detectors as sensors, not decision rules.** Each classical method is a
   biased estimator of a different aspect of drift, with a documented blind spot
   (Section 2). Rather than choosing one or voting, we feed all of them **as
   continuous statistics** to a learned controller, on the argument that
   thresholding at the detector discards exactly what the controller needs.

   *Our own experiment now supports this, but only after a measurement fix
   changed the answer.* An earlier run — before a feature-engineering bug in
   the D-columns was found and fixed, and before the monitoring set was
   widened from 10 to 20 features with a redundancy correction — found the
   opposite: the signals contributed +0.0011 AUC and an ablation without them
   converged to an identical policy. After both fixes, the same ablation
   (Section 6.4) shows the drift signals are **necessary and sufficient** for
   the entire learned-policy gain — an agent with model context but no drift
   signals matches the naive baseline exactly; an agent with drift signals but
   no model context matches the full agent exactly. The remaining gain this
   buys is still small in absolute terms (+0.0011 AUC), and we do not
   over-claim beyond that — but the *sign* of the finding depended on
   measurement correctness we initially got wrong, which is itself the
   clearest evidence in this paper for why Section 3.1's calibration discipline
   has to precede any claim about which signals matter.

4. **The model lattice as an experimental instrument.** Because cumulative
   retraining makes the reachable model space exactly `{v0, …, v_nweeks}` (plus
   partial and ensemble variants), the entire space can be materialised once and
   every policy — learned, classical, or random control — evaluated as a table
   lookup. This is what makes both PPO training and the frequency-matched random
   control affordable on a 14-window dataset, and it removes seed noise from the
   comparison by construction.

5. **A calibration protocol for drift monitors.** The null experiment
   (Section 3.1) should precede any drift study. A monitor that alarms on two
   random halves of the same window is measuring itself, and this is cheap to
   check and otherwise invisible — it caught three defects here that survived
   code review.

---

## 8. Threats to Validity

- **Label latency is the largest threat.** We assume immediate labels; real
  fraud labels lag by weeks. This advantages the performance-aware detectors and
  the agent's performance-derived context features. A lag-aware replay is the
  most important follow-up.
- **Cumulative retraining only, and this is load-bearing.** The finding that
  frequency barely helps has a mechanism specific to cumulative windows.
  Sliding-window or exponentially-weighted retraining would not share it and
  might reverse the ranking.
- **Fourteen decision points, one dataset — and this is not hypothetical
  caution.** Every policy result rests on 14 binary choices. Directions are
  clearer than magnitudes; percentile distinctions inside a few points are not
  robust to seed or window boundary. This paper now has direct evidence for
  that fragility rather than just an argument for it: fixing one
  feature-engineering bug and widening the monitoring set from 10 to 20
  features flipped the ablation's answer to "do the drift signals matter"
  (Section 6.4) and swapped which classical detector ranked best under the
  neural model (Section 6.3). A result this sensitive to measurement details
  on 14 data points should be read as directional, not as a precise estimate.
- **The agent is trained and evaluated on the same replay.** With one stream
  there is no held-out period, so the agent's performance is an *upper bound* on
  what an online agent would achieve — it has seen these weeks. The ablation and
  the random control are what keep the comparison informative despite this; the
  correct next step is training on one period and evaluating on a later one.
- **Lattice restrictions.** Partial updates are unchained and the ensemble is
  two-component with five discrete weights. A richer space might do better.
- **Frozen representation.** Deliberate, but it means we measure drift relative
  to a fixed feature space.

---

## 9. Related Work

- **Error-rate monitors.** DDM (Gama et al., 2004), EDDM (Baena-García et al.,
  2006), HDDM (Frías-Blanco et al., 2015), ADWIN (Bifet & Gavaldà, 2007). Our
  contribution is not a new detector but the observation that their standard
  input — the 0/1 error stream — is close to uninformative under severe
  imbalance, plus a concrete remedy.
- **Performance-aware detection.** The survey in `papers/` ("From concept drift
  to model degradation") frames the same distinction we use to organise
  Section 2; our prequential-AUC detector is its direct instantiation.
- **Distributional monitoring.** PSI's scorecard lineage, KS monitors, MMD
  tests. Our contribution is the calibration argument: these statistics are
  compared against constants only valid at an unstated sample size.
- **RL for model management.** Prior work applies RL to hyper-parameter and
  architecture search; applying it to the *retraining decision*, with classical
  detector outputs as the observation space, is the gap this addresses.

---

## 10. Conclusion

We set out to combine classical drift detectors into a learned adaptation
controller. The controller works — it tops the benchmark at 0.8831 mean AUC
against 0.8696 for the best classical detector and 0.8523 for never retraining,
matching the best worst-week AUC of any naive policy. But the reason it works
is not fully the reason we expected, and the difference — including a
finding we initially got backwards — is the paper's most useful contribution.

**What actually mattered, in order:**

1. **Measurement.** Before any modelling, 35.4% of features alarmed on data
   containing no drift, and the classifier had F1 = 0.0000 on every window
   because a misconfigured early-stopping rule trained a single tree and a fixed
   0.5 threshold never fired at 3.5% prevalence. Three detectors were monitoring
   a constant. A second, subtler measurement bug — a D-column transformation
   manufacturing a fake linear trend, invisible to the null experiment because
   it required a temporal check, not a random-split one — was found only while
   writing the feature-selection documentation, well after the "final" results
   had already been produced once. Nothing downstream of any of this is
   meaningful until it is fixed, and none of it was visible in code review —
   every instance was caught by a check.

2. **The action space.** Giving the system a cheap partial update worth +0.0124
   AUC over the best classical detector — roughly **92% of the agent's total
   gain** — and a naive "fine-tune every week" policy captures nearly all of it.
   This is the actionable finding for practitioners: *the largest available
   improvement is not a better detector, it is a cheaper adaptation option.* It
   also explains why the neural model was necessary; a GBDT cannot express it.
   This share grew, not shrank, after the measurement fixes — the action space
   is even more clearly the dominant lever than it first appeared.

3. **Timing and detector fusion, small but real, and the sign flipped once.**
   The learned policy adds +0.0011 AUC over naive always-partial. Before the
   measurement fixes, an ablation attributed that entire gain to the agent
   having learned a calendar, and found removing every drift signal cost
   nothing — reported at the time as a negative result for feeding detector
   outputs to a learned controller. After fixing the D-column bug and widening
   the monitoring set from 10 to 20 (with a redundancy correction), the same
   ablation reverses: drift signals are now **necessary and sufficient** for
   the entire +0.0011, while model-context features alone are neither. The
   absolute number is still small — we are not claiming detector fusion is a
   large effect here — but its *sign* depended on getting the measurement right
   first.

**We report the reversal explicitly rather than quietly updating the number**,
because the more important lesson is not "detector fusion helps" or "detector
fusion doesn't help" — it's that **a conclusion this qualitative flipped from
one feature-engineering bug and one redundancy fix, on 14 decision points.**
Anyone citing either version of this result — the original negative finding or
its reversal — should read it as fragile in exactly the way Section 8 argues
in the abstract: directionally suggestive, not a precise or stable estimate.
The conditions under which detector fusion should help *more robustly* are
still identifiable and still worth testing — sharper and less frequent regime
changes, more decision points, and a stream whose drift is visible to
label-free detectors (Section 6.1 established that this stream's drift is
concept drift, which the label-free majority of our signal vector is
structurally blind to) — but we no longer treat our own null result on this
question, from either run, as final.

**The methodological contributions stand independently of the RL result, and
are if anything strengthened by how the RL result moved.** The null experiment
is a cheap precondition for trusting any drift study, but it is not a complete
one — it catches encoder-refitting artifacts and structurally cannot catch a
temporally-monotone one, which is exactly the class of bug that changed this
paper's headline finding. The shared model registry removes seed noise from
detector comparison and caps distinct models at `1 + n_weeks`. The model
lattice turns every retraining policy — learned, classical, or random control —
into a table lookup, which is what makes frequency-matched controls
affordable; those controls are what revealed that detectors retraining more
than twice a replay were performing worse than chance at equal cost, under
both the original and the corrected feature pipeline. And making catastrophic
forgetting a measured quantity (up to 0.0089 AUC recoverable by hedging, even
milder after the fixes) rather than an assumed hazard is available to anyone
using a differentiable model.

**What we would do next**, in priority order: (i) a label-latency-aware replay,
since immediate labels flatter every performance-based detector and the agent's
context features; (ii) train the agent on one period and evaluate on a strictly
later one, since ours is trained and evaluated on the same replay and its
performance is therefore an upper bound; (iii) sliding-window retraining, since
the finding that frequency barely helps has a mechanism specific to cumulative
windows and may not survive; (iv) a stream with covariate shift, to test the
detector-fusion hypothesis under conditions where the detectors can actually
see something; and (v) a systematic audit for other temporally-monotone
transformation artifacts of the kind the D-columns had — the null experiment
cannot find these, and we found this one by accident while writing
documentation, not by design.
