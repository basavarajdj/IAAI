# Learning *When* and *How* to Adapt: Drift Detection and Reinforcement Learning for Fraud Model Retraining

**A research project presentation.** This document walks through the full pipeline —
data, features, modelling, twelve classical drift detectors evaluated as real
retraining policies, and a reinforcement-learning controller built to fix what
those detectors get wrong. Every number and figure below is generated from the
actual pipeline output (`reports/*.csv`), not illustrative.

Companion technical documents, for anyone who wants implementation detail this
presentation intentionally leaves out: [PAPER.md](PAPER.md) (full research
paper), [FEATURE_ENGINEERING_AND_MODELING.md](FEATURE_ENGINEERING_AND_MODELING.md),
[FEATURE_SELECTION_PROCESS.md](FEATURE_SELECTION_PROCESS.md),
[DRIFT_ANALYSIS_EXPLAINED.md](DRIFT_ANALYSIS_EXPLAINED.md).

**Last full rerun: 2026-08-04.** The classical drift-detection pipeline and
the RL experiment were both re-executed end to end on that date; every number
and chart below reflects that run. The pipeline is deterministic on fixed
input data (no random seeds vary run to run), so this rerun reproduced every
figure already in this document to at least four decimal places — see
"Reproducing this report" below for the exact commands.

---

## Reproducing this report

Three scripts, run in order, regenerate everything in this document — every
table, every number, every chart in `visuals/`. Nothing here is hand-edited
from a one-off session.

```bash
# 1. Classical drift-detector pipeline (~15-20 min).
#    Trains the shared LightGBM model registry, runs all 12 detectors over the
#    14-week replay, evaluates each as a retraining policy, writes reports/*.csv
python run_drift_analysis.py --top_k 20

# 2. RL experiment (~15-20 min).
#    Trains the neural model, builds the model lattice, trains the PPO agent,
#    runs the benchmark + ablation, writes reports/rl_*.csv and rl_experiment.json
python run_rl_experiment.py

# 3. Regenerate every chart in report/visuals/ from the fresh reports/*.csv
python report/generate_visuals.py
```

Requires the packages in `requirements.txt` (pandas, numpy, scipy,
scikit-learn, lightgbm, torch, shap, matplotlib) and the two IEEE-CIS CSVs in
`dataset/` (`train_transaction.csv`, `train_identity.csv`) — not included in
the repo; see the dataset's own source for access.

### Where results are stored

All output lands in `reports/` (raw CSV/JSON, git-ignored — regenerate rather
than diff) and `report/visuals/` (the PNGs this document embeds, checked in
so the document renders without rerunning anything):

| What | File |
|---|---|
| Per-week, per-detector alarm state (heatmap source) | `reports/method_week_matrix.csv` |
| Feature-level drift diagnostics, every window | `reports/unified_drift_report.{csv,json}` |
| Classical retraining-policy comparison (§3.5) | `reports/policy_comparison.csv` |
| Method-by-method 14-week deep dive (§3.3) | `reports/method_week_matrix.csv`, `reports/unified_drift_report.json` |
| Feature selection diagnostics — bagged importance, SHAP, redundancy (§1.4) | `reports/feature_selection_diagnostics.csv` |
| Detector pairwise agreement | `reports/detector_agreement.csv` |
| Method-level pros/cons metadata (§3.2 table) | `reports/method_profiles.csv` |
| Catastrophic-forgetting analysis (§5.5) | `reports/forgetting_analysis.csv` |
| RL agent vs. classical/naive policy comparison (§5.2, §5.3) | `reports/rl_policy_comparison.csv` |
| RL ablation — full / context-only / signals-only (§5.4) | `reports/rl_ablation.csv` |
| RL policy reliance / attributions (§5.4) | `reports/rl_policy_reliance.csv` |
| RL decision-by-decision trace | `reports/rl_decision_trace.csv` |
| Full RL experiment dump | `reports/rl_experiment.json` |
| Per-version, per-week AUC matrix | `reports/version_week_auc_matrix.csv` |
| Every chart in this document | `report/visuals/*.png`, built by `report/generate_visuals.py` |

Run logs land in `logs/` (`logs/run_drift_analysis_*.log`,
`logs/run_rl_experiment_*.log`) — useful for confirming a rerun actually
touched every detector rather than exiting early.

The 12 method-by-method tables in §3.3 are pasted-in markdown, not a live
include — `report/_gen_method_tables.py` regenerates them (prints to stdout)
from the same two files listed above. It's a one-off formatting helper, not
part of the three-step reproduction sequence: after a rerun, diff its output
against §3.3 to see whether any table actually needs updating before
re-pasting (on a deterministic pipeline with unchanged input data, it
shouldn't).

### Readable feature and signal names

Every chart renders engineered-feature names through `feature_engineering.feature_label()`
and RL state/signal names through the label table in `report/generate_visuals.py`,
so a chart shows "Var of \"Transaction amount\" per \"email domain x product\""
rather than `_var_TransactionAmt__P_emaildomain__ProductCD`. The raw names are
still what appears in the CSVs in `reports/` — the translation happens only at
chart-generation time.

---

## Table of Contents

1. [Data and Feature Selection](#1-data-and-feature-selection)
2. [Modelling](#2-modelling)
3. [Drift Detection Methods — Pros, Cons, and Results](#3-drift-detection-methods--pros-cons-and-results)
4. [How Reinforcement Learning Solves the Problem](#4-how-reinforcement-learning-solves-the-problem)
5. [What We Gain from RL — Architecture and Results](#5-what-we-gain-from-rl--architecture-and-results)

---

## 1. Data and Feature Selection

### 1.1 The dataset

We use the **IEEE-CIS Fraud Detection** dataset: 590,540 credit-card
transactions spanning roughly six months, starting late November 2017.
Two tables are joined on `TransactionID`:

| Table | Rows | Columns | Contents |
|---|---|---|---|
| `train_transaction` | 590,540 | 394 | Amount, product, card, address, email, engineered "C/D/M/V" blocks, `isFraud` label |
| `train_identity` | 144,233 | 41 | Device and network fingerprint (most transactions have **no** identity row — this table is sparse) |

**Positive-class prevalence is 3.50%.** This single fact shapes almost every
design decision downstream: a naive model predicts "not fraud" for everything
and still looks 96.5% accurate, so every evaluation choice (AUC over accuracy,
a *calibrated* decision threshold, a *class-balanced* error stream for the
monitors) exists specifically to route around this imbalance.

We replay the data as a stream: the **first 90 days** become a reference/
baseline window, and the remaining data is split into **14 consecutive 7-day
windows** — a weekly production cadence.

### 1.2 From 432 raw columns to 113 trained features

After the join, 432 raw predictor columns remain (excluding the join key and
the label). These fall into a few very differently-shaped blocks:

![Feature funnel: 432 raw columns to 113 trained features to 20 monitored features](visuals/01_feature_funnel.png)

| Raw block | Columns | What it is | What survives engineering |
|---|---|---|---|
| **V1–V339 (Vesta)** | 339 | Vesta's own opaque, highly collinear engineered features | Compressed to 4: 2 PCA components + a missing-count + a fixed-subset sum |
| **Identity / device** | 40 | Device type, device info, 38 anonymised `id_` network/behaviour signals | 38 (2 dropped as near-constant) |
| **C1–C14 (counters)** | 14 | Counts of things tied to the card/address (semantics undisclosed) | Replaced by 3 pattern features + 13 frequency encodings |
| **D1–D15 (timedeltas)** | 15 | Days since some prior event, per entity | 16 (14 columns + 2 missingness-pattern features; 1 dropped as near-constant) |
| **M1–M9 (match flags)** | 9 (8 used) | Whether name/address on the card matches billing/shipping | 8 (kept + 1 missingness-pattern feature) |
| **Everything else** | 15 | `TransactionAmt`, `ProductCD`, `card1`–`card6`, `addr1/2`, `dist1/2`, email domains, `TransactionDT` | Kept, transformed, or combined into ~31 features |

![Raw vs. final feature counts by category](visuals/02_feature_category_breakdown.png)

The 339-column Vesta block alone is 78% of the raw predictor space — which is
why aggressive compression, not per-column treatment, is the right strategy
for it specifically.

**What each surviving feature group actually carries as information:**

| Group | Information content |
|---|---|
| **Distance** | How far the billing address is from the shipping address (log-scaled) — a classic fraud tell (mismatched geography) |
| **Temporal** | Cyclic hour-of-day and weekday×hour — captures time-of-day fraud patterns without a hard 23→0 discontinuity |
| **Entity / causal sequence** | Seconds since this card last transacted, % change in amount vs. its last transaction, and its position in its own transaction history — velocity and behavioural-consistency signals, computed causally (no future leakage) |
| **Amount** | The transaction amount, its log, its cents portion, and how many decimal places were written — structuring/laundering signals live in the last two |
| **Aggregates** | Mean/max/variance of amount per email-domain × product-code group, frozen from the reference window — "is this amount typical for this kind of purchase?" |
| **Frequency** | How common a given counter value was during training — rare combinations stand out |
| **Counter / timedelta / match patterns** | Not just *how many* are zero/missing, but *which* — the pattern itself is informative, not just the count |
| **Vesta block** | 2 principal components of 339 opaque engineered features, plus how much of that block is missing — compressed, not discarded |

### 1.3 The representation is frozen — and this is the single most important decision in the whole project

Six of the engineering stages **learn** something from the data they see: the
group-aggregate maps, the frequency-encoding maps, the Vesta PCA rotation, the
near-constant-column filter, and the categorical-to-integer maps. All of these
are fit **once**, on the 90-day reference window, and **replayed unchanged**
on every later week — never refit. Refitting any of them per window
manufactures fake "drift" that is really just the encoder changing under your
feet (a category re-numbered by order of appearance, a PCA axis flipping
sign, a frequency changing because the window got smaller).

We proved this mattered with a **null experiment**: split one window into two
random halves — no real drift can exist between two random halves of the same
data — and count how often the pipeline alarms anyway.

| Encoder fitting | False-alarm rate |
|---|---|
| Refit per window (the naive design) | **35.4%** of features |
| Fit once, frozen (our design) | **0.9%** of features |

A pipeline that alarms on more than a third of its features when nothing has
happened is not measuring drift — it's measuring itself.

### 1.4 Selecting the monitoring set: 113 → 20 features

The model *trains* on all 113 features. But monitoring all 113 for drift
every week would be expensive (per-feature statistical tests, SHAP
attribution) and self-defeating (many of the 113 are near-duplicates of each
other, so a "6 of 10 features drifted" vote could really be one signal
counted six times).

The monitoring set is chosen by:

1. **Bagged importance** — 5 LightGBM fits on bootstrap resamples; a feature
   that every fit independently ranks highly is a property of the data, not
   of one fit's randomness.
2. **SHAP corroboration** — a second, cardinality-unbiased importance signal.
3. **A monitorability filter** — drop anything with fewer than 10 distinct
   values or near-zero variance (a statistical test can't say anything about
   those).
4. **Two-level redundancy pruning** — pairwise Spearman correlation ≥ 0.90 is
   rejected, **and** no more than 2 features from the same "numbered family"
   (e.g. `_freq_ref_C1`...`_freq_ref_C14` are one family) may be selected —
   this second rule was added after we found the *first* 10-feature monitoring
   set had **four** members of one family voting as if independent.

The set size was increased from an initial 10 to **20**. We checked whether
that is actually safe: a repeated null-experiment check (30 random splits, no
drift possible) found **0% false positives** on the 20-feature set — there is
no calibration cost to watching more features, only broader coverage.

> ### 🎤 Speaker Notes — Section 1
>
> - Open with the imbalance number (3.5%) — it explains almost every other
>   design choice later in the talk (calibrated threshold, class-balanced
>   error stream, AUC over accuracy), so plant it early.
> - The funnel chart (432 → 113 → 20) is the one visual to slow down on. Two
>   compressions are happening for *different reasons*: 432→113 is "make the
>   data usable for a model" (the Vesta block alone is 339 of those 432 raw
>   columns); 113→20 is "make it cheap and statistically sound to watch for
>   drift." Don't let the audience conflate the two.
> - The null-experiment number (35.4% → 0.9%) is the single most important
>   result to state plainly: it means that *most published drift-detection
>   demos, if they refit encoders per window, are measuring their own
>   pipeline, not the world.* This is the credibility foundation for
>   everything that follows in Section 3 — without it, nobody should trust
>   any of the confirmed-alarm counts we report there.
> - If asked "why 20 and not 50 or all 113?" — the honest answer is: 20 was a
>   deliberate, checked expansion from an initial 10, not a tuned optimum. We
>   verified it's *safe* (0% false positives under the null) but didn't
>   search for an "ideal" K. That's a fair thing to say to a professor —
>   better than pretending it was optimised.

---

## 2. Modelling

Two classifiers appear in this project, for two different reasons.

### 2.1 The classical model — LightGBM (used by all 12 drift detectors)

| Hyperparameter | Value | Why |
|---|---|---|
| Learning rate | 0.01 | Slow, stable learning |
| Boosting | Gradient-boosted trees, 64 leaves | Standard tabular baseline |
| `is_unbalance` | True | Built-in class weighting for 3.5% prevalence |
| Validation split | **Temporal** (last 20% by time), not random | A random split leaks same-day/same-card rows across the boundary via entity features, giving early stopping an over-optimistic target |
| Early-stopping patience | **100 rounds** | See the bug story below |
| Decision threshold | **Calibrated** by F1-maximising sweep on the validation split, not fixed at 0.5 | See the bug story below |

**A bug worth telling as a cautionary story.** The first version of this
project used an early-stopping patience of 5 rounds. At a learning rate of
0.01, validation AUC barely moves in the first few dozen rounds — so patience
5 halted training at **iteration 1**: every reported model was a single
decision tree. Combined with a fixed 0.5 decision threshold (which, at 3.5%
prevalence, puts almost no probability mass above the line), **every model
had F1 = 0.0000 on every single evaluation window.** Three of the twelve
drift detectors (DDM, EDDM, HDDM) watch the model's *error stream* — with a
constant-output model, that stream carries zero information, so those three
detectors were, unknowingly, monitoring a constant. Fixed by raising patience
to 100 and calibrating the threshold; validation F1 moved from 0.000 to ≈0.47.

A **shared model registry** trains at most **one model per week**, and every
detector that flags drift that week gets handed the identical fitted model —
not twelve independent, differently-seeded fits of the same data. This caps
the number of distinct models at `1 + n_weeks` (15 for this 14-week replay)
regardless of how many detectors are running, and guarantees that two
detectors retraining in the same week are compared against *bit-identical*
models.

### 2.2 The neural model — for the RL agent

A 3-layer MLP (256→128→64→1), BatchNorm + dropout, trained with a
positive-class-weighted loss and the same temporal-split/calibrated-threshold
discipline as the LightGBM model.

**Why switch models at all — the classical model isn't reused for RL?**
Because **a gradient-boosted forest cannot be partially updated.** Adding
trees to an existing booster is a different operation from adapting it — there
is no principled "fine-tune this forest on last month's data." With a GBDT,
the RL agent's action space collapses from four meaningful choices down to
two (retrain / don't), and the entire question this project asks — not just
*whether* to adapt, but *how* — disappears. A differentiable model makes a
cheap partial update a real, few-gradient-steps operation, and — critically —
makes its cost *measurable* (Section 5).

> ### 🎤 Speaker Notes — Section 2
>
> - The early-stopping bug is worth telling in full, even though it's
>   embarrassing, because it's the best evidence in the whole project for why
>   rigorous checking matters — three drift detectors were silently broken by
>   it, and none of that was visible from code review alone; a validation
>   metric caught it.
> - If your professor asks "why not just use LightGBM for the RL agent too,
>   with a smaller action space?" — that's a completely fair question, and the
>   honest answer is in Section 5's results: the action space (partial update)
>   turns out to be responsible for ~92% of the RL agent's advantage. Without
>   a differentiable model, that 92% simply isn't available to capture. This
>   is the strongest justification for the model switch, and it's an
>   *empirical* one, not an architectural preference.
> - The shared registry point is a systems/engineering contribution as much
>   as a modelling one — worth mentioning if the audience cares about
>   production feasibility, not just accuracy numbers.

---

## 3. Drift Detection Methods — Pros, Cons, and Results

We implemented and evaluated **12 classical drift detectors**, organised by
what they observe:

| # | Detector | Family | Observes | Needs labels? |
|---|---|---|---|---|
| 1 | KS test | Distributional | Feature marginals | No |
| 2 | PSI | Distributional | Binned marginals | No |
| 3 | Jensen-Shannon | Distributional | Binned marginals | No |
| 4 | DDM | Performance | Error stream | Yes |
| 5 | EDDM | Performance | Inter-error distances | Yes |
| 6 | HDDM | Performance | Error stream (Hoeffding bound) | Yes |
| 7 | ADWIN | Performance-adjacent | Prediction stream | No |
| 8 | SHAP / attribution | Explanation | Attribution distributions | No |
| 9 | Clustering | Representation | Joint feature geometry | No |
| 10 | Autoencoder | Representation | Reconstruction error | No |
| 11 | Prequential AUC | Performance | Ranking quality | Yes |
| 12 | Champion vs Challenger | Shadow model | Value of retraining | Yes |

Each was run as a genuine **retraining policy** over the 14-week replay — not
just "does it alarm," but "if you retrain whenever this detector confirms,
what AUC do you actually get." A **2-of-2 persistence gate** requires a
detector to fire in 2 consecutive windows before triggering a retrain, so a
one-off blip doesn't cause action.

### 3.1 Which detectors actually fired

![Confirmed alarms per detector, out of 14 weeks](visuals/03_confirmed_alarms_per_detector.png)

**Only 4 of the 12 detectors ever confirmed drift.** KS, PSI, Jensen-Shannon,
DDM, HDDM, SHAP, Clustering, and Autoencoder stayed silent for the entire
14-week replay.

The full week-by-week picture, every detector, every week:

![Heatmap of every detector's raw and confirmed alarm state, by week](visuals/05_method_week_heatmap.png)

**The headline finding: this is concept drift, not covariate shift.** Every
detector that looks only at *feature distributions* stayed silent, while
every detector that watches the *model's behaviour* (error rate, ranking
quality, prediction stream) fired repeatedly. In plain terms: the
transactions in week 9 look statistically like the transactions in the
reference window — same distributions of amount, product, device, etc. — but
*what makes a transaction fraudulent* changed. A production system that only
monitors input distributions (the most common setup, because it needs no
labels) would have shown a flat, reassuring dashboard for six months while
the model quietly decayed.

### 3.2 Method-by-method: pros, cons, and what actually happened here

The four detectors that vote across the monitored feature set (KS, PSI,
Jensen-Shannon, SHAP) never come close to their own 0.60 consensus line, in
any week:

![Weekly trend of the four feature-vote detectors against their consensus threshold](visuals/04_feature_vote_weekly_trend.png)

| Detector | Best at | Blind to / fails when | Result on this stream |
|---|---|---|---|
| **KS test** | Abrupt shift in a continuous feature | Concept drift entirely; over-sensitive at large sample sizes (flagged 14/14 weeks before we fixed the sample-size artifact) | **0/14 weeks.** Feature-crossing fraction stayed at 0.15–0.45, never near the 0.60 consensus line |
| **PSI** | Gradual population shift, interpretable magnitude | Concept drift; its 0.10/0.20 folklore bands assume an unstated sample size | **0/14 weeks.** Lowest fraction of the three distributional tests every week |
| **Jensen-Shannon** | Same job as PSI/KL, but bounded and symmetric | Concept drift | **0/14 weeks.** Closest of the three to the line (0.50 at week 12) but never crosses |
| **DDM** | Abrupt degradation on balanced problems | Hides degradation in the majority class; harder to trigger the longer a model is stable — backwards from what you want | **0/14 weeks confirmed.** |
| **EDDM** | Gradual degradation, earliest warning | Very noise-sensitive at default settings (13/14 weeks before correction) | **4/14 weeks confirmed** (2, 10, 12, 14) — the most retrain-happy detector this run |
| **HDDM** | Staying sensitive on long-stable models; distribution-free guarantee | Conservative by design | **0/14 weeks, and 0 even at the raw-alarm level** — the only detector that never once triggered, in any version of this experiment |
| **ADWIN** | Mean shift in the prediction stream, formal guarantee | Fires on turbulence, which isn't the same as staleness | **12/14 raw alarms, 5/14 confirmed** — as a *retraining policy* this is worse than randomly-timed retraining at equal cost |
| **SHAP / attribution** | The *only* label-free signal with real reach into concept drift — sees what the model relies on shifting | Expensive; blind to drift that doesn't change attributions | **0/14 weeks** at the (corrected) 20-feature set — an earlier, smaller, more redundant monitoring set had produced one confirmed alarm that turned out to be substantially a redundancy artifact |
| **Clustering** | Multivariate shifts no single-feature test can see | Needs standardisation or the largest-scale feature dominates | **0/14 confirmed**, but its one raw alarm (week 12) is the largest reading in its own table by a wide margin — corroborated independently by the autoencoder and by a raw feature outlier the same week |
| **Autoencoder** | Genuinely novel regions of feature space | Reconstruction-error tests over-trigger without an effect-size floor | **0/14 weeks**, but its reconstruction error also peaks sharply at week 12 — the same event Clustering flags |
| **Prequential AUC** | Measures what actually matters, directly | Strictly reactive — can't fire until damage is already visible | **12/14 raw, 6/14 confirmed** — the most persistence-confirmed alarms of any detector, but retrains 6 times for a worse mean AUC than the best 2-retrain policy |
| **Champion vs Challenger** | Answers the real question ("would retraining help") directly | Naive in-sample scoring inflates the challenger and causes constant false alarms — must use out-of-fold scoring | **2/14 confirmed** (weeks 2, 8) — the single best classical retraining policy this run |

### 3.3 Method-by-method deep dive: 14-week performance, thresholds, and why

Section 3.2 gave the one-line pros/cons/result for each detector. This section
shows the underlying week-by-week numbers behind those one-liners — the exact
metric value, the exact threshold it was measured against, and (where the
decision has more than one moving part) *why* a given week did or didn't
cross it. All 14 weeks, all 12 detectors, straight from
`reports/method_week_matrix.csv` and `reports/unified_drift_report.json` — the
same files listed in "Where results are stored" above.

A repeated pattern worth naming up front: for the six detectors that watch
model behaviour (DDM, EDDM, HDDM, ADWIN, Prequential AUC, Champion vs
Challenger), "raw alarm" means the underlying tracker crossed its line that
week; "confirmed" means it did so in **two consecutive weeks**, which is what
actually triggers a retrain (§3 intro, the persistence gate). A lone raw
alarm is common; a confirmed one is not — that gap is most of the story in
this section.

#### 3.3.1 KS test (distributional, label-free)

Votes across the 20 monitored features; a week "confirms" only if the
fraction crossing an individually significant, effect-size-gated KS test
reaches the 0.60 consensus line (chart already shown in §3.2, reproduced here
for this detector alone is unnecessary — the trend for all four vote-based
detectors is in [the shared chart](visuals/04_feature_vote_weekly_trend.png)).

| Week | Features crossed | Fraction | vs. 0.60 consensus | Raw alarm | Confirmed |
|---|---|---|---|---|---|
| 1 | 6/20 | 0.30 | -0.30 | No | No |
| 2 | 7/20 | 0.35 | -0.25 | No | No |
| 3 | 4/20 | 0.20 | -0.40 | No | No |
| 4 | 3/20 | 0.15 | -0.45 | No | No |
| 5 | 7/20 | 0.35 | -0.25 | No | No |
| 6 | 6/20 | 0.30 | -0.30 | No | No |
| 7 | 6/20 | 0.30 | -0.30 | No | No |
| 8 | 6/20 | 0.30 | -0.30 | No | No |
| 9 | 7/20 | 0.35 | -0.25 | No | No |
| 10 | 7/20 | 0.35 | -0.25 | No | No |
| 11 | 4/20 | 0.20 | -0.40 | No | No |
| 12 | 8/20 | 0.40 | -0.20 | No | No |
| 13 | 8/20 | 0.40 | -0.20 | No | No |
| 14 | 9/20 | 0.45 | -0.15 | No | No |

**Why it never fired.** KS tests a feature's *marginal* distribution — is
`TransactionAmt` shaped the same way this week as in the reference window?
Concept drift (§3.1) doesn't move marginals; it moves the *relationship*
between features and the label, which KS cannot see by construction. The
fraction crossing threshold drifts up slowly over the replay (0.30 → 0.45,
consistent with §3.4's seasonal-step finding) but never gets past roughly
three-quarters of the way to consensus.

#### 3.3.2 PSI (distributional, label-free)

| Week | Features crossed | Fraction | vs. 0.60 consensus | Raw alarm | Confirmed |
|---|---|---|---|---|---|
| 1 | 4/20 | 0.20 | -0.40 | No | No |
| 2 | 5/20 | 0.25 | -0.35 | No | No |
| 3 | 3/20 | 0.15 | -0.45 | No | No |
| 4 | 3/20 | 0.15 | -0.45 | No | No |
| 5 | 3/20 | 0.15 | -0.45 | No | No |
| 6 | 3/20 | 0.15 | -0.45 | No | No |
| 7 | 3/20 | 0.15 | -0.45 | No | No |
| 8 | 3/20 | 0.15 | -0.45 | No | No |
| 9 | 3/20 | 0.15 | -0.45 | No | No |
| 10 | 3/20 | 0.15 | -0.45 | No | No |
| 11 | 3/20 | 0.15 | -0.45 | No | No |
| 12 | 4/20 | 0.20 | -0.40 | No | No |
| 13 | 4/20 | 0.20 | -0.40 | No | No |
| 14 | 4/20 | 0.20 | -0.40 | No | No |

The pass/fail count above hides the actual PSI *magnitude*, which is the more
informative view for this specific detector — PSI is usually read as a
continuous score against the 0.10/0.20 scorecard folklore, not a binary flag:

![PSI week-over-week trend, against the scorecard folklore bands](visuals/11_psi_weekly_trend.png)

**Why it never fired.** The mean PSI across all 20 monitored features never
leaves the 0.09–0.15 band — under even the "moderate shift" folklore
threshold of 0.10 most weeks, let alone the pipeline's own bootstrap-null-
calibrated bar. The **max** PSI (dashed line) tells the real story: it swings
between 0.5 and 0.8 almost every week, entirely driven by one feature
(`_mcols_na_bin`, the match-flag missingness pattern — §3.4's seasonal-step
finding). One outlier feature out of twenty is nowhere near the 12-feature
consensus PSI needs to confirm.

#### 3.3.3 Jensen-Shannon (distributional, label-free)

| Week | Features crossed | Fraction | vs. 0.60 consensus | Raw alarm | Confirmed |
|---|---|---|---|---|---|
| 1 | 8/20 | 0.40 | -0.20 | No | No |
| 2 | 9/20 | 0.45 | -0.15 | No | No |
| 3 | 7/20 | 0.35 | -0.25 | No | No |
| 4 | 8/20 | 0.40 | -0.20 | No | No |
| 5 | 7/20 | 0.35 | -0.25 | No | No |
| 6 | 9/20 | 0.45 | -0.15 | No | No |
| 7 | 9/20 | 0.45 | -0.15 | No | No |
| 8 | 9/20 | 0.45 | -0.15 | No | No |
| 9 | 9/20 | 0.45 | -0.15 | No | No |
| 10 | 9/20 | 0.45 | -0.15 | No | No |
| 11 | 6/20 | 0.30 | -0.30 | No | No |
| 12 | 10/20 | 0.50 | -0.10 | No | No |
| 13 | 9/20 | 0.45 | -0.15 | No | No |
| 14 | 9/20 | 0.45 | -0.15 | No | No |

**Why it never fired.** Jensen-Shannon is the closest of the three
distributional tests to actually confirming — it reaches 0.50 at week 12,
one feature short of the 12/20 line — but it is measuring the same thing KS
and PSI measure (bounded, symmetric distributional distance), so it inherits
the same blind spot: it is structurally unable to see a relationship change
that leaves the feature marginals intact.

#### 3.3.4 SHAP / attribution (explanation, label-free)

The only label-free detector with a real mechanism for catching concept
drift — it watches what the model *relies on*, not what the data *looks
like* — but still never confirms at the corrected 20-feature monitoring set:

| Week | Features crossed | Fraction | vs. 0.60 consensus | Raw alarm | Confirmed |
|---|---|---|---|---|---|
| 1 | 5/20 | 0.25 | -0.35 | No | No |
| 2 | 4/20 | 0.20 | -0.40 | No | No |
| 3 | 1/20 | 0.05 | -0.55 | No | No |
| 4 | 5/20 | 0.25 | -0.35 | No | No |
| 5 | 2/20 | 0.10 | -0.50 | No | No |
| 6 | 1/20 | 0.05 | -0.55 | No | No |
| 7 | 1/20 | 0.05 | -0.55 | No | No |
| 8 | 4/20 | 0.20 | -0.40 | No | No |
| 9 | 2/20 | 0.10 | -0.50 | No | No |
| 10 | 3/20 | 0.15 | -0.45 | No | No |
| 11 | 1/20 | 0.05 | -0.55 | No | No |
| 12 | 1/20 | 0.05 | -0.55 | No | No |
| 13 | 4/20 | 0.20 | -0.40 | No | No |
| 14 | 6/20 | 0.30 | -0.30 | No | No |

SHAP crosses threshold on *fewer* features per week than any other
vote-based detector — it is the most conservative of the four by a wide
margin. The feature-level view explains why: how much each feature's
importance actually moves, averaged over all 14 weeks —

![SHAP: which features' influence on the model shifts the most](visuals/12_shap_importance_shift.png)

**Why it never fired.** Even the single most-shifting feature (the
match-flag missingness pattern) moves by only 0.019 in mean |importance
shift| — two orders of magnitude below the kind of swing that would flip a
model's reliance on a feature. The model's *attributions* are simply stable
across the replay, even in weeks where its *accuracy* is not — a genuinely
informative negative result: whatever is driving the AUC drops in §3.3.9 and
§3.3.10 below, it is not a change in which features the model leans on.

#### 3.3.5 DDM (performance, needs labels)

| Week | Mean error rate | DDM boundary (p_min + 3 × s_min) | Raw alarm | Confirmed | Model version after this week |
|---|---|---|---|---|---|
| 1 | 0.3310 | 0.3923 | No | No | v0 |
| 2 | 0.3503 | 0.3923 | No | No | v0 |
| 3 | 0.2945 | 0.3923 | No | No | v0 |
| 4 | 0.2700 | 0.3923 | No | No | v0 |
| 5 | 0.2877 | 0.3923 | No | No | v0 |
| 6 | 0.3160 | 0.3923 | No | No | v0 |
| 7 | 0.3331 | 0.3923 | No | No | v0 |
| 8 | 0.3431 | 0.3923 | No | No | v0 |
| 9 | 0.3240 | 0.3923 | No | No | v0 |
| 10 | 0.3105 | 0.3923 | No | No | v0 |
| 11 | 0.3172 | 0.3923 | No | No | v0 |
| 12 | 0.3104 | 0.3923 | No | No | v0 |
| 13 | 0.3091 | 0.3923 | No | No | v0 |
| 14 | 0.2857 | 0.3923 | No | No | v0 |

DDM and HDDM watch the identical error stream, so they're shown together —
the observed rate against both trackers' own end-of-week boundary:

![DDM & HDDM: the shared error-rate stream stays well under both boundaries](visuals/14_error_rate_trend.png)

**Why it never fired.** DDM's boundary (`p_min + 3·s_min`) is set from the
tracker's best-ever historical window, and its `s_min` term is a Bernoulli
standard deviation — wide, because this stream is noisy at ~30% error even
when nothing is wrong. Three standard deviations above that leaves a very
wide margin (0.39 vs. an observed 0.27–0.35 all replay), and — DDM's known
structural weakness (§3.2) — that margin only gets *harder* to close the
longer the model stays stable, which is backwards from what a retraining
trigger should do.

#### 3.3.6 EDDM (performance, needs labels)

The most retrain-happy detector this run, and the only one in this section
where a *falling* metric — not a rising one — signals drift:

| Week | Inter-error metric (p′+2s′) | Drift boundary (0.90 × running max) | Raw alarm | Confirmed | Model version after this week |
|---|---|---|---|---|---|
| 1 | 8.6330 | 10.8206 | Yes | No | v0 |
| 2 | 8.7439 | 10.8206 | Yes | **Yes** | v1 |
| 3 | 10.6994 | 10.1164 | No | No | v1 |
| 4 | 11.2809 | 10.1164 | No | No | v1 |
| 5 | 11.1322 | 10.1164 | No | No | v1 |
| 6 | 10.9475 | 10.1164 | No | No | v1 |
| 7 | 10.6051 | 10.1164 | No | No | v1 |
| 8 | 10.2722 | 10.1164 | No | No | v1 |
| 9 | 10.1934 | 10.1164 | Yes | No | v1 |
| 10 | 10.1243 | 10.1164 | Yes | **Yes** | v7 |
| 11 | 9.6959 | 11.0349 | Yes | No | v7 |
| 12 | 9.7206 | 11.0349 | Yes | **Yes** | v9 |
| 13 | 10.8996 | 11.3737 | Yes | No | v9 |
| 14 | 10.6470 | 11.3737 | Yes | **Yes** | v11 |

> EDDM alarms when the metric value *drops below* its boundary (errors
> bunching closer together = more frequent errors); it is the only detector
> in this section where "below the line" is the bad direction.

**Why it fired 4 of 14 weeks — more than any other detector.** The boundary
is 90% of the *largest* inter-error distance the tracker has ever seen, and
that ceiling only ever grows (or resets, on retrain). Once the model settles
into a good stretch, the running max climbs, and the very next mediocre week
looks small by comparison and trips the line — a mechanical consequence of
comparing every week against the *best* week ever seen rather than a typical
one. This is exactly the mirror image of DDM's problem: DDM gets harder to
trigger over time, EDDM gets easier.

#### 3.3.7 HDDM (performance, needs labels)

| Week | Mean error rate | Boundary (best-window mean + Hoeffding bound) | Raw alarm | Confirmed | Model version after this week |
|---|---|---|---|---|---|
| 1 | 0.3310 | 0.3465 | No | No | v0 |
| 2 | 0.3503 | 0.3354 | No | No | v0 |
| 3 | 0.2945 | 0.3512 | No | No | v0 |
| 4 | 0.2700 | 0.3306 | No | No | v0 |
| 5 | 0.2877 | 0.3229 | No | No | v0 |
| 6 | 0.3160 | 0.3213 | No | No | v0 |
| 7 | 0.3331 | 0.3202 | No | No | v0 |
| 8 | 0.3431 | 0.3192 | No | No | v0 |
| 9 | 0.3240 | 0.3184 | No | No | v0 |
| 10 | 0.3105 | 0.3178 | No | No | v0 |
| 11 | 0.3172 | 0.3171 | No | No | v0 |
| 12 | 0.3104 | 0.3165 | No | No | v0 |
| 13 | 0.3091 | 0.3159 | No | No | v0 |
| 14 | 0.2857 | 0.3158 | No | No | v0 |

(Trend chart shared with DDM, above.)

**Why it never fired — not even once, raw or confirmed, the only detector in
this project with that distinction.** Unlike DDM's fixed, wide margin,
HDDM's boundary is genuinely close to the observed rate by the second half of
the replay (within 0.01–0.02 from week 8 onward — visibly the two closest
lines in the chart above). It is still a distribution-free, Hoeffding-bound
test built for statistical soundness over sensitivity: the columns above are
an end-of-week *snapshot* of a stateful, per-sample tracker, and the true
decision rule requires the *cumulative* mean to clear an even wider combined
bound (its own historical best window's uncertainty plus the current
window's) — so "close" here does not mean "nearly triggered." Read together
with DDM, the two Bernoulli/Hoeffding-style detectors in this project simply
never found this error stream to look anomalous relative to its own history,
even in weeks the other nine detectors flagged loudly.

#### 3.3.8 ADWIN (performance-adjacent, label-free)

| Week | z-score | z-threshold | Raw alarm | Confirmed | Model version after this week |
|---|---|---|---|---|---|
| 1 | 0.101 | 3.090 | Yes | No | v0 |
| 2 | 1.374 | 3.090 | Yes | **Yes** | v1 |
| 3 | 1.004 | 3.090 | Yes | No | v1 |
| 4 | 9.521 | 3.090 | Yes | **Yes** | v2 |
| 5 | 5.348 | 3.090 | Yes | No | v2 |
| 6 | 3.150 | 3.090 | No | No | v2 |
| 7 | 5.289 | 3.090 | Yes | No | v2 |
| 8 | 6.328 | 3.090 | Yes | **Yes** | v5 |
| 9 | 3.585 | 3.090 | Yes | No | v5 |
| 10 | 2.421 | 3.090 | Yes | **Yes** | v7 |
| 11 | 6.372 | 3.090 | Yes | No | v7 |
| 12 | 9.461 | 3.090 | Yes | **Yes** | v9 |
| 13 | 1.602 | 3.090 | Yes | No | v9 |
| 14 | 0.055 | 3.090 | No | No | v9 |

![ADWIN: mean-shift z-score vs. its formal threshold](visuals/15_adwin_zscore_trend.png)

**Why it fired constantly, and why that's not actually good.** Notice the
raw column: 12 of 14 weeks are "Yes" — the z-score in row 1, column 1 of the
chart above is genuinely chaotic, swinging from near-zero to 9.5× the
threshold and back within a couple of weeks (weeks 3→4→6, or 10→11→12→13→14).
ADWIN has a real formal guarantee (it correctly detects *a* mean shift in the
prediction stream), but "a mean shift happened" and "the model is stale" are
different claims — this is the detector §3.5 shows loses to random-timed
retraining at the same budget (5.5th percentile), and this table is the
mechanism: it is reacting to week-to-week turbulence, not accumulated
staleness.

#### 3.3.9 Prequential AUC (performance, needs labels)

| Week | AUC drop vs. reference | Effective drop threshold | Raw alarm | Confirmed | Model version after this week |
|---|---|---|---|---|---|
| 1 | 0.0706 | 0.0136 | Yes | No | v0 |
| 2 | 0.0644 | 0.0137 | Yes | **Yes** | v1 |
| 3 | 0.0129 | 0.0097 | No | No | v1 |
| 4 | 0.0329 | 0.0109 | Yes | No | v1 |
| 5 | 0.0383 | 0.0125 | Yes | **Yes** | v3 |
| 6 | 0.0303 | 0.0121 | Yes | No | v3 |
| 7 | 0.0359 | 0.0141 | Yes | **Yes** | v4 |
| 8 | 0.0441 | 0.0128 | Yes | No | v4 |
| 9 | 0.0442 | 0.0151 | Yes | **Yes** | v6 |
| 10 | 0.0484 | 0.0166 | Yes | No | v6 |
| 11 | 0.0403 | 0.0125 | Yes | **Yes** | v8 |
| 12 | 0.0382 | 0.0132 | Yes | No | v8 |
| 13 | 0.0262 | 0.0115 | Yes | **Yes** | v10 |
| 14 | 0.0178 | 0.0316 | No | No | v10 |

![Prequential AUC: current performance vs. its own reference](visuals/16_prequential_auc_trend.png)

**Why it fired the most (6 confirmed, tied for most raw alarms too).**
Prequential AUC measures the thing that actually matters — ranking quality —
directly, with no proxy in between. The cost of that directness is visible in
the chart: current-week AUC sits *below* the reference line in nearly every
single week, because the reference resets to a fresh (higher) validation AUC
every time the detector retrains, and the gap immediately starts reopening.
With an AUC drop above its own effective threshold almost every week, the
2-of-2 persistence gate is nearly always satisfied — which is exactly why
§3.5 shows this policy retraining 6 times for a *worse* mean AUC than the
best 2-retrain policy: it is strictly reactive, confirming damage that has
already happened rather than anticipating it.

#### 3.3.10 Champion vs Challenger (shadow model, needs labels)

The single best classical retraining policy this run (§3.5) — and the table
below shows it has two independent ways to fire, not one:

| Week | AUC gap (challenger − champion) | Gap threshold (0.03) | AUC degradation from baseline | Degradation threshold (0.05) | Raw alarm | Confirmed | Model version after this week |
|---|---|---|---|---|---|---|---|
| 1 | 0.0035 | 0.0300 | 0.0706 | 0.0500 | Yes | No | v0 |
| 2 | 0.0287 | 0.0300 | 0.0644 | 0.0500 | Yes | **Yes** | v1 |
| 3 | -0.0020 | 0.0300 | 0.0129 | 0.0500 | No | No | v1 |
| 4 | 0.0176 | 0.0300 | 0.0329 | 0.0500 | No | No | v1 |
| 5 | 0.0037 | 0.0300 | 0.0383 | 0.0500 | No | No | v1 |
| 6 | 0.0270 | 0.0300 | 0.0418 | 0.0500 | No | No | v1 |
| 7 | 0.0228 | 0.0300 | 0.0502 | 0.0500 | Yes | No | v1 |
| 8 | 0.0207 | 0.0300 | 0.0607 | 0.0500 | Yes | **Yes** | v5 |
| 9 | -0.0114 | 0.0300 | 0.0413 | 0.0500 | No | No | v5 |
| 10 | 0.0215 | 0.0300 | 0.0477 | 0.0500 | No | No | v5 |
| 11 | 0.0189 | 0.0300 | 0.0406 | 0.0500 | No | No | v5 |
| 12 | 0.0475 | 0.0300 | 0.0493 | 0.0500 | Yes | No | v5 |
| 13 | 0.0159 | 0.0300 | 0.0362 | 0.0500 | No | No | v5 |
| 14 | -0.1485 | 0.0300 | 0.0296 | 0.0500 | No | No | v5 |

> Fires when EITHER the gap clears 0.03 (and its own bootstrap standard
> error, so a noisy small-sample gap doesn't count) OR degradation from
> baseline clears 0.05. In every week this method actually raised a raw
> alarm, the **degradation** column is what crossed — the gap column alone
> never independently clears 0.03-with-significance in this replay.

How the champion (currently deployed) and challenger (freshly retrained,
scored out-of-fold to avoid the overfitting bias described in §2.1) actually
compared, week by week:

![Champion vs. Challenger: would retraining actually help this week?](visuals/17_champion_challenger.png)

**Why only 2 of 14 confirmed, despite 4 raw alarms.** The persistence gate
is the whole story here: weeks 1 and 7 raise a raw alarm (degradation just
past 0.05) but are immediately followed by a week that drops back under
threshold (week 2 still qualifies and confirms; week 8 also still qualifies
and confirms) — but weeks 4–6 and 9–13 never sustain two in a row. This is
the most direct of the twelve questions ("would retraining actually help
this week?") and the persistence gate keeps it from overreacting to any
single noisy week — the discipline that makes it the best classical policy
in §3.5, at only 2 retrains.

#### 3.3.11 Clustering (representation, label-free)

K-Means with a **fixed k = 5** clusters — not learned or tuned per week,
fit once on the reference window and never refit, exactly like every other
frozen encoder in this pipeline (§1.3):

| Week | Centroid distance ratio | Distance threshold (1.5) | Cluster-assignment PSI | PSI threshold (0.2) | Raw alarm | Confirmed | Model version after this week |
|---|---|---|---|---|---|---|---|
| 1 | 1.035 | 1.500 | 0.165 | 0.200 | No | No | v0 |
| 2 | 1.066 | 1.500 | 0.193 | 0.200 | No | No | v0 |
| 3 | 1.057 | 1.500 | 0.117 | 0.200 | No | No | v0 |
| 4 | 1.059 | 1.500 | 0.143 | 0.200 | No | No | v0 |
| 5 | 1.088 | 1.500 | 0.125 | 0.200 | No | No | v0 |
| 6 | 1.089 | 1.500 | 0.083 | 0.200 | No | No | v0 |
| 7 | 1.101 | 1.500 | 0.100 | 0.200 | No | No | v0 |
| 8 | 1.094 | 1.500 | 0.100 | 0.200 | No | No | v0 |
| 9 | 1.097 | 1.500 | 0.110 | 0.200 | No | No | v0 |
| 10 | 1.109 | 1.500 | 0.095 | 0.200 | No | No | v0 |
| 11 | 1.363 | 1.500 | 0.026 | 0.200 | No | No | v0 |
| 12 | 2.949 | 1.500 | 0.034 | 0.200 | **Yes** | No | v0 |
| 13 | 1.158 | 1.500 | 0.094 | 0.200 | No | No | v0 |
| 14 | 1.175 | 1.500 | 0.086 | 0.200 | No | No | v0 |

> Fires when EITHER series crosses its own threshold — two independent
> triggers sharing one fixed clustering.

![Clustering (k=5 fixed): both drift signals vs. their thresholds](visuals/13_clustering_trend.png)

**Why it never confirmed, despite one large raw alarm.** Every week but one,
the mean distance from a point to its nearest of the 5 reference centroids
sits at a steady ~1.0–1.1× the reference baseline — genuinely stable
multivariate geometry. Week 12 is a sharp, isolated spike to 2.95× (almost
double the drift threshold) and is the largest reading in this detector's
entire table by a wide margin — but week 13 snaps straight back to 1.16×.
The 2-of-2 persistence gate is doing exactly its job: a single-week outlier,
however large, does not get to trigger a retrain on its own. (This same week
12 event independently shows up in the autoencoder, below, and in the raw
top-drifting-feature data in §3.4 — three unrelated methods agreeing makes
it a real, corroborated one-week anomaly, just not a *sustained* one.)

#### 3.3.12 Autoencoder (representation, label-free)

A bottleneck MLP trained once on the reference window, monitoring
reconstruction-error z-score on every later week:

| Week | Reconstruction-error z-score | Threshold (3.0) | Raw alarm | Confirmed |
|---|---|---|---|---|
| 1 | 0.100 | 3.000 | No | No |
| 2 | 0.295 | 3.000 | No | No |
| 3 | 0.110 | 3.000 | No | No |
| 4 | 0.091 | 3.000 | No | No |
| 5 | 0.156 | 3.000 | No | No |
| 6 | 0.279 | 3.000 | No | No |
| 7 | 0.207 | 3.000 | No | No |
| 8 | 0.200 | 3.000 | No | No |
| 9 | 0.198 | 3.000 | No | No |
| 10 | 0.180 | 3.000 | No | No |
| 11 | 0.454 | 3.000 | No | No |
| 12 | 2.458 | 3.000 | No | No |
| 13 | 0.282 | 3.000 | No | No |
| 14 | 0.331 | 3.000 | No | No |

![Autoencoder: reconstruction-error z-score vs. threshold](visuals/18_autoencoder_zscore_trend.png)

**Why it never fired, even at its closest.** The same week-12 event visible
in the clustering table above shows up here too — the z-score jumps roughly
15× its typical level, to 2.46, its only reading anywhere near the 3.0
threshold in the entire replay — but even that spike falls short, and every
other week sits below 0.5. Two representation-based detectors independently
noticing the same single week, and neither one confirming it, is itself a
useful result: it says the week-12 event was real but genuinely
one-off, not the start of a sustained shift — exactly the kind of event a
persistence-gated policy is designed to not overreact to.


### 3.4 Which specific features are actually drifting, and how often

Rather than just counting *how many* features cross threshold each week, we
tracked *which* ones:

![Top individually-drifting features across all four feature-vote detectors](visuals/06_top_drifting_features.png)

Three features cross their own threshold in **14 of 14 weeks**, individually,
under multiple detectors at once: `_mcols_na_bin` (the missingness pattern
across match-flag columns) and the two Vesta PCA components. We investigated
each rather than assuming they were real drift, and found two different
mechanisms:

- **A real bug, found and fixed.** Two "days-since-event" columns (`D2`,
  `D15`) were computed as `raw_value − days_elapsed_since_dataset_start`.
  Since the raw value is roughly *stationary* over time but the subtracted
  quantity grows *linearly*, this manufactured a fake trend out of nothing —
  `D2` alone went from crossing threshold in 14/14 weeks to 2/14 once we
  removed the bad subtraction. This is the same "monotone-in-time proxy"
  problem that `TransactionDT` itself is deliberately excluded from the model
  for — just reintroduced here through a transformation, not a raw
  timestamp.
- **Real, but not drift in the usual sense.** `_mcols_na_bin` and the Vesta
  components show a one-time *step* between the 90-day reference window and
  week 1 that then stays flat — not a progressive ramp. The most defensible
  explanation: the reference window spans the dataset's **holiday-season**
  start (late November–February), and later weeks don't. The "drift" these
  three features report is most likely the reference window itself being
  seasonally unrepresentative of the rest of the year — a real, corroborated
  signal, but not the kind of drift a retraining policy should chase every
  week.

### 3.5 Classical detectors as retraining policies — the surprising result

Detecting drift and knowing when to *retrain* are not the same question.
Turning each detector into a real policy and measuring out-of-sample AUC:

| Policy | Retrains | Mean AUC | vs. randomly-timed policies of equal cost |
|---|---|---|---|
| **Champion vs Challenger** | 2 | **0.8819** | 81st percentile |
| EDDM | 3 | 0.8776 | 21.5th percentile |
| Prequential AUC | 6 | 0.8760 | 20th percentile |
| ADWIN | 5 | 0.8733 | 5.5th percentile |
| *Always retrain (all 13 weeks)* | 13 | 0.8726 | — |
| *Never retrain* | 0 | 0.8725 | — |

Two findings stand out:

1. **Retraining every single week buys statistically nothing** — 13 retrains
   improve mean AUC by +0.0001 over never retraining. Two *well-timed*
   retrains improve it by +0.0094 — roughly two orders of magnitude the
   benefit, at 15% of the cost.
2. **Detectors that retrain more than twice a replay are worse than a coin
   flip at the same budget.** ADWIN, at 5 retrains, sits at just the 5.5th
   percentile of randomly-timed policies spending the same budget — you would
   have beaten it ~94.5% of the time by picking retraining weeks at random.
   Without that random-policy comparison, ADWIN would look like a success
   story (it does beat never-retraining) — this is exactly why the comparison
   matters.

> ### 🎤 Speaker Notes — Section 3
>
> - Lead with the confirmed-alarms bar chart, then immediately pivot to the
>   heatmap — the bar chart tells you *what*, the heatmap tells you *when and
>   how persistent*. The heatmap is the visual to leave on screen the longest;
>   it's the one that makes "concept drift, not covariate shift" visually
>   obvious (the top rows — all distributional/representation detectors — are
>   just empty).
> - The D-column bug story is a great moment to pause on if the professor is
>   evaluating rigor, not just results — it demonstrates we didn't just trust
>   "the model looks done," we went back and audited *why* specific features
>   kept flagging, and found a real defect. This is the kind of thing a
>   reviewer wants to see: results that survived being doubted.
> - Be ready for "so which detector should I use in production?" The honest
>   answer, and the one to give: *it depends what kind of drift you expect,*
>   and on this dataset the answer would have been "none of the label-free
>   ones" — which is uncomfortable, because label-free monitoring (no waiting
>   for ground truth) is what most production systems actually run. That
>   discomfort is exactly the motivation for Section 4.
> - The random-control percentile point (ADWIN at the 5.5th percentile) is the
>   single most persuasive number in this section for a skeptical audience —
>   it's the concrete evidence that "detects real drift" and "makes a good
>   retraining trigger" are different claims.
> - §3.3 (the 12-detector deep dive) is reference material, not a slide to
>   present linearly — don't walk through all twelve in a talk. Keep three in
>   your pocket for questions: PSI (§3.3.2, the mean-vs-max chart is the best
>   single illustration of "one outlier feature can't win a 20-feature vote"),
>   Clustering (§3.3.11, the week-12 spike that two independent
>   representation-based detectors both saw and neither confirmed — a clean
>   example of the persistence gate working as designed), and Champion vs
>   Challenger (§3.3.10, the two-independent-triggers table is the best
>   concrete illustration of why this detector's discipline beats the others).

---

## 4. How Reinforcement Learning Solves the Problem

### 4.1 The reframing

Every one of the 12 detectors above answers the same kind of question: **"has
drift occurred?"** — a one-shot, binary classification problem. But the
operator's actual question is different: **"given everything I've observed,
and the model I currently have, what should I do this week?"** — a
*sequential decision* problem. These differ in three concrete ways no
detector's design accounts for:

1. **The right action depends on the model's own state, not just the data.**
   The same drift signal justifies an urgent retrain if the model is six
   months stale, and justifies nothing if it was rebuilt last week. No
   detector tracks its own model's age.
2. **The choice isn't binary.** Between "do nothing" and "rebuild from
   scratch" there is a cheap fine-tune and a free ensemble re-weighting — real
   options every classical detector's design ignores.
3. **Actions have delayed consequences.** Retraining during a turbulent week
   permanently folds that turbulence into a cumulative training set, and the
   cost shows up weeks later, not immediately. Section 3.5 already showed
   this mechanism at work — busy detectors retrain into noise.

This is precisely the structure of a **Markov Decision Process**, so we frame
it as one and learn the policy with reinforcement learning, instead of hand-
tuning a threshold.

### 4.2 The MDP formulation

```mermaid
flowchart LR
    S["State s_t
    11 drift-detector signals
    + 6 model-context features"] --> AG["Agent
    drift encoder → policy head"]
    AG --> ACT["Action a_t
    do nothing / partial update /
    full retrain / hedge ensemble"]
    ACT --> ENV["Environment
    (the model lattice)"]
    ENV --> RW["Reward r_t
    100 × (AUC gain over never-retrain)
    − 100 × action cost"]
    RW -.becomes next week's state.-> S
```

**State** deliberately mixes *all twelve* detectors' continuous outputs
(not booleans — "PSI is 0.19" and "PSI is 0.02" are both "no drift" to a
threshold rule, and obviously different to a learner) with model-context
features: weeks since the last full retrain, weeks since the last partial
update, the current ensemble weight, recent AUC and F1, and progress through
the replay.

**Action space** — four choices, only possible because the underlying model
is differentiable (Section 2.2):

| Action | Effect | Relative cost |
|---|---|---|
| Do nothing | Keep the current model | Free |
| Partial update | Fine-tune the last *full* model on the recent 4 weeks | Cheap |
| Full retrain | Rebuild on all data seen so far | Expensive |
| Hedge ensemble | Shift weight from the current model toward the stable baseline | Free |

**Reward** is realised performance improvement over a never-retrain baseline,
minus the action's cost. A decision made in week *t* is graded starting week
*t+1* — grading it on its own week would let the agent retrain on data it has
already been scored against.

### 4.3 Why this needed a model change, and why PPO

We use **Proximal Policy Optimization (PPO)**: the action space is small and
discrete, episodes are only 14 steps, so sample efficiency isn't the binding
constraint — *stability* is, and PPO's clipped objective prevents any single
noisy batch from over-correcting the policy, which matters when the entire
dataset is one 14-week trajectory replayed under different choices.

**Making PPO trainable on 14 weeks at all** required one further trick: under
this action set, the model in force at any point is fully determined by three
numbers — `(last full-retrain week, last partial-update week, ensemble
weight)`. That space is small and enumerable, so we precompute **every**
reachable model once (the "model lattice"), cache what each one scores on
every future week, and the training environment becomes a lookup table.
Episodes then cost nothing to simulate, which is what makes thousands of PPO
episodes affordable on a 14-window dataset.

> ### 🎤 Speaker Notes — Section 4
>
> - The three-point list (model-state-dependence, non-binary choice, delayed
>   consequences) is the intellectual core of the whole project — if the
>   audience only remembers one slide's worth of content, make it this one.
>   Everything in Section 3 is evidence *for* points 2 and 3 specifically
>   (busy detectors retrain into noise = delayed consequences; no detector
>   knows if the model is already fresh = state-dependence).
> - The mermaid diagram is a genuine causal loop, not just decoration — trace
>   it out loud: state → action → reward → *becomes* next state. Emphasise
>   that the reward is graded one week *later* than the decision — that's the
>   detail that prevents the agent from cheating.
> - If asked "why not just use a bigger/more complex model for the agent" —
>   the model lattice trick is the answer: the *environment* being reducible
>   to a lookup table is what makes this tractable at all on 14 data points;
>   a bigger network wouldn't fix the more fundamental problem of having very
>   little data to learn from, and might overfit worse.
> - Good moment to explicitly name the limitation: 14 decision points is
>   very few for a reinforcement-learning problem. We address this directly
>   in Section 5 rather than hiding it — the ablation study result there is
>   presented as fragile *because* of this small-sample reality.

---

## 5. What We Gain from RL — Architecture and Results

### 5.1 Agent architecture

```mermaid
flowchart TD
    IN["Drift signals + model context
    (17-dim input)"] --> ENC["Drift Encoder
    2-layer MLP, tanh activation, shared"]
    ENC --> POL["Policy Head
    4 actions (categorical distribution)"]
    ENC --> VAL["Value Head
    V(s), a single scalar"]
```

The encoder is **shared** between the policy and value heads deliberately:
with only 14 data points to learn from, the value head's gradient becomes
extra supervision for the same small encoder, rather than each head having to
learn its own representation from scratch. Trained with PPO: clipped
surrogate objective, generalised advantage estimation, an entropy bonus to
keep exploring, and gradient-norm clipping for stability. Exploration is
epsilon-greedy during training (annealed from 0.30 to 0.02) and Thompson
sampling (drawing from the learned action distribution rather than always
taking the top choice) as a production-safe alternative.

### 5.2 Benchmark results

Every policy below — classical detector, naive control, or the RL agent —
acts on the **same neural classifier and the same precomputed model lattice**,
so differences come from decisions, not training randomness or lucky seeds.

![RL policy comparison bar chart](visuals/07_rl_policy_comparison.png)

The RL agent tops the benchmark at **0.8831 mean AUC**, against **0.8696**
for the best classical detector (ADWIN, under this model) and **0.8523** for
never retraining.

### 5.3 The honest decomposition — where does that gain actually come from?

The chart above hides an uncomfortable detail: the **"always partial
update"** policy — fine-tune every single week, no learning, no drift signals,
no decisions at all — reaches **0.8820**, just 0.0011 AUC behind the full
learned agent.

![Gain decomposition: action space vs. learned policy](visuals/08_gain_decomposition.png)

**Roughly 92% of the RL agent's advantage over the best classical detector
comes from simply *having* a cheap adaptation action available — not from
the learning itself.** A naive "fine-tune every week" policy captures almost
all of it. This is worth stating plainly rather than leading only with the
headline number, because "RL beats every classical detector by 0.0135 AUC" is
true and, on its own, would give a misleading impression of what earned that
result.

### 5.4 Do the drift detectors actually matter? Yes — but we only know that because we checked, and got it wrong once first

We trained three identical agents, differing only in what they can observe:

![Ablation: full agent vs. context-only vs. signals-only](visuals/09_ablation.png)

The pattern is exact, not approximate: **"context only" (no drift signals)
lands on precisely the naive always-partial-update policy — it gains nothing
from having model-context features without drift signals. "Signals only" (no
model context) lands on precisely the full agent's performance** — the drift
signals alone are sufficient to recover the entire learned-policy advantage.

**This result reversed once already**, on an earlier, buggy version of the
feature pipeline (before the D-column bug in Section 3.4 was fixed, and
before the monitoring set was widened from 10 to 20 features). That earlier
run found the *opposite* — model context alone reproduced almost all of the
full agent's performance, and dropping the drift signals cost almost
nothing. We initially reported that as a negative result for feeding
detector outputs to a learned controller. **We no longer believe that
conclusion** — it was measured on a feature pipeline with a real defect in
it. We consider the *reversal itself* more important than either individual
result: a single feature-engineering bug and one redundancy fix were enough
to flip a qualitative conclusion, on only 14 data points. Any claim drawn
from an experiment this small should be treated as directionally suggestive,
not as a settled fact — and this project has direct, empirical evidence for
that caution, not just a theoretical worry about it.

**What the agent's decisions actually rely on:**

![Policy reliance: which inputs drive the agent's actions](visuals/10_policy_reliance.png)

`progress` (position in the replay — essentially, "what week is it") is still
the single largest driver, but far less dominant than in the earlier, buggy
run (previously 4× the next-largest input; now under 3×) — and **two genuine
drift signals now sit inside the top six inputs**, rather than being crowded
out entirely.

### 5.5 Catastrophic forgetting — measured, not assumed

Because the agent can compare "trust the fine-tuned model fully" against
"blend it back toward the stable baseline," forgetting becomes a number
instead of an assumption:

| Quantity | Value |
|---|---|
| Cases evaluated | 139 |
| Cases where hedging would have recovered some AUC | 139 (all of them) |
| Maximum AUC recoverable by hedging | 0.0089 |
| Mean AUC recoverable (where positive) | 0.0020 |
| Best correction is usually gentle (α = 0.75) | 119 of 139 cases |

Forgetting is real but mild here — a direct consequence of a design choice:
partial updates always branch from the last **full** model, never chain from
the previous partial update, so damage from a bad fine-tune cannot compound
across weeks. Notably, the trained agent **never actually used the free hedge
action** in its final policy, despite this measured, recoverable benefit —
a real but modest missed opportunity in what the learned policy captured.

### 5.6 Summary — what RL actually bought us

1. **A better ceiling.** 0.8831 vs. 0.8696 best-classical vs. 0.8523
   never-retrain, with the best worst-week performance of any policy — the
   week fraud losses would actually spike.
2. **Mostly, a cheaper way to adapt, not a smarter detector.** ~92% of the
   gain over the best classical detector is available to *any* system with a
   partial-update option, whether or not it uses learning at all.
3. **A small but now-real role for the drift detectors** — necessary and
   sufficient for the last ~8% of the gain, once measurement was fixed.
4. **Forgetting turned from an assumed hazard into a measured quantity** —
   up to 0.0089 AUC recoverable by hedging, on this stream.
5. **A methodological lesson bigger than any single number**: a project this
   size (14 decision points) can flip its own qualitative conclusions from
   one bug fix. That fragility is not a flaw to hide — it's the strongest
   argument in the whole project for calibration checks (Section 1.3) and
   ablations (Section 5.4) as required steps, not optional polish.

> ### 🎤 Speaker Notes — Section 5
>
> - Show the benchmark chart first, let it land, *then* show the gain-
>   decomposition chart immediately after — the sequence "RL wins" →
>   "here's why that's not the whole story" is the intended rhetorical
>   structure, and it's much more convincing delivered in that order than if
>   you lead with the caveat.
> - The ablation reversal (5.4) is the most sophisticated result in the
>   project and the one most likely to draw follow-up questions. Have the
>   specific numbers ready: pre-fix, signals contributed +0.0011 AUC and
>   removing them cost nothing; post-fix, removing them costs the *entire*
>   learned-policy gain. Be ready to say plainly: "we do not know if a third
>   bug fix would flip it again — that's exactly the point we're making about
>   small-sample fragility."
> - If your professor pushes on "so is RL worth it or not" — the honest,
>   defensible answer: *worth it for the ceiling and for making forgetting
>   measurable; not worth it if the only alternative under consideration is
>   'give a classical detector a cheap partial-update option too' — most of
>   the win is available without any learning at all.* That's a more
>   interesting and more defensible claim than "RL wins," and it's the actual
>   finding.
> - Close on point 5 in the summary — a project that can honestly report its
>   own result reversing is a stronger research artifact than one that
>   reports a single clean number, and that's a good note to end a
>   presentation on for an academic audience.
