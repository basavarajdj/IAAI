# Selecting What to Monitor: From 114 Trained Features to 20 Watched Ones

This document answers three questions that came up while reviewing the drift
pipeline: **how** the monitored feature set is chosen, **whether 20 is the
right size**, and **whether the 60% consensus threshold is well-calibrated**.
It's a companion to [FEATURE_ENGINEERING_AND_MODELING.md](FEATURE_ENGINEERING_AND_MODELING.md)
(which covers how the 114 *trained* features are built) and to PAPER.md
§2.1/§6.1 (which report the results this process produces).

Everything here is the *monitoring* problem, not the *modelling* problem. The
model trains on all 114 features. Monitoring all 114 for drift would (a)
multiply the multiple-testing burden the FDR correction has to absorb, (b)
mostly duplicate signal, since many of the 114 are near-duplicates of each
other, and (c) be expensive — SHAP/gradient attribution for 114 features,
every week, against every reference. A smaller, deliberately-curated set is
selected instead.

---

## 1. Why not just take the top-K by importance?

The original design did exactly that: one LightGBM fit, ranked by gain, top-K
kept. Three things go wrong with it, all in `feature_selection.py`'s opening
docstring and all real:

1. **It's unstable.** Gain comes from one stochastic fit (row/column
   subsampling, a seed). A different seed can substantially change the
   top-10.
2. **It's redundant.** Gain splits across correlated features roughly at
   random, and correlated features drift *together* — so a "6 of 10 features
   drifted" consensus vote can be six readings of the same underlying signal,
   not six independent pieces of evidence.
3. **It ignores monitorability.** A near-constant or very low-cardinality
   feature can carry real gain and still be untestable — a KS test on a
   column with three distinct values is dominated by ties.

## 2. The actual selection process

`select_monitoring_features()` ([feature_selection.py](../feature_selection.py))
runs four stages:

**Stage 1 — Bagged importance.** Fit 5 LightGBM models on bootstrap
resamples of the reference window, each with a different seed. Rank features
within each fit; aggregate by mean reciprocal rank, and separately record
each feature's *selection frequency* — how often it lands in the top-25
across the 5 bags. A feature every bag independently ranks highly is a
property of the data; a feature only one bag likes is that bag's noise.

**Stage 2 — SHAP corroboration.** Global mean |SHAP| on a sample from the
reference model, as a second, split-count-independent view. Gain is biased
toward high-cardinality features; SHAP isn't, so disagreement between the two
is a useful red flag (not currently acted on automatically, but reported in
the diagnostics for inspection).

**Stage 3 — Monitorability filter.** Drop anything a distributional test
can't meaningfully evaluate: fewer than 10 distinct values, or effectively
zero variance in the reference window.

**Stage 4 — Redundancy pruning, two mechanisms.** Walk the ranked candidate
list top-down; skip a candidate if:
- its Spearman |ρ| against an already-selected feature is ≥ 0.90, **or**
- it would be the 3rd feature from the same *numbered family* (added this
  session — see §4).

A composite score (`0.7 × selection_frequency + 0.3 × normalised MRR`) sets
the walk order, so agreement-across-bags dominates and reciprocal rank only
breaks ties.

Everything is reported: `reports/feature_selection_diagnostics.csv` has one
row per candidate feature with its score, selection frequency, SHAP
importance, cardinality, and whether it was selected; the JSON report's
`feature_selection` block records the stability index, every redundancy/
family-cap rejection with what it was rejected against, and the max pairwise
correlation that survived into the final set.

## 3. The redundancy problem that motivated the family cap

Before this session, the 10-feature monitoring set included **four** members
of the same family: `_freq_ref_C1`, `_freq_ref_C2`, `_freq_ref_C5`,
`_freq_ref_C14` — four count-encodings of different raw `C` columns, all from
the same underlying "card/address counter" block. None of the six pairwise
correlations among them individually cleared the 0.90 pruning threshold, but
the set as a whole was not four independent votes — it was one family voting
four times. `max_pairwise_rho_among_selected` was **0.817**, uncomfortably
close to the 0.90 cutoff, for a set that pairwise pruning alone had already
"cleared."

**The fix:** `_feature_family()` strips trailing digits from a feature name
(`_freq_ref_C1` and `_freq_ref_C14` both map to `_freq_ref_C`; `D2` and `D15`
both map to `D`), and the greedy selection loop caps how many features from
one family can be selected (`max_per_family=2`, default). Pairwise
correlation catches *strong* duplicates; the family cap catches *chains* of
moderate correlation that individually clear a threshold but collectively
don't.

**Measured effect**, same dataset, before → after:

| | Before (10 features, no family cap) | After (20 features, family cap = 2) |
|---|---|---|
| `_freq_ref_C*` members selected | 4 (C1, C2, C5, C14) | 2 (C1, C14) |
| `max_pairwise_rho_among_selected` | 0.817 | 0.817 → see note |
| Family-cap rejections | n/a (mechanism didn't exist) | 14 (of 20 candidates considered) |

*(Max pairwise ρ is numerically the same value in both runs because it comes
from the same C1×C5-adjacent pair pattern recurring at the larger K; the
family cap controls how many *members* of a family can vote, not the
strength of any single pair — a pairwise-only fix would need `redundancy_rho`
tightened toward ~0.75 instead, which risks discarding real signal from
weakly-correlated-but-genuinely-different features. The family cap is the
more targeted fix for chains within a known numbered block.)*

## 4. Should the monitoring set be bigger than 10? — Yes, and here's the evidence

The concern with enlarging K is usually: *more features monitored, more
chances to alarm on noise.* That concern is directly checkable, and the
check says no:

**Null-experiment result** (`python validate_monitor.py --calibrate_consensus`,
30 independent random splits of the 90-day baseline window — no drift can
exist by construction): **0 of the 20 monitored features crossed their own
threshold in any of the 30 trials**, for both KS and PSI. The per-feature
false-positive rate on this exact monitoring set is empirically 0%. There is
no calibration cost to watching more features, only a coverage benefit (the
model trains on 114; a 20-feature monitor still only watches 17.5% of it) and
a compute cost (proportional to K, dominated by the SHAP/gradient-attribution
detector).

10 → 20 was chosen as a round increase that stays well inside "no calibration
cost, meaningfully broader coverage," not derived from an optimum.

### A caveat the diagnostics surface: the last few slots are weak

`reports/feature_selection_diagnostics.csv`, sorted by score, for the current
20-feature set:

| Feature | Selection frequency (of 5 bags) | Score |
|---|---:|---:|
| `_mcols_na_bin` … `_card3__card5` (12 features) | **1.0** | 1.00 → 0.71 |
| `_vcols_dec0` | 0.8 | 0.57 |
| `_var_TransactionAmt__P_emaildomain__ProductCD`, `addr1`, `_ccols_sum` | 0.6 | 0.43 |
| `_max_TransactionAmt__P_emaildomain__ProductCD` | 0.4 | 0.29 |
| `_vcols_dec1` | 0.2 | 0.15 |
| `P_emaildomain`, `DeviceInfo` | **0.0** | 0.01, 0.005 |

The last two features, `P_emaildomain` and `DeviceInfo`, were selected by
**zero** of the five bootstrap fits — they entered the monitoring set purely
through the "if pruning starved the set, top up by score" fallback
([feature_selection.py:273](../feature_selection.py#L273)), after the family cap
and redundancy pruning had exhausted the pool of well-supported candidates.
They are not stable, importance-backed picks; they are filler.

This is a genuine limitation, not a reason to revert to 10 — the first 14–16
features (frequency ≥ 0.4) are well-supported, and DeviceInfo's inclusion
surfaced a real, separate problem covered in §6. **Recommended follow-up**
(not implemented this session, to keep this change bounded and verifiable):
widen the Stage-1 candidate pool (`top_m`) so the starvation fallback has more
real candidates to draw from before falling back to near-zero-frequency
features, or cap `top_k` at the point where `selection_frequency` drops below
some floor (e.g. 0.4) rather than a fixed round number.

## 5. Is the 60% consensus threshold right? — It can't be null-calibrated, and here's why that's informative

`MIN_FEATURE_DRIFT_FRACTION = 0.6` means a feature-vote detector (KS, PSI,
JS, SHAP) only raises a raw flag if **12 of the 20** monitored features
individually cross their own threshold. Where does 0.6 come from, and is it
too conservative?

**The null-calibration attempt returns a degenerate answer, which is itself
the finding.** `validate_monitor.py --calibrate_consensus` repeats the null
split (§4) and measures what fraction of the 20 monitored features cross
*by chance* under provable non-drift. That fraction was 0 in all 30 trials —
not "usually low," literally zero every time. There is no non-zero null floor
to set a percentile against; the naive "p99 of the null distribution" logic
degenerates to a meaningless near-0 suggestion (this case is now detected and
called out explicitly rather than silently returned, see
`calibrate_consensus_threshold()`).

**What that means for the threshold:** the individual per-feature tests
already have ~0% inherent false-positive rate, so *any* fixed
`MIN_FEATURE_DRIFT_FRACTION > 0` is safe against pure measurement noise. The
question "should it be 0.6 or 0.3?" is therefore not a calibration question —
it's a policy question about how much *breadth* of real, individually-true
(non-null) movement should count as "the population changed," as opposed to
"a handful of features moved for unrelated reasons." That breadth question is
answered by the *actual replay*, not by a null split — see §6.

**Why lowering it to "3–4 of 20" (15–20%) would be a mistake:** in the real
14-week replay, `_mcols_na_bin`, `_vcols_dec0`, and `_vcols_dec1` cross their
individual thresholds in **14 of 14 weeks each** (§6) — a real, persistent,
non-null signal in 3 of the 20 monitored features essentially all the time.
A consensus bar at or below that level would make KS/PSI/JS fire on almost
every week for reasons unrelated to any given week's actual content,
reproducing the same "constantly-alarming, uninformative detector" failure
mode the null-experiment correction (PAPER.md §3.1) was built to eliminate —
just triggered by real per-feature background movement instead of a
measurement artifact. **0.6 was kept unchanged** after this analysis: it
already sits well above the observed non-null floor (3/20 = 15%), and the
null check confirms there is no false-positive argument for moving it either
direction.

## 6. What's actually driving the persistent per-feature crossings — real, seasonal, or artifact?

Three features cross in all or nearly all of the 14 weeks, across multiple
independent detectors (KS, PSI, and/or JS, and/or SHAP). Each was
investigated individually rather than assumed; the mechanisms differ:

### 6a. `D2`, `D15` — was an outright bug, now fixed

Before this session, `D2` and `D15` (both selected once K rose to 20) crossed
their KS/PSI/JS threshold in **14 of 14 weeks**, unconditionally. Investigation
(comparing each D-column's *raw* weekly mean against its *transformed* value)
found the cause: `feature_engineering.py`'s `_d_features` computed
`D_i ← D_i.fillna(0) - _days`. Raw `D2`'s weekly mean is roughly stationary
across the whole replay (160–190 throughout); `_days` — the row's absolute
day offset from the dataset start — grows linearly, from ~40 in the reference
window to ~180 by week 14. Subtracting a linearly-growing quantity from a
stationary one manufactures a linear trend: the transformed `D2`'s mean fell
in a straight line from +40 to −74 over the replay. **This was not drift. It
was arithmetic.** The same failure mode `TransactionDT`/`TransactionID` are
excluded from the feature matrix for — a column that is monotone in elapsed
time — reintroduced by a transformation on an otherwise legitimate column.
Full detail and the fix: FEATURE_ENGINEERING_AND_MODELING.md §2.9.

**After the fix** (subtraction removed, `D_i` passed through with only NaN
filled), `D2`'s KS crossing rate fell from 14/14 to 2/14. `D15` still crosses
more often than that under KS (7/14) and consistently under JS (14/14, see
§6c) — its *raw*, unmodified weekly mean does drift mildly upward over the
replay (151 → 196, a real if slow trend, not the clean artifact `D2` had),
which is a candidate for genuine mild covariate movement rather than a
pipeline defect. This is flagged as real signal, not re-investigated further
here.

### 6b. `_mcols_na_bin` — real, persistent, plausibly the reference window itself

`_mcols_na_bin` (which of the 8 match-flag columns are missing, encoded as a
15-distinct-value pattern string, frozen-factorized) crosses in **14 of 14
weeks** under both KS and PSI. Two candidate mechanisms were checked:

- **Unseen-category accumulation** (the mechanism that explains `_uid2`'s
  known issue, and DeviceInfo's below): ruled out. The fraction of rows
  mapping to the "unseen" sentinel code is ~0.00–0.02% every week — hardly
  ever a new pattern appears that the 90-day reference didn't already see.
- **A genuine shift in the *frequency* of already-known patterns**: this is
  what's left, and it's corroborated by two independent statistical tests
  (KS and PSI) agreeing every week, which argues against a single-test
  artifact.

Low cardinality (15 values) means this isn't a monitorability problem either.
The most defensible reading: transactions in the 90-day reference window
(which, given the dataset's `2017-11-30` start date, spans the November–
December holiday shopping season) have a genuinely different missingness
pattern across M1–M9 than transactions in any subsequent week — i.e. a
**level shift tied to the reference window's own seasonal composition**,
not a continuously evolving drift. Consistent with this: the crossing rate
doesn't visibly intensify or fade across weeks 1→14 (it's persistent, not
progressive) — a genuinely continuous drift would be expected to show *some*
week-to-week gradient; a seasonal reference-vs-rest split would not.

### 6c. `_vcols_dec0`, `_vcols_dec1` (Vesta PCA components) — level shift, likely the same seasonal effect, plus one real anomaly

Both components cross in 14 of 14 weeks under KS and PSI. Their weekly means
jump from ~0 in the reference window (expected — PCA is centred on the
reference by construction) to roughly 0.3–0.5 in week 1, and then stay in
that band with no further trend — again a **step, not a ramp**, consistent
with the same reference-window-composition explanation as §6b rather than
continuous drift.

One genuine, separate finding surfaced while checking this: `_vcols_sum`
(sum over a fixed Vesta subset) spikes from a baseline of ~1,300–2,000 in
most weeks to **12,300 in week 11** and **73,100 in week 12** — a 6–37×
outlier, clearly not part of the steady-state pattern. This lines up exactly
with independent evidence already in PAPER.md §6.1's per-method results:
`clustering`'s one raw alarm and the autoencoder's peak reconstruction error
both land at week 12. Three independent signals (a raw feature-sum outlier,
representation-level clustering distance, and autoencoder reconstruction
error) agreeing on the same week is good evidence of a real, if narrow,
anomalous event around weeks 11–12 — not an artifact and not routine
seasonality, but a one-off regime disturbance.

### 6d. `DeviceInfo` — a second real limitation, structurally like the known `_uid2` issue

`DeviceInfo` crosses in 10/14 weeks under KS and, notably, is the **most
frequent SHAP-attribution driver** (12/14 weeks) — despite entering the
monitoring set with **zero bagged-importance support** (§4). Investigation:

- Reference-window cardinality is enormous relative to the reference sample:
  **1,401 distinct non-null values** in a ~316,000-row window, with 75.5% of
  rows null to begin with.
- The frozen-factorize "unseen" sentinel fraction in the replay weeks is
  **79–89%** — i.e., 4 times out of 5, a `DeviceInfo` string encountered in a
  monitored week was never seen in the reference window at all.

This is not new devices "appearing over time" in a meaningful sense (the
unseen rate doesn't ramp up progressively across weeks 1→14; it's already
~89% in week 1) — it's that `DeviceInfo` is a **near-unique free-text
identifier** (device/browser fingerprint strings), structurally the same
problem already diagnosed and fixed for `_uid2`
([feature_engineering.py:184](../feature_engineering.py#L184)): a frozen
ordinal code on a column with this little per-value support assigns most of
any future window to one arbitrary sentinel value, and the resulting
"distribution" is mostly a comparison of two different unseen-rates, not of
two different real behaviours.

**This was not fixed this session** (unlike §6a), to keep the change scope
verified and bounded — but it should be. The monitorability filter currently
only checks a cardinality/variance *floor* (`n_distinct ≥ 10`); it has no
*ceiling*, so a near-unique identifier like `DeviceInfo` passes it as easily
as a well-behaved moderate-cardinality categorical. A blanket cardinality cap
is the wrong fix, though — `card1` (11,315 distinct values, numeric/ordinal
in nature) is a legitimately useful, currently-selected, well-supported
monitored feature, and a raw cardinality ceiling would exclude it too. The
distinguishing factor is that `card1` is dense (every value recurs across
many rows and windows) while `DeviceInfo` is sparse (most values are
singletons). **Recommended fix:** add a sparsity check to the monitorability
filter — something like "average rows-per-distinct-value in the reference
window" — rather than a raw cardinality ceiling, or frequency-encode
`DeviceInfo` (as is already done for `_P_emaildomain__ProductCD`) instead of
factorizing it, before allowing it into the monitored candidate pool.

### 6e. The `TransactionAmt`-aggregate features — likely the same reference-composition effect

`_var_TransactionAmt__P_emaildomain__ProductCD`,
`_max_TransactionAmt__P_emaildomain__ProductCD`, and
`_mean_TransactionAmt__P_emaildomain__ProductCD` each cross in 11–12 of 14
weeks under KS and JS. These are aggregates frozen on the reference window,
keyed by an email-domain × product-code cross (§2.6 of the engineering doc).
The same seasonal-composition hypothesis as §6b applies directly: if the
mix of products and email providers used during the November–December
reference window differs from the mix in later, non-holiday weeks, every row
in a later week is being compared against a group-average computed under a
different product/email mix — a real compositional difference, most
plausibly seasonal, not a measurement defect.

---

## 7. Summary table

| Feature(s) | Weeks crossed | Mechanism | Verdict | Action taken |
|---|---|---|---|---|
| `D2`, `D15` (partially) | was 14/14 | `D_i - _days` manufactured a linear trend | **Bug** | Fixed — subtraction removed |
| `_mcols_na_bin` | 14/14 | Reference window's own missingness-pattern mix differs from the replay's | Likely seasonal (reference = holiday season) | Documented; no fix needed (real signal) |
| `_vcols_dec0`, `_vcols_dec1` | 14/14 | Same reference-composition effect (level shift, not a ramp) | Likely seasonal | Documented |
| `_vcols_sum` (weeks 11–12 only) | 2/14, but a 6–37× outlier | Corroborated by clustering + autoencoder at the same weeks | Real, narrow anomaly | Documented |
| `DeviceInfo` | 10/14 (KS), 12/14 (SHAP driver) | Near-unique identifier; frozen factorize assigns 79–89% of rows to "unseen" | **Artifact** (same class as the known `_uid2` issue) | Documented, not yet fixed — see §6d for the recommended fix |
| `_*_TransactionAmt__P_emaildomain__ProductCD` aggregates | 11–12/14 | Reference-window product/email mix differs from later weeks | Likely seasonal | Documented |

Two of six recurring "always drifts" patterns turned out to be measurement
problems (one fixed, one identified and deferred); the rest are corroborated,
plausible, real signal — most of it consistent with the reference window's
own seasonal composition rather than a continuously evolving population.
