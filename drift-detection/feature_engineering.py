"""
Feature Engineering — Stateful (fit / transform) Pipeline
=========================================================

Why this module is a class and not a function
---------------------------------------------
The original implementation exposed a single ``apply_feature_engineering(df)``
that was called independently on the baseline window and on every subsequent
weekly window. Several of its steps are *data-dependent encoders* — they learn
something from the batch they are handed:

    * ``pd.factorize`` on object columns   → category ⇒ integer code
    * ``value_counts`` count encoding      → category ⇒ raw occurrence count
    * ``groupby(...).agg`` aggregates      → group  ⇒ aggregate value
    * ``PCA`` / ``MinMaxScaler`` on V-cols → rotation matrix
    * the "drop columns that are ≥98% one value" redundancy filter

Refitting those per window is a silent correctness bug for drift analysis:

    1. **Factorize codes are assignment-order dependent.** ``pd.factorize``
       numbers categories by order of first appearance. The email domain that
       is code 3 in the baseline can be code 17 next week without a single
       real-world thing having changed. Every distribution test downstream
       then sees a "shifted" feature.
    2. **Raw count encodings scale with batch size.** The baseline is ~90 days
       of transactions; each monitored window is 7. A category appearing at a
       perfectly constant *rate* has a count ~13x smaller in the weekly batch.
       KS/PSI/KL cannot help but scream drift.
    3. **PCA components are sign- and rotation-arbitrary.** Refitting on each
       window can flip a component's sign, inverting the feature.
    4. **A per-batch redundancy filter changes the schema itself**, so the
       aligned matrix is silently padded with zeros for columns dropped this
       week but not last week.

All four produce drift signal that is an artifact of the *encoder*, not of the
data-generating process. That is precisely the failure mode a drift study is
supposed to measure, so it must be removed before any detector is evaluated.

``FeatureEngineer`` fixes this by learning every encoder **once** on the
reference window (``fit_transform``) and replaying it unchanged on later
windows (``transform``). Encoders are re-fit only when the pipeline
deliberately retrains a model, which is an explicit, logged event.

Causal sequence features
------------------------
``_day_lag_uid2`` / ``_amount_lag_pct_uid2`` are per-entity lags. Computed
inside a 7-day window they can only ever see that window's history, whereas the
baseline saw 90 days — another window-length artifact. ``add_causal_sequence_features``
computes them **once** over the full chronologically sorted frame. They remain
strictly causal (each row only references earlier rows), so this is not
leakage: in production the prior transaction for a card is genuinely available.
"""

import datetime
import gc
import logging

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

CCOLS = [f'C{i}' for i in range(1, 15)]
DCOLS = [f'D{i}' for i in range(1, 16)]
MCOLS = ['M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']
VCOLS = [f'V{i}' for i in range(1, 340)]

START_DATE = '2017-11-30'

# Columns removed from the model matrix once their derived features exist.
#
# ``TransactionDT`` and ``TransactionID`` both increase monotonically with time.
# Left in, they are perfect proxies for "when did this happen": the model can
# split on them, and every monitored window occupies a disjoint range from the
# reference, so every distributional test reports a KS statistic of 1.0 forever
# regardless of whether anything drifted. Their legitimate signal (hour of day,
# day of week, transaction ordering) is already carried by the cyclic time
# features and ``_uid2_seq_index``.
#
# ``_uid1``/``_uid2`` are near-unique entity identifiers. A frozen ordinal code
# maps essentially every row of a future window to the unseen sentinel, so the
# column is informative in-sample and constant at inference — it flatters the
# baseline model and then contributes nothing. Entity behaviour is carried by
# the causal sequence features instead.
DROP_COLS = ['_days', 'TransactionDT', 'TransactionID', '_uid1', '_uid2']


# Human-readable names for engineered columns. The raw names are terse by
# necessity (they encode the transformation), but a drift report that says
# "_mcols_na_bin drifted" is unreadable to anyone who did not write the feature
# pipeline — and drift reports are read by risk and ops teams, not just by the
# person who built the model.
FEATURE_LABELS = {
    '_log_dist_1_2': 'Billing-to-shipping distance (log)',
    '_hour_cos': 'Hour of day (cyclic, cosine)',
    '_hour_sin': 'Hour of day (cyclic, sine)',
    '_weekday__hour': 'Weekday x hour slot',
    '_P_emaildomain__ProductCD': 'Email domain x product code',
    '_card3__card5': 'Card issuer x card sub-type',
    '_day_lag_uid2': 'Seconds since this card last transacted',
    '_amount_lag_pct_uid2': 'Amount change vs this card previous transaction',
    '_uid2_seq_index': 'Transaction number in this card history',
    '_amount_decimal': 'Cents portion of the amount',
    '_amount_decimal_len': 'Number of decimal places in the amount',
    '_log_amount': 'Transaction amount (log)',
    '_all_na': 'Count of missing fields in the row',
    '_ccols_nonzero': 'Count of non-zero counter features',
    '_ccols_sum': 'Sum of counter features',
    '_ccols_0_bin': 'Which counter features are zero (pattern)',
    '_dcol_na': 'Count of missing timedelta features',
    '_dcols_na_bin': 'Which timedelta features are missing (pattern)',
    '_mcols_na_bin': 'Which match-flag features are missing (pattern)',
    '_vcols_dec0': 'Vesta feature block, principal component 1',
    '_vcols_dec1': 'Vesta feature block, principal component 2',
    '_vcols_na': 'Count of missing Vesta features',
    '_vcols_sum': 'Sum of Vesta features',
    '_unseen__uid2': 'Card not seen during training',
    '_unseen__P_emaildomain__ProductCD': 'Email x product pair not seen in training',
    'uid2': 'card identity',
    'P_emaildomain__ProductCD': 'email domain x product',
    'TransactionAmt': 'Transaction amount',
    'ProductCD': 'Product code',
    'card1': 'Card identifier 1',
    'card2': 'Card identifier 2',
    'card3': 'Card issuer country code',
    'card5': 'Card sub-type code',
    'card4': 'Card network (Visa/Mastercard/...)',
    'card6': 'Card funding type (debit/credit)',
    'addr1': 'Billing address region',
    'addr2': 'Billing address country',
    'dist1': 'Billing-to-shipping distance',
    'dist2': 'Secondary distance measure',
    'P_emaildomain': 'Purchaser email domain',
    'R_emaildomain': 'Recipient email domain',
}


def feature_label(name):
    """Readable name for a column, falling back to a derived description."""
    if name in FEATURE_LABELS:
        return FEATURE_LABELS[name]

    # Generated families: describe the transformation rather than give up.
    # Recurse so the inner column gets its own readable name.
    if name.startswith('_freq_ref_'):
        base = feature_label(name[len('_freq_ref_'):])
        return f'How common this value of "{base}" was in training'
    for agg in ('max', 'mean', 'var'):
        prefix = f'_{agg}_'
        if name.startswith(prefix):
            col, _, group = name[len(prefix):].partition('__')
            return (f'{agg.capitalize()} of "{feature_label(col)}" '
                    f'per "{feature_label(group.lstrip("_"))}"')
    if name.startswith('C') and name[1:].isdigit():
        return f'Address/card counter {name[1:]}'
    if name.startswith('D') and name[1:].isdigit():
        return f'Days since prior event {name[1:]}'
    if name.startswith('M') and name[1:].isdigit():
        return f'Match flag {name[1:]}'
    if name.startswith('V') and name[1:].isdigit():
        return f'Vesta engineered feature {name[1:]}'
    if name.startswith('id_'):
        return f'Identity/device attribute {name[3:]}'
    return name


def label_features(names):
    return [feature_label(n) for n in names]

# Columns whose count encoding is learned on the reference window.
#
# ``_uid2`` is deliberately absent. It is close to an entity identifier, so a
# frozen frequency map assigns ~0 to almost every row of a future window while
# the reference window is spread across many small values. Under the null
# experiment (one window split into two random halves) that alone produced a KS
# statistic of 0.74 — the encoding is not transferable across windows at all.
# Entity behaviour is instead captured by the causal sequence features
# (``_day_lag_uid2``, ``_amount_lag_pct_uid2``, ``_uid2_seq_index``), which are
# computed over the full frame and therefore remain comparable across windows.
COUNT_ENCODE_COLS = CCOLS + ['_P_emaildomain__ProductCD']

# Group keys used for aggregate features. ``_uid2`` is excluded for the same
# reason: an aggregate keyed on a near-unique id degenerates to a single fill
# value for the majority of any future window, producing a point mass whose KS
# distance from the reference equals the unseen-entity rate (~0.61 here)
# regardless of whether anything actually drifted. Such a feature also flatters
# the baseline model, which sees it computed in-sample, and then silently
# becomes a constant at inference.
AGG_SPECS = [
    (['_P_emaildomain__ProductCD'], ['TransactionAmt'], ['max', 'mean', 'var']),
]

# Unseen-category sentinel for the frozen factorize maps. Kept distinct from
# the valid code range (0..n-1) so "a category the reference window never saw"
# is itself an observable, monitorable signal rather than silently colliding
# with an existing category.
UNSEEN_CODE = -1


def encode_loop(df, col, drop=True):
    """Cyclic (sin/cos) encoding of a periodic integer column."""
    period = df[col].max()
    period = period if period and period > 0 else 1
    df[col + '_cos'] = np.cos(2 * np.pi * df[col] / period)
    df[col + '_sin'] = np.sin(2 * np.pi * df[col] / period)
    if drop:
        df.drop(col, axis=1, inplace=True)
    return df


def add_causal_sequence_features(df, time_col='TransactionDT'):
    """Add per-entity lag features over the FULL chronological frame.

    Must be called once, on the complete time-sorted dataset, before any
    windowing. Each output references only rows earlier in time than the row
    it annotates, so no future information leaks; computing it globally simply
    prevents the lag horizon from being truncated to the window length.
    """
    df = df.sort_values(time_col)

    days = df[time_col] // (24 * 60 * 60)
    uid1 = (days - df['D1']).astype(str) + '__' + df['P_emaildomain'].astype(str)
    uid2 = df['card1'].astype(str) + '__' + df['addr1'].astype(str) + '__' + uid1

    df['_uid1'] = uid1
    df['_uid2'] = uid2

    grouped = df.groupby('_uid2', sort=False)
    df['_day_lag_uid2'] = df[time_col] - grouped[time_col].shift(1)
    df['_amount_lag_pct_uid2'] = np.abs(
        grouped['TransactionAmt'].pct_change(fill_method=None)
    )
    # Transaction index within the entity's own history — a cheap, scale-free
    # tenure proxy that (unlike a raw count encoding) does not depend on how
    # much data the current window happens to contain.
    df['_uid2_seq_index'] = grouped.cumcount()

    logger.info("Causal sequence features computed over the full frame.")
    return df


class FeatureEngineer:
    """Fit encoders once on a reference window; replay them on later windows.

    Usage:
        fe = FeatureEngineer()
        ref = fe.fit_transform(baseline_df)     # learns + applies
        wk  = fe.transform(week_df)             # applies only

    ``fe.feature_schema`` is the authoritative column order after fitting;
    ``transform`` always returns exactly those columns.
    """

    def __init__(self, redundancy_threshold=0.98, n_v_components=2, random_state=42):
        self.redundancy_threshold = redundancy_threshold
        self.n_v_components = n_v_components
        self.random_state = random_state
        self.is_fitted = False

        # Learned state
        self.category_maps = {}        # col -> {category value: code}
        self.freq_maps = {}            # col -> {value: relative frequency}
        self.agg_maps = {}             # (period, col, agg) -> {group: value}
        self.agg_globals = {}          # (period, col, agg) -> global fallback value
        self.known_groups = {}         # period col -> set of groups seen at fit time
        self.v_scaler = None
        self.v_pca = None
        self.v_pca_cols = []
        self.v_sum_cols = []
        self.redundant_cols = []
        self.feature_schema = None
        self.hour_period = 23.0

    # ── public API ───────────────────────────────────────────────
    def fit_transform(self, df):
        logger.info("Fitting feature engineering state on reference window...")
        out = self._build(df, fitting=True)
        self.is_fitted = True
        self.feature_schema = [c for c in out.columns if c != 'isFraud']
        logger.info(
            f"Feature engineering fitted — {len(self.feature_schema)} features, "
            f"{len(self.redundant_cols)} redundant columns dropped."
        )
        return out

    def transform(self, df):
        if not self.is_fitted:
            raise RuntimeError("FeatureEngineer.transform called before fit_transform.")
        out = self._build(df, fitting=False)
        target = out['isFraud'] if 'isFraud' in out.columns else None
        out = out.reindex(columns=self.feature_schema)
        if target is not None:
            out['isFraud'] = target.values
        return out

    # ── internals ────────────────────────────────────────────────
    def _build(self, all_data, fitting):
        all_data = all_data.copy()

        self._distance_features(all_data)
        all_data = self._date_features(all_data, fitting)
        self._combination_features(all_data)
        self._identity_features(all_data)
        self._amount_features(all_data)
        all_data = self._aggregate_features(all_data, fitting)
        self._count_encoding(all_data, fitting)
        all_data = self._c_features(all_data)
        self._d_features(all_data)
        self._m_features(all_data)
        all_data = self._v_features(all_data, fitting)
        all_data = self._redundancy_filter(all_data, fitting)
        all_data.drop(DROP_COLS, axis=1, errors='ignore', inplace=True)
        all_data = self._factorize(all_data, fitting)

        gc.collect()
        return all_data

    def _distance_features(self, df):
        df['_log_dist_1_2'] = np.log1p(
            np.where(df['dist1'].isna(), df['dist2'], df['dist1'])
        )

    def _date_features(self, df, fitting):
        startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
        trandate = df['TransactionDT'].apply(lambda x: startdate + datetime.timedelta(seconds=x))

        df['_days'] = df['TransactionDT'] // (24 * 60 * 60)
        df['_hour'] = trandate.dt.hour
        # Freeze the cyclic period at fit time. Deriving it from ``.max()`` per
        # window means a quiet week with no 23:00 transactions silently rescales
        # every hour encoding.
        if fitting:
            self.hour_period = 23.0
        period = self.hour_period
        df['_hour_cos'] = np.cos(2 * np.pi * df['_hour'] / period)
        df['_hour_sin'] = np.sin(2 * np.pi * df['_hour'] / period)
        df.drop('_hour', axis=1, inplace=True)

        df['_weekday__hour'] = (
            trandate.dt.dayofweek.astype(str) + '_' + trandate.dt.hour.astype(str)
        )
        return df

    def _combination_features(self, df):
        df['_P_emaildomain__ProductCD'] = (
            df['P_emaildomain'].astype(str) + '__' + df['ProductCD'].astype(str)
        )
        df['_card3__card5'] = df['card3'].astype(str) + '__' + df['card5'].astype(str)

    def _identity_features(self, df):
        # add_causal_sequence_features normally supplies these over the full
        # frame; recompute locally only if the caller skipped that step.
        if '_uid2' in df.columns:
            return
        df['_uid1'] = (df['_days'] - df['D1']).astype(str) + '__' + df['P_emaildomain'].astype(str)
        df['_uid2'] = (
            df['card1'].astype(str) + '__' + df['addr1'].astype(str) + '__' + df['_uid1']
        )
        grouped = df.groupby('_uid2', sort=False)
        df['_day_lag_uid2'] = df['TransactionDT'] - grouped['TransactionDT'].shift(1)
        df['_amount_lag_pct_uid2'] = np.abs(
            grouped['TransactionAmt'].pct_change(fill_method=None)
        )
        df['_uid2_seq_index'] = grouped.cumcount()

    def _amount_features(self, df):
        amt = df['TransactionAmt']
        df['_amount_decimal'] = ((amt - amt.astype(int)) * 1000).astype(int)
        df['_amount_decimal_len'] = amt.astype(str).str.split('.').str[-1].str.len().fillna(0)
        df['_log_amount'] = np.log1p(amt.clip(lower=0))

    def _aggregate_features(self, df, fitting):
        """Group aggregates, learned on the reference window and replayed after.

        Recomputing ``groupby.agg`` per window makes the aggregate a function of
        how many rows of that group happen to fall inside the window, which is a
        window-length artifact rather than a behavioural change.

        Groups absent from the reference map must not fall through to 0.
        ``_uid2`` is close to an entity identifier, so most groups in a future
        window were never seen during fitting. Filling those with 0 puts them on
        a different scale from every mapped row, and a null experiment (one
        window split into two random halves, where no drift is possible) showed
        these aggregates reaching KS D = 0.61 purely from that fill. Unseen
        groups are instead filled with the reference *global* value of the same
        aggregate, which keeps the feature on one scale, and the "this entity is
        new" signal is moved into its own explicit indicator column where it can
        be monitored on purpose rather than by accident.
        """
        for periods, columns, aggs in AGG_SPECS:
            for period in periods:
                for col in columns:
                    if col not in df.columns or period not in df.columns:
                        continue
                    for a in aggs:
                        key = (period, col, a)
                        out_col = f'_{a}_{col}_{period}'
                        if fitting:
                            mapping = df.groupby(period)[col].agg(a)
                            self.agg_maps[key] = mapping
                            global_val = pd.to_numeric(df[col], errors='coerce').agg(a)
                            self.agg_globals[key] = float(global_val) if pd.notna(global_val) else 0.0
                            df[out_col] = df[period].map(mapping)
                        else:
                            mapping = self.agg_maps.get(key)
                            df[out_col] = (
                                df[period].map(mapping) if mapping is not None else np.nan
                            )
                        df[out_col] = df[out_col].fillna(self.agg_globals.get(key, 0.0))

        # Explicit, monitorable "not seen during fitting" indicators. The
        # new-entity signal is genuinely interesting, so it gets its own column
        # rather than leaking into every aggregate as a fill artifact.
        for period in ('_uid2', '_P_emaildomain__ProductCD'):
            if period not in df.columns:
                continue
            if fitting:
                self.known_groups[period] = set(df[period].unique())
            known = self.known_groups.get(period, set())
            df[f'_unseen_{period}'] = (~df[period].isin(known)).astype(np.int8)

        return df

    def _count_encoding(self, df, fitting):
        """Frequency encoding — relative and FROZEN.

        The original stored raw ``value_counts``. Those are proportional to the
        window's row count, so a category with a perfectly stable rate looks
        ~13x rarer in a 7-day window than in the 90-day baseline. Storing the
        *relative* frequency learned on the reference window removes the
        window-length dependence.

        Storing the relative frequency is necessary but not sufficient: it must
        also be frozen. A per-window relative frequency is still window-size
        dependent, because its resolution floor is 1/n — a category seen once
        in a 2k-row window encodes as 5e-4, while the same category seen once
        in a 24k-row window encodes as 4e-5, a 10x difference with no change in
        the world. An interim version of this module emitted such a
        ``_freq_now_`` feature alongside the frozen one; a null experiment
        (splitting a single window into two random halves, where no real drift
        can exist) showed it reaching KS D = 0.88 — by far the largest false
        signal in the whole feature set. It was removed.
        """
        for f in COUNT_ENCODE_COLS:
            if f not in df.columns:
                continue
            if fitting:
                self.freq_maps[f] = (df[f].value_counts(dropna=False) / len(df)).to_dict()
            ref_map = self.freq_maps.get(f, {})
            df[f'_freq_ref_{f}'] = df[f].map(ref_map).fillna(0.0).astype(np.float32)

        df['_all_na'] = df.isna().sum(axis=1).astype(np.int16)

    def _c_features(self, df):
        present = [c for c in CCOLS if c in df.columns]
        if not present:
            return df
        arr = df[present].to_numpy()
        df['_ccols_nonzero'] = (np.nan_to_num(arr) != 0).sum(axis=1).astype(np.int8)
        df['_ccols_sum'] = np.nansum(arr, axis=1).astype(np.float32)
        df['_ccols_0_bin'] = ''
        for c in present:
            df['_ccols_0_bin'] += (df[c] == 0).astype(int).astype(str)
        df.drop(present, axis=1, inplace=True)
        return df

    def _d_features(self, df):
        """D1-D15 are already relative ("days since some prior event") — they
        do not need, and must not get, a further absolute-time anchor.

        An earlier version of this method computed ``D_i - _days``, apparently
        intending to normalise the timedelta. Empirically, raw D-columns are
        roughly stationary across the replay (D2's weekly mean stays in
        160-190 throughout), while ``_days`` — the row's absolute day index —
        grows linearly from ~40 to ~180. Subtracting a monotonically growing
        quantity from a stationary one manufactures a monotonic trend: D2's
        transformed mean fell from +40 at the reference window to -74 by week
        14, and every distributional detector correctly flagged it as
        drifting every single week for the entire replay. That is not real
        drift, it's the same "monotone-in-time proxy" failure mode that
        ``TransactionDT``/``TransactionID`` are excluded from the matrix for
        (see ``DROP_COLS`` above) — just introduced here by a transformation
        rather than by leaving a raw column in. Only the missingness fill
        remains; the columns are otherwise passed through unchanged.
        """
        present = [c for c in DCOLS if c in df.columns]
        if not present:
            return
        df['_dcol_na'] = df[present].isna().sum(axis=1).astype(np.int8)
        df['_dcols_na_bin'] = ''
        for c in present:
            df['_dcols_na_bin'] += df[c].isna().astype(int).astype(str)
        for f in present:
            df[f] = df[f].fillna(0)

    def _m_features(self, df):
        df['_mcols_na_bin'] = ''
        for c in MCOLS:
            if c in df.columns:
                df['_mcols_na_bin'] += df[c].isna().astype(int).astype(str)

    def _v_features(self, df, fitting):
        """PCA compression of the V block, with a FROZEN rotation.

        A PCA refit per window yields components whose sign and orientation are
        arbitrary, so the compressed feature can invert between two windows in
        which nothing actually changed.
        """
        pca_cols = [f for f in VCOLS if f in df.columns]
        if not pca_cols:
            return df

        if fitting:
            self.v_pca_cols = pca_cols
            self.v_scaler = preprocessing.MinMaxScaler()
            self.v_pca = PCA(n_components=self.n_v_components, random_state=self.random_state)
            scaled = self.v_scaler.fit_transform(df[pca_cols].fillna(-1))
            comps = self.v_pca.fit_transform(scaled)
            self.v_sum_cols = [
                c for c in pca_cols
                if c not in ('V144', 'V145', 'V150', 'V151', 'V159', 'V160', 'V307')
            ]
        else:
            block = df.reindex(columns=self.v_pca_cols).fillna(-1)
            scaled = self.v_scaler.transform(block)
            comps = self.v_pca.transform(scaled)
            pca_cols = [c for c in self.v_pca_cols if c in df.columns]

        for i in range(comps.shape[1]):
            df[f'_vcols_dec{i}'] = comps[:, i]
        df['_vcols_na'] = df[pca_cols].isna().sum(axis=1).astype(np.int16)
        sum_cols = [c for c in self.v_sum_cols if c in df.columns]
        df['_vcols_sum'] = df[sum_cols].sum(axis=1).astype(np.float32)
        df.drop(VCOLS, axis=1, errors='ignore', inplace=True)
        return df

    def _redundancy_filter(self, df, fitting):
        """Drop near-constant columns — decided ONCE, on the reference window.

        Deciding per window makes the output schema itself drift, so the aligned
        matrix gets zero-padded differently from week to week.
        """
        if fitting:
            drop = []
            for c in df.columns:
                if c == 'isFraud':
                    continue
                vc = df[c].value_counts(normalize=True).values
                if len(vc) > 0 and vc[0] >= self.redundancy_threshold:
                    drop.append(c)
            self.redundant_cols = drop
        return df.drop(columns=[c for c in self.redundant_cols if c in df.columns], errors='ignore')

    def _factorize(self, df, fitting):
        """Frozen ordinal encoding of object columns.

        ``pd.factorize`` assigns codes by order of first appearance, so the same
        category receives a different integer in every window. Freezing the
        map at fit time makes the encoded value comparable across time, which is
        a precondition for any distributional drift test to mean anything.
        """
        # NOTE: test for "not numeric", not for ``dtype == 'object'``. Under
        # pandas 3.x string columns carry a dedicated ``str`` dtype rather than
        # ``object``, so an object-only check silently skips every categorical —
        # leaving them to be factorised per window further downstream, which is
        # exactly the artifact this method exists to prevent.
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
                continue
            if fitting:
                codes, uniques = pd.factorize(df[col])
                self.category_maps[col] = {v: i for i, v in enumerate(uniques)}
                df[col] = codes
            else:
                mapping = self.category_maps.get(col, {})
                df[col] = df[col].map(mapping).fillna(UNSEEN_CODE).astype(np.int32)
        return df


# ──────────────────────────────────────────────────────────────────
# Backwards-compatible shim
# ──────────────────────────────────────────────────────────────────
def apply_feature_engineering(all_data):
    """Legacy stateless entry point — fits and transforms in one shot.

    Retained so older scripts keep running, but it reintroduces exactly the
    per-window refit problem documented at the top of this module. New code
    should use ``FeatureEngineer`` and call ``transform`` on monitored windows.
    """
    return FeatureEngineer().fit_transform(all_data)
