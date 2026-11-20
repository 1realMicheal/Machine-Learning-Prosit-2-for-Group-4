"""V2 F1 feature adapter. Original dataset fields -> exact engineered feature contract.

Point-in-time: only previously disbursed loans on strictly earlier calendar dates.
No target, due-date outcome, current Total_Amount, country_id, lender_id or repayment
metadata is included as a predictive feature. Compatible with F1 from V2 notebook.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

BORROW = [
    'previous_loan_count','previous_valid_amount_count','previous_total_amount_borrowed',
    'previous_mean_loan_amount','previous_min_loan_amount','previous_max_loan_amount',
    'previous_mean_duration','days_since_previous_loan','previous_loans_last_90d',
    'previous_loans_last_180d','previous_mean_days_between_loans',
    'previous_mean_amount_last_3','previous_most_recent_loan_amount',
    'previous_amount_trend_last_3',
]
FLAGS = ['has_previous_loan','has_previous_valid_amount','has_previous_valid_duration',
         'has_multiple_previous_loan_dates']
CALENDAR = ['disbursement_year','disbursement_month','disbursement_quarter',
            'disbursement_dayofweek','disbursement_day','disbursement_is_weekend']
EXPECTED_F1 = ['loan_type','duration'] + CALENDAR + BORROW + FLAGS
RAW_HISTORY_FIELDS = ['customer_id','disbursement_date','Total_Amount','duration']
RAW_APPLICATION_FIELDS = ['customer_id','disbursement_date','loan_type','duration']


def _as_day(v, *, field):
    d = pd.to_datetime(v, errors='coerce')
    if pd.isna(d):
        raise ValueError(f"{field} must be a valid date.")
    return pd.Timestamp(d).normalize()


def _mean(values):
    v = np.asarray(values,dtype=float)
    v = v[np.isfinite(v)]
    return float(v.mean()) if len(v) else np.nan


def _history_preparation(history, *, assume_only_customer=False):
    """Normalize raw original columns and retain source order for tie handling.

    Missing/invalid nonpositive prior amounts & durations become NaN exactly like V2.
    No current-target or future-record values are fed to any estimator.
    """
    if history is None:
        history = pd.DataFrame(columns=RAW_HISTORY_FIELDS)
    if not isinstance(history, pd.DataFrame):
        history = pd.DataFrame(history)
    history = history.copy()
    required = RAW_HISTORY_FIELDS[1:] if assume_only_customer else RAW_HISTORY_FIELDS
    missing = [c for c in required if c not in history.columns]
    if missing:
        raise ValueError(f"History is missing original dataset fields: {missing}")
    if 'customer_id' not in history:
        history['customer_id'] = ''
    history['customer_id'] = history['customer_id'].astype('string').str.strip()
    history['_day'] = pd.to_datetime(history['disbursement_date'], errors='coerce').dt.normalize()
    history['_amount'] = pd.to_numeric(history['Total_Amount'], errors='coerce')
    history['_duration'] = pd.to_numeric(history['duration'], errors='coerce')
    history.loc[~np.isfinite(history['_amount']) | (history['_amount']<=0),'_amount'] = np.nan
    history.loc[~np.isfinite(history['_duration']) | (history['_duration']<=0),'_duration'] = np.nan
    history['_order'] = np.arange(len(history))
    return history


def make_f1_features(*, customer_id, disbursement_date, loan_type, duration,
                     history=None, required_features=None, history_is_customer_only=False):
    """One row of model inputs from a proposed loan and earlier raw loan events.

    For manual past loans, pass a history dataframe containing the original
    `disbursement_date`, `Total_Amount`, `duration`, optionally `customer_id`.
    Unknown history is not interpreted as 'no previous loans': use no-history
    only when borrower is confirmed new or history is known complete.
    """
    day = _as_day(disbursement_date,field='disbursement_date')
    loan_type = str(loan_type).strip()
    if not loan_type or loan_type.lower() in ('nan','none','<na>'):
        raise ValueError('loan_type is required.')
    dur = pd.to_numeric(duration, errors='coerce')
    if pd.isna(dur) or not np.isfinite(dur) or dur <= 0:
        raise ValueError('Current duration must be a positive number of days.')
    sid = str(customer_id).strip()
    if not sid or sid.lower() in ('nan','none','<na>'):
        raise ValueError('customer_id is required for safe history matching.')
    h = _history_preparation(history, assume_only_customer=history_is_customer_only)
    if not history_is_customer_only:
        h = h.loc[h.customer_id.eq(sid).fillna(False)].copy()
    h = h.loc[h._day.notna() & (h._day < day)].copy()
    h = h.sort_values(['_day','_order'],kind='stable')
    n = len(h)
    amounts = h._amount.to_numpy(dtype=float)
    durations = h._duration.to_numpy(dtype=float)
    valid_a = amounts[np.isfinite(amounts)]
    valid_d = durations[np.isfinite(durations)]
    date_array = h._day.to_numpy(dtype='datetime64[ns]')
    distinct = np.unique(date_array)
    f = {
        'loan_type':loan_type, 'duration':float(dur),
        'disbursement_year':float(day.year),
        'disbursement_month':float(day.month),
        'disbursement_quarter':float((day.month-1)//3+1),
        'disbursement_dayofweek':float(day.dayofweek),
        'disbursement_day':float(day.day),
        'disbursement_is_weekend':float(day.dayofweek>=5),
        'previous_loan_count':n,
        'previous_valid_amount_count':len(valid_a) if n else 0,
        'previous_total_amount_borrowed':float(valid_a.sum()) if len(valid_a) else np.nan,
        'previous_mean_loan_amount':_mean(valid_a),
        'previous_min_loan_amount':float(valid_a.min()) if len(valid_a) else np.nan,
        'previous_max_loan_amount':float(valid_a.max()) if len(valid_a) else np.nan,
        'previous_mean_duration':_mean(valid_d),
        'days_since_previous_loan':float((day - pd.Timestamp(distinct[-1])).days) if n else np.nan,
        'previous_loans_last_90d':int(np.sum(date_array>=np.datetime64(day-pd.Timedelta(days=90)))) if n else 0,
        'previous_loans_last_180d':int(np.sum(date_array>=np.datetime64(day-pd.Timedelta(days=180)))) if n else 0,
        'previous_mean_days_between_loans':float(np.diff(distinct).astype('timedelta64[D]').astype(float).mean()) if len(distinct)>1 else np.nan,
        'previous_mean_amount_last_3':_mean(amounts[-3:]) if n else np.nan,
        'previous_most_recent_loan_amount':np.nan,
        'previous_amount_trend_last_3':np.nan,
        'has_previous_loan':int(n>0),
        'has_previous_valid_amount':int(len(valid_a)>0),
        'has_previous_valid_duration':int(len(valid_d)>0),
        'has_multiple_previous_loan_dates':int(len(distinct)>1),
    }
    if n:
        # Original V2: latest date must contain exactly one earlier loan.
        on_latest = np.flatnonzero(date_array==distinct[-1])
        if len(on_latest)==1 and np.isfinite(amounts[on_latest[0]]):
            f['previous_most_recent_loan_amount'] = float(amounts[on_latest[0]])
        last3 = np.arange(max(0,n-3),n)
        if (len(last3)>=2 and len(np.unique(date_array[last3]))==len(last3)
            and np.isfinite(amounts[last3[0]]) and np.isfinite(amounts[last3[-1]])):
            f['previous_amount_trend_last_3'] = float(amounts[last3[-1]]-amounts[last3[0]])
    feats = list(required_features) if required_features is not None else EXPECTED_F1
    unsupported = [x for x in feats if x not in f]
    if unsupported:
        raise ValueError(f'Unexpected trained feature contract: {unsupported}. Use the matching V2 F1 models.')
    return pd.DataFrame([{k:f[k] for k in feats}])


def raw_batch_to_f1(applications, *, history=None, required_features=None, use_batch_as_history=True,
                    max_applications=2000):
    """Convert original-format rows to F1 model rows.

    Rows in the application batch can contribute history to *later-date* rows
    when use_batch_as_history=True; current and same-day rows are excluded.
    For unseen test data, provide actual historical raw loan events when possible.
    """
    if not isinstance(applications,pd.DataFrame):
        raise ValueError('Applications must be a DataFrame.')
    missing=[x for x in RAW_APPLICATION_FIELDS if x not in applications]
    if missing: raise ValueError(f'Raw application CSV missing: {missing}')
    if len(applications)>max_applications:
        raise ValueError(f'Demo limit is {max_applications:,} rows per batch.')
    source = applications.copy()
    if history is not None:
        hist = history.copy()
        if use_batch_as_history and all(c in source.columns for c in RAW_HISTORY_FIELDS):
            hist = pd.concat([hist,source[RAW_HISTORY_FIELDS]],ignore_index=True)
    elif use_batch_as_history and all(c in source.columns for c in RAW_HISTORY_FIELDS):
        hist = source[RAW_HISTORY_FIELDS].copy()
    else:
        hist = pd.DataFrame(columns=RAW_HISTORY_FIELDS)
    prepared = _history_preparation(hist)
    by_customer = {str(cid):part.copy() for cid,part in prepared.groupby('customer_id',dropna=False)}
    rows=[]; errors=[]
    for idx,(_,app) in enumerate(source.iterrows(),start=0):
        try:
            cid = str(app['customer_id']).strip()
            past = by_customer.get(cid, prepared.iloc[0:0])
            feat = make_f1_features(
                customer_id=cid, disbursement_date=app['disbursement_date'],
                loan_type=app['loan_type'], duration=app['duration'],
                history=past, required_features=required_features,
                history_is_customer_only=True)
            rows.append(feat.iloc[0].to_dict()); errors.append('')
        except Exception as e:
            rows.append({name:np.nan for name in (required_features or EXPECTED_F1)})
            errors.append(str(e))
    return pd.DataFrame(rows,columns=required_features or EXPECTED_F1),pd.Series(errors,name='input_error')


def engineered_batch_to_f1(df,*,required_features):
    missing=[x for x in required_features if x not in df]
    if missing:raise ValueError(f'Engineered CSV missing features: {missing}')
    X=df[required_features].copy()
    for c in required_features:
        if c=='loan_type':
            X[c]=X[c].astype('string').replace({'':pd.NA}).astype(object)
        else:
            raw=X[c]
            num=pd.to_numeric(raw,errors='coerce')
            invalid=raw.notna() & num.isna() & raw.astype('string').str.strip().ne('')
            if invalid.any():raise ValueError(f'{c}: {int(invalid.sum())} invalid numbers')
            X[c]=num.replace([np.inf,-np.inf],np.nan)
    return X


def convert_kes_to_ghs(kes, ghs_per_kes):
    rate=float(ghs_per_kes)
    if not math.isfinite(rate) or rate<=0:
        raise ValueError('Enter a positive exchange rate in GHS per 1 KES.')
    amounts=np.asarray(kes,dtype=float)
    if not np.isfinite(amounts).all():
        raise ValueError('KES predictions must be finite.')
    return np.maximum(0,amounts)*rate
