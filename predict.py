"""
Churn prediction inference script.
Loads model.joblib + model_meta.json and scores a batch of raw user records.

Usage:
    uv run predict.py users.csv          # CSV with raw user fields, prints churn proba
    uv run predict.py --demo             # score 5 synthetic records to verify the setup

Input CSV columns (all optional — missing ones are filled with safe defaults):
    msno, registration_init_time, city, bd, gender, registered_via,
    log_days, log_total_secs, log_mean_secs,
    log_num_25, log_num_50, log_num_75, log_num_985, log_num_100, log_num_uniq,
    log_last_date, log_first_date,
    txn_count, txn_last_plan_days, txn_last_price, txn_last_list_price,
    txn_last_auto_renew, txn_cancel_count, txn_last_date
"""

import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder

MODEL_PATH = "model.joblib"
META_PATH = "model_meta.json"


def build_features(df: pd.DataFrame, ref_date: pd.Timestamp) -> np.ndarray:
    f = pd.DataFrame(index=df.index)

    reg_time = df["registration_init_time"] if "registration_init_time" in df.columns else pd.Series(pd.NaT, index=df.index)
    f["tenure_days"] = (ref_date - pd.to_datetime(reg_time, errors="coerce")).dt.days

    f["city"] = df.get("city", pd.Series(-1, index=df.index)).fillna(-1)
    f["age"] = df.get("bd", pd.Series(-1, index=df.index)).clip(0, 100).fillna(-1)
    f["gender"] = LabelEncoder().fit_transform(
        df.get("gender", pd.Series("unknown", index=df.index)).fillna("unknown").astype(str)
    )
    f["registered_via"] = df.get("registered_via", pd.Series(-1, index=df.index)).fillna(-1)

    total_plays = sum(
        df.get(c, pd.Series(0, index=df.index)).fillna(0)
        for c in ["log_num_25", "log_num_50", "log_num_75", "log_num_985", "log_num_100"]
    )
    f["log_days"] = df.get("log_days", pd.Series(0, index=df.index)).fillna(0)
    f["log_total_secs"] = df.get("log_total_secs", pd.Series(0, index=df.index)).fillna(0)
    f["log_mean_secs"] = df.get("log_mean_secs", pd.Series(0, index=df.index)).fillna(0)
    num_100 = df.get("log_num_100", pd.Series(0, index=df.index)).fillna(0)
    f["completion_rate"] = (num_100 / total_plays.replace(0, np.nan)).fillna(0)
    num_uniq = df.get("log_num_uniq", pd.Series(0, index=df.index)).fillna(0)
    f["unique_ratio"] = (num_uniq / total_plays.replace(0, np.nan)).fillna(0)

    last_listen = pd.to_datetime(df["log_last_date"] if "log_last_date" in df.columns else pd.Series(pd.NaT, index=df.index), errors="coerce")
    first_listen = pd.to_datetime(df["log_first_date"] if "log_first_date" in df.columns else pd.Series(pd.NaT, index=df.index), errors="coerce")
    f["days_since_last_listen"] = (ref_date - last_listen).dt.days.fillna(999)
    f["listening_span_days"] = (last_listen - first_listen).dt.days.fillna(0)

    f["txn_count"] = df.get("txn_count", pd.Series(0, index=df.index)).fillna(0)
    f["txn_last_plan_days"] = df.get("txn_last_plan_days", pd.Series(30, index=df.index)).fillna(30)
    f["txn_last_price"] = df.get("txn_last_price", pd.Series(0, index=df.index)).fillna(0)
    list_price = df.get("txn_last_list_price", pd.Series(0, index=df.index)).fillna(0)
    f["discount"] = (list_price - f["txn_last_price"]).clip(lower=0)
    f["txn_last_auto_renew"] = df.get("txn_last_auto_renew", pd.Series(0, index=df.index)).fillna(0)
    cancel_count = df.get("txn_cancel_count", pd.Series(0, index=df.index)).fillna(0)
    f["txn_cancel_count"] = cancel_count
    last_txn = pd.to_datetime(df["txn_last_date"] if "txn_last_date" in df.columns else pd.Series(pd.NaT, index=df.index), errors="coerce")
    f["days_since_last_txn"] = (ref_date - last_txn).dt.days.fillna(999)
    f["log_total_secs_log"] = np.log1p(f["log_total_secs"])
    txn_count_safe = f["txn_count"].replace(0, 1)
    f["cancel_rate"] = cancel_count / txn_count_safe
    f["days_since_last_listen_log"] = np.log1p(f["days_since_last_listen"])

    return f.values.astype(np.float32)


def load_model():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as fh:
        meta = json.load(fh)
    return model, meta


def score(df: pd.DataFrame) -> pd.DataFrame:
    model, meta = load_model()
    ref_date = pd.Timestamp(meta["ref_date"])
    X = build_features(df, ref_date)
    proba = model.predict_proba(X)[:, 1]
    out = pd.DataFrame({"churn_probability": proba}, index=df.index)
    if "msno" in df.columns:
        out.insert(0, "msno", df["msno"].values)
    return out


def _demo():
    # Use raw column names that build_features expects (dates are ISO strings)
    demo_users = pd.DataFrame([
        {"msno": "active_user",  "txn_last_auto_renew": 1, "txn_cancel_count": 0,
         "txn_last_date": "2017-03-26", "log_last_date": "2017-03-28", "log_first_date": "2016-01-01",
         "log_days": 90,  "log_total_secs": 500000, "txn_count": 24,
         "registration_init_time": "2014-06-01"},
        {"msno": "at_risk_user", "txn_last_auto_renew": 0, "txn_cancel_count": 3,
         "txn_last_date": "2017-02-10", "log_last_date": "2017-02-15", "log_first_date": "2016-06-01",
         "log_days": 10,  "log_total_secs": 50000,  "txn_count": 6,
         "registration_init_time": "2016-01-01"},
        {"msno": "lapsed_user",  "txn_last_auto_renew": 0, "txn_cancel_count": 5,
         "txn_last_date": "2016-12-01", "log_last_date": "2016-12-10", "log_first_date": "2016-09-01",
         "log_days": 2,   "log_total_secs": 10000,  "txn_count": 3,
         "registration_init_time": "2016-08-01"},
        {"msno": "loyal_user",   "txn_last_auto_renew": 1, "txn_cancel_count": 0,
         "txn_last_date": "2017-03-29", "log_last_date": "2017-03-30", "log_first_date": "2014-01-01",
         "log_days": 300, "log_total_secs": 2000000, "txn_count": 48,
         "registration_init_time": "2013-01-01"},
        {"msno": "new_user",     "txn_last_auto_renew": 1, "txn_cancel_count": 0,
         "txn_last_date": "2017-03-20", "log_last_date": "2017-03-22", "log_first_date": "2017-03-01",
         "log_days": 5,   "log_total_secs": 30000,  "txn_count": 1,
         "registration_init_time": "2017-03-01"},
    ])
    results = score(demo_users)
    print("\nDemo churn scores:")
    print(results.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "--demo":
        _demo()
    else:
        input_path = sys.argv[1]
        df = pd.read_csv(input_path)
        results = score(df)
        results.to_csv(sys.stdout, index=False)
