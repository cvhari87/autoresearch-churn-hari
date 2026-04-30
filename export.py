"""
Serialize the trained churn model and feature metadata for inference.
Usage: uv run export.py
Produces: model.joblib, model_meta.json
"""

import json
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from prepare import load_data, evaluate_auc, RANDOM_SEED

start = time.time()

REF_DATE = pd.Timestamp("2017-03-31")

df = load_data()

features = pd.DataFrame(index=df.index)

features["tenure_days"] = (
    REF_DATE - pd.to_datetime(df["registration_init_time"], errors="coerce")
).dt.days

features["city"] = df["city"].fillna(-1)
features["age"] = df["bd"].clip(0, 100).fillna(-1)
features["gender"] = LabelEncoder().fit_transform(df["gender"].fillna("unknown").astype(str))
features["registered_via"] = df["registered_via"].fillna(-1)

total_plays = (
    df["log_num_25"].fillna(0) + df["log_num_50"].fillna(0) +
    df["log_num_75"].fillna(0) + df["log_num_985"].fillna(0) +
    df["log_num_100"].fillna(0)
)
features["log_days"] = df["log_days"].fillna(0)
features["log_total_secs"] = df["log_total_secs"].fillna(0)
features["log_mean_secs"] = df["log_mean_secs"].fillna(0)
features["completion_rate"] = (df["log_num_100"].fillna(0) / total_plays.replace(0, np.nan)).fillna(0)
if "log_num_uniq" in df.columns:
    features["unique_ratio"] = (df["log_num_uniq"].fillna(0) / total_plays.replace(0, np.nan)).fillna(0)
else:
    features["unique_ratio"] = 0.0

features["days_since_last_listen"] = (
    REF_DATE - pd.to_datetime(df["log_last_date"], errors="coerce")
).dt.days.fillna(999)
features["listening_span_days"] = (
    pd.to_datetime(df["log_last_date"], errors="coerce") -
    pd.to_datetime(df["log_first_date"], errors="coerce")
).dt.days.fillna(0)

features["txn_count"] = df["txn_count"].fillna(0)
features["txn_last_plan_days"] = df["txn_last_plan_days"].fillna(30)
features["txn_last_price"] = df["txn_last_price"].fillna(0)
features["discount"] = (
    df["txn_last_list_price"].fillna(0) - df["txn_last_price"].fillna(0)
).clip(lower=0)
features["txn_last_auto_renew"] = df["txn_last_auto_renew"].fillna(0)
features["txn_cancel_count"] = df["txn_cancel_count"].fillna(0)
features["days_since_last_txn"] = (
    REF_DATE - pd.to_datetime(df["txn_last_date"], errors="coerce")
).dt.days.fillna(999)
features["log_total_secs_log"] = np.log1p(df["log_total_secs"].fillna(0))
cancel_count = df["txn_cancel_count"].fillna(0)
txn_count_safe = df["txn_count"].fillna(1).replace(0, 1)
features["cancel_rate"] = cancel_count / txn_count_safe
features["days_since_last_listen_log"] = np.log1p(features["days_since_last_listen"])

y = df["is_churn"].values
X = features.values.astype(np.float32)
feature_names = features.columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)

model = lgb.LGBMClassifier(
    n_estimators=1500,
    learning_rate=0.05,
    num_leaves=63,
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    class_weight="balanced",
    random_state=RANDOM_SEED,
    n_jobs=-1,
    verbose=-1,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(period=-1)],
)

val_auc = evaluate_auc(model, X_val, y_val)

joblib.dump(model, "model.joblib")

meta = {
    "feature_names": feature_names,
    "ref_date": REF_DATE.isoformat(),
    "val_auc": round(val_auc, 6),
    "best_iteration": int(model.best_iteration_) if model.best_iteration_ else model.n_estimators,
    "trained_at": pd.Timestamp.now().isoformat(),
}
with open("model_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

duration = time.time() - start
print(f"val_auc:       {val_auc:.6f}")
print(f"model saved:   model.joblib  ({joblib.__version__ if hasattr(joblib, '__version__') else 'ok'})")
print(f"meta saved:    model_meta.json")
print(f"features:      {len(feature_names)}")
print(f"duration_s:    {duration:.1f}")
