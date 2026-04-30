"""
Autoresearch churn training script. Single-file, CPU-friendly.
Usage: uv run train.py

This is the file the agent modifies. Everything below the imports is fair game:
feature engineering, model family, hyperparameters, threshold strategy.

The only constraints:
- Do not modify prepare.py or its evaluate_auc() function.
- Do not add new dependencies beyond pyproject.toml.
- The script must print the summary block in the exact format below.
"""

import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import lightgbm as lgb

from prepare import load_data, evaluate_auc, RANDOM_SEED

start = time.time()

# ---------------------------------------------------------------------------
# Load raw joined data
# ---------------------------------------------------------------------------

df = load_data()

# ---------------------------------------------------------------------------
# Feature engineering — modify freely
# ---------------------------------------------------------------------------

REF_DATE = pd.Timestamp("2017-03-31")  # approximate end of observation window

features = pd.DataFrame(index=df.index)

# --- Membership tenure ---
features["tenure_days"] = (
    REF_DATE - pd.to_datetime(df["registration_init_time"], errors="coerce")
).dt.days

# --- Demographics ---
features["city"] = df["city"].fillna(-1)
features["age"] = df["bd"].clip(0, 100).fillna(-1)
features["gender"] = LabelEncoder().fit_transform(df["gender"].fillna("unknown").astype(str))
features["registered_via"] = df["registered_via"].fillna(-1)

# --- Listening behavior ---
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

# --- Recency ---
features["days_since_last_listen"] = (
    REF_DATE - pd.to_datetime(df["log_last_date"], errors="coerce")
).dt.days.fillna(999)
features["listening_span_days"] = (
    pd.to_datetime(df["log_last_date"], errors="coerce") -
    pd.to_datetime(df["log_first_date"], errors="coerce")
).dt.days.fillna(0)

# --- Transactions ---
features["txn_count"] = df["txn_count"].fillna(0)
features["txn_last_plan_days"] = df["txn_last_plan_days"].fillna(30)
features["txn_last_price"] = df["txn_last_price"].fillna(0)
features["discount"] = (
    df["txn_last_list_price"].fillna(0) - df["txn_last_price"].fillna(0)
).clip(lower=0)
features["txn_last_auto_renew"] = df["txn_last_auto_renew"].fillna(0)
features["txn_cancel_count"] = df["txn_cancel_count"].fillna(0)

# ---------------------------------------------------------------------------
# Train / val split (fixed — do not change seed or fraction)
# ---------------------------------------------------------------------------

y = df["is_churn"].values
X = features.values.astype(np.float32)
feature_names = features.columns.tolist()

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
)

# ---------------------------------------------------------------------------
# Model — modify freely
# ---------------------------------------------------------------------------

model = lgb.LGBMClassifier(
    n_estimators=500,
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
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
)

# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

val_auc = evaluate_auc(model, X_val, y_val)
train_auc = evaluate_auc(model, X_train, y_train)
duration = time.time() - start
best_iter = model.best_iteration_ if model.best_iteration_ else model.n_estimators

# ---------------------------------------------------------------------------
# Summary — keep this block intact so the agent can parse it
# ---------------------------------------------------------------------------

importances = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])
print("\nFeature importances (top 10):")
for name, imp in importances[:10]:
    print(f"  {name:<30} {imp}")

print("---")
print(f"val_auc:        {val_auc:.6f}")
print(f"train_auc:      {train_auc:.6f}")
print(f"best_iteration: {best_iter}")
print(f"n_features:     {len(feature_names)}")
print(f"model:          LightGBM")
print(f"duration_s:     {duration:.1f}")
