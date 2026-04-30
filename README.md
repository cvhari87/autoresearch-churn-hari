# autoresearch-churn

Autonomous churn prediction research on the [KKBox Music Streaming dataset](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge) (WSDM 2017 Kaggle competition).

An AI agent runs the full experiment loop — hypothesis → feature engineering → train → evaluate → commit or revert — without human intervention. The goal is to maximise validation AUC on a fixed train/val split (seed=42, 80/20).

---

## Results

| Experiment | val_auc | Δ | Description |
|---|---|---|---|
| Baseline | 0.9120 | — | LightGBM defaults, 14 features |
| +cancel signals | 0.9722 | **+0.0602** | Added `txn_last_auto_renew` and `txn_cancel_count` |
| +recency/txn | 0.9798 | +0.0076 | Added `days_since_last_txn`, log-transform `total_secs` |
| +cancel rate | 0.9800 | +0.0002 | Added `cancel_rate` ratio, log-transform recency |
| +AUC early stop | **0.9800** | +0.0000 | Fixed early stopping to optimise AUC not log-loss |

**Best val_auc: 0.980011** — branch `autoresearch/apr30`, commit `0812a84`

### Key finding

The single biggest improvement (+0.060 AUC) came from two features that were computed in `prepare.py` but never wired into `train.py`:

- `txn_last_auto_renew` — whether the user's last subscription had auto-renew enabled
- `txn_cancel_count` — total number of explicit cancellations in transaction history

These two fields encode the user's *stated intent* around renewal, making them far more predictive than any behavioral proxy.

---

## How it works

```
prepare.py   ← fixed infrastructure, do not modify
train.py     ← the agent modifies this file only
results.tsv  ← experiment log (commit, val_auc, model, status, description)
CHECKPOINT.md← agent state: best result, what was tried, what to try next
```

The agent follows a strict loop:

1. Form a hypothesis
2. Edit `train.py`
3. `git commit`
4. `uv run train.py > run.log`
5. Parse `val_auc` from the log
6. If improved → keep commit, log result; otherwise → `git reset --hard HEAD~1`

All experiments are directly comparable because the train/val split is fixed.

---

## Setup

**Prerequisites:** Python ≥ 3.10, [uv](https://github.com/astral-sh/uv), a [Kaggle API key](https://www.kaggle.com/settings/account) at `~/.kaggle/kaggle.json`, and competition rules accepted at [kkbox-churn-prediction-challenge](https://www.kaggle.com/competitions/kkbox-churn-prediction-challenge/rules).

```bash
# 1. Clone and install
git clone https://github.com/cvhari87/autoresearch-churn-hari.git
cd autoresearch-churn-hari
uv sync

# 2. Download and prepare data (one-time, ~10 min)
uv run prepare.py

# 3. Run the current best model
uv run train.py
```

Expected output:

```
Feature importances (top 10):
  days_since_last_txn            3687
  tenure_days                    3238
  completion_rate                2426
  ...
---
val_auc:        0.980011
train_auc:      0.986415
best_iteration: 456
n_features:     22
model:          LightGBM
duration_s:     18.4
```

---

## Feature set (22 features)

| Group | Features |
|---|---|
| Membership | `tenure_days`, `city`, `age`, `gender`, `registered_via` |
| Listening | `log_days`, `log_total_secs`, `log_total_secs_log`, `log_mean_secs`, `completion_rate`, `unique_ratio` |
| Recency | `days_since_last_listen`, `days_since_last_listen_log`, `listening_span_days` |
| Transactions | `txn_count`, `txn_last_plan_days`, `txn_last_price`, `discount`, `txn_last_auto_renew`, `txn_cancel_count`, `days_since_last_txn`, `cancel_rate` |

---

## Model

LightGBM classifier with AUC-optimised early stopping.

```
n_estimators=1500   learning_rate=0.05   num_leaves=63
subsample=0.8       colsample_bytree=0.8
reg_alpha=0.1       reg_lambda=0.1
class_weight=balanced   eval_metric=auc   early_stopping_rounds=100
```

The model converges at ~456 iterations. `class_weight='balanced'` handles the ~6% churn rate.

---

## About the dataset

KKBox is a music streaming subscription service in Taiwan and Southeast Asia. The dataset covers user listening logs, transaction history, and member profiles. The prediction target is whether a user will churn (fail to renew) within 30 days of their subscription expiry.

- ~700k users in training set
- ~6% churn rate (imbalanced)
- Key signals: auto-renew status, cancellation history, listening recency, subscription tenure
