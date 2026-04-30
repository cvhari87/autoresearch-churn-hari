# Checkpoint — autoresearch/apr30

**Date**: 2026-04-30  
**Branch**: autoresearch/apr30  
**Current best val_auc**: 0.980011  
**Best commit**: 0812a84

## Results so far

| commit  | val_auc  | status  | description |
|---------|----------|---------|-------------|
| bb1f485 | 0.912000 | keep    | baseline |
| 8504ca6 | 0.972195 | keep    | add txn_last_auto_renew and txn_cancel_count |
| ad02cf1 | 0.979772 | keep    | add days_since_last_txn and log-transform total_secs |
| 11137c1 | 0.979979 | keep    | add cancel_rate ratio and log-transform days_since_last_listen |
| 6619fc9 | 0.979459 | discard | increase n_estimators to 1500 only (logloss-based early stopping wrong) |
| 0812a84 | 0.980011 | keep    | eval_metric=auc for early stopping, n_estimators=1500 (best_iter=456) |

## Current feature set (22 features)

Membership: tenure_days, city, age, gender, registered_via  
Listening: log_days, log_total_secs, log_total_secs_log, log_mean_secs, completion_rate, unique_ratio  
Recency: days_since_last_listen, days_since_last_listen_log, listening_span_days  
Transactions: txn_count, txn_last_plan_days, txn_last_price, discount, txn_last_auto_renew, txn_cancel_count, days_since_last_txn, cancel_rate  

## Current model config

LightGBM, n_estimators=1500, lr=0.05, num_leaves=63, subsample=0.8, colsample=0.8,
reg_alpha/lambda=0.1, class_weight=balanced, eval_metric=auc, early_stopping=100

## Key insights

- The single biggest win was adding `txn_last_auto_renew` and `txn_cancel_count` (+0.060 AUC). These were in `prepare.py` but missing from the baseline.
- Early stopping must use `eval_metric='auc'` — default logloss optimizes the wrong thing and costs ~0.0005 AUC.
- Model converges at ~456 iterations with AUC-based early stopping.
- Top features by importance: days_since_last_txn, tenure_days, completion_rate, log_days, txn_count, cancel_rate

## Next experiments to try

1. **Listening intensity**: `log_total_secs / log_days` (avg secs per active day) — engagement consistency
2. **Cancel-no-renew flag**: `txn_cancel_count * (1 - txn_last_auto_renew)` — highest-risk segment
3. **Lower lr + more trees**: lr=0.02, n_estimators=3000 — slower convergence may find better optimum
4. **Larger num_leaves**: try 127 — model may benefit from more complex trees given 22 features
5. **Engagement drop flag**: `days_since_last_listen / listening_span_days` — recent silence relative to history
6. **Drop redundant features**: `log_total_secs` and `log_total_secs_log` are correlated — try dropping original
7. **Try XGBoost** if LightGBM plateau persists
