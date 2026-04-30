# skill: classical-ml-experimentation

Guidelines for autonomous ML experimentation on tabular churn/classification tasks.
Follow this document alongside `program.md` when running the experiment loop.

---

## Model ladder

Always progress through models in this order. Do not skip ahead — each rung establishes
a baseline that makes the next rung's gain (or lack thereof) meaningful.

### Rung 1 — Logistic Regression (mandatory first baseline)

Before any tree-based model, fit a logistic regression on the current feature set.
Use `class_weight='balanced'` and `max_iter=1000`. Scale features with `StandardScaler`.

**Why**: LR is the interpretability floor. If LR AUC is close to the tree model, the
tree is not finding complex interactions — simpler is better. If LR is far behind, the
trees are doing real work.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_SEED)
)
```

Log result as model=`LogReg`. Record the gap vs. tree models — it's the "complexity tax."

### Rung 2 — LightGBM (primary workhorse)

Once LR baseline is recorded, switch to LightGBM. Use AUC-based early stopping
(`eval_metric="auc"`) so the stopping criterion matches the evaluation metric.

Key defaults to start with:
```
n_estimators=1500, learning_rate=0.05, num_leaves=63,
subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1,
class_weight="balanced", early_stopping_rounds=100
```

### Rung 3 — XGBoost (challenger)

Try XGBoost if LightGBM plateaus for 5+ experiments. Use:
```python
import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=1500, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(),
    eval_metric="auc", early_stopping_rounds=100,
    random_state=RANDOM_SEED
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```

Log model=`XGBoost`. Only keep if it beats the best LightGBM result.

---

## Feature engineering ladder

Try transformations in this order. Commit each, keep only if it improves val_auc.

1. **Log transforms** — apply `np.log1p()` to any right-skewed count or monetary column
2. **Ratios** — e.g. cancel_count / txn_count, listen_secs / active_days
3. **Interaction terms** — multiply two high-importance features (e.g. cancel_rate × days_since_last_listen)
4. **Cyclical time features** — for month/day-of-week fields: `sin(2π·x/period)` and `cos(2π·x/period)`
5. **Binning** — if a continuous feature has a non-linear relationship to churn, try pd.qcut with 5–10 bins
6. **Missing-value flags** — `feature_was_null` binary indicator for columns with >5% nulls

Add features one at a time. Do not bundle multiple untested features into one commit.

---

## Feature leakage audit

Run this audit **before locking in the best model** and after a significant AUC jump.

### Protocol

For each feature in the current feature set:

1. Temporarily remove the feature from `features` (comment it out).
2. Retrain and record val_auc.
3. Restore the feature.
4. Do NOT commit individual ablation runs — run them in a scratch loop without committing.

### Leakage signal

A feature is a **leakage suspect** if removing it causes val_auc to drop by **> 0.010**.

When you find a suspect:
- Ask: could this value be known at prediction time in production? If yes, it is legitimate signal.
- Ask: is this value computed from the label period or future data? If yes, it is leakage — remove it.
- Log your conclusion in `results.tsv` as a `leakage-audit` row with status `keep` or `removed`.

### Audit output format

After the audit, print a summary block:

```
=== Leakage Audit ===
feature                        auc_without  delta    verdict
txn_last_auto_renew            0.921        -0.059   OK (legitimate signal)
days_since_last_txn            0.968        -0.012   SUSPECT — verify
...
```

Log one audit row per flagged feature in `results.tsv`:

```
<hash>  <auc>  LightGBM  leakage-audit  audit: <feature> delta=-0.012, verdict=OK
```

---

## Stopping criteria

See `program.md` for the authoritative stopping rules. Summary:
- Stop when improvement across last 5 experiments is < 0.001 total
- Stop when 50 experiments are reached
- Run the leakage audit before declaring the best model final

---

## Simplicity tie-breaker

If two models are within 0.001 AUC of each other, prefer the simpler one:
- Fewer features beats more features
- LR beats LightGBM beats XGBoost (all else equal)
- Fewer engineered features beats more

Document the chosen model and the runner-up in `CHECKPOINT.md`.
