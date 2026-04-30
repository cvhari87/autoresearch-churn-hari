# autoresearch-churn

Autonomous churn prediction research on the KKBox Music Streaming dataset (WSDM 2017 Kaggle competition).

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr21`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read all of these:
   - `README.md` — repository context.
   - `prepare.py` — fixed infrastructure: data loading, aggregation, `evaluate_auc()`. Do not modify.
   - `train.py` — the file you modify. Feature engineering, model, hyperparameters.
4. **Verify data exists**: Check that `~/.cache/autoresearch-churn/data.pkl` exists. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row.
6. **Confirm and go**: Confirm setup looks good, then begin.

## Experimentation

Each experiment modifies `train.py` and runs it. The full training loop completes in seconds to low minutes on CPU — no time budget constraint is needed.

**What you CAN do:**

- Modify `train.py` — this is the only file you edit. Everything is fair game:
  - Feature engineering: interaction terms, ratios, binning, log transforms, recency/frequency/monetary features, temporal features
  - Model family: LightGBM, XGBoost, RandomForest, GradientBoosting, LogisticRegression, or ensembles
  - Hyperparameters: n_estimators, learning_rate, depth, regularization, subsampling
  - Class imbalance strategies: class_weight, scale_pos_weight, threshold tuning

**What you CANNOT do:**

- Modify `prepare.py`. It is read-only.
- Change the train/val split (seed=42, 20% holdout). The split is fixed in train.py — do not alter it.
- Call `evaluate_auc()` on training data to guide decisions — only `X_val/y_val` counts.
- Install packages not in `pyproject.toml`.

**The goal: get the highest val_auc.** The train/val split is fixed (seed=42), so all experiments are directly comparable.

**Simplicity criterion**: All else equal, simpler is better. A 0.0005 AUC gain that adds 30 lines of complex feature code is not worth it. Removing a feature that doesn't hurt AUC is a win. Prefer interpretable features — this is ultimately a business model, not a Kaggle leaderboard submission.

**The first run**: Always establish the baseline first — run `train.py` as-is without any changes and record the result.

## Output format

The script prints a summary like:

```
---
val_auc:        0.845678
train_auc:      0.912345
best_iteration: 312
n_features:     18
model:          LightGBM
duration_s:     14.3
```

Extract the key metric:

```
grep "^val_auc:" run.log
```

## Logging results

Log to `results.tsv` (tab-separated) after every experiment.

```
commit	val_auc	model	status	description
```

1. git commit hash (short, 7 chars)
2. val_auc achieved (e.g. 0.845678) — use 0.000000 for crashes
3. model family used (e.g. LightGBM, XGBoost, LogReg)
4. status: `keep`, `discard`, or `crash`
5. short description of what this experiment tried

Example:
```
commit	val_auc	model	status	description
a1b2c3d	0.832100	LightGBM	keep	baseline
b2c3d4e	0.841200	LightGBM	keep	add cancel_rate and discount features
c3d4e5f	0.829000	LogReg	discard	logistic regression with same features
d4e5f6g	0.844500	LightGBM	keep	log-transform total_secs, increase n_estimators to 1000
```

## The experiment loop

LOOP FOREVER:

1. Check the current git state.
2. Form a hypothesis — what feature, transformation, or model change might improve val_auc?
3. Modify `train.py` with the change.
4. `git commit -m "experiment: <short description>"`
5. Run: `uv run train.py > run.log 2>&1`
6. Read result: `grep "^val_auc:" run.log`
7. If the grep is empty, it crashed — run `tail -n 50 run.log` to diagnose. Fix if trivial, otherwise skip.
8. Log the result to `results.tsv`.
9. If val_auc improved → keep the commit, advance.
10. If val_auc equal or worse → `git reset --hard HEAD~1`, move on.

**Do not pause to ask if you should continue.** The human may be away. Run until interrupted.

**Crash policy**: If a crash is a trivial bug (typo, missing import), fix and re-run. If the idea itself is broken, log as crash and move on.

**Stuck policy**: If several experiments in a row don't improve AUC, try a different axis:
- Shift from hyperparameter tuning to new features
- Try a completely different model family
- Revisit the raw columns in `prepare.py` for features you haven't used yet (check `df.columns`)
- Try feature selection (drop low-importance features and simplify)

**Context on the data** (use this to generate hypotheses):
- KKBox is a music streaming subscription service in Taiwan/SE Asia
- `is_churn = 1` means the user did not renew their subscription in the next 30 days
- Key behavioral signals in the data: listening recency, frequency, completion rate, unique song diversity
- Key subscription signals: auto-renew status, cancellation history, days to expiry, discount history
- Users with high `days_since_last_listen` and `txn_cancel_count` are the intuitive high-risk segment
- Churn rate is roughly 6% (imbalanced — handle appropriately)
