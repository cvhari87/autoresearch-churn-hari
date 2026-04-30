"""
One-time data preparation for autoresearch-churn.
Downloads the KKBox Churn Prediction dataset via the Kaggle CLI,
aggregates the raw tables into user-level base features, and saves a
single processed file that train.py reads.

Usage:
    uv run prepare.py

Requires ~/.kaggle/kaggle.json with your Kaggle API credentials.
Data is cached at ~/.cache/autoresearch-churn/.

DO NOT MODIFY THIS FILE. It is fixed infrastructure.
The evaluation function evaluate_auc() is the ground-truth metric.
"""

import os
import subprocess
import zipfile
import pickle
import py7zr
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ---------------------------------------------------------------------------
# Constants (fixed — do not modify)
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-churn")
RAW_DIR = os.path.join(CACHE_DIR, "raw")
DATA_PATH = os.path.join(CACHE_DIR, "data.pkl")

COMPETITION = "kkbox-churn-prediction-challenge"
VAL_FRACTION = 0.20
RANDOM_SEED = 42

# Listening log columns: count of songs played to each completion bracket
LOG_PLAY_COLS = ["num_25", "num_50", "num_75", "num_985", "num_100", "num_uniq"]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download():
    os.makedirs(RAW_DIR, exist_ok=True)
    # Skip if already downloaded
    if os.path.exists(os.path.join(RAW_DIR, "train_v2.csv")):
        print("Raw files already present, skipping download.")
        return
    print(f"Downloading KKBox dataset from Kaggle competition: {COMPETITION}")
    try:
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", RAW_DIR],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        err = (e.stderr or "").lower()
        if "403" in err or "forbidden" in err:
            raise SystemExit(
                "\n[prepare.py] 403 Forbidden — you need to accept the competition rules first.\n"
                f"Go to: https://www.kaggle.com/competitions/{COMPETITION}/rules\n"
                "Click 'I Understand and Accept', then re-run: uv run prepare.py"
            ) from None
        if "401" in err or "unauthorized" in err:
            raise SystemExit(
                "\n[prepare.py] 401 Unauthorized — Kaggle credentials not found or invalid.\n"
                "Make sure C:\\Users\\madhu\\.kaggle\\kaggle.json exists and contains your API key.\n"
                "Get a new key at: https://www.kaggle.com/settings/account"
            ) from None
        raise SystemExit(f"\n[prepare.py] Kaggle download failed:\n{e.stderr}") from None
    print("Extracting ...")
    for fname in os.listdir(RAW_DIR):
        fpath = os.path.join(RAW_DIR, fname)
        if fname.endswith(".zip"):
            with zipfile.ZipFile(fpath) as z:
                z.extractall(RAW_DIR)
        elif fname.endswith(".7z"):
            with py7zr.SevenZipFile(fpath, mode="r") as z:
                z.extractall(RAW_DIR)
    print("Extraction complete.")


# ---------------------------------------------------------------------------
# Aggregation helpers (produce one row per user)
# ---------------------------------------------------------------------------

def _agg_user_logs() -> pd.DataFrame:
    """Aggregate daily listening logs to user level. Reads both v1 and v2.
    Aggregates chunk-by-chunk to avoid loading full files into RAM."""
    agg = None
    total_rows = 0

    for fname in ["user_logs.csv", "user_logs_v2.csv"]:
        fpath = os.path.join(RAW_DIR, fname)
        if not os.path.exists(fpath):
            continue
        print(f"  Reading {fname} in chunks ...")
        for i, chunk in enumerate(pd.read_csv(fpath, chunksize=100_000)):
            total_rows += len(chunk)
            chunk["date"] = pd.to_datetime(chunk["date"].astype(str), format="%Y%m%d", errors="coerce")

            present_play_cols = [c for c in LOG_PLAY_COLS if c in chunk.columns]
            chunk_agg = chunk.groupby("msno").agg(
                log_days=("date", "count"),
                log_last_date=("date", "max"),
                log_first_date=("date", "min"),
                log_total_secs=("total_secs", "sum"),
                log_mean_secs=("total_secs", "mean"),
                **{f"log_{c}": (c, "sum") for c in present_play_cols},
            ).reset_index()

            if agg is None:
                agg = chunk_agg
            else:
                combined = pd.concat([agg, chunk_agg], ignore_index=True)
                present_log_cols = [c for c in LOG_PLAY_COLS if f"log_{c}" in combined.columns]
                agg = combined.groupby("msno").agg(
                    log_days=("log_days", "sum"),
                    log_last_date=("log_last_date", "max"),
                    log_first_date=("log_first_date", "min"),
                    log_total_secs=("log_total_secs", "sum"),
                    log_mean_secs=("log_mean_secs", "mean"),
                    **{f"log_{c}": (f"log_{c}", "sum") for c in present_log_cols},
                ).reset_index()

            if (i + 1) % 10 == 0:
                print(f"    ... {total_rows:,} rows processed")

    print(f"  Total log rows processed: {total_rows:,}")
    return agg


def _agg_transactions() -> pd.DataFrame:
    """Aggregate transaction history to user level."""
    dfs = []
    for fname in ["transactions.csv", "transactions_v2.csv"]:
        fpath = os.path.join(RAW_DIR, fname)
        if not os.path.exists(fpath):
            continue
        print(f"  Reading {fname} ...")
        df = pd.read_csv(fpath, parse_dates=["transaction_date", "membership_expire_date"])
        dfs.append(df)
    txn = pd.concat(dfs, ignore_index=True).sort_values("transaction_date")

    agg = txn.groupby("msno").agg(
        txn_count=("transaction_date", "count"),
        txn_last_date=("transaction_date", "max"),
        txn_last_plan_days=("payment_plan_days", "last"),
        txn_last_price=("actual_amount_paid", "last"),
        txn_last_list_price=("plan_list_price", "last"),
        txn_last_auto_renew=("is_auto_renew", "last"),
        txn_cancel_count=("is_cancel", "sum"),
    ).reset_index()
    return agg


def _load_members() -> pd.DataFrame:
    fpath = os.path.join(RAW_DIR, "members_v3.csv")
    if not os.path.exists(fpath):
        fpath = os.path.join(RAW_DIR, "members.csv")
    print(f"  Reading members ...")
    members = pd.read_csv(fpath)
    members["registration_init_time"] = pd.to_datetime(
        members["registration_init_time"].astype(str), format="%Y%m%d", errors="coerce"
    )
    return members


# ---------------------------------------------------------------------------
# Main prepare
# ---------------------------------------------------------------------------

def prepare():
    _download()

    print("\nAggregating user logs ...")
    logs_agg = _agg_user_logs()

    print("\nAggregating transactions ...")
    txn_agg = _agg_transactions()

    print("\nLoading members ...")
    members = _load_members()

    print("\nLoading labels ...")
    labels_path = os.path.join(RAW_DIR, "train_v2.csv")
    if not os.path.exists(labels_path):
        labels_path = os.path.join(RAW_DIR, "train.csv")
    labels = pd.read_csv(labels_path)  # columns: msno, is_churn

    # Join all tables on msno
    df = labels.merge(members, on="msno", how="left")
    df = df.merge(logs_agg, on="msno", how="left")
    df = df.merge(txn_agg, on="msno", how="left")

    print(f"\nJoined dataset: {df.shape[0]:,} users, {df.shape[1]} columns")
    print(f"Churn rate: {df['is_churn'].mean():.3f}")

    # Save raw joined dataframe (train.py does feature engineering)
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(DATA_PATH, "wb") as f:
        pickle.dump(df, f)
    print(f"\nSaved to {DATA_PATH}")


# ---------------------------------------------------------------------------
# Runtime utilities — used by train.py, DO NOT MODIFY
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    """Returns the raw joined dataframe. train.py engineers features from this."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Data not found at {DATA_PATH}. Run: uv run prepare.py"
        )
    with open(DATA_PATH, "rb") as f:
        return pickle.load(f)


def evaluate_auc(model, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """ROC-AUC on the validation set. This is the ground-truth metric — do not modify."""
    proba = model.predict_proba(X_val)[:, 1]
    return roc_auc_score(y_val, proba)


RANDOM_SEED = RANDOM_SEED  # re-export for train.py


if __name__ == "__main__":
    prepare()
