import os
import time
import requests
import pandas as pd
from datetime import datetime
from typing import Optional
import matplotlib.pyplot as plt
import numpy as np



# ============================================================
# Riksbank Turnover Statistics API
# - Daily endpoint, max 31 days per call
# - Auth: Ocp-Apim-Subscription-Key
# - Your plan: 200 calls/min (~3.33 calls/sec)
# ============================================================

BASE_URL = "https://api.riksbank.se/turnover-statistics/v1"
MARKET = "fi"
FREQUENCY = "daily"

# Put the key in an env var if you can (recommended)
#   export RIKSBANK_API_KEY="..."
API_KEY = os.getenv("RIKSBANK_API_KEY") or "0460e512bfd8437598e82f4df5aa3437"

# Rate-limit safety: keep below 200/min.
# 0.35 sec ~ 2.86 calls/sec ~ 171 calls/min
MIN_SECONDS_BETWEEN_CALLS = 0.35

# Requests config
TIMEOUT = (10, 60)  # (connect_timeout, read_timeout)
MAX_RETRIES = 6

session = requests.Session()
session.headers.update({
    "Accept": "application/json",
    "Ocp-Apim-Subscription-Key": API_KEY,
})

_last_call_ts = 0.0

def _rate_limit_sleep():
    global _last_call_ts
    now = time.time()
    wait = MIN_SECONDS_BETWEEN_CALLS - (now - _last_call_ts)
    if wait > 0:
        time.sleep(wait)
    _last_call_ts = time.time()

def _parse_date(d: str) -> pd.Timestamp:
    return pd.to_datetime(d, format="%Y-%m-%d")

def fetch_window_daily(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch one window (<= 31 days) of daily data.
    Returns empty DF for 404 (no data).
    Retries on 429 / 5xx / timeouts with backoff.
    """
    url = f"{BASE_URL}/markets/{MARKET}/frequencies/{FREQUENCY}"
    params = {"start_date": start_date, "end_date": end_date}

    backoff = 1.0
    for attempt in range(1, MAX_RETRIES + 1):
        _rate_limit_sleep()
        try:
            r = session.get(url, params=params, timeout=TIMEOUT)
        except (requests.Timeout, requests.ConnectionError):
            if attempt == MAX_RETRIES:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)
            continue

        # No data in this window
        if r.status_code == 404:
            return pd.DataFrame()

        # Rate limited: respect Retry-After if present
        if r.status_code == 429:
            retry_after = float(r.headers.get("Retry-After", "2"))
            time.sleep(retry_after + 0.25)
            continue

        # Transient server errors: backoff + retry
        if 500 <= r.status_code < 600:
            if attempt == MAX_RETRIES:
                r.raise_for_status()
            time.sleep(backoff)
            backoff = min(backoff * 2.0, 30.0)
            continue

        r.raise_for_status()

        data = r.json()
        if isinstance(data, list):
            return pd.DataFrame(data)
        return pd.DataFrame()

    return pd.DataFrame()

def fetch_all_daily(start_date: str, end_date: str, window_days: int = 31) -> pd.DataFrame:
    """
    Fetch daily data across a long span by chunking into <= 31-day windows (inclusive).
    """
    start = _parse_date(start_date)
    end = _parse_date(end_date)
    if end < start:
        raise ValueError("end_date must be >= start_date")

    dfs = []
    cur = start
    while cur <= end:
        win_end = min(cur + pd.Timedelta(days=window_days - 1), end)
        df = fetch_window_daily(cur.strftime("%Y-%m-%d"), win_end.strftime("%Y-%m-%d"))
        if not df.empty:
            dfs.append(df)
        cur = win_end + pd.Timedelta(days=1)

    if not dfs:
        return pd.DataFrame()

    out = pd.concat(dfs, ignore_index=True).drop_duplicates()
    return out

# ============================================================
# 1) Download daily data (2000-2025)
# ============================================================
df_all = fetch_all_daily("2000-01-01", "2025-12-31")
print("Downloaded:", df_all.shape)
print("Columns:", list(df_all.columns))

# ============================================================
# 2) Filter to GVB + ILB, spot, secondary counterparties
# ============================================================
SECONDARY_CPTY = {"REP", "OMM", "BROKNO", "CUSE", "CUFO"}

df_filt = df_all[
    df_all["Asset"].isin(["GVB", "ILB"]) &
    (df_all["Contract"] == "SP") &
    df_all["Counterparty"].isin(SECONDARY_CPTY)
].copy()

df_filt["Period"] = pd.to_datetime(df_filt["Period"], errors="coerce")
df_filt = df_filt.dropna(subset=["Period"]).sort_values("Period")

print("Filtered shape:", df_filt.shape)
print(df_filt.head())
# Start sample from 2003-11-03
df_filt = df_filt[df_filt["Period"] >= "2003-11-03"].copy()

# 1) Count transactions per date and asset
df_counts = (
    df_filt
    .groupby(["Period", "Asset"])
    .size()
    .unstack("Asset")
    .fillna(0)
)

# Ensure both columns exist
for col in ["GVB", "ILB"]:
    if col not in df_counts.columns:
        df_counts[col] = 0

# 2) Total number of transactions per day
df_counts["Total_transactions"] = df_counts["GVB"] + df_counts["ILB"]

# 3) ILB transaction share
df_counts["ILB_tx_share"] = df_counts["ILB"] / df_counts["Total_transactions"]

# Rename Period -> Month
df_counts.index.name = "Month"

# ILB share (transaction count based)
df_counts["ILB_share"] = df_counts["ILB"] / (df_counts["ILB"] + df_counts["GVB"])

# Log share (avoid log(0))
df_counts["liquidity_ma3"] = np.log(
    df_counts["ILB_share"].replace(0, np.nan)
)

# Monthly average
liq_monthly = (
    df_counts["liquidity_ma3"]
    .resample("M")
    .mean()
)

# 3-month moving average
liq_monthly_ma3 = liq_monthly.rolling(3).mean()
liq_monthly_ma3 = liq_monthly_ma3.dropna()
liq_monthly_ma3.to_excel(
    "turnover_monthly_with_liquidity.xlsx",
    sheet_name="turnover_monthly"
)