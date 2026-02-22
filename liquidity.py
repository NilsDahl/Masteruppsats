import os
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Now to the liquidity variable ----
# Turnover statistics from Riksbanken

BASE_URL = "https://api.riksbank.se/turnover-statistics/v1"
MARKET = "fi"
FREQUENCY = "monthly"

# Put your key in your shell before running:
os.environ["RIKSBANK_API_KEY"] = "0460e512bfd8437598e82f4df5aa3437"
API_KEY = os.getenv("RIKSBANK_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing RIKSBANK_API_KEY env var. Set it before running.")

session = requests.Session()
session.headers.update({
    "Accept": "application/json",
    "Ocp-Apim-Subscription-Key": API_KEY
})

def ym_add(ym, months):
    y, m = map(int, ym.split("-"))
    total = y * 12 + (m - 1) + months
    ny, nm0 = divmod(total, 12)
    return f"{ny:04d}-{nm0+1:02d}"

def fetch_window(start_ym, end_ym, max_retries=6):
    url = f"{BASE_URL}/markets/{MARKET}/frequencies/{FREQUENCY}"
    params = {"start_date": start_ym, "end_date": end_ym}

    for attempt in range(max_retries):
        r = session.get(url, params=params, timeout=60)

        if r.status_code == 404:
            return pd.DataFrame()

        if r.status_code == 429:
            # Respect server guidance if provided
            retry_after = r.headers.get("Retry-After")
            wait = int(retry_after) if (retry_after and retry_after.isdigit()) else (1 + attempt)
            time.sleep(wait)
            continue

        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data if isinstance(data, list) else [])

    raise RuntimeError(f"Too many 429s for window {start_ym} -> {end_ym}")

def fetch_all_monthly(start_ym, end_ym):
    dfs = []
    cur = start_ym

    while True:
        win_end = ym_add(cur, 11)
        if win_end > end_ym:
            win_end = end_ym

        df = fetch_window(cur, win_end)
        if not df.empty:
            dfs.append(df)

        if win_end == end_ym:
            break
        cur = ym_add(win_end, 1)

    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True).drop_duplicates()

# 1) Download (26 calls for 2000–2025)
df_all = fetch_all_monthly("2000-01", "2025-12")
print("Downloaded:", df_all.shape)
print(df_all.head())
print(df_all.columns)

# we only want GVB and ILB, no forwards
SECONDARY_CPTY = {"REP", "OMM", "BROKNO", "CUSE", "CUFO"}

df_filt = df_all[
    df_all["Asset"].isin(["GVB", "ILB"]) &
    (df_all["Contract"] == "SP") &
    df_all["Counterparty"].isin(SECONDARY_CPTY)
].copy()

print("Filtered shape:", df_filt.shape)
print(df_filt.head())

# Make sure date is datetime
df_filt["Period"] = pd.to_datetime(df_filt["Period"], errors="coerce")
df_filt = df_filt.dropna(subset=["Period"])

# Month-end timestamp
df_filt["Month"] = df_filt["Period"].dt.to_period("M").dt.to_timestamp("M")

counts = (
    df_filt
    .groupby(["Month", "Asset"])
    .size()
    .unstack(fill_value=0)
    .sort_index()
)

counts = counts.reindex(columns=["ILB", "GVB"], fill_value=0)
counts["L_share"] = counts["ILB"] / (counts["ILB"] + counts["GVB"])

# Aggregate turnover
turnover_monthly = (
    df_filt
    .groupby(["Month", "Asset"])["Amount"]
    .sum()
    .unstack("Asset")
    .sort_index()
)

print(turnover_monthly.head())

# Liquidity: higher = more illiquid ILB relative to GVB
turnover_monthly["liquidity"] = -np.log(
    turnover_monthly["ILB"] / turnover_monthly["GVB"]
)

# 3-month moving average
turnover_monthly["liquidity_ma3"] = (
    turnover_monthly["liquidity"]
    .rolling(window=3, min_periods=2)
    .mean()
)

plt.figure(figsize=(10,5))
plt.plot(turnover_monthly.index, turnover_monthly["liquidity"], alpha=0.4, label="Raw")
plt.plot(turnover_monthly.index, turnover_monthly["liquidity_ma3"], lw=2, label="3m MA")
plt.title("Relative Illiquidity (ILB vs GVB)")
plt.legend()
plt.tight_layout()
plt.show()

turnover_monthly.to_excel(
    "turnover_monthly_with_liquidity.xlsx",
    sheet_name="turnover_monthly"
)