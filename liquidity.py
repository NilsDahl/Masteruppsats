import os
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_URL  = "https://api.riksbank.se/turnover-statistics/v1"
MARKET    = "fi"
FREQUENCY = "monthly"

os.environ["RIKSBANK_API_KEY"] = "0460e512bfd8437598e82f4df5aa3437"
API_KEY = os.getenv("RIKSBANK_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing RIKSBANK_API_KEY env var.")

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

# Download
df_all = fetch_all_monthly("2000-01", "2025-12")
print("Downloaded:", df_all.shape)

SECONDARY_CPTY = {"REP", "OMM", "BROKNO", "CUSE", "CUFO"}

df_filt = df_all[
    df_all["Asset"].isin(["GVB", "ILB"]) &
    (df_all["Contract"] == "SP") &
    df_all["Counterparty"].isin(SECONDARY_CPTY)
].copy()

df_filt["Period"] = pd.to_datetime(df_filt["Period"], errors="coerce")
df_filt = df_filt.dropna(subset=["Period"])
df_filt["Month"] = df_filt["Period"].dt.to_period("M").dt.to_timestamp("M")
df_filt["Month"] = df_filt["Month"] + pd.offsets.MonthEnd(0)

# Aggregate turnover - Indicator 2
turnover_monthly = (
    df_filt
    .groupby(["Month", "Asset"])["Amount"]
    .sum()
    .unstack("Asset")
    .sort_index()
)

turnover_monthly["liquidity"] = -np.log(
    turnover_monthly["ILB"] / turnover_monthly["GVB"]
)
turnover_monthly["liquidity_ma3"] = (
    turnover_monthly["liquidity"]
    .rolling(window=3, min_periods=2)
    .mean()
)# Load Indicator 1: NSS yield fitting errors
rmse = pd.read_excel(
    "zero_yields_SGBIL.xlsx",
    sheet_name="fit_params",
    usecols=["date", "rmse_yield_bp_ext"]
)
rmse["date"] = pd.to_datetime(rmse["date"]) + pd.offsets.MonthEnd(0)
rmse = rmse.set_index("date")["rmse_yield_bp_ext"].sort_index()
rmse.name = "rmse_bp"
 
# Load Indicator 3: Bid-ask spread ratio (from Stens excel)
spread_df = pd.read_excel(
    "price_factors.xlsx",
    sheet_name="liquidity",
    usecols=[0, 1],   # column A = Date, column B = your series
)
spread_df.columns = ["Date", "ILLIQUIDITY_REAL"]  # rename for consistency
spread_df["Date"] = pd.to_datetime(spread_df["Date"]) + pd.offsets.MonthEnd(0)
spread_df = spread_df.dropna(subset=["ILLIQUIDITY_REAL"])
spread_df = spread_df.set_index("Date")["ILLIQUIDITY_REAL"].sort_index()
spread_df.name = "illiquidity_real"
 
# Align all three indicators on common index
ind1 = rmse.dropna()
ind2 = turnover_monthly["liquidity_ma3"].dropna()
ind3 = spread_df.dropna()
common_idx = ind1.index.intersection(ind2.index).intersection(ind3.index)

ind1 = ind1.loc[common_idx]
ind2 = ind2.loc[common_idx]
ind3 = ind3.loc[common_idx]
 
print(f"\nComposite index sample: {common_idx.min().date()} -> {common_idx.max().date()}")
print(f"  N = {len(common_idx)} months")
 
# Standardize all three indicators
ind1_z = (ind1 - ind1.mean()) / ind1.std(ddof=0)
ind2_z = (ind2 - ind2.mean()) / ind2.std(ddof=0)
ind3_z = ind3
 
# Weights: ind3 (bid-ask spread) = 50%, ind1 (NSS RMSE) = 25%, ind2 (turnover) = 25%
composite = 1/3 * ind1_z + 1/3 * ind2_z + 1/3 * ind3_z
composite = (composite - composite.min())   
composite.name = "Liquidity_MA3"
composite = composite.sort_index()

print("\nComposite liquidity index summary:")
print(composite.describe().round(4))
 
# Save
turnover_monthly["rmse_bp"]           = rmse
turnover_monthly["illiquidity_real"]   = spread_df
turnover_monthly["composite_liq"]      = composite
 
turnover_monthly.to_excel(
    "turnover_monthly_with_liquidity.xlsx",
    sheet_name="turnover_monthly"
)
print("\nSaved: turnover_monthly_with_liquidity.xlsx")
 
# Diagnostic plots — 4 panels
fig, axes = plt.subplots(4, 1, figsize=(12, 13), sharex=True)
 
axes[0].plot(ind1.index, ind1.values, color="#e07b39", lw=1.8, label="NSS RMSE (bp)")
axes[0].set_ylabel("Basis points")
axes[0].set_title("Indicator 1 (weight 33%): NS yield fitting error (SGBIL)")
axes[0].legend(fontsize=10)
axes[0].yaxis.grid(True, linestyle=":", alpha=0.6)
 
axes[1].plot(ind2.index, ind2.values, color="#2a6ebb", lw=1.8, label="Turnover ratio (3m MA)")
axes[1].set_ylabel("−log(ILB/GVB)")
axes[1].set_title("Indicator 2 (weight 33%): Relative turnover illiquidity")
axes[1].legend(fontsize=10)
axes[1].yaxis.grid(True, linestyle=":", alpha=0.6)
 
axes[2].plot(ind3.index, ind3.values, color="#8b5cf6", lw=1.8, label="Bid-ask spread illiquidity (real)")
axes[2].set_ylabel("Spread level")
axes[2].set_title("Indicator 3 (weight 33%): Bid-ask spread illiquidity (real)")
axes[2].legend(fontsize=10)
axes[2].yaxis.grid(True, linestyle=":", alpha=0.6)
 
axes[3].fill_between(composite.index, composite.values, alpha=0.25, color="#3aaa6e")
axes[3].plot(composite.index, composite.values, color="#3aaa6e", lw=2.0,
             label="Composite liquidity index")
axes[3].axhline(0, color="black", lw=0.7, linestyle=":")
axes[3].set_ylabel("Index level")
axes[3].set_title("Composite SGBIL Liquidity Index (33% / 33% / 33%)")
axes[3].legend(fontsize=10)
axes[3].yaxis.grid(True, linestyle=":", alpha=0.6)
 
fig.suptitle("Swedish ILB Liquidity Index Construction", fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("liquidity_index_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: liquidity_index_diagnostics.png")

