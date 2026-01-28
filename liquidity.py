
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---- Now to the liqudity variable ----
# start with importing turnover statistics from Riksbanken

### Riksbanken only allow downloading 12 months at a time
### The problem is RIksbanken only allow 5 calls per minute, so we need one call per year and 26 calls 
### so the importaking takes 7 minutes

BASE_URL = "https://api.riksbank.se/turnover-statistics/v1"
MARKET = "fi"
FREQUENCY = "monthly"

SLEEP_BETWEEN_CALLS = 15  # stays under 5 requests/minute
session = requests.Session()
session.headers.update({"Accept": "application/json"})

def ym_add(ym, months):
    y, m = map(int, ym.split("-"))
    total = y * 12 + (m - 1) + months
    ny, nm0 = divmod(total, 12)
    return f"{ny:04d}-{nm0+1:02d}"

def fetch_window(start_ym, end_ym):
    url = f"{BASE_URL}/markets/{MARKET}/frequencies/{FREQUENCY}"
    params = {"start_date": start_ym, "end_date": end_ym}

    r = session.get(url, params=params)
    time.sleep(SLEEP_BETWEEN_CALLS)

    # Some windows may return 404 = no data
    if r.status_code == 404:
        return pd.DataFrame()

    r.raise_for_status()
    return pd.DataFrame(r.json())

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

## -------  end api call funtions   --------------------------------------------------------

# 1) Download (26 calls for 2000–2025)
df_all = fetch_all_monthly("2000-01", "2025-12")
print("Downloaded:", df_all.shape)
print(df_all.head())
print(df_all.columns)


# we only want gvb and IL, no forwards
# Counterparties we keep (secondary market), primary market and riksbanken excluded
SECONDARY_CPTY = {
    "REP",      # dealers
    "OMM",      # other market makers
    "BROKNO",   # interdealer brokers
    "CUSE",     # Swedish customers
    "CUFO"      # Non-Swedish customers
}

df_filt = df_all[
    df_all["Asset"].isin(["GVB", "ILB"]) &
    (df_all["Contract"] == "SP") &
    df_all["Counterparty"].isin(SECONDARY_CPTY)
].copy()

print("Filtered shape:", df_filt.shape)
print(df_filt.head())

# Make sure date is datetime
df_filt["Period"] = pd.to_datetime(df_filt["Period"])

# Month-end timestamp
df_filt["Month"] = df_filt["Period"].dt.to_period("M").dt.to_timestamp("M")

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
plt.show()

turnover_monthly.to_excel(
    "turnover_monthly_with_liquidity.xlsx",
    sheet_name="turnover_monthly"
)