import os
import time
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ── Load NSS RMSE ─────────────────────────────────────────────────────────────
rmse = pd.read_excel(
    "zero_yields_SGBIL.xlsx",
    sheet_name="fit_params",
    usecols=["date", "rmse_yield_bp_ext"]
)
rmse["date"] = pd.to_datetime(rmse["date"]) + pd.offsets.MonthEnd(0)
rmse = rmse.set_index("date")["rmse_yield_bp_ext"].sort_index()
rmse.name = "rmse_bp"
rmse = rmse.dropna()

# ── Regime-adjust: cap at 99th pct of pre-hike-cycle sample ──────────────────
# The 2022-2024 period saw unprecedented Riksbank rate hikes (+425 bps in
# 18 months), which mechanically inflates NSS fitting errors regardless of
# true market liquidity conditions. Large fitting errors during rapid rate
# cycles reflect model-fitting difficulty, not search frictions or market
# dysfunction. Following Alexandersson (2018), we interpret fitting errors
# as liquidity proxies only when they are not driven by rate volatility.
# We therefore cap the RMSE at the 99th percentile of the pre-hike-cycle
# sample (pre-2022), ensuring the index reflects genuine liquidity stress
# episodes (2008-2009, 2011-2012) rather than the monetary policy regime
# shift of 2022-2024.

cap_date      = "2021-12-31"
pre_hike_rmse = rmse.loc[:cap_date]
cap_level     = pre_hike_rmse.quantile(0.99)
print(f"\nRMSE cap level (99th pct pre-2022): {cap_level:.2f} bp")


rmse_capped = rmse.clip(upper=cap_level)
rmse_capped.name = "rmse_bp_capped"

# ── Standardize and shift to non-negative ─────────────────────────────────────
# Standardized so it enters the VAR on a comparable scale to the yield PCs.
# Shifted so minimum = 0, enforcing positivity: illiquidity raises real yields.
composite = (rmse_capped - rmse_capped.mean()) / rmse_capped.std(ddof=0)
composite = composite - composite.min()
composite.name = "composite_liq"
composite = composite.sort_index()

print(f"\ncomposite_liq sample: {composite.index.min().date()} -> {composite.index.max().date()}")
print(f"  N = {len(composite)} months")
print("\nComposite liquidity index summary:")
print(composite.describe().round(4))

# ── Save ──────────────────────────────────────────────────────────────────────
out = pd.DataFrame({
    "rmse_bp":        rmse,
    "rmse_bp_capped": rmse_capped,
    "composite_liq":  composite,
})

out.to_excel(
    "turnover_monthly_with_liquidity.xlsx",
    sheet_name="turnover_monthly",
    index_label="Month"
)
print("\nSaved: turnover_monthly_with_liquidity.xlsx")

# ── Diagnostic plots ──────────────────────────────────────────────────────────
stress_events = [
    ("2008-09-30", "Lehman",    "red"),
    ("2011-08-31", "EZ crisis", "purple"),
    ("2020-03-31", "COVID",     "orange"),
]

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Panel 1: Raw RMSE with cap line
axes[0].plot(rmse.index, rmse.values, color="#e07b39", lw=1.8, label="NSS RMSE (raw)")
axes[0].axhline(cap_level, color="crimson", linestyle="--", lw=1.4,
                label=f"Cap = {cap_level:.1f} bp (99th pct pre-2022)")
for date, label, color in stress_events:
    axes[0].axvline(pd.Timestamp(date), color=color, linestyle=":", lw=1.2, label=label)
axes[0].set_ylabel("Basis points")
axes[0].set_title("Step 1: Raw NSS SGBIL yield fitting error")
axes[0].legend(fontsize=9, ncol=2)
axes[0].yaxis.grid(True, linestyle=":", alpha=0.6)

# Panel 2: Capped RMSE
axes[1].plot(rmse_capped.index, rmse_capped.values, color="#e07b39", lw=1.8,
             label="NSS RMSE (capped at 99th pct pre-2022)")
for date, label, color in stress_events:
    axes[1].axvline(pd.Timestamp(date), color=color, linestyle=":", lw=1.2, label=label)
axes[1].set_ylabel("Basis points")
axes[1].set_title("Step 2: Rate-cycle-adjusted RMSE (2022 spike flattened)")
axes[1].legend(fontsize=9, ncol=2)
axes[1].yaxis.grid(True, linestyle=":", alpha=0.6)

# Panel 3: Final composite_liq
axes[2].fill_between(composite.index, composite.values, alpha=0.25, color="#3aaa6e")
axes[2].plot(composite.index, composite.values, color="#3aaa6e", lw=2.0,
             label="composite_liq (standardized, min-shifted)")
axes[2].axhline(0, color="black", lw=0.7, linestyle=":")
for date, label, color in stress_events:
    axes[2].axvline(pd.Timestamp(date), color=color, linestyle=":", lw=1.2, label=label)
axes[2].set_ylabel("Index level")
axes[2].set_title("Step 3: Final composite_liq — ready for model")
axes[2].legend(fontsize=9, ncol=2)
axes[2].yaxis.grid(True, linestyle=":", alpha=0.6)

fig.suptitle("SGBIL Liquidity Index: Regime-Adjusted NSS RMSE",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("liquidity_index_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: liquidity_index_diagnostics.png")