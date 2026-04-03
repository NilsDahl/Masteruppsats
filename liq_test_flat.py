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

# ── Flatten post-hike-cycle period ────────────────────────────────────────────
# From 2022 onward the Riksbank hiked aggressively (+425 bps in 18 months),
# causing SGBIL NSS fitting errors to spike mechanically. This spike reflects
# two non-liquidity forces: (1) inflation uncertainty making real cash flows
# harder to model, and (2) pure curve-fitting difficulty when the yield level
# shifts rapidly. We therefore treat the post-2021 RMSE as uninformative for
# liquidity identification and hold the index flat at its December 2021 value.
# This is equivalent to assuming the structural liquidity conditions of the
# SGBIL market in 2022-2025 are unknown from the RMSE signal alone, and
# avoids contaminating the LP estimate with inflation and rate-regime effects.

flatten_date = "2021-12-31"
flat_value   = rmse.loc[:flatten_date].iloc[-1]   # last observed pre-hike value
print(f"\nFlat value from {flatten_date}: {flat_value:.2f} bp")

rmse_flat = rmse.copy()
rmse_flat.loc[rmse_flat.index > pd.Timestamp(flatten_date)] = flat_value
rmse_flat.name = "rmse_bp_flat"

# ── Standardize and shift to non-negative ─────────────────────────────────────
# Standardize using the full (flattened) series so scale is consistent.
# Shift so minimum = 0, enforcing positivity of the liquidity index.
composite = (rmse_flat - rmse_flat.mean()) / rmse_flat.std(ddof=0)
composite = composite - composite.min()
composite.name = "composite_liq"
composite = composite.sort_index()

print(f"\ncomposite_liq sample: {composite.index.min().date()} -> {composite.index.max().date()}")
print(f"  N = {len(composite)} months")
print("\nComposite liquidity index summary:")
print(composite.describe().round(4))

# Spot check: values at key dates
for label, date in [
    ("Pre-Lehman  (2008-08)", "2008-08-31"),
    ("Lehman      (2008-10)", "2008-10-31"),
    ("EZ crisis   (2011-09)", "2011-09-30"),
    ("COVID       (2020-03)", "2020-03-31"),
    ("Hike start  (2022-02)", "2022-02-28"),
    ("Hike peak   (2023-06)", "2023-06-30"),
]:
    ts = pd.Timestamp(date)
    if ts in composite.index:
        print(f"  {label}: {composite.loc[ts]:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
out = pd.DataFrame({
    "rmse_bp":        rmse,
    "rmse_bp_flat":   rmse_flat,
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
    ("2008-09-30", "Lehman",       "red"),
    ("2011-08-31", "EZ crisis",    "purple"),
    ("2020-03-31", "COVID",        "orange"),
    ("2022-01-31", "Hike cycle",   "navy"),
]

fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Panel 1: Raw RMSE
axes[0].plot(rmse.index, rmse.values, color="#e07b39", lw=1.8, label="NSS RMSE (raw)")
axes[0].axvline(pd.Timestamp(flatten_date), color="navy", linestyle="--",
                lw=1.4, label="Flatten from here (2022-01)")
for date, label, color in stress_events:
    axes[0].axvline(pd.Timestamp(date), color=color, linestyle=":", lw=1.2)
axes[0].set_ylabel("Basis points")
axes[0].set_title("Step 1: Raw NSS SGBIL yield fitting error")
axes[0].legend(fontsize=9, ncol=2)
axes[0].yaxis.grid(True, linestyle=":", alpha=0.6)

# Panel 2: Flattened RMSE
axes[1].plot(rmse_flat.index, rmse_flat.values, color="#e07b39", lw=1.8,
             label=f"NSS RMSE (flat at {flat_value:.1f} bp post-2021)")
axes[1].axvline(pd.Timestamp(flatten_date), color="navy", linestyle="--", lw=1.4)
for date, label, color in stress_events:
    axes[1].axvline(pd.Timestamp(date), color=color, linestyle=":", lw=1.2,
                    label=label)
axes[1].set_ylabel("Basis points")
axes[1].set_title("Step 2: Post-hike-cycle RMSE held flat at Dec 2021 level")
axes[1].legend(fontsize=9, ncol=2)
axes[1].yaxis.grid(True, linestyle=":", alpha=0.6)

# Panel 3: Final composite_liq
axes[2].fill_between(composite.index, composite.values, alpha=0.25, color="#3aaa6e")
axes[2].plot(composite.index, composite.values, color="#3aaa6e", lw=2.0,
             label="composite_liq (standardized, min-shifted)")
axes[2].axhline(0, color="black", lw=0.7, linestyle=":")
axes[2].axvline(pd.Timestamp(flatten_date), color="navy", linestyle="--", lw=1.4,
                label="Post-2021 flattened")
for date, label, color in stress_events[:-1]:   # skip hike cycle marker, already shown
    axes[2].axvline(pd.Timestamp(date), color=color, linestyle=":", lw=1.2,
                    label=label)
axes[2].set_ylabel("Index level")
axes[2].set_title("Step 3: Final composite_liq — ready for model")
axes[2].legend(fontsize=9, ncol=2)
axes[2].yaxis.grid(True, linestyle=":", alpha=0.6)

fig.suptitle("SGBIL Liquidity Index: NSS RMSE with Post-2021 Flat Specification",
             fontsize=13, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("liquidity_index_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: liquidity_index_diagnostics.png")