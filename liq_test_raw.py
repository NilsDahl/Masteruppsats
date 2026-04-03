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

# ── Standardize and shift to non-negative ─────────────────────────────────────
composite = (rmse - rmse.mean()) / rmse.std(ddof=0)
composite = composite - composite.min()
composite.name = "composite_liq"

# ── Save ──────────────────────────────────────────────────────────────────────
out = pd.DataFrame({
    "rmse_bp":       rmse,
    "composite_liq": composite,
})

out.to_excel(
    "turnover_monthly_with_liquidity.xlsx",
    sheet_name="turnover_monthly",
    index_label="Month"
)
print("Saved: turnover_monthly_with_liquidity.xlsx")
print(composite.describe().round(4))

# ── Diagnostic plot ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)

axes[0].plot(rmse.index, rmse.values, color="#e07b39", lw=1.8, label="NSS RMSE (bp, raw)")
axes[0].set_ylabel("Basis points")
axes[0].set_title("NSS SGBIL yield fitting error (raw)")
axes[0].legend(fontsize=10)
axes[0].yaxis.grid(True, linestyle=":", alpha=0.6)

axes[1].fill_between(composite.index, composite.values, alpha=0.25, color="#3aaa6e")
axes[1].plot(composite.index, composite.values, color="#3aaa6e", lw=2.0,
             label="composite_liq (standardized, min-shifted)")
axes[1].axhline(0, color="black", lw=0.7, linestyle=":")
axes[1].set_ylabel("Index level")
axes[1].set_title("composite_liq — ready for model")
axes[1].legend(fontsize=10)
axes[1].yaxis.grid(True, linestyle=":", alpha=0.6)

for ax in axes:
    for date, label, color in [
        ("2008-09-30", "Lehman",    "red"),
        ("2011-08-31", "EZ crisis", "purple"),
        ("2020-03-31", "COVID",     "orange"),
    ]:
        ax.axvline(pd.Timestamp(date), color=color, linestyle=":", lw=1.2, label=label)

axes[0].legend(fontsize=9, ncol=2)
axes[1].legend(fontsize=9, ncol=2)

plt.tight_layout()
plt.savefig("liquidity_index_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: liquidity_index_diagnostics.png")