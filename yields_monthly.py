import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from nelson_siegel_svensson.calibrate import calibrate_nss_ols, calibrate_ns_ols
from nelson_siegel_svensson import NelsonSiegelCurve, NelsonSiegelSvenssonCurve


# ============================================================
# 0) Settings
# ============================================================
BOND_XLSX = "BondCurves.xlsx"

# Choose curve model per block: "NS" or "NSS"
MODEL_NOMINAL = "NSS"   # suggested for nominal (more pillars)
MODEL_REAL    = "NS"    # suggested for real (few pillars)

# Grids in MONTHS (what you asked for)
GRID_NOMINAL_MONTHS = np.arange(6, 121, 1)     # 6..120 months
GRID_REAL_MONTHS    = np.arange(24, 121, 1)    # 24..120 months

# Output files
OUTPUT_XLSX_NOM = "nominal_eom_zero_coupon_yields_monthly_grid.xlsx"
OUTPUT_XLSX_REAL = "real_eom_zero_coupon_yields_monthly_grid.xlsx"


# ============================================================
# 1) Read and prep raw data
# ============================================================
xls = pd.ExcelFile(BOND_XLSX)
print(xls.sheet_names)  # sanity check

df_il  = pd.read_excel(xls, sheet_name=0)  # BEI + nominal pillars
df_nom = pd.read_excel(xls, sheet_name=1)  # nominal ZC pillars (0.5y..10y)

# --- Rename IL sheet columns (BEI + nominal ZC pillars) ---
df_il = df_il.rename(columns={
    "Data": "date",
    "Sweden, Break-Even Inflation Spot Rate, 1 Year": "bei_1y",
    "Sweden, Break-Even Inflation Spot Rate, 2 Years": "bei_2y",
    "Sweden, Break-Even Inflation Spot Rate, 5 Years": "bei_5y",
    "Sweden, Break-Even Inflation Spot Rate, 10 Years": "bei_10y",
    "Sweden, Government bonds, Zero coupon yield, 1 year": "zc_nom_1y",
    "Sweden, Government bonds, Zero coupon yield, 2 years": "zc_nom_2y",
    "Sweden, Government bonds, Zero coupon yield, 5 years": "zc_nom_5y",
    "Sweden, Government bonds, Zero coupon yield, 10 years": "zc_nom_10y",
})

# Build real ZC yields from nominal ZC minus BEI
df_il["zc_real_1y"]  = df_il["zc_nom_1y"]  - df_il["bei_1y"]
df_il["zc_real_2y"]  = df_il["zc_nom_2y"]  - df_il["bei_2y"]
df_il["zc_real_5y"]  = df_il["zc_nom_5y"]  - df_il["bei_5y"]
df_il["zc_real_10y"] = df_il["zc_nom_10y"] - df_il["bei_10y"]

df_il = df_il[["date", "zc_real_1y", "zc_real_2y", "zc_real_5y", "zc_real_10y"]]

# --- Rename nominal sheet columns (pillars 0.5y..10y) ---
df_nom = df_nom.rename(columns={
    "Date": "date",
    "Sweden, Government bonds, Zero coupon yield, 6 months": "y_0.5y",
    "Sweden, Government bonds, Zero coupon yield, 1 year": "y_1y",
    "Sweden, Government bonds, Zero coupon yield, 2 years": "y_2y",
    "Sweden, Government bonds, Zero coupon yield, 3 years": "y_3y",
    "Sweden, Government bonds, Zero coupon yield, 4 years": "y_4y",
    "Sweden, Government bonds, Zero coupon yield, 5 years": "y_5y",
    "Sweden, Government bonds, Zero coupon yield, 6 years": "y_6y",
    "Sweden, Government bonds, Zero coupon yield, 7 years": "y_7y",
    "Sweden, Government bonds, Zero coupon yield, 8 years": "y_8y",
    "Sweden, Government bonds, Zero coupon yield, 9 years": "y_9y",
    "Sweden, Government bonds, Zero coupon yield, 10 years": "y_10y",
})

df_nom = df_nom[
    ["date",
     "y_0.5y", "y_1y", "y_2y", "y_3y", "y_4y",
     "y_5y", "y_6y", "y_7y", "y_8y", "y_9y", "y_10y"]
]

# Date handling
df_il["date"]  = pd.to_datetime(df_il["date"], errors="coerce")
df_nom["date"] = pd.to_datetime(df_nom["date"], errors="coerce")

df_il  = df_il.dropna(subset=["date"]).set_index("date").sort_index()
df_nom = df_nom.dropna(subset=["date"]).set_index("date").sort_index()

# End-of-month sampling
df_il_eom  = df_il.resample("M").last()
df_nom_eom = df_nom.resample("M").last()


# ============================================================
# 2) Maturity maps (pillars) in YEARS
# ============================================================
tau_real = {
    "zc_real_1y": 1.0,
    "zc_real_2y": 2.0,
    "zc_real_5y": 5.0,
    "zc_real_10y": 10.0,
}

tau_nom = {
    "y_0.5y": 0.5,
    "y_1y": 1.0,
    "y_2y": 2.0,
    "y_3y": 3.0,
    "y_4y": 4.0,
    "y_5y": 5.0,
    "y_6y": 6.0,
    "y_7y": 7.0,
    "y_8y": 8.0,
    "y_9y": 9.0,
    "y_10y": 10.0,
}

# Convert desired grids in months -> years for the curve evaluation
grid_nom_years = GRID_NOMINAL_MONTHS / 12.0
grid_real_years = GRID_REAL_MONTHS / 12.0


# ============================================================
# 3) Fit helpers (same idea as yours, but grid is passed in YEARS and
#    output columns are in MONTHS)
# ============================================================
def fit_nss_panel_monthgrid(df_eom_panel, tau_map, grid_years, grid_months, tau0=(2.0, 5.0), min_points=4):
    """
    NSS fit date-by-date. Returns:
      params_df: beta0,beta1,beta2,beta3,tau1,tau2
      fitted_df: yields on month grid columns y_6m ... y_120m (in %)

    min_points:
      For NSS you generally want >=4 points. With fewer, NSS is often unstable.
    """
    cols = [c for c in tau_map.keys() if c in df_eom_panel.columns]
    t_all = np.array([tau_map[c] for c in cols], float)

    params, fitted = [], []
    tau_prev = tau0

    for dt, row in df_eom_panel[cols].iterrows():
        y_raw = pd.to_numeric(row, errors="coerce").to_numpy(float)
        mask = np.isfinite(y_raw)

        if mask.sum() < min_points:
            params.append([np.nan] * 6)
            fitted.append([np.nan] * len(grid_years))
            continue

        t = t_all[mask]
        y = y_raw[mask] / 100.0  # % -> decimal

        curve, status = calibrate_nss_ols(t, y, tau0=tau_prev)
        if not status.success:
            params.append([np.nan] * 6)
            fitted.append([np.nan] * len(grid_years))
            tau_prev = tau0
            continue

        params.append([curve.beta0, curve.beta1, curve.beta2, curve.beta3, curve.tau1, curve.tau2])
        fitted.append((curve(grid_years) * 100.0).tolist())  # back to %

        tau_prev = (curve.tau1, curve.tau2)

    params_df = pd.DataFrame(params, index=df_eom_panel.index,
                             columns=["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"])
    fitted_df = pd.DataFrame(fitted, index=df_eom_panel.index,
                             columns=[f"y_{m}m" for m in grid_months])
    return params_df, fitted_df


def fit_ns_panel_monthgrid(df_eom_panel, tau_map, grid_years, grid_months, tau0=1.5, min_points=3):
    """
    NS fit date-by-date. Returns:
      params_df: beta0,beta1,beta2,tau
      fitted_df: yields on month grid columns y_6m ... y_120m (in %)
    """
    cols = [c for c in tau_map.keys() if c in df_eom_panel.columns]
    t_all = np.array([tau_map[c] for c in cols], float)

    params, fitted = [], []

    for dt, row in df_eom_panel[cols].iterrows():
        y_raw = pd.to_numeric(row, errors="coerce").to_numpy(float)
        mask = np.isfinite(y_raw)

        if mask.sum() < min_points:
            params.append([np.nan] * 4)
            fitted.append([np.nan] * len(grid_years))
            continue

        t = t_all[mask]
        y = y_raw[mask] / 100.0  # % -> decimal

        curve, status = calibrate_ns_ols(t, y, tau0=tau0)
        if not status.success:
            params.append([np.nan] * 4)
            fitted.append([np.nan] * len(grid_years))
            continue

        params.append([curve.beta0, curve.beta1, curve.beta2, curve.tau])
        fitted.append((curve(grid_years) * 100.0).tolist())  # back to %

    params_df = pd.DataFrame(params, index=df_eom_panel.index,
                             columns=["beta0", "beta1", "beta2", "tau"])
    fitted_df = pd.DataFrame(fitted, index=df_eom_panel.index,
                             columns=[f"y_{m}m" for m in grid_months])
    return params_df, fitted_df


# ============================================================
# 4) Fit REAL curve on 24..120m grid
# ============================================================
if MODEL_REAL.upper() == "NS":
    params_real, fitted_real = fit_ns_panel_monthgrid(
        df_eom_panel=df_il_eom,
        tau_map=tau_real,
        grid_years=grid_real_years,
        grid_months=GRID_REAL_MONTHS,
        tau0=1.5,
        min_points=3
    )
elif MODEL_REAL.upper() == "NSS":
    params_real, fitted_real = fit_nss_panel_monthgrid(
        df_eom_panel=df_il_eom,
        tau_map=tau_real,
        grid_years=grid_real_years,
        grid_months=GRID_REAL_MONTHS,
        tau0=(2.0, 5.0),
        min_points=4
    )
else:
    raise ValueError("MODEL_REAL must be 'NS' or 'NSS'")

print("Real curve fitted on months:", GRID_REAL_MONTHS[0], "..", GRID_REAL_MONTHS[-1])


# ============================================================
# 5) Fit NOMINAL curve on 6..120m grid  (NEW: you asked for this)
# ============================================================
if MODEL_NOMINAL.upper() == "NS":
    params_nom, fitted_nom = fit_ns_panel_monthgrid(
        df_eom_panel=df_nom_eom,
        tau_map=tau_nom,
        grid_years=grid_nom_years,
        grid_months=GRID_NOMINAL_MONTHS,
        tau0=1.5,
        min_points=3
    )
elif MODEL_NOMINAL.upper() == "NSS":
    params_nom, fitted_nom = fit_nss_panel_monthgrid(
        df_eom_panel=df_nom_eom,
        tau_map=tau_nom,
        grid_years=grid_nom_years,
        grid_months=GRID_NOMINAL_MONTHS,
        tau0=(2.0, 5.0),
        min_points=4
    )
else:
    raise ValueError("MODEL_NOMINAL must be 'NS' or 'NSS'")

print("Nominal curve fitted on months:", GRID_NOMINAL_MONTHS[0], "..", GRID_NOMINAL_MONTHS[-1])


# ============================================================
# 6) Quick sanity plots (optional)
# ============================================================
# Plot a few sample dates of nominal fitted curve (dense-ish monthly grid already)
sample_dates = fitted_nom.index[::12]  # every 12 months
plt.figure(figsize=(10, 6))
for dt in sample_dates[:6]:
    y = fitted_nom.loc[dt].to_numpy(float)
    plt.plot(GRID_NOMINAL_MONTHS, y, alpha=0.8)
plt.title(f"Nominal fitted ZC yields ({MODEL_NOMINAL}) on monthly maturity grid")
plt.xlabel("Maturity (months)")
plt.ylabel("Yield (%)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
for dt in fitted_real.index[::12][:6]:
    y = fitted_real.loc[dt].to_numpy(float)
    plt.plot(GRID_REAL_MONTHS, y, alpha=0.8)
plt.title(f"Real fitted ZC yields ({MODEL_REAL}) on monthly maturity grid")
plt.xlabel("Maturity (months)")
plt.ylabel("Yield (%)")
plt.tight_layout()
plt.show()


# ============================================================
# 7) Export
# ============================================================
# Export fitted monthly-grid yields (in %)
fitted_nom.to_excel(OUTPUT_XLSX_NOM, sheet_name="yields_monthgrid")
params_nom.to_excel(OUTPUT_XLSX_NOM.replace(".xlsx", "_params.xlsx"), sheet_name="params")
print(f"Wrote: {OUTPUT_XLSX_NOM}")
print(f"Wrote: {OUTPUT_XLSX_NOM.replace('.xlsx','_params.xlsx')}")

fitted_real.to_excel(OUTPUT_XLSX_REAL, sheet_name="yields_monthgrid")
params_real.to_excel(OUTPUT_XLSX_REAL.replace(".xlsx", "_params.xlsx"), sheet_name="params")
print(f"Wrote: {OUTPUT_XLSX_REAL}")
print(f"Wrote: {OUTPUT_XLSX_REAL.replace('.xlsx','_params.xlsx')}")
