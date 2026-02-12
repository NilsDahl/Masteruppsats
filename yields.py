import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from nelson_siegel_svensson.calibrate import calibrate_nss_ols, calibrate_ns_ols
from nelson_siegel_svensson import NelsonSiegelCurve, NelsonSiegelSvenssonCurve


# ----------------------------
# Read and prep
# ----------------------------
xls = pd.ExcelFile("BondCurves.xlsx")
print(xls.sheet_names)  # sanity check
df_il = pd.read_excel(xls, sheet_name=0)
df_nom = pd.read_excel(xls, sheet_name=1)

df_il = df_il.rename(columns={
    "Data": "date",
    # Break-even inflation
    "Sweden, Break-Even Inflation Spot Rate, 1 Year": "bei_1y",
    "Sweden, Break-Even Inflation Spot Rate, 2 Years": "bei_2y",
    "Sweden, Break-Even Inflation Spot Rate, 5 Years": "bei_5y",
    "Sweden, Break-Even Inflation Spot Rate, 10 Years": "bei_10y",
    # Nominal zero-coupon yields
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

df_il["date"] = pd.to_datetime(df_il["date"])
df_nom["date"] = pd.to_datetime(df_nom["date"])

df_il = df_il.dropna(subset=["date"]).set_index("date").sort_index()
df_nom = df_nom.dropna(subset=["date"]).set_index("date").sort_index()

df_il_eom = df_il.resample("M").last()
df_nom_eom = df_nom.resample("M").last()

final_nominal_eom_zc_yields = df_nom_eom


# ----------------------------
# Real (IL) setup
# ----------------------------
tau_il = {
    "zc_real_1y": 1.0,
    "zc_real_2y": 2.0,
    "zc_real_5y": 5.0,
    "zc_real_10y": 10.0,
}
grid_il = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], float)


# ----------------------------
# Fit helpers
# ----------------------------
def fit_nss_panel(df_eom_panel, tau_map, grid, tau0=(2.0, 5.0), min_points=3):
    """
    NSS fit date-by-date with warm-start of (tau1,tau2).
    min_points default set to 3 to allow fitting when one pillar is missing.
    NOTE: NSS with only 3 points can be unstable; NS is usually preferred in that case.
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
            fitted.append([np.nan] * len(grid))
            continue

        t = t_all[mask]
        y = y_raw[mask] / 100.0  # % -> decimal

        curve, status = calibrate_nss_ols(t, y, tau0=tau_prev)
        if not status.success:
            params.append([np.nan] * 6)
            fitted.append([np.nan] * len(grid))
            tau_prev = tau0
            continue

        params.append([curve.beta0, curve.beta1, curve.beta2, curve.beta3, curve.tau1, curve.tau2])
        fitted.append((curve(grid) * 100.0).tolist())  # back to %

        tau_prev = (curve.tau1, curve.tau2)

    params_df = pd.DataFrame(params, index=df_eom_panel.index,
                             columns=["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"])
    fitted_df = pd.DataFrame(fitted, index=df_eom_panel.index,
                             columns=[f"y_{m:g}y" for m in grid])
    return params_df, fitted_df


def fit_ns_panel(df_eom_panel, tau_map, grid, tau0=1.5, min_points=3):
    """
    NS fit date-by-date; tau is estimated (tau0 is initial guess).
    min_points default set to 3 to allow fitting when one pillar is missing.
    """
    cols = [c for c in tau_map.keys() if c in df_eom_panel.columns]
    t_all = np.array([tau_map[c] for c in cols], float)

    params, fitted = [], []

    for dt, row in df_eom_panel[cols].iterrows():
        y_raw = pd.to_numeric(row, errors="coerce").to_numpy(float)
        mask = np.isfinite(y_raw)

        if mask.sum() < min_points:
            params.append([np.nan] * 4)
            fitted.append([np.nan] * len(grid))
            continue

        t = t_all[mask]
        y = y_raw[mask] / 100.0  # % -> decimal

        curve, status = calibrate_ns_ols(t, y, tau0=tau0)
        if not status.success:
            params.append([np.nan] * 4)
            fitted.append([np.nan] * len(grid))
            continue

        params.append([curve.beta0, curve.beta1, curve.beta2, curve.tau])
        fitted.append((curve(grid) * 100.0).tolist())  # back to %

    params_df = pd.DataFrame(params, index=df_eom_panel.index,
                             columns=["beta0", "beta1", "beta2", "tau"])
    fitted_df = pd.DataFrame(fitted, index=df_eom_panel.index,
                             columns=[f"y_{m:g}y" for m in grid])
    return params_df, fitted_df


def rmse_panel(actual, fitted, col_map):
    rmses = []
    actual_cols = list(col_map.keys())
    fitted_cols = list(col_map.values())

    common_idx = actual.index.intersection(fitted.index)
    for dt in common_idx:
        y_true = actual.loc[dt, actual_cols].to_numpy(float)
        y_hat = fitted.loc[dt, fitted_cols].to_numpy(float)

        mask = np.isfinite(y_true) & np.isfinite(y_hat)
        if mask.sum() == 0:
            continue

        rmses.append(np.sqrt(np.mean((y_true[mask] - y_hat[mask]) ** 2)))

    return float(np.mean(rmses)) if len(rmses) else np.nan


col_map_real = {
    "zc_real_1y": "y_1y",
    "zc_real_2y": "y_2y",
    "zc_real_5y": "y_5y",
    "zc_real_10y": "y_10y",
}


# ----------------------------
# Run grids and keep everything from the loops
# ----------------------------
ns_tau_grid = [1.0, 1.5, 2.0]
nss_tau_grid = [(1.0, 5.0), (2.0, 5.0), (2.0, 7.0), (2.0, 10.0)]

fits = {}
rmse_results = {}

# NS
for tau0 in ns_tau_grid:
    name = f"REAL NS est τ (tau0={tau0:g})"
    params, fitted = fit_ns_panel(df_il_eom, tau_il, grid_il, tau0=tau0)  # min_points default = 3
    fits[name] = (params, fitted)
    rmse_results[name] = rmse_panel(df_il_eom, fitted, col_map_real)

# NSS 
for (t1, t2) in nss_tau_grid:
    name = f"REAL NSS (tau0=({t1:g},{t2:g}))"
    params, fitted = fit_nss_panel(df_il_eom, tau_il, grid_il, tau0=(t1, t2))  # min_points default = 3
    fits[name] = (params, fitted)
    rmse_results[name] = rmse_panel(df_il_eom, fitted, col_map_real)

rmse_table = pd.Series(rmse_results).sort_values()
print(rmse_table)


# ----------------------------
# Plotting helpers
# ----------------------------
def pick_dates_every_n_months(df, step_months=12):
    idx = df.index.sort_values()
    return idx[::step_months] if len(idx) else []


def plot_curve_samples_from_params(
    df_actual,
    params_df,
    tau_map,
    dates,
    title,
    model="NS",  # "NS" or "NSS"
    dense_grid=None,
):
    if dense_grid is None:
        dense_grid = np.linspace(0.25, 11, 250)

    pillar_cols = [c for c in tau_map.keys() if c in df_actual.columns]
    pillar_t = np.array([tau_map[c] for c in pillar_cols], float)

    plt.figure(figsize=(10, 6))

    for dt in dates:
        if dt not in df_actual.index or dt not in params_df.index:
            continue

        y_pillars = df_actual.loc[dt, pillar_cols].to_numpy(float)
        row = params_df.loc[dt]

        if model.upper() == "NS":
            if not np.isfinite(row[["beta0", "beta1", "beta2", "tau"]]).all():
                continue
            curve = NelsonSiegelCurve(row.beta0, row.beta1, row.beta2, row.tau)

        elif model.upper() == "NSS":
            if not np.isfinite(row[["beta0", "beta1", "beta2", "beta3", "tau1", "tau2"]]).all():
                continue
            curve = NelsonSiegelSvenssonCurve(row.beta0, row.beta1, row.beta2, row.beta3, row.tau1, row.tau2)

        else:
            raise ValueError("model must be 'NS' or 'NSS'")

        y_fit_dense = curve(dense_grid) * 100.0
        plt.plot(dense_grid, y_fit_dense, alpha=0.8)
        plt.scatter(pillar_t, y_pillars, s=25, alpha=0.9)

    plt.title(title)
    plt.xlabel("Maturity (years)")
    plt.ylabel("Yield (%)")
    plt.tight_layout()
    plt.show()

# ----------------------------
# Example: plot any specification against actual yields
# Here i pick the best NSS (lowest RMSE) vs best NS (lowest RMSE)
# ----------------------------
sample_dates = pick_dates_every_n_months(df_il_eom, step_months=12)
dense_grid = np.linspace(0.25, 11, 250)

run_to_plot = "REAL NS est τ (tau0=1.5)"
params_df_plot, _ = fits[run_to_plot]

plot_curve_samples_from_params(
    df_actual=df_il_eom,
    params_df=params_df_plot,
    tau_map=tau_il,
    dates=sample_dates,
    title=f"REAL curve samples every 12 months: {run_to_plot} vs observed pillars",
    model="NS",
    dense_grid=dense_grid
)

run_to_plot = "REAL NSS (tau0=(1,5))"
params_df_plot, _ = fits[run_to_plot]

plot_curve_samples_from_params(
    df_actual=df_il_eom,
    params_df=params_df_plot,
    tau_map=tau_il,
    dates=sample_dates,
    title=f"REAL curve samples every 12 months: {run_to_plot} vs observed pillars",
    model="NSS",
    dense_grid=dense_grid
)

# ----------------------------
# Nominal: plot each maturity (1y–10y) over time
# ----------------------------
nom_cols_1_10 = [f"y_{m}y" for m in range(1, 11)]
nom_ts = final_nominal_eom_zc_yields[nom_cols_1_10].dropna(how="all")

plt.figure(figsize=(12, 6))
for c in nom_cols_1_10:
    if c in nom_ts.columns:
        plt.plot(nom_ts.index, nom_ts[c], alpha=0.9)
plt.title("Nominal zero-coupon yields by maturity (EOM)")
plt.xlabel("Date")
plt.ylabel("Yield (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# Real (fitted): plot each maturity (1y–10y) over time
# ----------------------------
run_to_plot = "REAL NS est τ (tau0=1.5)"
params_real, fitted_real = fits[run_to_plot]

real_cols_1_10 = [f"y_{m}y" for m in range(1, 11)]
real_ts = fitted_real[real_cols_1_10].dropna(how="all")

plt.figure(figsize=(12, 6))
for c in real_cols_1_10:
    if c in real_ts.columns:
        plt.plot(real_ts.index, real_ts[c], alpha=0.9)
plt.title("Real zero-coupon yields by maturity (fitted NS, EOM)")
plt.xlabel("Date")
plt.ylabel("Real yield (%)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# Excel writing: export the NS (tau0=1.5) specification
# ----------------------------
OUTPUT_XLSX_real = "real_eom_zero_coupon_yields.xlsx"
run_to_export = "REAL NS est τ (tau0=1.5)"
params_export, fitted_export = fits[run_to_export]

fitted_export.to_excel(OUTPUT_XLSX_real, sheet_name="yields")
print(f"Wrote: {OUTPUT_XLSX_real} ({run_to_export})")

OUTPUT_XLSX_NOM = "nominal_eom_zero_coupon_yields.xlsx"
final_nominal_eom_zc_yields.to_excel(OUTPUT_XLSX_NOM, sheet_name="yields")
print(f"Wrote: {OUTPUT_XLSX_NOM}")

