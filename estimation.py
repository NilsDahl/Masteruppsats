##
##   THIS SCRIPT WAS BUILT USING GLOBAL VARIABLES AND DATS
##   WAS LATER REWRITTEN (BY CLAUDE) IN A COMPACT SCRIPT WITH A ESTIMATION FUNCTION
##

import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.optimize import minimize
from scipy.optimize import least_squares

# Helpers: handling of Excel sheets

def read_excel_date_index(path, sheet_name=0, date_col="date"):
    """
    Like read_excel_date_index but normalizes dates to period-month
    to handle last-trading-day vs calendar-end-of-month mismatches.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    # Normalize to month-end so SGB and US files always align
    df[date_col] = df[date_col] + pd.offsets.MonthEnd(0)
    df = df.set_index(date_col).sort_index()
    return df

def month_cols(df):
    """Return columns like y_6m, y_120m."""
    return [c for c in df.columns if isinstance(c, str) and c.startswith("y_") and c.endswith("m")]

def col_to_m(c):
    """Convert column name y_24m -> 24."""
    return int(c.split("_")[1][:-1])

# User inputs

factors_path = "price_factors.xlsx"

nominal_zc_monthgrid_path = "zero_yields_SGB.xlsx"
real_zc_monthgrid_path    = "zero_yields_SGBIL_2.xlsx"

state_factors = [
    "PC1_level", "PC2_slope", "PC3_curvature",
    "composite_liq", "Real_PC1", "Real_PC2"
]

NOM_RET_MONTHS_FULLYEARS = np.arange(12, 121, 12)   # 12, 34, ..., 120
REAL_RET_MONTHS_FULLYEARS = np.arange(24, 121, 12)  # 24, 36, ..., 120

# Load factors + inflation + short rate and build dataset

df_f = read_excel_date_index(factors_path, sheet_name=0)

df_inf = (
    read_excel_date_index("CPI_and_rate.xlsx", sheet_name="inflation")
    .rename(columns={"monthly log": "inflation"})
)

# Short rate from CPI_and_rate.xlsx, sheet "rate")
df_r = (
    read_excel_date_index("short_rate.xlsx", sheet_name=0)  
    .rename(columns={"y_1m": "short_rate"})
)
df_r["short_rate_dec"] = df_r["short_rate"] / 12.0   # monthly decimal
df_r = df_r[["short_rate_dec"]]

# Merge into one aligned panel
df_model_data = df_f.join([df_inf, df_r], how="inner").sort_index()

# Convert inflation from percent to decimals and shift to align with rx_{t->t+1}
infl_m = df_model_data["inflation"] / 100.0
df_model_data["infl_1m"] = infl_m.shift(-1)

# Load month-grid yields (nominal + real), convert % -> decimals

df_nom_y = read_excel_date_index(nominal_zc_monthgrid_path, sheet_name=0) 
df_real_y = read_excel_date_index(real_zc_monthgrid_path, sheet_name=0) 

# Survey inflation expectations (Swedish Money Market Players, quarterly, %)
df_survey = pd.read_excel(
    "CPI_Inflation_Expectations.xlsx",
    sheet_name="CPI Expectations",
    header=1
)
df_survey["date"] = pd.to_datetime(df_survey["Month"], format="%b-%Y")
df_survey = (df_survey.drop(columns=["Month"]).set_index("date").sort_index())
# Convert % to decimals to match model units
df_survey = df_survey / 100.0

# Helpers for building log prices and excess returns from yields

# Build log prices from month-grid yields
def build_logP_from_yield_monthgrid(df_y):
    """
    Build log prices from annualized yields (decimals):
      logP_t(n months) = -(n/12) * y_t(n)
    """
    cols = month_cols(df_y)
    m_list = sorted([col_to_m(c) for c in cols])

    logP = pd.DataFrame(index=df_y.index)
    for m in m_list:
        col = f"y_{m}m"
        tau_years = m / 12.0
        logP[f"logP_{m}m"] = -tau_years * df_y[col]
    return logP

df_logP_nom = build_logP_from_yield_monthgrid(df_nom_y)
df_logP_real = build_logP_from_yield_monthgrid(df_real_y)

# One-month excess returns rx_{t->t+1} using short rate
def rx1m_from_logP(logP_df: pd.DataFrame, short_rate_dec: pd.Series, ret_months: np.ndarray):
    """
    rx_{t->t+1}(n) = logP_{t+1}(n-1) - logP_t(n) - r_t
    """
    out = {}
    for n in ret_months:
        col_n = f"logP_{n}m"
        col_n1 = f"logP_{n-1}m"
        if col_n in logP_df.columns and col_n1 in logP_df.columns:
            out[n] = logP_df[col_n1].shift(-1) - logP_df[col_n] - short_rate_dec
    return pd.DataFrame(out, index=logP_df.index).iloc[:-1]

# Align short rate to nominal logP index
r_t = df_model_data["short_rate_dec"].reindex(df_logP_nom.index)

rx1m_nom = rx1m_from_logP(df_logP_nom, r_t, NOM_RET_MONTHS_FULLYEARS)
rx1m_real = rx1m_from_logP(df_logP_real, 
                           r_t.reindex(df_logP_real.index), 
                           REAL_RET_MONTHS_FULLYEARS)

# Stack into one wide panel
rx1m_nom_stacked = rx1m_nom.copy()
rx1m_nom_stacked.columns = [f"nom_{int(c)}m" for c in rx1m_nom_stacked.columns]

rx1m_real_stacked = rx1m_real.copy()
rx1m_real_stacked.columns = [f"real_{int(c)}m" for c in rx1m_real_stacked.columns]

rx1m_stacked = rx1m_nom_stacked.join(rx1m_real_stacked, how="inner").sort_index()


# VAR(1) on state vector X_t (monthly)
X_var = df_model_data[state_factors].dropna().copy()
var_res = VAR(X_var).fit(1)

Phi   = var_res.coefs[0]
mu    = var_res.intercept
Sigma = var_res.sigma_u

K = len(state_factors)
mu_x = np.linalg.solve(np.eye(K) - Phi, mu)

print(var_res.summary())

# OLS regression of short rate on state vector X_t 
#     short_rate_dec(t) = delta0 + delta1' X_t + error

df_rate_reg = (
    df_model_data[["short_rate_dec"]]
    .join(df_model_data[state_factors], how="inner")
    .dropna()
    .sort_index()
)

y_rate_monthly = df_rate_reg["short_rate_dec"]
X_rate = sm.add_constant(df_rate_reg[state_factors], has_constant="add")

ols_rate = sm.OLS(y_rate_monthly, X_rate).fit()

delta0 = ols_rate.params["const"]
delta1 = ols_rate.params[state_factors]

print(ols_rate.summary())

# Equation (28) from Abrahams style stacked regression (monthly one-step)
#     Regress on X_t and X_{t+1}

df = df_model_data.join(rx1m_stacked, how="inner").sort_index()

ret_cols = rx1m_stacked.columns.tolist()
state_cols = state_factors

# X_{t+1} as one-month lead
X_lead1 = df[state_cols].shift(-1).add_suffix("_lead1")

# Long panel of returns
rx_long = (
    df[ret_cols]
    .stack()
    .rename("rx1m")
    .reset_index()
    .rename(columns={"level_1": "asset"})
)

rx_long["type"] = rx_long["asset"].str.split("_").str[0]  # nom / real
rx_long["maturity_m"] = rx_long["asset"].str.split("_").str[1].str.replace("m", "").astype(int)

# Regressors at date t
regressors_infl = pd.concat(
    [
        df[["infl_1m"]],
        df[state_cols],
        X_lead1
    ],
    axis=1
).reset_index()

df_ols_long = rx_long.merge(regressors_infl, on="date", how="left").dropna()
df_ols_long = df_ols_long.sort_values(["date", "type", "maturity_m"]).reset_index(drop=True)

# Construct R^pi:
df_ols_long["R_pi"] = df_ols_long["rx1m"]
mask_real = df_ols_long["type"] == "real"
df_ols_long.loc[mask_real, "R_pi"] = df_ols_long.loc[mask_real, "rx1m"] + df_ols_long.loc[mask_real, "infl_1m"]

# Regressor columns
x_now_cols  = state_cols
x_lead_cols = [c + "_lead1" for c in state_cols]
x_cols = x_now_cols + x_lead_cols

# Run separate OLS per asset

params_list = []
resid_list = []

asset_order = (
    df_ols_long[["asset", "maturity_m", "type"]]
    .drop_duplicates()
    .sort_values(["type", "maturity_m"])
    ["asset"]
    .tolist()
)

for asset in asset_order:
    g = df_ols_long[df_ols_long["asset"] == asset].sort_values("date").copy()

    y = g["R_pi"].astype(float).values
    X = sm.add_constant(g[x_cols].astype(float), has_constant="add")

    fit = sm.OLS(y, X).fit()

    p = fit.params.copy()
    p.name = asset
    params_list.append(p)

    g["resid"] = fit.resid
    resid_list.append(g[["date", "asset", "resid"]])

params_df = pd.DataFrame(params_list).reindex(asset_order)

resid_df = pd.concat(resid_list, ignore_index=True)
resid_df["date"] = pd.to_datetime(resid_df["date"])

E_hat = (
    resid_df.pivot(index="date", columns="asset", values="resid")
    .sort_index()
    .reindex(columns=asset_order)
)
# Build Sigma_e_hat
T_e = E_hat.shape[0]
Sigma_e_hat = (E_hat.to_numpy().T @ E_hat.to_numpy()) / T_e

# Sigma e hat can be nearly singular (common if assets are very correlated, 
# or sample small, or some columns almost linear com
# bos). Then inversion is unstable.
# Eigen floor (ridge) to ensure SPD and invertible

eigvals = np.linalg.eigvalsh(Sigma_e_hat)
lam_min = np.min(eigvals)
print("Sigma_e_hat eig min/max:", lam_min, np.max(eigvals))
ridge = max(1e-10, 1e-6 * np.max(eigvals))  
Sigma_e_ridge = Sigma_e_hat + ridge * np.eye(Sigma_e_hat.shape[0])
W = np.linalg.inv(Sigma_e_ridge)  # now safe

# Recover Phi_tilde via GLS identity (Abrahams-style)
#     coeff_now  = -B * Phi_tilde
#     coeff_lead =  B

B_ols = params_df[x_lead_cols].to_numpy(dtype=float)      # (N, K)
coeff_now = params_df[x_now_cols].to_numpy(dtype=float)   # (N, K)
Bphi_target = -coeff_now                                   # (N, K)

lhs = B_ols.T @ W @ B_ols
rhs = B_ols.T @ W @ Bphi_target

phi_gls = np.linalg.solve(lhs, rhs)

eig = np.linalg.eigvals(phi_gls)
print("max |eig(phi_gls)|:", np.max(np.abs(eig)))

# Recover α_gls and B_gls via "SUR" on [1, Z_t]
#     Model: R_pi,t = α + B Z_t + e_t 

# Build Y panel first to define return dates
Rpi_wide = (
    df_ols_long.pivot(index="date", columns="asset", values="R_pi")
    .sort_index()
    .reindex(columns=asset_order)
)
Rpi_wide.index = pd.to_datetime(Rpi_wide.index)
ret_dates = Rpi_wide.index

# Define X_t and X_{t+1}, aligned to return dates
X_minus = df_model_data[state_factors].reindex(ret_dates)              # X_t
X_now   = df_model_data[state_factors].shift(-1).reindex(ret_dates)    # X_{t+1} aligned to t

mask_z = X_minus.notna().all(axis=1) & X_now.notna().all(axis=1)
X_minus = X_minus.loc[mask_z]
X_now   = X_now.loc[mask_z]

# Z_t = -Phi_tilde X_t + X_{t+1}
Z = X_now.to_numpy() - X_minus.to_numpy() @ phi_gls.T

X_sur = np.column_stack([np.ones(Z.shape[0]), Z])
z_dates = X_minus.index

# Align Y to z_dates and enforce complete panel across assets
Rpi_wide = Rpi_wide.reindex(z_dates)

mask_y = Rpi_wide.notna().all(axis=1).to_numpy()
Y = Rpi_wide.loc[mask_y].to_numpy()
X_sur = X_sur[mask_y, :]

T_eff, N = Y.shape
K = len(state_factors)

print("SUR design X_sur:", X_sur.shape, "Y:", Y.shape)

# Multivariate OLS
C_hat = np.linalg.solve(X_sur.T @ X_sur, X_sur.T @ Y)

alpha_gls = C_hat[0, :]
B_gls     = C_hat[1:, :].T

# Recover mu_tilde_gls via Eq. (30) from Abrahams using alpha_gls, B_gls, Sigma_e_hat, and Sigma_hat

# Build gamma_hat_gls (Eq. 27): gamma_i = b_i' Σ b_i for each row b_i of B_gls
#    Result shape (N,)
gamma_hat_gls = np.einsum("ik,kl,il->i", B_gls, Sigma, B_gls)

# Form RHS vector (alpha + 1/2 gamma), shape (N,)
rhs = alpha_gls + 0.5 * gamma_hat_gls

M = B_gls.T @ W @ B_gls          # (K, K)
v = B_gls.T @ W @ rhs            # (K,)

mu_tilde_gls = -np.linalg.solve(M, v)      # (K,)

print("mu_tilde_gls shape:", mu_tilde_gls.shape)
print("mu_tilde_gls:", mu_tilde_gls)

# pi0, pi1 via Least Squares on real-bond 1m excess returns 

# Fixed inputs from earlier steps
mu_tilde_use  = mu_tilde_gls
Sigma_use     = Sigma
Phi_tilde_use = phi_gls

# short rate must be 1-month, same units as rx/logP (monthly decimal)
r_1m = df_model_data["short_rate_dec"].astype(float)

# delta0_m, delta1_m from r_t = delta0 + delta1' X_t
delta0_m = delta0 
delta1_m = delta1.to_numpy(dtype=float)

# Align (rx_real, X_t, X_{t+1}, r_t) on the SAME t-index
rx_real_df = rx1m_real.sort_index()

X_df = df_model_data[state_factors].astype(float).copy()
X_t_df   = X_df.reindex(rx_real_df.index)
X_tp1_df = X_df.shift(-1).reindex(rx_real_df.index)

mask = rx_real_df.notna().all(1) & X_t_df.notna().all(1) & X_tp1_df.notna().all(1)

rx_real_df = rx_real_df.loc[mask]
X_t_df     = X_t_df.loc[mask]
X_tp1_df   = X_tp1_df.loc[mask]
r_t        = r_1m.reindex(rx_real_df.index).to_numpy()

rx_real = rx_real_df.to_numpy(float)  # (T, NR)
X_t     = X_t_df.to_numpy(float)      # (T, K)
X_tp1   = X_tp1_df.to_numpy(float)    # (T, K)

T_pi, NR = rx_real.shape
K = X_t.shape[1]

real_maturities = REAL_RET_MONTHS_FULLYEARS
max_n = int(np.max(real_maturities))

print("Aligned shapes:", rx_real.shape, X_t.shape, X_tp1.shape, r_t.shape)

# Start values fpr pi0 and pi1 from inflation regression 
tmp = df_model_data[["infl_1m"] + state_factors].dropna()
X_pi = sm.add_constant(tmp[state_factors].to_numpy(float), has_constant="add")
y_pi = tmp["infl_1m"].to_numpy(float)
b_pi = np.linalg.lstsq(X_pi, y_pi, rcond=None)[0]
x0 = np.r_[float(b_pi[0]), b_pi[1:].astype(float)]

# Bounds + clip x0 into bounds

pi0_lo, pi0_hi = -0.20, 0.20
pi1_lo, pi1_hi = -1.0,  1.0

lb = np.r_[pi0_lo, np.full(K, pi1_lo)]
ub = np.r_[pi0_hi, np.full(K, pi1_hi)]
x0 = np.minimum(np.maximum(x0, lb + 1e-12), ub - 1e-12)

print("x0:", x0)

# Recursions + residuals 

def tips_AB(pi0, pi1):
    A = np.zeros(max_n + 1, float)
    B = np.zeros((max_n + 1, K), float)
    delta0_R = delta0_m - pi0

    for n in range(1, max_n + 1):
        B_pi_prev = B[n-1] + pi1
        Bn = Phi_tilde_use.T @ B_pi_prev - delta1_m
        An = A[n-1] + B_pi_prev @ mu_tilde_use + 0.5*(B_pi_prev @ Sigma_use @ B_pi_prev) - delta0_R

        B[n] = Bn
        A[n] = An

    return A, B

def rx_hat_from_AB(A, B):
    out = np.empty((T_pi, NR), float)
    for j, n in enumerate(real_maturities):
        v = (A[n-1] + X_tp1 @ B[n-1]) - (A[n] + X_t @ B[n]) - r_t
        out[:, j] = v
    return out

def residuals_pi(params):
    pi0 = float(params[0])
    pi1 = np.asarray(params[1:], float)

    A, B = tips_AB(pi0, pi1)

    rx_hat = rx_hat_from_AB(A, B)

    res = (rx_real - rx_hat) 
    return res.ravel()

res = least_squares(
    residuals_pi,
    x0,
    method="trf",
    bounds=(lb, ub),
    x_scale="jac",
    diff_step=1e-6,
    ftol=1e-12, xtol=1e-12, gtol=1e-12,
    max_nfev=20000,
    verbose=2
)

pi0_hat = float(res.x[0])
pi1_hat = res.x[1:].astype(float)

print("success:", res.success, "|", res.message)
print("pi0_hat:", pi0_hat)
print("pi1_hat:", pi1_hat)


# Plot estimated inflation vs observed monthly inflation
# (uses pi0_hat, pi1_hat, mu_tilde_gls, phi_gls, Sigma, delta0_m, delta1_m, df_model_data)
# Inflation: model vs observed
X_pi_df = df_model_data[state_factors].astype(float)
pi_hat_1m = pd.Series(pi0_hat + X_pi_df.to_numpy() @ pi1_hat, index=X_pi_df.index, name="pi_hat_1m")
pi_obs_1m = df_model_data["infl_1m"].astype(float).rename("pi_obs_1m")

tmp_pi = pd.concat([pi_obs_1m, pi_hat_1m], axis=1).dropna()

plt.figure(figsize=(10,4))
plt.plot(tmp_pi.index, tmp_pi["pi_obs_1m"]*100, linewidth=2, label="Observed inflation (1m)")
plt.plot(tmp_pi.index, tmp_pi["pi_hat_1m"]*100, linewidth=2, label="Model-implied inflation (1m)")
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Monthly inflation: observed vs model-implied")
plt.ylabel("Percent")
plt.legend()
plt.tight_layout()
plt.show()

# Code to check final fit and plot results
real_maturities = REAL_RET_MONTHS_FULLYEARS
K = len(state_factors)
A, B = tips_AB(pi0_hat, pi1_hat)
rx_hat = rx_hat_from_AB(A, B)
err = rx_real - rx_hat  # (T, NR)
err_df = pd.DataFrame(err, index=rx_real_df.index, columns=[f"{int(n)}m" for n in real_maturities])

# Plot all maturities on one plot
plt.figure(figsize=(11,5))
for c in err_df.columns:
    plt.plot(err_df.index, err_df[c], linewidth=1, alpha=0.8, label=c)
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Real bond 1m excess return errors (Observed - Model)")
plt.ylabel("Decimal return")
plt.legend(title="Maturity", ncol=3, fontsize=9)
plt.tight_layout()
plt.show()

################------------------####################
# Price yields under Q and P + BEI decomposition (liq-adj)
# Requires already defined: df_model_data, df_nom_y, df_real_y, state_factors,
# Phi, mu_x, Sigma, phi_gls, mu_tilde_gls, delta0_m, delta1_m, pi0_hat, pi1_hat

# maturities
nom_months  = np.array(sorted([col_to_m(c) for c in month_cols(df_nom_y)]), dtype=int)
real_months = np.array(sorted([col_to_m(c) for c in month_cols(df_real_y)]), dtype=int)
max_n = int(max(nom_months.max(), real_months.max()))
K = len(state_factors)

# shared rate/inflation parameters (monthly)
delta0 = float(delta0_m)
delta1 = delta1_m.astype(float)
pi0 = float(pi0_hat)
pi1 = pi1_hat.astype(float)

# Helpers: AB recursions + yield builder + excess return builder

def AB_nominal(Phi_dyn, mu_dyn, Sigma, delta0, delta1, max_n):
    A = np.zeros(max_n + 1, float)
    B = np.zeros((max_n + 1, K), float)
    for n in range(1, max_n + 1):
        Bn = Phi_dyn.T @ B[n-1] - delta1
        An = A[n-1] + B[n-1] @ mu_dyn + 0.5 * (B[n-1] @ Sigma @ B[n-1]) - delta0
        A[n], B[n] = An, Bn
    return A, B

def AB_real(Phi_dyn, mu_dyn, Sigma, delta0, delta1, pi0, pi1, max_n):
    A = np.zeros(max_n + 1, float)
    B = np.zeros((max_n + 1, K), float)
    delta0_R = delta0 - pi0
    for n in range(1, max_n + 1):
        B_pi_prev = B[n-1] + pi1
        Bn = Phi_dyn.T @ B_pi_prev - delta1
        An = A[n-1] + B_pi_prev @ mu_dyn + 0.5 * (B_pi_prev @ Sigma @ B_pi_prev) - delta0_R
        A[n], B[n] = An, Bn
    return A, B

def yhat_from_AB(A, B, X_index, months):
    X_pr = df_model_data[state_factors].reindex(X_index).astype(float).dropna()
    X_arr = X_pr.to_numpy(float)
    out = pd.DataFrame(index=X_pr.index)
    for n in months:
        tau = n / 12.0
        out[f"y_{n}m"] = -(A[n] + X_arr @ B[n]) / tau
    return out, X_pr

def rxhat_nominal_from_AB(B, Phi_tilde, mu_tilde, Sigma, X_t_df, X_tp1_df, months):
    idx = X_t_df.index.intersection(X_tp1_df.index)
    X_t_arr = X_t_df.reindex(idx).to_numpy(float)
    X_tp1_arr = X_tp1_df.reindex(idx).to_numpy(float)

    out = pd.DataFrame(index=idx)
    for n in months:
        b_prev = B[n - 1]
        alpha = -(b_prev @ mu_tilde + 0.5 * (b_prev @ Sigma @ b_prev))
        out[f"rx_{n}m"] = alpha - (X_t_arr @ Phi_tilde.T @ b_prev) + (X_tp1_arr @ b_prev)
    return out

def rxhat_real_from_AB(B, Phi_tilde, mu_tilde, Sigma, pi0, pi1, X_t_df, X_tp1_df, months):
    idx = X_t_df.index.intersection(X_tp1_df.index)
    X_t_arr = X_t_df.reindex(idx).to_numpy(float)
    X_tp1_arr = X_tp1_df.reindex(idx).to_numpy(float)

    out = pd.DataFrame(index=idx)
    for n in months:
        b_prev = B[n - 1]
        b_pi_prev = b_prev + pi1
        alpha = -(pi0 + b_pi_prev @ mu_tilde + 0.5 * (b_pi_prev @ Sigma @ b_pi_prev))
        out[f"rx_{n}m"] = alpha - (X_t_arr @ Phi_tilde.T @ b_pi_prev) + (X_tp1_arr @ b_prev)
    return out

# Q measure objects (risk-neutral): Phi_tilde, mu_tilde

Phi_Q = phi_gls
mu_Q  = mu_tilde_gls
Sigma_Q = Sigma

A_nom_Q, B_nom_Q = AB_nominal(Phi_Q, mu_Q, Sigma_Q, delta0, delta1, max_n)
A_real_Q, B_real_Q = AB_real(Phi_Q, mu_Q, Sigma_Q, delta0, delta1, pi0, pi1, max_n)

yhat_Q,  Xpr_nom_Q  = yhat_from_AB(A_nom_Q,  B_nom_Q,  df_nom_y.index,  nom_months)
yhatR_Q, Xpr_real_Q = yhat_from_AB(A_real_Q, B_real_Q, df_real_y.index, real_months)

# Observed yields aligned to model indices
yobs_Q  = df_nom_y.reindex(yhat_Q.index)[yhat_Q.columns]
yobsR_Q = df_real_y.reindex(yhatR_Q.index)[yhatR_Q.columns]

# P measure objects (physical): Phi, mu_x

Phi_P = Phi
mu_P  = mu
Sigma_P = Sigma

A_nom_P, B_nom_P = AB_nominal(Phi_P, mu_P, Sigma_P, delta0, delta1, max_n)
A_real_P, B_real_P = AB_real(Phi_P, mu_P, Sigma_P, delta0, delta1, pi0, pi1, max_n)

yhat_P,  Xpr_nom_P  = yhat_from_AB(A_nom_P,  B_nom_P,  df_nom_y.index,  nom_months)
yhatR_P, Xpr_real_P = yhat_from_AB(A_real_P, B_real_P, df_real_y.index, real_months)

# Build model-implied excess returns under Q
X_df = df_model_data[state_factors].astype(float).copy()
X_t_df = X_df.copy()
X_tp1_df = X_df.shift(-1)

rxhat_nom_Q = rxhat_nominal_from_AB(
    B=B_nom_Q,
    Phi_tilde=Phi_Q,
    mu_tilde=mu_Q,
    Sigma=Sigma_Q,
    X_t_df=X_t_df,
    X_tp1_df=X_tp1_df,
    months=NOM_RET_MONTHS_FULLYEARS
)
rxhat_real_Q = rxhat_real_from_AB(
    B=B_real_Q,
    Phi_tilde=Phi_Q,
    mu_tilde=mu_Q,
    Sigma=Sigma_Q,
    pi0=pi0,
    pi1=pi1,
    X_t_df=X_t_df,
    X_tp1_df=X_tp1_df,
    months=REAL_RET_MONTHS_FULLYEARS
)
# Observed excess returns aligned to model-implied series
rxobs_nom_Q = rx1m_nom.copy()
rxobs_nom_Q.columns = [f"rx_{int(c)}m" for c in rxobs_nom_Q.columns]
rxobs_nom_Q = rxobs_nom_Q.reindex(rxhat_nom_Q.index)[rxhat_nom_Q.columns]

rxobs_real_Q = rx1m_real.copy()
rxobs_real_Q.columns = [f"rx_{int(c)}m" for c in rxobs_real_Q.columns]
rxobs_real_Q = rxobs_real_Q.reindex(rxhat_real_Q.index)[rxhat_real_Q.columns]

# Quick plots: model vs observed yields
mat_labels = {12: "1-year", 24: "2-year", 60: "5-year", 120: "10-year"}

def plot_3panel(mats, obs_df, hat_df, title, prefix):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for ax, m in zip(axes, mats):
        col = f"{prefix}{m}m"
        if col not in obs_df.columns:
            ax.set_visible(False)
            continue
        ax.plot(obs_df.index, obs_df[col] * 10_000,
                label="Observed", linewidth=1.8, color="#2a6ebb")
        ax.plot(hat_df.index, hat_df[col] * 10_000,
                label="Model",    linewidth=1.8, color="#e07b39")
        ax.set_title(mat_labels[m], fontsize=11)
        ax.set_ylabel("Basis points", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.axhline(0, color="black", lw=0.7, linestyle=":")
        ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()

plot_3panel([12, 60, 120], yobs_Q,       yhat_Q,       "Nominal Yield Fit",            "y_")
plot_3panel([24, 60, 120], yobsR_Q,      yhatR_Q,      "Real Yield Fit",               "y_")
plot_3panel([12, 60, 120], rxobs_nom_Q,  rxhat_nom_Q,  "Nominal 1m Excess Return Fit", "rx_")
plot_3panel([24, 60, 120], rxobs_real_Q, rxhat_real_Q, "Real 1m Excess Return Fit",    "rx_")


# ──------- BEI Decomposition ────────────────────────────────────────────────────────
liq_idx = state_factors.index("composite_liq")
assert state_factors[liq_idx] == "composite_liq"

common = sorted(
    set(yhat_Q.columns)
    .intersection(yhatR_Q.columns)
    .intersection(yhat_P.columns)
    .intersection(yhatR_P.columns)
)

# Single common row index across all four yield frames — prevents silent NaN
# misalignment when nominal and real yield data cover slightly different date ranges
date_idx = (
    yhat_Q.index
    .intersection(yhatR_Q.index)
    .intersection(yhat_P.index)
    .intersection(yhatR_P.index)
)

# BEI under Q and P
bei_Q = yhat_Q.loc[date_idx, common] - yhatR_Q.loc[date_idx, common]
bei_P = yhat_P.loc[date_idx, common] - yhatR_P.loc[date_idx, common]

# Liquidity components of nominal and real yields
liq_nom_Q  = pd.DataFrame(index=date_idx)
liq_real_Q = pd.DataFrame(index=date_idx)
liq_nom_P  = pd.DataFrame(index=date_idx)
liq_real_P = pd.DataFrame(index=date_idx)

# Reindex factor realisations to date_idx so all liq frames share the same row index
Xpr_nom_Q_al  = df_model_data[state_factors].reindex(date_idx).astype(float)
Xpr_real_Q_al = df_model_data[state_factors].reindex(date_idx).astype(float)

for n in nom_months:
    col = f"y_{n}m"
    if col in common:
        tau = n / 12.0
        liq_nom_Q[col] = -(B_nom_Q[n][liq_idx] * Xpr_nom_Q_al.iloc[:, liq_idx].values) / tau
        liq_nom_P[col] = -(B_nom_P[n][liq_idx] * Xpr_nom_Q_al.iloc[:, liq_idx].values) / tau

for n in real_months:
    col = f"y_{n}m"
    if col in common:
        tau = n / 12.0
        liq_real_Q[col] = -(B_real_Q[n][liq_idx] * Xpr_real_Q_al.iloc[:, liq_idx].values) / tau
        liq_real_P[col] = -(B_real_P[n][liq_idx] * Xpr_real_Q_al.iloc[:, liq_idx].values) / tau

# Liquidity-adjusted BEI (strip out liq component from both nominal and real)
bei_Q_adj = (yhat_Q.loc[date_idx, common] - liq_nom_Q[common]) - (yhatR_Q.loc[date_idx, common] - liq_real_Q[common])
bei_P_adj = (yhat_P.loc[date_idx, common] - liq_nom_P[common]) - (yhatR_P.loc[date_idx, common] - liq_real_P[common])

# Decomposition:
#   BEI         = E^P[π] + IRP + LP
#   LP          = BEI_Q      - BEI_Q_adj   (liquidity distortion in raw breakeven)
#   IRP         = BEI_Q_adj  - BEI_P_adj   (Q vs P wedge on liq-adjusted BEI)
#   E^P[π]      = BEI_P_adj               (liq-adjusted P-measure expected inflation)
LP      = bei_Q     - bei_Q_adj
IRP     = bei_Q_adj - bei_P_adj
E_inf   = bei_P_adj                        # liq-adjusted expected inflation under P
BEI_obs = bei_Q                            # raw (unadjusted) breakeven = sum of all three

# Plots 

plot_mats   = [24, 60, 120]
mat_labels  = {24: "2-year", 60: "5-year", 120: "10-year"}

colors = {
    "BEI":   "#2c2c2c",
    "E_inf": "#e07b39",
    "IRP":   "#2a6ebb",
    "LP":    "#3aaa6e",
}

# Map zero-coupon maturity months to survey column names
survey_col_map = {24: "2 Year", 60: "5 Year"}

for m in plot_mats:
    col = f"y_{m}m"
    if col not in common:
        continue

    # Align all series to a common index and convert to bp
    idx = (
        BEI_obs[col].dropna().index
        .intersection(IRP[col].dropna().index)
        .intersection(LP[col].dropna().index)
        .intersection(E_inf[col].dropna().index)
    )

    s_bei   = BEI_obs[col].loc[idx] * 10_000
    s_irp   = IRP[col].loc[idx]     * 10_000
    s_lp    = LP[col].loc[idx]      * 10_000
    s_einf  = E_inf[col].loc[idx]   * 10_000

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(idx, s_bei,  color=colors["BEI"],   linewidth=2.0, linestyle="--",
            label="Breakeven inflation (raw)")
    ax.plot(idx, s_einf, color=colors["E_inf"], linewidth=2.0,
            label="Expected inflation ($E^P[\\pi]$)")
    ax.plot(idx, s_irp,  color=colors["IRP"],   linewidth=2.0,
            label="Inflation risk premium")
    ax.plot(idx, s_lp,   color=colors["LP"],    linewidth=2.0,
            label="Liquidity premium")

    # Overlay survey expectations where available (2y and 5y horizons)
    if m in survey_col_map:
        s_survey = df_survey[survey_col_map[m]].dropna() * 10_000
        ax.plot(s_survey.index, s_survey.values, color="#9b2335", linewidth=1.6,
                linestyle=(0, (4, 2)), marker="o", markersize=2.5,
                label="Survey expected inflation")

    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)

    ax.set_title(
        f"BEI Decomposition — {mat_labels[m]} maturity ({m}m)",
        fontsize=13, fontweight="bold"
    )
    ax.set_ylabel("Basis points", fontsize=11)
    ax.set_xlabel("")
    ax.legend(frameon=True, fontsize=9, loc="upper right", ncol=3)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    # Sanity check printed below each plot
    check = (s_einf + s_irp + s_lp - s_bei).abs().max()
    print(f"{mat_labels[m]}: max decomposition residual = {check:.4f} bp")

    plt.tight_layout()
    plt.show()


# ── 5–10y Forward BEI Decomposition ─────────────────────────────────────────

def to_fwd_510(zc_df):
    """5-10y forward rate: [10*y120 - 5*y60] / 5 = 2*y120 - y60."""
    return 2.0 * zc_df["y_120m"] - zc_df["y_60m"]

def fwd_liq_510(liq_df):
    """Apply same linear operator to liquidity component DataFrames."""
    return 2.0 * liq_df["y_120m"] - liq_df["y_60m"]

# Forward rates under Q and P (nominal and real)
fwd_nom_Q  = to_fwd_510(yhat_Q.loc[date_idx])
fwd_real_Q = to_fwd_510(yhatR_Q.loc[date_idx])
fwd_nom_P  = to_fwd_510(yhat_P.loc[date_idx])
fwd_real_P = to_fwd_510(yhatR_P.loc[date_idx])

# Liquidity components of 5-10y forward yields under Q and P
fwd_liq_nom_Q  = fwd_liq_510(liq_nom_Q)
fwd_liq_real_Q = fwd_liq_510(liq_real_Q)
fwd_liq_nom_P  = fwd_liq_510(liq_nom_P)
fwd_liq_real_P = fwd_liq_510(liq_real_P)

# Raw forward BEI under Q and P
fwd_bei_Q = fwd_nom_Q - fwd_real_Q
fwd_bei_P = fwd_nom_P - fwd_real_P

# Liquidity-adjusted forward BEI under Q and P
fwd_bei_Q_adj = (fwd_nom_Q - fwd_liq_nom_Q) - (fwd_real_Q - fwd_liq_real_Q)
fwd_bei_P_adj = (fwd_nom_P - fwd_liq_nom_P) - (fwd_real_P - fwd_liq_real_P)

# Decomposition: BEI = E_inf + IRP + LP
fwd_LP    = fwd_bei_Q     - fwd_bei_Q_adj   # liquidity distortion in raw breakeven
fwd_IRP   = fwd_bei_Q_adj - fwd_bei_P_adj   # Q vs P wedge on liq-adjusted BEI
fwd_E_inf = fwd_bei_P_adj                   # liq-adjusted expected inflation under P

# Align to common index
idx = (
    fwd_bei_Q.dropna().index
    .intersection(fwd_IRP.dropna().index)
    .intersection(fwd_LP.dropna().index)
    .intersection(fwd_E_inf.dropna().index)
)

s_bei  = fwd_bei_Q.loc[idx]  * 10_000
s_irp  = fwd_IRP.loc[idx]    * 10_000
s_lp   = fwd_LP.loc[idx]     * 10_000
s_einf = fwd_E_inf.loc[idx]  * 10_000

# Sanity check
check = (s_einf + s_irp + s_lp - s_bei).abs().max()
print(f"5-10y forward: max decomposition residual = {check:.4f} bp")

# Plot
fig, ax = plt.subplots(figsize=(11, 4))

ax.plot(idx, s_bei,  color=colors["BEI"],   linewidth=2.0, linestyle="--",
        label="5y5y forward breakeven (raw)")
ax.plot(idx, s_einf, color=colors["E_inf"], linewidth=2.0,
        label="Expected inflation ($E^P[\\pi]$)")
ax.plot(idx, s_irp,  color=colors["IRP"],   linewidth=2.0,
        label="Inflation risk premium")
ax.plot(idx, s_lp,   color=colors["LP"],    linewidth=2.0,
        label="Liquidity premium")

ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
ax.set_title("BEI Decomposition — 5y5y forward",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Basis points", fontsize=11)
ax.legend(frameon=True, fontsize=10, loc="upper right", ncol=2)
ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()

# ── 2y2y Forward BEI Decomposition ───────────────────────────────────────────
# 2y2y forward rate: rate from year 2 to year 4
#   f(2y,2y) = (4*y_48m - 2*y_24m) / 2 = 2*y_48m - y_24m

def to_fwd_2y2y(zc_df):
    """2y2y forward rate: 2*y_48m - y_24m."""
    return 2.0 * zc_df["y_48m"] - zc_df["y_24m"]

def fwd_liq_2y2y(liq_df):
    """Apply same linear operator to liquidity component DataFrames."""
    return 2.0 * liq_df["y_48m"] - liq_df["y_24m"]

# Forward rates under Q and P (nominal and real)
fwd2_nom_Q  = to_fwd_2y2y(yhat_Q.loc[date_idx])
fwd2_real_Q = to_fwd_2y2y(yhatR_Q.loc[date_idx])
fwd2_nom_P  = to_fwd_2y2y(yhat_P.loc[date_idx])
fwd2_real_P = to_fwd_2y2y(yhatR_P.loc[date_idx])

# Liquidity components of 2y2y forward yields under Q and P
fwd2_liq_nom_Q  = fwd_liq_2y2y(liq_nom_Q)
fwd2_liq_real_Q = fwd_liq_2y2y(liq_real_Q)
fwd2_liq_nom_P  = fwd_liq_2y2y(liq_nom_P)
fwd2_liq_real_P = fwd_liq_2y2y(liq_real_P)

# Raw forward BEI under Q and P
fwd2_bei_Q = fwd2_nom_Q - fwd2_real_Q
fwd2_bei_P = fwd2_nom_P - fwd2_real_P

# Liquidity-adjusted forward BEI under Q and P
fwd2_bei_Q_adj = (fwd2_nom_Q - fwd2_liq_nom_Q) - (fwd2_real_Q - fwd2_liq_real_Q)
fwd2_bei_P_adj = (fwd2_nom_P - fwd2_liq_nom_P) - (fwd2_real_P - fwd2_liq_real_P)

# Decomposition: BEI = E_inf + IRP + LP
fwd2_LP    = fwd2_bei_Q     - fwd2_bei_Q_adj   # liquidity distortion in raw breakeven
fwd2_IRP   = fwd2_bei_Q_adj - fwd2_bei_P_adj   # Q vs P wedge on liq-adjusted BEI
fwd2_E_inf = fwd2_bei_P_adj                    # liq-adjusted expected inflation under P

# Align to common index
idx2 = (
    fwd2_bei_Q.dropna().index
    .intersection(fwd2_IRP.dropna().index)
    .intersection(fwd2_LP.dropna().index)
    .intersection(fwd2_E_inf.dropna().index)
)

s2_bei  = fwd2_bei_Q.loc[idx2]  * 10_000
s2_irp  = fwd2_IRP.loc[idx2]    * 10_000
s2_lp   = fwd2_LP.loc[idx2]     * 10_000
s2_einf = fwd2_E_inf.loc[idx2]  * 10_000

# Sanity check
check2 = (s2_einf + s2_irp + s2_lp - s2_bei).abs().max()
print(f"2y2y forward: max decomposition residual = {check2:.4f} bp")

# Plot
fig, ax = plt.subplots(figsize=(11, 4))

ax.plot(idx2, s2_bei,  color=colors["BEI"],   linewidth=2.0, linestyle="--",
        label="2y2y forward breakeven (raw)")
ax.plot(idx2, s2_einf, color=colors["E_inf"], linewidth=2.0,
        label="Expected inflation ($E^P[\\pi]$)")
ax.plot(idx2, s2_irp,  color=colors["IRP"],   linewidth=2.0,
        label="Inflation risk premium")
ax.plot(idx2, s2_lp,   color=colors["LP"],    linewidth=2.0,
        label="Liquidity premium")

ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
ax.set_title("BEI Decomposition — 2y2y forward",
             fontsize=13, fontweight="bold")
ax.set_ylabel("Basis points", fontsize=11)
ax.legend(frameon=True, fontsize=10, loc="upper right", ncol=2)
ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()

################------------------####################
# Fit diagnostics tables for Latex 
# One table for yields (nominal + real), one for excess returns

from scipy.stats import skew, kurtosis

def pricing_errors(obs_df, hat_df, maturities, prefix, scale=10_000):
    rows = []
    for m in maturities:
        col = f"{prefix}{m}m"
        if col not in obs_df.columns or col not in hat_df.columns:
            continue
        err = (obs_df[col] - hat_df[col]).dropna() * scale
        rows.append({
            "n": m,
            "Mean":  round(err.mean(), 3),
            "Std":   round(err.std(),  3),
            "Skew":  round(skew(err),  3),
            "Kurt":  round(kurtosis(err, fisher=False), 3),
        })
    return pd.DataFrame(rows).set_index("n")

def two_panel_latex(df_top, df_bot, panel_top, panel_bot, caption, label, note=None):
    col_fmt = "r" + "r" * len(df_top.columns)
    cols = list(df_top.columns)
    header = "$n$ (months) & " + " & ".join(cols) + r" \\"

    def rows_block(df):
        lines = []
        for idx, row in df.iterrows():
            vals = " & ".join([f"{idx}"] + [f"{v:.3f}" for v in row.values])
            lines.append(f"        {vals} \\\\")
        return "\n".join(lines)

    tex = rf"""
\begin{{table}}[htbp]
    \centering
    \caption{{{caption}}}
    \label{{{label}}}
    \begin{{tabular}}{{{col_fmt}}}
    \toprule
        {header}
    \midrule
        \multicolumn{{{1 + len(cols)}}}{{l}}{{\small\textit{{Panel A: {panel_top}}}}} \\
    \midrule
{rows_block(df_top)}
    \midrule
        \multicolumn{{{1 + len(cols)}}}{{l}}{{\small\textit{{Panel B: {panel_bot}}}}} \\
    \midrule
{rows_block(df_bot)}
    \bottomrule
    \end{{tabular}}"""

    if note:
        tex += rf"""
    \begin{{minipage}}{{\linewidth}}
        \vspace{{4pt}}
        \footnotesize \textit{{Note:}} {note}
    \end{{minipage}}"""

    tex += "\n\\end{table}"
    return tex

# Yield table
tbl_nom_y  = pricing_errors(yobs_Q,   yhat_Q,   [12, 24, 36, 60, 84, 120], "y_")
tbl_real_y = pricing_errors(yobsR_Q,  yhatR_Q,  [24, 36, 60, 84, 120],     "y_")

# Return table
tbl_nom_rx  = pricing_errors(rxobs_nom_Q,  rxhat_nom_Q,  [12, 24, 36, 60, 84, 120], "rx_")
tbl_real_rx = pricing_errors(rxobs_real_Q, rxhat_real_Q, [24, 36, 60, 84, 120],     "rx_")

note_y = ("Mean, Std, Skew, and Kurt refer to the sample mean, standard deviation, "
          "skewness, and kurtosis of yield pricing errors in basis points. "
          "Sample: 2004:01--2025:12.")

note_rx = ("Mean, Std, Skew, and Kurt refer to the sample mean, standard deviation, "
           "skewness, and kurtosis of excess return pricing errors in basis points. "
           "Sample: 2004:01--2025:12.")

print(two_panel_latex(
    tbl_nom_y, tbl_real_y,
    panel_top = "Nominal SGB yield pricing errors",
    panel_bot = "Real SGBIL yield pricing errors",
    caption   = ("Yield fit diagnostics. Panel A reports pricing errors for nominal "
                 "SGB yields and Panel B for real SGBIL yields."),
    label     = "tab:yield_fit",
    note      = note_y
))

print(two_panel_latex(
    tbl_nom_rx, tbl_real_rx,
    panel_top = "Nominal SGB excess return pricing errors",
    panel_bot = "Real SGBIL excess return pricing errors",
    caption   = ("Excess return fit diagnostics. Panel A reports pricing errors for "
                 "nominal SGB returns and Panel B for real SGBIL returns."),
    label     = "tab:ret_fit",
    note      = note_rx
))


# Compute realized average inflation over n months ahead
def realized_avg_inflation(infl_monthly, horizon_m):
    # Rolling forward sum / horizon, then annualize (*12) and convert to bp (*10000)
    realized = (
        infl_monthly
        .rolling(window=horizon_m)
        .mean()
        .shift(-horizon_m)   # align: value at t = avg inflation from t to t+n
        * 12                 # annualize
        * 10_000             # to basis points
    )
    return realized

horizon_map = {
    "2 Year": 24,
    "5 Year": 60,
}

fname_map = {
    "2 Year": "survey_vs_model_2y.png",
    "5 Year": "survey_vs_model_5y.png",
}

for survey_col, m in horizon_map.items():
    col = f"y_{m}m"

    s_model    = E_inf[col].dropna()            * 10_000
    s_survey   = df_survey[survey_col].dropna() * 10_000
    s_bei      = BEI_obs[col].dropna()          * 10_000
    s_realized = realized_avg_inflation(infl_m, m).dropna()

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(s_bei.index,      s_bei,      color="#2c2c2c", linewidth=1.5,
            linestyle="--",   label="Breakeven inflation (raw)")
    ax.plot(s_model.index,    s_model,    color="#e07b39", linewidth=2.0,
            label="Model-implied expected inflation")
    ax.plot(s_survey.index,   s_survey,   color="#9b2335", linewidth=1.8,
            linestyle=(0, (4, 2)), marker="o", markersize=2.5,
            label="Survey expected inflation")
    ax.plot(s_realized.index, s_realized, color="#3aaa6e", linewidth=1.8,
            linestyle="-",    label=f"Realized inflation ({survey_col})")

    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_title(
        f"Expected vs Realized Inflation — {survey_col} horizon",
        fontsize=13, fontweight="bold", pad=10
    )
    ax.set_ylabel("Basis points", fontsize=11)
    ax.legend(frameon=True, fontsize=10, ncol=2)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.show()


def rmse(a, b):
    diff = a - b
    return np.sqrt((diff ** 2).mean())

horizon_map = {"2 Year": 24, "5 Year": 60}

rows = []
for label, m in horizon_map.items():
    col = f"y_{m}m"

    realized  = realized_avg_inflation(infl_m, m)
    model_fc  = E_inf[col]       * 10_000
    bei_fc    = BEI_obs[col]     * 10_000
    survey_fc = (df_survey[label] * 10_000).copy()
    survey_fc.index = survey_fc.index + pd.offsets.MonthEnd(0)  # snap to month-end
    rw_fc     = infl_m.rolling(m).mean() * 12 * 10_000

    common = (realized.dropna().index
              .intersection(model_fc.dropna().index)
              .intersection(rw_fc.dropna().index))

    r  = realized.loc[common]
    mf = model_fc.loc[common]
    rw = rw_fc.loc[common]

    survey_common = realized.dropna().index.intersection(survey_fc.dropna().index)

    row = {
        "Horizon":        label,
        "Model forecast": rmse(mf, r),
        "Random walk":    rmse(rw, r),
        "Breakevens":     rmse(bei_fc.loc[common.intersection(bei_fc.dropna().index)],
                               realized.loc[common.intersection(bei_fc.dropna().index)])
                          if len(common.intersection(bei_fc.dropna().index)) > 10 else np.nan,
        "Survey":         rmse(survey_fc.loc[survey_common],
                               realized.loc[survey_common])
                          if len(survey_common) > 10 else np.nan,
    }

    rows.append(row)

is_rmse = pd.DataFrame(rows).set_index("Horizon")

print("=== In-sample RMSE (basis points) ===")
print(is_rmse.round(1).to_string())
print("\nNote: Survey RMSE computed over available survey dates only.")