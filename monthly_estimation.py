import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox


# ============================================================
# Helper: robust date parsing for Excel sheets
# ============================================================
def read_excel_date_index(path, sheet_name=0, date_col="date"):
    """
    Read an Excel sheet and return a DataFrame with a clean DatetimeIndex.
    Steps:
    1) Read sheet into a DataFrame
    2) Parse date_col to datetime (coerce errors)
    3) Drop rows where date cannot be parsed
    4) Set DatetimeIndex and sort
    """
    df = pd.read_excel(path, sheet_name=sheet_name)

    if date_col not in df.columns:
        raise ValueError(f"Missing '{date_col}' column in {path} / sheet={sheet_name}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).set_index(date_col).sort_index()
    return df


def month_cols(df):
    """Return columns like y_6m, y_120m."""
    return [c for c in df.columns if isinstance(c, str) and c.startswith("y_") and c.endswith("m")]


def col_to_m(c):
    """Convert column name y_24m -> 24."""
    return int(c.split("_")[1][:-1])


# ============================================================
# 0) User inputs
# ============================================================
factors_path = "model_factors_nominal_real_liquidity.xlsx"
macro_path   = "CPI_and_rate.xlsx"

nominal_zc_monthgrid_path = "nominal_eom_zero_coupon_yields_monthly_grid.xlsx"
real_zc_monthgrid_path    = "real_eom_zero_coupon_yields_monthly_grid.xlsx"

state_factors = [
    "PC1_level", "PC2_slope", "PC3_curvature",
    "Liquidity_MA3", "Real_PC1", "Real_PC2"
]

# Given your available grids:
# Nominal has y_6m..y_120m => rx for 7..120 (need n and n-1)
# Real has y_24m..y_120m   => rx for 25..120 (need n and n-1)
NOM_RET_MONTHS = np.arange(6, 121)
REAL_RET_MONTHS = np.arange(24, 121)
NOM_RET_MONTHS_FULLYEARS = np.arange(12, 121, 12)   # 24, 36, ..., 120
REAL_RET_MONTHS_FULLYEARS = np.arange(24, 121, 12)  # 36, 48, ..., 120

# ============================================================
# 1) Load factors + macro (inflation + short rate) and build dataset
# ============================================================
df_f = read_excel_date_index(factors_path, sheet_name=0)

df_inf = (
    read_excel_date_index(macro_path, sheet_name="inflation")
    .rename(columns={"monthly log": "inflation"})
)

# Short rate from CPI_and_rate.xlsx, sheet "rate" (NOT in decimals per your note)
df_r = (
    read_excel_date_index("short_rate.xlsx", sheet_name=0)   # or sheet_name="Sheet1"
    .rename(columns={"y_1m": "short_rate"})
)

# Merge into one aligned panel
df_model_data = df_f.join([df_inf, df_r], how="inner").sort_index()

# Keep required columns
keep_cols = df_f.columns.tolist() + ["inflation", "short_rate"]
df_model_data = df_model_data[keep_cols].copy()

# Convert inflation from percent to decimals
infl_m = df_model_data["inflation"] / 100.0

# One-month inflation aligned with rx_{t->t+1}
df_model_data["infl_1m"] = infl_m.shift(-1)

# Convert short rate from percent to decimals
df_model_data["short_rate_dec"] = df_model_data["short_rate"] / 100.0


# ============================================================
# 2) Load month-grid yields (nominal + real), convert % -> decimals
# ============================================================
df_nom_y = read_excel_date_index(nominal_zc_monthgrid_path, sheet_name=0) / 100.0
df_real_y = read_excel_date_index(real_zc_monthgrid_path, sheet_name=0) / 100.0

if len(month_cols(df_nom_y)) == 0:
    raise ValueError("No nominal month-grid columns found. Expected y_6m..y_120m.")
if len(month_cols(df_real_y)) == 0:
    raise ValueError("No real month-grid columns found. Expected y_24m..y_120m.")


# ============================================================
# 3) Build log prices from month-grid yields
# ============================================================
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


# ============================================================
# 4) One-month excess returns rx_{t->t+1} using CPI_and_rate short rate
# ============================================================
def rx1m_from_logP(logP_df: pd.DataFrame, short_rate_dec: pd.Series, ret_months: np.ndarray):
    """
    rx_{t->t+1}(n) = logP_{t+1}(n-1) - logP_t(n) - r_t
    short_rate_dec is annualized (decimal).
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


# ============================================================
# 5) VAR(1) on state vector X_t (monthly)
# ============================================================
X_var = df_model_data[state_factors].dropna().copy()
var_res = VAR(X_var).fit(1)

Phi = var_res.coefs[0]
mu  = var_res.intercept
Sigma = var_res.sigma_u

K = len(state_factors)
mu_x = np.linalg.solve(np.eye(K) - Phi, mu)

print(var_res.summary())


# ============================================================
# 6) OLS regression of short rate on state vector X_t  (ADDED BACK)
#     short_rate_dec(t) = delta0 + delta1' X_t + error
# ============================================================
df_rate_reg = (
    df_model_data[["short_rate_dec"]]
    .join(df_model_data[state_factors], how="inner")
    .dropna()
    .sort_index()
)

y_rate = df_rate_reg["short_rate_dec"]
X_rate = sm.add_constant(df_rate_reg[state_factors], has_constant="add")

ols_rate = sm.OLS(y_rate, X_rate).fit()

delta0 = ols_rate.params["const"]
delta1 = ols_rate.params[state_factors]

print(ols_rate.summary())


# ============================================================
# 7) Equation (28)-style stacked regression (monthly one-step)
#     Regress on X_t and X_{t+1}
# ============================================================
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
regressors = pd.concat(
    [
        df[["infl_1m"]],
        df[state_cols],
        X_lead1
    ],
    axis=1
).reset_index()

df_ols_long = rx_long.merge(regressors, on="date", how="left").dropna()
df_ols_long = df_ols_long.sort_values(["date", "type", "maturity_m"]).reset_index(drop=True)

# Construct R^pi:
df_ols_long["R_pi"] = df_ols_long["rx1m"]
mask_real = df_ols_long["type"] == "real"
df_ols_long.loc[mask_real, "R_pi"] = df_ols_long.loc[mask_real, "rx1m"] + df_ols_long.loc[mask_real, "infl_1m"]

# Regressor columns
x_now_cols  = state_cols
x_lead_cols = [c + "_lead1" for c in state_cols]
x_cols = x_now_cols + x_lead_cols


# ============================================================
# 8) Run separate OLS per asset
# ============================================================
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


# ============================================================
# 9) Recover Phi_tilde via GLS identity (Abrahams-style)
#     coeff_now  = -B * Phi_tilde
#     coeff_lead =  B
# ============================================================
B_ols = params_df[x_lead_cols].to_numpy()        # B
Bphi_ols = -params_df[x_now_cols].to_numpy()     # B * Phi_tilde

T = E_hat.shape[0]
Sigma_e_hat = (E_hat.T @ E_hat) / T
W = np.linalg.inv(Sigma_e_hat)

phi_gls = np.linalg.inv(B_ols.T @ W @ B_ols) @ (B_ols.T @ W @ Bphi_ols)

print("phi_gls shape:", phi_gls.shape)
print("VAR Phi shape:", Phi.shape)

# ============================================================
# 10) Quick residual plot
# ============================================================
plt.figure(figsize=(10, 4))
if E_hat.shape[1] > 0:
    some_asset = E_hat.columns[0]
    E_hat[some_asset].plot()
    plt.title(f"Residuals: {some_asset}")
    plt.axhline(0, linestyle="--")
    plt.show()
else:
    print("No residual series to plot.")

# ============================================================
# 11) Recover α_gls and B_gls via "SUR" on [1, Z_t]
#     Model: R_pi,t = α + B Z_t + e_t
# ============================================================
# 11.1 Drop rows where X_{t+1} is missing (last observation)
mask_z = X_minus.notna().all(axis=1) & X_now.notna().all(axis=1)
X_minus = X_minus.loc[mask_z]
X_now   = X_now.loc[mask_z]

# Recompute Z with the filtered index to be safe
Z = X_now.to_numpy() - X_minus.to_numpy() @ phi_gls.T   # T x K

# Design matrix: [1, Z]
X_sur = np.column_stack([np.ones(Z.shape[0]), Z])        # T x (1+K)

# The dates corresponding to rows of X_sur
z_dates = pd.to_datetime(X_minus.index)

# 11.2 Build Y = R_pi in wide form (T x N), using the same asset order as before
Rpi_wide = (
    df_ols_long.pivot(index="date", columns="asset", values="R_pi")
    .sort_index()
    .reindex(columns=asset_order)
)
Rpi_wide.index = pd.to_datetime(Rpi_wide.index)

# Align Y to the Z dates
Rpi_wide = Rpi_wide.reindex(z_dates)

# Keep only rows where we have returns for all assets
mask_y = Rpi_wide.notna().all(axis=1).to_numpy()
Y = Rpi_wide.loc[mask_y].to_numpy()          # T_eff x N
X_sur = X_sur[mask_y, :]                     # T_eff x (1+K)

T_eff, N = Y.shape
K = len(state_factors)

print("SUR design X_sur:", X_sur.shape, "Y:", Y.shape)

# 11.3 Multivariate OLS (same point estimates as SUR here because regressors are common)
# Coefficient matrix C_hat is (1+K) x N
C_hat = np.linalg.inv(X_sur.T @ X_sur) @ (X_sur.T @ Y)

# Parse coefficients into α and B
alpha_gls = C_hat[0, :]          # N
B_gls     = C_hat[1:, :].T       # N x K

print("alpha_gls:", alpha_gls.shape, "B_gls:", B_gls.shape)

# 11.4 Residuals and updated Sigma_e (recommended for eq (30))
E_sur = Y - X_sur @ C_hat                      # T_eff x N
Sigma_e_hat_sur = (E_sur.T @ E_sur) / T_eff    # N x N

# ============================================================
# 12) Compute mu_tilde_gls from eq (30)
#     mu_tilde_gls = - (B' Σ_e^{-1} B)^{-1} B' Σ_e^{-1} (α + 1/2 γ)
# ============================================================

# γ_i = b_i' Σ b_i where Σ is the VAR shock covariance (K x K)
gamma_gls = np.einsum("ik,kl,il->i", B_gls, Sigma, B_gls)   # N
#   γ_i = B_i' Σ B_i   for i = 1,...,N

W_sur = np.linalg.inv(Sigma_e_hat_sur)

BW_B   = B_gls.T @ W_sur @ B_gls                            # K x K
BW_vec = B_gls.T @ W_sur @ (alpha_gls + 0.5 * gamma_gls)    # K

mu_tilde_gls = -np.linalg.inv(BW_B) @ BW_vec                # K

print("mu_tilde_gls:", mu_tilde_gls)
