import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.optimize import minimize
from scipy.optimize import least_squares

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
    read_excel_date_index("short_rate.xlsx", sheet_name=0)  
    .rename(columns={"y_1m": "short_rate"})
)
df_r["short_rate_dec"] = df_r["short_rate"] / 100.0 / 12.0   # monthly decimal
df_r = df_r[["short_rate_dec"]]

# Merge into one aligned panel
df_model_data = df_f.join([df_inf, df_r], how="inner").sort_index()

# Keep required columns
keep_cols = df_f.columns.tolist() + ["inflation", "short_rate_dec"]
df_model_data = df_model_data[keep_cols].copy()

# Convert inflation from percent to decimals
infl_m = df_model_data["inflation"] / 100.0

# One-month inflation aligned with rx_{t->t+1}
df_model_data["infl_1m"] = infl_m.shift(-1)

# ============================================================
# 2) Load month-grid yields (nominal + real), convert % -> decimals
# ============================================================
df_nom_y = read_excel_date_index(nominal_zc_monthgrid_path, sheet_name=0) / 100.0
df_real_y = read_excel_date_index(real_zc_monthgrid_path, sheet_name=0) / 100.0

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
# 4) One-month excess returns rx_{t->t+1} using short rate
# ============================================================
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
# 6) OLS regression of short rate on state vector X_t 
#     short_rate_dec(t) = delta0 + delta1' X_t + error
# ============================================================
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

phi_gls = -np.linalg.inv(B_ols.T @ W @ B_ols) @ (B_ols.T @ W @ Bphi_ols)

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
# 11.0 Build Y panel first to define return dates
Rpi_wide = (
    df_ols_long.pivot(index="date", columns="asset", values="R_pi")
    .sort_index()
    .reindex(columns=asset_order)
)
Rpi_wide.index = pd.to_datetime(Rpi_wide.index)
ret_dates = Rpi_wide.index

# 11.1 Define X_t and X_{t+1}, aligned to return dates
X_minus = df_model_data[state_factors].reindex(ret_dates)              # X_t
X_now   = df_model_data[state_factors].shift(-1).reindex(ret_dates)    # X_{t+1} aligned to t

mask_z = X_minus.notna().all(axis=1) & X_now.notna().all(axis=1)
X_minus = X_minus.loc[mask_z]
X_now   = X_now.loc[mask_z]

# Z_t = -Phi_tilde X_t + X_{t+1}
Z = X_now.to_numpy() - X_minus.to_numpy() @ phi_gls.T

X_sur = np.column_stack([np.ones(Z.shape[0]), Z])
z_dates = X_minus.index

# 11.2 Align Y to z_dates and enforce complete panel across assets
Rpi_wide = Rpi_wide.reindex(z_dates)

mask_y = Rpi_wide.notna().all(axis=1).to_numpy()
Y = Rpi_wide.loc[mask_y].to_numpy()
X_sur = X_sur[mask_y, :]

T_eff, N = Y.shape
K = len(state_factors)

print("SUR design X_sur:", X_sur.shape, "Y:", Y.shape)

# 11.3 Multivariate OLS
C_hat = np.linalg.inv(X_sur.T @ X_sur) @ (X_sur.T @ Y)

alpha_gls = C_hat[0, :]
B_gls     = C_hat[1:, :].T

# ============================================================
# 12) Recover mu_tilde_gls via Eq. (30) using alpha_gls, B_gls, Sigma_e_hat, and Sigma_hat
# ============================================================

# 1) Build gamma_hat_gls (Eq. 27): gamma_i = b_i' Σ b_i for each row b_i of B_gls
#    Result shape (N,)
gamma_hat_gls = np.einsum("ik,kl,il->i", B_gls, Sigma, B_gls)

# 2) Form RHS vector (alpha + 1/2 gamma), shape (N,)
rhs = alpha_gls + 0.5 * gamma_hat_gls

# 3) Compute mu_tilde_gls (Eq. 30)
Sigma_e_inv = np.linalg.inv(Sigma_e_hat)

M = B_gls.T @ Sigma_e_inv @ B_gls          # (K, K)
v = B_gls.T @ Sigma_e_inv @ rhs            # (K,)

mu_tilde_gls = -np.linalg.solve(M, v)      # (K,)

print("mu_tilde_gls shape:", mu_tilde_gls.shape)
print("mu_tilde_gls:", mu_tilde_gls)

# ============================================================
# 13) Build required inputs for pi0, pi1 minimization (minimal)
# ============================================================
# 13.0 Inputs
Phi_tilde_use = phi_gls
mu_tilde_use  = mu_tilde_gls
Sigma_use     = Sigma

# short rate must be MONTHLY decimal already
short_rate_monthly = df_model_data["short_rate_dec"]

# 13.1 Monthly short-rate regression: r_1m = delta0_m + delta1_m' X_t
df_rate_reg = (
    df_model_data[state_factors]
    .join(short_rate_monthly.rename("r_1m"), how="inner")
    .dropna()
    .sort_index()
)

y_rate = df_rate_reg["r_1m"].astype(float).to_numpy()
X_rate = sm.add_constant(df_rate_reg[state_factors].astype(float), has_constant="add").to_numpy()
b_rate = np.linalg.lstsq(X_rate, y_rate, rcond=None)[0]

delta0_m = float(b_rate[0])
delta1_m = b_rate[1:].astype(float)

print("Rate regression done (monthly). delta0_m:", delta0_m)

# ----------------------------
# 13.2 Align rx_real, X_t, X_{t+1}, r_t on SAME t-dates
# ----------------------------
rx_real_df = rx1m_real.copy().sort_index()

X_t_df   = df_model_data[state_factors].reindex(rx_real_df.index)
X_tp1_df = df_model_data[state_factors].shift(-1).reindex(rx_real_df.index)

mask = (
    rx_real_df.notna().all(axis=1)
    & X_t_df.notna().all(axis=1)
    & X_tp1_df.notna().all(axis=1)
)

rx_real_df = rx_real_df.loc[mask]
X_t_df     = X_t_df.loc[mask]
X_tp1_df   = X_tp1_df.loc[mask]

# Align r_t AFTER filtering
r_t_aligned = short_rate_monthly.reindex(rx_real_df.index).to_numpy(dtype=float)

rx_real = rx_real_df.to_numpy(dtype=float)  # (T, NR)
X_t     = X_t_df.to_numpy(dtype=float)      # (T, K)
X_tp1   = X_tp1_df.to_numpy(dtype=float)    # (T, K)

T_pi, NR = rx_real.shape
K = X_t.shape[1]
real_maturities = REAL_RET_MONTHS_FULLYEARS
max_n = int(np.max(real_maturities))

print("Step 13 aligned shapes:", rx_real.shape, X_t.shape, X_tp1.shape, r_t_aligned.shape)

# ----------------------------
# 13.3 Starting values (inflation regression)
# ----------------------------
tmp_pi = df_model_data[["infl_1m"] + state_factors].dropna().copy()
y_pi = tmp_pi["infl_1m"].astype(float).to_numpy()
X_pi = sm.add_constant(tmp_pi[state_factors].astype(float), has_constant="add").to_numpy()
b_pi = np.linalg.lstsq(X_pi, y_pi, rcond=None)[0]

x0 = np.r_[float(b_pi[0]), b_pi[1:].astype(float)]
print("Initial π from inflation regression:", x0)

# ----------------------------
# 13.4 Safe recursions + safe model rx (prevents NaN/Inf under finite differences)
# ----------------------------
# Residual scaling for conditioning
rx_std = np.std(rx_real, axis=0, ddof=0)
rx_std[~np.isfinite(rx_std)] = 1.0
rx_std[rx_std == 0] = 1.0

# Safety caps (log-units)
A_CAP  = 1e4
B_CAP  = 1e3
RX_CAP = 1e3
PEN    = 1e6

def tips_loadings_A_B_safe(pi0, pi1, mu_tilde, Phi_tilde, Sigma, delta0_m, delta1_m, max_n):
    A = np.zeros(max_n + 1, dtype=float)
    B = np.zeros((max_n + 1, len(pi1)), dtype=float)
    delta0_R = delta0_m - pi0

    for n in range(1, max_n + 1):
        B_pi_prev = B[n-1] + pi1
        Bn = Phi_tilde.T @ B_pi_prev - delta1_m
        An = A[n-1] + B_pi_prev @ mu_tilde + 0.5 * (B_pi_prev @ Sigma @ B_pi_prev) - delta0_R

        # hard guards (critical for TRF finite differences)
        if (not np.isfinite(Bn).all()) or (not np.isfinite(An)):
            return None, None
        if np.max(np.abs(Bn)) > B_CAP or abs(An) > A_CAP:
            return None, None

        B[n] = Bn
        A[n] = An

    return A, B

def model_rx_real_safe(A, B, X_t, X_tp1, r_t, maturities_months):
    T = X_t.shape[0]
    NR = len(maturities_months)
    out = np.empty((T, NR), dtype=float)

    for j, n in enumerate(maturities_months):
        v = (A[n-1] + X_tp1 @ B[n-1]) - (A[n] + X_t @ B[n]) - r_t
        if not np.isfinite(v).all():
            return None
        if np.max(np.abs(v)) > RX_CAP:
            return None
        out[:, j] = v

    return out

def residuals_pi(params):
    pi0 = float(params[0])
    pi1 = np.asarray(params[1:], dtype=float)

    A_R, B_R = tips_loadings_A_B_safe(
        pi0, pi1,
        mu_tilde_use, Phi_tilde_use, Sigma_use,
        delta0_m, delta1_m,
        max_n
    )
    if A_R is None:
        return np.full(T_pi * NR, PEN, dtype=float)

    rx_hat = model_rx_real_safe(A_R, B_R, X_t, X_tp1, r_t_aligned, real_maturities)
    if rx_hat is None:
        return np.full(T_pi * NR, PEN, dtype=float)

    res = (rx_real - rx_hat) / rx_std
    if not np.isfinite(res).all():
        return np.full(T_pi * NR, PEN, dtype=float)

    return res.ravel()
# ----------------------------
# 13.5 Solve with TRF + bounds, small diff_step (prevents NaN Jacobian)
# ----------------------------
pi0_lo, pi0_hi = -0.05, 0.05
pi1_lo, pi1_hi = -1.0, 1.0

lb = np.r_[pi0_lo, np.full(K, pi1_lo)]
ub = np.r_[pi0_hi, np.full(K, pi1_hi)]

res = least_squares(
    residuals_pi,
    x0,
    method="trf",
    bounds=(lb, ub),
    loss="linear",
    x_scale="jac",
    diff_step=1e-6,          # IMPORTANT
    ftol=1e-12, xtol=1e-12, gtol=1e-12,
    max_nfev=20000,
    verbose=2
)

pi0_hat = float(res.x[0])
pi1_hat = res.x[1:].astype(float)

print("LS success:", res.success)
print("status:", res.status, "message:", res.message)
print("pi0_hat:", pi0_hat)
print("pi1_hat:", pi1_hat)
print("SSE (scaled):", float(np.sum(res.fun**2)))


####### ============================================================
## ML  
####### ============================================================

# ============================================================
# INITIAL VALUES Double check 
# ============================================================

Phi_init   = Phi.copy()                # VAR(1) transition matrix
mu_init    = mu.copy()                 # VAR intercept
Sigma_init = Sigma.copy()              # VAR innovation covariance

K = Phi_init.shape[0]

print("Phi_init shape:", Phi_init.shape)
print("mu_init shape:", mu_init.shape)
print("Sigma_init shape:", Sigma_init.shape)

delta0_init = float(delta0)
delta1_init = delta1.to_numpy(dtype=float)

print("delta0_init:", delta0_init)
print("delta1_init shape:", delta1_init.shape)

Phi_tilde_init = phi_gls.copy()
print("Phi_tilde_init shape:", Phi_tilde_init.shape)

alpha_init = alpha_gls.copy()
B_init     = B_gls.copy()

print("alpha_init shape:", alpha_init.shape)
print("B_init shape:", B_init.shape)

mu_tilde_init = mu_tilde_gls.copy()
print("mu_tilde_init shape:", mu_tilde_init.shape)

pi0_init = float(b_pi[0])
pi1_init = b_pi[1:].astype(float)

print("pi0_init:", pi0_init)
print("pi1_init shape:", pi1_init.shape)


