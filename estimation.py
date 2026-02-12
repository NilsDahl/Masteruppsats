import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Helper: robust date parsing for Excel sheets
# ============================================================
def read_excel_date_index(path, sheet_name=0, date_col="date"):
    """
    Read an Excel sheet and return a DataFrame with a clean DatetimeIndex.
    What we do:
    1) Read sheet into a normal DataFrame
    2) Convert date_col to datetime with errors='coerce'
    3) Drop rows where date could not be parsed
    4) Set DatetimeIndex and sort
    """
    df = pd.read_excel(path, sheet_name=sheet_name)

    if date_col not in df.columns:
        raise ValueError(f"Missing '{date_col}' column in {path} / sheet={sheet_name}")

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df = df.set_index(date_col).sort_index()

    return df


# ============================================================
# 0) User inputs
# ============================================================
factors_path = "model_factors_nominal_real_liquidity.xlsx"
macro_path   = "CPI_and_rate.xlsx"

nominal_zc_path = "nominal_eom_zero_coupon_yields.xlsx"
real_zc_path    = "real_eom_zero_coupon_yields.xlsx"

# Holding period (months). H=12 means returns from t to t+12.
H = 12

# State vector (factors)
state_factors = [
    "PC1_level", "PC2_slope", "PC3_curvature",
    "Liquidity_MA3", "Real_PC1", "Real_PC2"
]

# Short rate proxy used in rx construction (here: 1y yield)
y1_rate_col = "y_1y"


# ============================================================
# 1) Load factors + macro and build model dataset
# ============================================================
# Load factors (sheet_name=0 means first sheet)
df_f = read_excel_date_index(factors_path, sheet_name=0)

# Load inflation and rename the inflation column
df_inf = (
    read_excel_date_index(macro_path, sheet_name="inflation")
    .rename(columns={"monthly log": "inflation"})
)

# Load short rate and rename
df_r = (
    read_excel_date_index(macro_path, sheet_name="rate")
    .rename(columns={"rate": "short_rate"})
)

# Merge everything into one aligned panel on the date index
df_model_data = df_f.join([df_inf, df_r], how="inner").sort_index()

# Keep only what you need (all factor columns + inflation + short rate)
keep_cols = df_f.columns.tolist() + ["inflation", "short_rate"]
df_model_data = df_model_data[keep_cols].copy()

# Inflation handling
# Assumption: inflation column is in percent. If it is already decimal, remove "/ 100.0".
infl_m = df_model_data["inflation"] / 100.0

# IMPORTANT ALIGNMENT FOR HOLDING PERIOD:
# For a return rx(t -> t+H), the matching inflation is typically sum_{t+1..t+H}.
# Implementation:
# - shift(-1): move inflation(t+1) to index t
# - rolling(H).sum(): sum inflation(t+1..t+H) aligned to the end of that shifted window
# - shift(-(H-1)): bring the sum back to index t
df_model_data["infl_12m"] = infl_m.shift(-1).rolling(H).sum().shift(-(H - 1))


# ============================================================
# 2) Load nominal/real zero coupon yields and build log prices
# ============================================================
df_nominal = read_excel_date_index(nominal_zc_path, sheet_name=0)
df_real    = read_excel_date_index(real_zc_path, sheet_name=0)

# Convert yields from percent to decimals (remove /100 if already decimals)
df_nominal_dec = df_nominal / 100.0
df_real_dec    = df_real / 100.0

# Map yield column names to maturity in years
# Adjust if your files use different column names
mat_years_nom = {"y_0.5y": 0.5} | {f"y_{i}y": i for i in range(1, 11)}
mat_years_real = {f"y_{i}y": i for i in range(2, 11)}  # your real curve starts at 2y

# Build log prices using continuous-compounding approximation:
# logP(t,n) = -n * y(t,n)
df_logP_nominal = pd.DataFrame(index=df_nominal_dec.index)
df_logP_real    = pd.DataFrame(index=df_real_dec.index)

for col, n_years in mat_years_nom.items():
    if col in df_nominal_dec.columns:
        df_logP_nominal[f"logP_{n_years}y"] = -n_years * df_nominal_dec[col]

for col, n_years in mat_years_real.items():
    if col in df_real_dec.columns:
        df_logP_real[f"logP_{n_years}y"] = -n_years * df_real_dec[col]


# ============================================================
# 3) Construct holding period excess returns rx(t -> t+H)
# ============================================================
def rx_h_from_logP(logP_df: pd.DataFrame, y1_series: pd.Series, h: int = 12) -> pd.DataFrame:
    """
    Approximate log excess holding period return for maturity n (in years):

      rx_{t->t+h}(n) = logP_{t+h}(n-1) - logP_t(n) - y1_t

    Notes:
    - Uses 1y yield y1_t as the "risk-free" proxy.
    - Requires both maturities n and (n-1) to exist in logP_df.
    - h is in months, while n is in years (this is your existing design choice).

    Returns:
    - DataFrame indexed by date t, columns are maturities n (as floats), values are rx(t->t+h).
    """
    # Extract maturity numbers from column names (e.g. "logP_2.0y" -> 2.0)
    mats = sorted(float(c.split("_")[1][:-1]) for c in logP_df.columns)

    # Work on a copy with numeric maturity columns
    lp = logP_df.copy()
    lp.columns = mats

    out = {}
    for n in mats:
        if (n - 1) in lp.columns:
            # logP_{t+h}(n-1) - logP_t(n) - y1_t
            out[n] = lp[n - 1].shift(-h) - lp[n] - y1_series

    # Drop last h rows since shift(-h) generates NaNs at the end
    return pd.DataFrame(out, index=logP_df.index).iloc[:-h]

# Risk-free proxy aligned to nominal index
rf1y_nom = df_nominal_dec[y1_rate_col].reindex(df_logP_nominal.index)

# Holding period excess returns (nominal)
rx12_nominal = rx_h_from_logP(df_logP_nominal, rf1y_nom, h=H)

# For real returns, still use nominal 1y yield as short-rate proxy (as in your original setup)
rf1y_real = rf1y_nom.reindex(df_logP_real.index)
rx12_real = rx_h_from_logP(df_logP_real, rf1y_real, h=H)

# Stack nominal + real into one wide panel with clear column names
rx12_nominal_stacked = rx12_nominal.copy()
rx12_nominal_stacked.columns = [f"nom_{c}" for c in rx12_nominal_stacked.columns]

rx12_real_stacked = rx12_real.copy()
rx12_real_stacked.columns = [f"real_{c}" for c in rx12_real_stacked.columns]

rx12_stacked = rx12_nominal_stacked.join(rx12_real_stacked, how="inner").sort_index()


# ============================================================
# 4) Estimate VAR(1) on state vector X_t
# ============================================================
# Build the state vector time series and fit a VAR(1)
X_var = df_model_data[state_factors].dropna().copy()
var_res = VAR(X_var).fit(1)

# VAR(1) objects
Phi = var_res.coefs[0]       # K x K transition matrix
mu  = var_res.intercept      # K intercept
Sigma = var_res.sigma_u      # K x K shock covariance

K = len(state_factors)

# Unconditional mean of VAR(1): mu_x = (I - Phi)^(-1) * mu
mu_x = np.linalg.solve(np.eye(K) - Phi, mu)

var_res.summary()


# ============================================================
# 5) OLS of 1y yield on state vector X_t (delta0, delta1)
# ============================================================
# Regression: y_1y(t) = delta0 + delta1' X_t + error
df_reg = (
    df_nominal_dec[[y1_rate_col]]
    .join(df_model_data[state_factors], how="inner")
    .dropna()
    .sort_index()
)

y_y1 = df_reg[y1_rate_col]
X_y1 = sm.add_constant(df_reg[state_factors])

ols_y1 = sm.OLS(y_y1, X_y1).fit()

delta0 = ols_y1.params["const"]
delta1 = ols_y1.params[state_factors]

print(ols_y1.summary())


# ============================================================
# 6) Equation (28) style stacking regression with H=12
#    Alignment in this code:
#    - return at date t is rx(t -> t+H)
#    - regress on X_t and X_{t+H} (lead H months)
# ============================================================
df = df_model_data.join(rx12_stacked, how="inner").sort_index()

ret_cols = rx12_stacked.columns.tolist()
state_cols = state_factors

# Lead state vector by H months -> X_{t+H}
X_leadH = df[state_cols].shift(-H).add_suffix(f"_lead{H}")

# Long format: one row per (date, asset)
rx_long = (
    df[ret_cols]
    .stack()
    .rename("rxH")
    .reset_index()
    .rename(columns={"level_1": "asset"})
)

# Parse asset metadata
rx_long["type"] = rx_long["asset"].str.split("_").str[0]         # "nom" or "real"
rx_long["maturity"] = rx_long["asset"].str.split("_").str[1].astype(float)

# Build regressors at the same date index t
regressors = pd.concat(
    [
        df[["infl_12m"]],      # inflation over t+1..t+H
        df[state_cols],        # X_t
        X_leadH                # X_{t+H}
    ],
    axis=1
).reset_index()

# Merge returns and regressors on date
df_ols_long = rx_long.merge(regressors, on="date", how="left").dropna()
df_ols_long = df_ols_long.sort_values(["date", "type", "maturity"]).reset_index(drop=True)

# Build R^pi:
# - nominal: R_pi = rxH
# - real:    R_pi = rxH + inflation(t+1..t+H)
df_ols_long["R_pi"] = df_ols_long["rxH"]
mask_real = df_ols_long["type"] == "real"
df_ols_long.loc[mask_real, "R_pi"] = (
    df_ols_long.loc[mask_real, "rxH"] + df_ols_long.loc[mask_real, "infl_12m"]
)

# Regression columns: current + lead
x_now_cols  = state_cols
x_lead_cols = [c + f"_lead{H}" for c in state_cols]
x_cols = x_now_cols + x_lead_cols


# ============================================================
# 7) Run separate OLS per asset (maturity x type)
#    Store coefficients and residuals
# ============================================================
params_list = []
resid_list = []

asset_order = (
    df_ols_long[["asset", "maturity", "type"]]
    .drop_duplicates()
    .sort_values(["type", "maturity"])
    ["asset"]
    .tolist()
)

for asset in asset_order:
    g = df_ols_long[df_ols_long["asset"] == asset].sort_values("date").copy()

    y = g["R_pi"].astype(float).values
    X = sm.add_constant(g[x_cols].astype(float), has_constant="add")

    fit = sm.OLS(y, X).fit()

    # Save params
    p = fit.params.copy()
    p.name = asset
    params_list.append(p)

    # Save residuals
    g["resid"] = fit.resid
    resid_list.append(g[["date", "asset", "resid"]])

params_df = pd.DataFrame(params_list).reindex(asset_order)

resid_df = pd.concat(resid_list, ignore_index=True)
resid_df["date"] = pd.to_datetime(resid_df["date"])

# Pivot residuals into wide E_hat: T x N
E_hat = (
    resid_df.pivot(index="date", columns="asset", values="resid")
    .sort_index()
    .reindex(columns=asset_order)
)


# ============================================================
# 8) Build objects used for initial conditions step (GLS-style)
# ============================================================
# B_ols: coefficients on current X_t (N x K)
B_ols = params_df[x_now_cols].to_numpy()

# B_lead_ols: coefficients on X_{t+H} (N x K)
B_lead_ols = -params_df[x_lead_cols].to_numpy()

# Residual covariance (N x N)
T = E_hat.shape[0]
Sigma_e_hat = (E_hat.T @ E_hat) / T

# GLS weight matrix
W = np.linalg.inv(Sigma_e_hat)

# "Phi-like" mapping from current block to lead block (interpret as an H-step object)
phi_gls_H = np.linalg.inv(B_ols.T @ W @ B_ols) @ (B_ols.T @ W @ B_lead_ols)


# ============================================================
# 9) Quick sanity plot of residuals for one asset
# ============================================================
plt.figure(figsize=(10, 4))
if "nom_5.0" in E_hat.columns:
    E_hat["nom_5.0"].plot()
    plt.title("Residuals: nom_5.0")
    plt.axhline(0, linestyle="--")
    plt.show()
else:
    print("Asset nom_5.0 not found in E_hat columns.")

