import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import numpy as np
import matplotlib.pyplot as plt


### Building a df with all factors + inflation + real rate
factors_path = "model_factors_nominal_real_liquidity.xlsx"
macro_path   = "CPI_and_rate.xlsx"

df_f = pd.read_excel(factors_path).set_index("date").sort_index()

df_inf = (pd.read_excel(macro_path, sheet_name="inflation")
          .set_index("date")
          .rename(columns={"monthly log": "inflation"}))

df_r = (pd.read_excel(macro_path, sheet_name="rate")
        .set_index("date")
        .rename(columns={"rate": "short_rate"}))

df_model_data = df_f.join([df_inf, df_r], how="inner")
df_model_data = df_out[df_f.columns.tolist() + ["inflation", "short_rate"]]
infl_m = df_model_data["inflation"] / 100.0
df_model_data["infl_12m"] = infl_m.rolling(12).sum().shift(-11)

df_nominal = pd.read_excel("nominal_eom_zero_coupon_yields.xlsx").set_index("date").sort_index()
df_real = pd.read_excel("real_eom_zero_coupon_yields.xlsx").set_index("date").sort_index()

### Create dfs with log prices
mat_years_nom = (
    {"y_0.5y": 0.5}
    | {f"y_{i}y": i for i in range(1, 11)}
)
mat_years_real = (
    {f"y_{i}y": i for i in range(2, 11)}
)

df_nominal_dec = df_nominal.copy() / 100.0
df_real_dec = df_real.copy() / 100.0

df_logP_nominal = pd.DataFrame(index=df_nominal.index)
df_logP_real = pd.DataFrame(index=df_real.index)

for col, n_years in mat_years_nom.items():
    df_logP_nominal[f"logP_{n_years}y"] = -n_years * df_nominal_dec[col]
for col, n_years in mat_years_real.items():
    df_logP_real[f"logP_{n_years}y"] = -n_years * df_real_dec[col]

### now create excess returns

def rx12_from_logP(logP, y1, h=12):
    mats = sorted(float(c.split("_")[1][:-1]) for c in logP.columns)  # logP_0.5y, logP_1y, ...
    lp = logP.copy(); lp.columns = mats
    return pd.DataFrame({n: lp[n-1].shift(-h) - lp[n] - y1 for n in mats if (n-1) in mats}).iloc[:-h]

rf1y = df_nominal_dec["y_1y"].reindex(df_logP_nominal.index)
rx12_nominal = rx12_from_logP(df_logP_nominal, rf1y)
rx12_real    = rx12_from_logP(df_logP_real, rf1y.reindex(df_logP_real.index))

# --- stack nominal + real excess returns into one panel ---
rx12_nominal_stacked = rx12_nominal.copy()
rx12_nominal_stacked.columns = [f"nom_{c}" for c in rx12_nominal_stacked.columns]

rx12_real_stacked = rx12_real.copy()
rx12_real_stacked.columns = [f"real_{c}" for c in rx12_real_stacked.columns]

rx12_stacked = rx12_nominal_stacked.join(rx12_real_stacked, how="inner")

### 
state_factors = [
    "PC1_level","PC2_slope","PC3_curvature",
    "Liquidity_MA3","Real_PC1","Real_PC2"
]

# VAR on the state vector
X = df_model_data[state_factors].dropna()
factor_var_model = VAR(X)
results_factor_var_model = factor_var_model.fit(1)
results_factor_var_model.summary()
Phi = results_factor_var_model.coefs[0]
mu = results_factor_var_model.intercept
K = len(state_factors)
mu_x = np.linalg.solve(np.eye(K) - Phi, results_factor_var_model.intercept)
Sigma = results_factor_var_model.sigma_u

# OLS on 1y rate
y1_rate = "y_1y"  
df_reg = df_nominal_dec[[y1_rate]].join(df_model_data[state_factors], how="inner").dropna()
y = df_reg[y1_rate]
X = sm.add_constant(df_reg[state_factors])  # adds δ0
ols_res = sm.OLS(y, X).fit()
delta0 = ols_res.params["const"] 
delta1 = ols_res.params[state_factors]          
print(ols_res.summary())

## Equation 28 in supplementary 
df = df_model_data.copy().join(rx12_stacked, how="inner")
ret_cols = list(rx12_stacked.columns)
state_cols = [c for c in df.columns if c not in (["inflation", "short_rate", "infl_12m"] + ret_cols)]
X_lag = df[state_cols].shift(-12).add_suffix("_lag1")


rx_long = (
    df[ret_cols]
    .stack()
    .rename("rx12")
    .reset_index()
    .rename(columns={"level_1": "asset"})
)
rx_long["type"] = rx_long["asset"].str.split("_").str[0]
rx_long["maturity"] = rx_long["asset"].str.split("_").str[1].astype(float)
regressors = pd.concat([df[["infl_12m"]], df[state_cols], X_lag], axis=1).reset_index()
df_ols_long = rx_long.merge(regressors, on="date", how="left").dropna()
df_ols_long = df_ols_long.sort_values(["date", "maturity"]).reset_index(drop=True)
df_ols_long["R_pi"] = df_ols_long["rx12"]
mask_real = df_ols_long["type"] == "real"
df_ols_long.loc[mask_real, "R_pi"] += df_ols_long.loc[mask_real, "infl_12m"]

y_col = "R_pi"
x_now_cols = state_cols
x_lag_cols = [c + "_lag1" for c in state_cols]
x_cols = x_now_cols + x_lag_cols 

params_list = []
resid_list = []

asset_order = (
    df_ols_long[["asset", "maturity", "type"]]
    .drop_duplicates()
    .sort_values(["type", "maturity"])
    ["asset"]
    .tolist()
)

for m in asset_order:
    g = df_ols_long[df_ols_long["asset"] == m].sort_values("date")

    y = g[y_col].astype(float).values
    X = sm.add_constant(g[x_cols].astype(float), has_constant="add")

    fit = sm.OLS(y, X).fit()

    p = fit.params.copy()
    p.name = m
    params_list.append(p)

    g = g.copy()
    g["resid"] = fit.resid
    resid_list.append(g[["date", "asset", "resid"]])

params_df = pd.DataFrame(params_list).reindex(asset_order)
resid_df = pd.concat(resid_list)
resid_df["date"] = pd.to_datetime(resid_df["date"]).dt.to_period("M").dt.to_timestamp("M")
E_hat = resid_df.pivot(index="date", columns="asset", values="resid").sort_index()
E_hat = E_hat.reindex(columns=asset_order)

# B_ols: N x K (coefficients on current X)
B_ols = params_df[x_now_cols].to_numpy()

# -BPhi_ols: N x K (coefficients on lagged X)
Bphi_ols = -params_df[x_lag_cols].to_numpy()

# Sigma_e_hat: N x N
T = E_hat.shape[0]
Sigma_e_hat = (E_hat.T @ E_hat) / T

# recover phi 
W = np.linalg.inv(Sigma_e_hat)   # N x N
phi_gls = np.linalg.inv(B_ols.T @ W @ B_ols) @ (B_ols.T @ W @ Bphi_ols)  # K x 

### 
plt.figure(figsize=(10,4))
E_hat["nom_5.0"].plot()
plt.title("Residuals: nom_5.0")
plt.axhline(0, linestyle="--")
plt.show()
###


