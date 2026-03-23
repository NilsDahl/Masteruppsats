import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
 
def read_excel_date_index(path, sheet_name=0, date_col="date"):
    df = pd.read_excel(path, sheet_name=sheet_name)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    df[date_col] = df[date_col] + pd.offsets.MonthEnd(0)
    df = df.set_index(date_col).sort_index()
    return df
 
def month_cols(df):
    return [c for c in df.columns if isinstance(c, str) and c.startswith("y_") and c.endswith("m")]
 
def col_to_m(c):
    return int(c.split("_")[1][:-1])
 
 
# ══════════════════════════════════════════════════════════════════════════════
# USER INPUTS
# ══════════════════════════════════════════════════════════════════════════════
 
factors_path              = "pcas_and_liquidity.xlsx"
nominal_zc_monthgrid_path = "zero_yields_SGB.xlsx"
real_zc_monthgrid_path    = "zero_yields_SGBIL.xlsx"
 
state_factors = [
    "PC1_level", "PC2_slope", "PC3_curvature",
    "composite_liq", "Real_PC1", "Real_PC2"
]
 
NOM_RET_MONTHS_FULLYEARS  = np.arange(12, 121, 12)
REAL_RET_MONTHS_FULLYEARS = np.arange(24, 121, 12)
 
 
# ══════════════════════════════════════════════════════════════════════════════
# LOAD ALL RAW DATA ONCE
# ══════════════════════════════════════════════════════════════════════════════
 
df_f = read_excel_date_index(factors_path, sheet_name=0)
 
df_inf = (
    read_excel_date_index("CPI_and_rate.xlsx", sheet_name="inflation")
    .rename(columns={"monthly log": "inflation"})
)
 
df_r = (
    read_excel_date_index("short_rate.xlsx", sheet_name=0)
    .rename(columns={"y_1m": "short_rate"})
)
df_r["short_rate_dec"] = df_r["short_rate"] / 12.0
df_r = df_r[["short_rate_dec"]]
 
df_nom_y  = read_excel_date_index(nominal_zc_monthgrid_path, sheet_name=0)
df_real_y = read_excel_date_index(real_zc_monthgrid_path,    sheet_name=0)
 
df_survey = pd.read_excel(
    "CPI_Inflation_Expectations.xlsx",
    sheet_name="CPI Expectations",
    header=1
)
df_survey["date"] = pd.to_datetime(df_survey["Month"], format="%b-%Y")
df_survey = df_survey.drop(columns=["Month"]).set_index("date").sort_index()
df_survey = df_survey / 100.0
 
 
# ══════════════════════════════════════════════════════════════════════════════
# CORE ESTIMATION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
 
def estimate_model(cutoff_date,
                   df_f, df_inf, df_r,
                   df_nom_y, df_real_y,
                   state_factors,
                   NOM_RET_MONTHS_FULLYEARS,
                   REAL_RET_MONTHS_FULLYEARS,
                   verbose=False):
 
    cutoff     = pd.Timestamp(cutoff_date) + pd.offsets.MonthEnd(0)
    df_f_      = df_f.loc[:cutoff].copy()
    df_inf_    = df_inf.loc[:cutoff].copy()
    df_r_      = df_r.loc[:cutoff].copy()
    df_nom_y_  = df_nom_y.loc[:cutoff].copy()
    df_real_y_ = df_real_y.loc[:cutoff].copy()
 
    df_model_data = df_f_.join([df_inf_, df_r_], how="inner").sort_index()
    infl_m = df_model_data["inflation"] / 100.0
    df_model_data["infl_1m"] = infl_m.shift(-1)
 
    # ── Log prices and excess returns ─────────────────────────────────────────
    def build_logP_from_yield_monthgrid(df_y):
        logP = pd.DataFrame(index=df_y.index)
        for m in sorted([col_to_m(c) for c in month_cols(df_y)]):
            logP[f"logP_{m}m"] = -(m / 12.0) * df_y[f"y_{m}m"]
        return logP
 
    def rx1m_from_logP(logP_df, short_rate_dec, ret_months):
        out = {}
        for n in ret_months:
            col_n, col_n1 = f"logP_{n}m", f"logP_{n-1}m"
            if col_n in logP_df.columns and col_n1 in logP_df.columns:
                out[n] = logP_df[col_n1].shift(-1) - logP_df[col_n] - short_rate_dec
        return pd.DataFrame(out, index=logP_df.index).iloc[:-1]
 
    df_logP_nom  = build_logP_from_yield_monthgrid(df_nom_y_)
    df_logP_real = build_logP_from_yield_monthgrid(df_real_y_)
    r_t          = df_model_data["short_rate_dec"].reindex(df_logP_nom.index)
 
    rx1m_nom  = rx1m_from_logP(df_logP_nom,  r_t,                             NOM_RET_MONTHS_FULLYEARS)
    rx1m_real = rx1m_from_logP(df_logP_real, r_t.reindex(df_logP_real.index), REAL_RET_MONTHS_FULLYEARS)
 
    rx1m_nom_stacked  = rx1m_nom.copy();  rx1m_nom_stacked.columns  = [f"nom_{int(c)}m"  for c in rx1m_nom_stacked.columns]
    rx1m_real_stacked = rx1m_real.copy(); rx1m_real_stacked.columns = [f"real_{int(c)}m" for c in rx1m_real_stacked.columns]
    rx1m_stacked = rx1m_nom_stacked.join(rx1m_real_stacked, how="inner").sort_index()
 
    # ── VAR(1) ────────────────────────────────────────────────────────────────
    var_res = VAR(df_model_data[state_factors].dropna().copy()).fit(1)
    Phi     = var_res.coefs[0]
    mu      = var_res.intercept
    Sigma   = var_res.sigma_u
    K       = len(state_factors)
    mu_x    = np.linalg.solve(np.eye(K) - Phi, mu)
    if verbose: print(var_res.summary())
 
    # ── Short rate OLS ────────────────────────────────────────────────────────
    df_rate_reg = df_model_data[["short_rate_dec"]].join(df_model_data[state_factors], how="inner").dropna()
    ols_rate    = sm.OLS(df_rate_reg["short_rate_dec"],
                         sm.add_constant(df_rate_reg[state_factors], has_constant="add")).fit()
    delta0_m    = ols_rate.params["const"]
    delta1_m    = ols_rate.params[state_factors]
    if verbose: print(ols_rate.summary())
 
    # ── Stacked regression (Eq. 28) ───────────────────────────────────────────
    df        = df_model_data.join(rx1m_stacked, how="inner").sort_index()
    X_lead1   = df[state_factors].shift(-1).add_suffix("_lead1")
    rx_long   = (df[rx1m_stacked.columns.tolist()].stack().rename("rx1m")
                 .reset_index().rename(columns={"level_1": "asset"}))
    rx_long["type"]       = rx_long["asset"].str.split("_").str[0]
    rx_long["maturity_m"] = rx_long["asset"].str.split("_").str[1].str.replace("m", "").astype(int)
 
    regressors_infl = pd.concat([df[["infl_1m"]], df[state_factors], X_lead1], axis=1).reset_index()
    df_ols_long     = rx_long.merge(regressors_infl, on="date", how="left").dropna()
    df_ols_long     = df_ols_long.sort_values(["date", "type", "maturity_m"]).reset_index(drop=True)
 
    df_ols_long["R_pi"] = df_ols_long["rx1m"]
    mask_real = df_ols_long["type"] == "real"
    df_ols_long.loc[mask_real, "R_pi"] = (df_ols_long.loc[mask_real, "rx1m"]
                                           + df_ols_long.loc[mask_real, "infl_1m"])
 
    x_now_cols  = state_factors
    x_lead_cols = [c + "_lead1" for c in state_factors]
    x_cols      = x_now_cols + x_lead_cols
 
    asset_order = (df_ols_long[["asset", "maturity_m", "type"]]
                   .drop_duplicates().sort_values(["type", "maturity_m"])["asset"].tolist())
 
    params_list, resid_list = [], []
    for asset in asset_order:
        g   = df_ols_long[df_ols_long["asset"] == asset].sort_values("date").copy()
        fit = sm.OLS(g["R_pi"].astype(float).values,
                     sm.add_constant(g[x_cols].astype(float), has_constant="add")).fit()
        p   = fit.params.copy(); p.name = asset; params_list.append(p)
        g["resid"] = fit.resid; resid_list.append(g[["date", "asset", "resid"]])
 
    params_df = pd.DataFrame(params_list).reindex(asset_order)
    resid_df  = pd.concat(resid_list, ignore_index=True)
    resid_df["date"] = pd.to_datetime(resid_df["date"])
 
    E_hat       = (resid_df.pivot(index="date", columns="asset", values="resid")
                   .sort_index().reindex(columns=asset_order))
    Sigma_e_hat = (E_hat.to_numpy().T @ E_hat.to_numpy()) / E_hat.shape[0]
    eigvals     = np.linalg.eigvalsh(Sigma_e_hat)
    # adding a ridge
    W = np.linalg.inv(Sigma_e_hat + max(1e-10, 1e-9 * np.max(eigvals)) * np.eye(Sigma_e_hat.shape[0]))
    if verbose: print("Sigma_e_hat eig min/max:", np.min(eigvals), np.max(eigvals))
 
    # ── Phi_tilde via GLS ─────────────────────────────────────────────────────
    B_ols   = params_df[x_lead_cols].to_numpy(dtype=float)
    phi_gls = np.linalg.solve(B_ols.T @ W @ B_ols,
                               B_ols.T @ W @ (-params_df[x_now_cols].to_numpy(dtype=float)))
    if verbose: print("max |eig(phi_gls)|:", np.max(np.abs(np.linalg.eigvals(phi_gls))))
 
    # ── alpha_gls, B_gls via SUR ──────────────────────────────────────────────
    Rpi_wide = (df_ols_long.pivot(index="date", columns="asset", values="R_pi")
                .sort_index().reindex(columns=asset_order))
    Rpi_wide.index = pd.to_datetime(Rpi_wide.index)
 
    X_minus = df_model_data[state_factors].reindex(Rpi_wide.index)
    X_now_  = df_model_data[state_factors].shift(-1).reindex(Rpi_wide.index)
    mask_z  = X_minus.notna().all(axis=1) & X_now_.notna().all(axis=1)
    X_minus, X_now_ = X_minus.loc[mask_z], X_now_.loc[mask_z]
 
    Z       = X_now_.to_numpy() - X_minus.to_numpy() @ phi_gls.T
    X_sur   = np.column_stack([np.ones(Z.shape[0]), Z])
    Rpi_wide = Rpi_wide.reindex(X_minus.index)
    mask_y   = Rpi_wide.notna().all(axis=1).to_numpy()
    C_hat    = np.linalg.solve((X_sur[mask_y]).T @ X_sur[mask_y],
                                (X_sur[mask_y]).T @ Rpi_wide.loc[mask_y].to_numpy())
    alpha_gls = C_hat[0, :]
    B_gls     = C_hat[1:, :].T
 
    gamma_hat_gls = np.einsum("ik,kl,il->i", B_gls, Sigma, B_gls)
    mu_tilde_gls  = -np.linalg.solve(B_gls.T @ W @ B_gls,
                                      B_gls.T @ W @ (alpha_gls + 0.5 * gamma_hat_gls))
    if verbose: print("mu_tilde_gls:", mu_tilde_gls)
 
    # ── pi0, pi1 via least squares on real excess returns ─────────────────────
    rx_real_df = rx1m_real.sort_index()
    X_df       = df_model_data[state_factors].astype(float)
    X_t_df_    = X_df.reindex(rx_real_df.index)
    X_tp1_df_  = X_df.shift(-1).reindex(rx_real_df.index)
    mask       = rx_real_df.notna().all(1) & X_t_df_.notna().all(1) & X_tp1_df_.notna().all(1)
    rx_real_df, X_t_df_, X_tp1_df_ = rx_real_df.loc[mask], X_t_df_.loc[mask], X_tp1_df_.loc[mask]
    r_t_pi     = df_model_data["short_rate_dec"].astype(float).reindex(rx_real_df.index).to_numpy()
 
    rx_real  = rx_real_df.to_numpy(float)
    X_t_pi   = X_t_df_.to_numpy(float)
    X_tp1_pi = X_tp1_df_.to_numpy(float)
    T_pi, NR = rx_real.shape
    max_n    = int(np.max(REAL_RET_MONTHS_FULLYEARS))
 
    tmp  = df_model_data[["infl_1m"] + state_factors].dropna()
    X_pi = sm.add_constant(tmp[state_factors].to_numpy(float), has_constant="add")
    b_pi = np.linalg.lstsq(X_pi, tmp["infl_1m"].to_numpy(float), rcond=None)[0]
    x0   = np.clip(np.r_[float(b_pi[0]), b_pi[1:].astype(float)],
                   np.r_[-0.20, np.full(K, -1.0)] + 1e-12,
                   np.r_[ 0.20, np.full(K,  1.0)] - 1e-12)
    lb   = np.r_[-0.20, np.full(K, -1.0)]
    ub   = np.r_[ 0.20, np.full(K,  1.0)]
    #liq_idx = state_factors.index("composite_liq")
    #lb[1 + liq_idx] = -1e-8
    #ub[1 + liq_idx] =  1e-8
    #x0[1 + liq_idx] =  0.0

    def tips_AB_local(pi0, pi1):
        A, B = np.zeros(max_n + 1, float), np.zeros((max_n + 1, K), float)
        d0_R = delta0_m - pi0
        for n in range(1, max_n + 1):
            Bp = B[n-1] + pi1
            B[n] = phi_gls.T @ Bp - delta1_m.to_numpy()
            A[n] = A[n-1] + Bp @ mu_tilde_gls + 0.5*(Bp @ Sigma @ Bp) - d0_R
        return A, B
 
    def rx_hat_local(A, B):
        out = np.empty((T_pi, NR), float)
        for j, n in enumerate(REAL_RET_MONTHS_FULLYEARS):
            out[:, j] = (A[n-1] + X_tp1_pi @ B[n-1]) - (A[n] + X_t_pi @ B[n]) - r_t_pi
        return out
 
    res     = least_squares(lambda p: (rx_real - rx_hat_local(*tips_AB_local(float(p[0]), p[1:].astype(float)))).ravel(),
                            x0, method="trf", bounds=(lb, ub), x_scale="jac", diff_step=1e-6,
                            ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=20000,
                            verbose=2 if verbose else 0)
    pi0_hat = float(res.x[0])
    pi1_hat = res.x[1:].astype(float)
    if verbose: print("pi0_hat:", pi0_hat, "\npi1_hat:", pi1_hat)
 
    # ── AB recursions + yield pricing ─────────────────────────────────────────
    nom_months_  = np.array(sorted([col_to_m(c) for c in month_cols(df_nom_y_)]),  dtype=int)
    real_months_ = np.array(sorted([col_to_m(c) for c in month_cols(df_real_y_)]), dtype=int)
    max_n_y      = int(max(nom_months_.max(), real_months_.max()))
    delta0_      = float(delta0_m)
    delta1_      = delta1_m.to_numpy(float)
 
    def AB_nominal(Phi_dyn, mu_dyn, d0, d1, max_n):
        A, B = np.zeros(max_n + 1, float), np.zeros((max_n + 1, K), float)
        for n in range(1, max_n + 1):
            B[n] = Phi_dyn.T @ B[n-1] - d1
            A[n] = A[n-1] + B[n-1] @ mu_dyn + 0.5*(B[n-1] @ Sigma @ B[n-1]) - d0
        return A, B
 
    def AB_real(Phi_dyn, mu_dyn, d0, d1, p0, p1, max_n):
        A, B = np.zeros(max_n + 1, float), np.zeros((max_n + 1, K), float)
        d0_R = d0 - p0
        for n in range(1, max_n + 1):
            Bp = B[n-1] + p1
            B[n] = Phi_dyn.T @ Bp - d1
            A[n] = A[n-1] + Bp @ mu_dyn + 0.5*(Bp @ Sigma @ Bp) - d0_R
        return A, B
 
    def yhat_from_AB(A, B, X_index, months):
        X_pr  = df_model_data[state_factors].reindex(X_index).astype(float).dropna()
        X_arr = X_pr.to_numpy(float)
        out   = pd.DataFrame(index=X_pr.index)
        for n in months:
            out[f"y_{n}m"] = -(A[n] + X_arr @ B[n]) / (n / 12.0)
        return out, X_pr
 
    A_nom_Q,  B_nom_Q  = AB_nominal(phi_gls, mu_tilde_gls, delta0_, delta1_, max_n_y)
    A_real_Q, B_real_Q = AB_real(   phi_gls, mu_tilde_gls, delta0_, delta1_, pi0_hat, pi1_hat, max_n_y)
    A_nom_P,  B_nom_P  = AB_nominal(Phi,     mu,           delta0_, delta1_, max_n_y)
    A_real_P, B_real_P = AB_real(   Phi,     mu,           delta0_, delta1_, pi0_hat, pi1_hat, max_n_y)
 
    yhat_Q,  Xpr_nom_Q  = yhat_from_AB(A_nom_Q,  B_nom_Q,  df_nom_y_.index,  nom_months_)
    yhatR_Q, Xpr_real_Q = yhat_from_AB(A_real_Q, B_real_Q, df_real_y_.index, real_months_)
    yhat_P,  Xpr_nom_P  = yhat_from_AB(A_nom_P,  B_nom_P,  df_nom_y_.index,  nom_months_)
    yhatR_P, Xpr_real_P = yhat_from_AB(A_real_P, B_real_P, df_real_y_.index, real_months_)
 
    # ── BEI decomposition ─────────────────────────────────────────────────────
    liq_idx = state_factors.index("composite_liq")
    common  = sorted(set(yhat_Q.columns).intersection(yhatR_Q.columns)
                     .intersection(yhat_P.columns).intersection(yhatR_P.columns))
 
    liq_nom_Q  = pd.DataFrame(index=yhat_Q.index)
    liq_real_Q = pd.DataFrame(index=yhatR_Q.index)
    liq_nom_P  = pd.DataFrame(index=yhat_P.index)
    liq_real_P = pd.DataFrame(index=yhatR_P.index)
 
    for n in nom_months_:
        col = f"y_{n}m"
        if col in common:
            tau = n / 12.0
            liq_nom_Q[col] = -(B_nom_Q[n][liq_idx] * Xpr_nom_Q.iloc[:, liq_idx].values) / tau
            liq_nom_P[col] = -(B_nom_P[n][liq_idx] * Xpr_nom_P.iloc[:, liq_idx].values) / tau
 
    for n in real_months_:
        col = f"y_{n}m"
        if col in common:
            tau = n / 12.0
            liq_real_Q[col] = -(B_real_Q[n][liq_idx] * Xpr_real_Q.iloc[:, liq_idx].values) / tau
            liq_real_P[col] = -(B_real_P[n][liq_idx] * Xpr_real_P.iloc[:, liq_idx].values) / tau
 
    bei_Q     = yhat_Q[common]  - yhatR_Q[common]
    bei_Q_adj = (yhat_Q[common] - liq_nom_Q[common]) - (yhatR_Q[common] - liq_real_Q[common])
    bei_P_adj = (yhat_P[common] - liq_nom_P[common]) - (yhatR_P[common] - liq_real_P[common])
 
    # ── Excess return helpers (for fit diagnostics) ───────────────────────────
    def rxhat_nominal_from_AB(B, Phi_tilde, mu_tilde, X_t_df, X_tp1_df, months):
        idx = X_t_df.index.intersection(X_tp1_df.index)
        Xt  = X_t_df.reindex(idx).to_numpy(float)
        Xtp1 = X_tp1_df.reindex(idx).to_numpy(float)
        out = pd.DataFrame(index=idx)
        for n in months:
            bp = B[n-1]
            out[f"rx_{n}m"] = -(bp @ mu_tilde + 0.5*(bp @ Sigma @ bp)) - (Xt @ Phi_tilde.T @ bp) + (Xtp1 @ bp)
        return out
 
    def rxhat_real_from_AB(B, Phi_tilde, mu_tilde, p0, p1, X_t_df, X_tp1_df, months):
        idx = X_t_df.index.intersection(X_tp1_df.index)
        Xt  = X_t_df.reindex(idx).to_numpy(float)
        Xtp1 = X_tp1_df.reindex(idx).to_numpy(float)
        out = pd.DataFrame(index=idx)
        for n in months:
            bp  = B[n-1]
            bpp = bp + p1
            out[f"rx_{n}m"] = -(p0 + bpp @ mu_tilde + 0.5*(bpp @ Sigma @ bpp)) - (Xt @ Phi_tilde.T @ bpp) + (Xtp1 @ bp)
        return out
 
    X_t_df_full  = df_model_data[state_factors].astype(float)
    X_tp1_df_full = X_t_df_full.shift(-1)
 
    rxhat_nom_Q  = rxhat_nominal_from_AB(B_nom_Q,  phi_gls, mu_tilde_gls,
                                          X_t_df_full, X_tp1_df_full, NOM_RET_MONTHS_FULLYEARS)
    rxhat_real_Q = rxhat_real_from_AB(   B_real_Q, phi_gls, mu_tilde_gls,
                                          pi0_hat, pi1_hat,
                                          X_t_df_full, X_tp1_df_full, REAL_RET_MONTHS_FULLYEARS)
 
    rxobs_nom_Q  = rx1m_nom.copy();  rxobs_nom_Q.columns  = [f"rx_{int(c)}m" for c in rxobs_nom_Q.columns]
    rxobs_real_Q = rx1m_real.copy(); rxobs_real_Q.columns = [f"rx_{int(c)}m" for c in rxobs_real_Q.columns]
    rxobs_nom_Q  = rxobs_nom_Q.reindex(rxhat_nom_Q.index)[rxhat_nom_Q.columns]
    rxobs_real_Q = rxobs_real_Q.reindex(rxhat_real_Q.index)[rxhat_real_Q.columns]
 
    return dict(
        # decomposition outputs
        E_inf        = bei_P_adj,
        IRP          = bei_Q_adj - bei_P_adj,
        LP           = bei_Q - bei_Q_adj,
        BEI_obs      = bei_Q,
        # inflation fit
        infl_m       = infl_m,
        pi0_hat      = pi0_hat,
        pi1_hat      = pi1_hat,
        # parameters
        phi_gls      = phi_gls,
        mu_tilde_gls = mu_tilde_gls,
        Phi          = Phi,
        mu           = mu,
        mu_x         = mu_x,
        Sigma        = Sigma,
        delta0       = delta0_m,
        delta1       = delta1_m,
        # model data (needed for downstream plots)
        df_model_data = df_model_data,
        rx1m_nom      = rx1m_nom,
        rx1m_real     = rx1m_real,
        rx_real_df    = rx_real_df,
        # yield fit objects
        yhat_Q  = yhat_Q,   yhatR_Q = yhatR_Q,
        yhat_P  = yhat_P,   yhatR_P = yhatR_P,
        yobs_Q  = df_nom_y_.reindex(yhat_Q.index)[yhat_Q.columns],
        yobsR_Q = df_real_y_.reindex(yhatR_Q.index)[yhatR_Q.columns],
        rxhat_nom_Q  = rxhat_nom_Q,  rxobs_nom_Q  = rxobs_nom_Q,
        rxhat_real_Q = rxhat_real_Q, rxobs_real_Q = rxobs_real_Q,
        # liquidity components (needed for 5-10y forward plot)
        liq_nom_Q  = liq_nom_Q,  liq_real_Q = liq_real_Q,
        liq_nom_P  = liq_nom_P,  liq_real_P = liq_real_P,
        # AB arrays + month arrays (needed for OOS re-scoring)
        A_nom_Q = A_nom_Q, B_nom_Q = B_nom_Q,
        A_nom_P = A_nom_P, B_nom_P = B_nom_P,
        B_real_Q = B_real_Q,
        B_real_P = B_real_P,
        nom_months  = nom_months_,
        real_months = real_months_,
        liq_idx = liq_idx
    )
 
 
# ══════════════════════════════════════════════════════════════════════════════
# OOS RE-SCORING HELPER
# ══════════════════════════════════════════════════════════════════════════════
 
def rescore_E_inf(res_oos, df_f, state_factors):
    """Re-apply vintage parameters to the full factor history."""
    phi_gls      = res_oos["phi_gls"]
    mu_tilde_gls = res_oos["mu_tilde_gls"]
    Phi          = res_oos["Phi"]
    mu           = res_oos["mu"]
    Sigma        = res_oos["Sigma"]
    delta0       = float(res_oos["delta0"])
    delta1       = res_oos["delta1"].to_numpy(float)
    pi0          = float(res_oos["pi0_hat"])
    pi1          = res_oos["pi1_hat"].astype(float)
    nom_months_  = res_oos["nom_months"]
    real_months_ = res_oos["real_months"]
    K            = len(state_factors)
    liq_idx      = state_factors.index("composite_liq")
    max_n_y      = int(max(nom_months_.max(), real_months_.max()))
 
    def AB_nominal(Phi_dyn, mu_dyn, d0, d1, max_n):
        A, B = np.zeros(max_n + 1, float), np.zeros((max_n + 1, K), float)
        for n in range(1, max_n + 1):
            B[n] = Phi_dyn.T @ B[n-1] - d1
            A[n] = A[n-1] + B[n-1] @ mu_dyn + 0.5*(B[n-1] @ Sigma @ B[n-1]) - d0
        return A, B
 
    def AB_real(Phi_dyn, mu_dyn, d0, d1, p0, p1, max_n):
        A, B = np.zeros(max_n + 1, float), np.zeros((max_n + 1, K), float)
        d0_R = d0 - p0
        for n in range(1, max_n + 1):
            Bp = B[n-1] + p1
            B[n] = Phi_dyn.T @ Bp - d1
            A[n] = A[n-1] + Bp @ mu_dyn + 0.5*(Bp @ Sigma @ Bp) - d0_R
        return A, B
 
    X_full  = df_f[state_factors].dropna()
    X_arr   = X_full.to_numpy(float)
    liq_vals = X_full.iloc[:, liq_idx].values
 
    def yhat_full(A, B, months):
        out = pd.DataFrame(index=X_full.index)
        for n in months:
            out[f"y_{n}m"] = -(A[n] + X_arr @ B[n]) / (n / 12.0)
        return out
 
    A_nom_Q,  B_nom_Q  = AB_nominal(phi_gls, mu_tilde_gls, delta0, delta1, max_n_y)
    A_real_Q, B_real_Q = AB_real(   phi_gls, mu_tilde_gls, delta0, delta1, pi0, pi1, max_n_y)
    A_nom_P,  B_nom_P  = AB_nominal(Phi,     mu,           delta0, delta1, max_n_y)
    A_real_P, B_real_P = AB_real(   Phi,     mu,           delta0, delta1, pi0, pi1, max_n_y)
 
    yhat_Q_  = yhat_full(A_nom_Q,  B_nom_Q,  nom_months_)
    yhatR_Q_ = yhat_full(A_real_Q, B_real_Q, real_months_)
    yhat_P_  = yhat_full(A_nom_P,  B_nom_P,  nom_months_)
    yhatR_P_ = yhat_full(A_real_P, B_real_P, real_months_)
 
    common = sorted(set(yhat_Q_.columns).intersection(yhatR_Q_.columns)
                    .intersection(yhat_P_.columns).intersection(yhatR_P_.columns))
 
    liq_nom_P_  = pd.DataFrame(index=X_full.index)
    liq_real_P_ = pd.DataFrame(index=X_full.index)
    for n in nom_months_:
        col = f"y_{n}m"
        if col in common:
            liq_nom_P_[col] = -(B_nom_P[n][liq_idx] * liq_vals) / (n / 12.0)
    for n in real_months_:
        col = f"y_{n}m"
        if col in common:
            liq_real_P_[col] = -(B_real_P[n][liq_idx] * liq_vals) / (n / 12.0)
 
    return ((yhat_P_[common] - liq_nom_P_[common]) -
            (yhatR_P_[common] - liq_real_P_[common]))
 
 
# ══════════════════════════════════════════════════════════════════════════════
# FULL-SAMPLE ESTIMATION
# ══════════════════════════════════════════════════════════════════════════════
 
results = estimate_model(
    cutoff_date               = "2099-12-31",
    df_f                      = df_f,
    df_inf                    = df_inf,
    df_r                      = df_r,
    df_nom_y                  = df_nom_y,
    df_real_y                 = df_real_y,
    state_factors             = state_factors,
    NOM_RET_MONTHS_FULLYEARS  = NOM_RET_MONTHS_FULLYEARS,
    REAL_RET_MONTHS_FULLYEARS = REAL_RET_MONTHS_FULLYEARS,
    verbose                   = True,
)
 
# Unpack everything needed by downstream code
E_inf         = results["E_inf"]
IRP           = results["IRP"]
LP            = results["LP"]
BEI_obs       = results["BEI_obs"]
infl_m        = results["infl_m"]
pi0_hat       = results["pi0_hat"]
pi1_hat       = results["pi1_hat"]
phi_gls       = results["phi_gls"]
mu_tilde_gls  = results["mu_tilde_gls"]
Phi           = results["Phi"]
mu            = results["mu"]
Sigma         = results["Sigma"]
delta0_m      = results["delta0"]
delta1_m      = results["delta1"]
df_model_data = results["df_model_data"]
rx1m_nom      = results["rx1m_nom"]
rx1m_real     = results["rx1m_real"]
rx_real_df    = results["rx_real_df"]
yhat_Q        = results["yhat_Q"]
yhatR_Q       = results["yhatR_Q"]
yhat_P        = results["yhat_P"]
yhatR_P       = results["yhatR_P"]
yobs_Q        = results["yobs_Q"]
yobsR_Q       = results["yobsR_Q"]
rxhat_nom_Q   = results["rxhat_nom_Q"]
rxobs_nom_Q   = results["rxobs_nom_Q"]
rxhat_real_Q  = results["rxhat_real_Q"]
rxobs_real_Q  = results["rxobs_real_Q"]
liq_nom_Q     = results["liq_nom_Q"]
liq_real_Q    = results["liq_real_Q"]
liq_nom_P     = results["liq_nom_P"]
liq_real_P    = results["liq_real_P"]
B_real_Q      = results["B_real_Q"]
liq_idx       = results["liq_idx"]
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PLOT: MONTHLY INFLATION FIT
# ══════════════════════════════════════════════════════════════════════════════
 
X_pi_df   = df_model_data[state_factors].astype(float)
pi_hat_1m = pd.Series(pi0_hat + X_pi_df.to_numpy() @ pi1_hat, index=X_pi_df.index, name="pi_hat_1m")
pi_obs_1m = df_model_data["infl_1m"].astype(float).rename("pi_obs_1m")
tmp_pi    = pd.concat([pi_obs_1m, pi_hat_1m], axis=1).dropna()
 
plt.figure(figsize=(10, 4))
plt.plot(tmp_pi.index, tmp_pi["pi_obs_1m"] * 100, linewidth=2, label="Observed inflation (1m)")
plt.plot(tmp_pi.index, tmp_pi["pi_hat_1m"] * 100, linewidth=2, label="Model-implied inflation (1m)")
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Monthly inflation: observed vs model-implied")
plt.ylabel("Percent")
plt.legend()
plt.tight_layout()
plt.show()

# Real bond excess return errors
rx_real_arr = rx_real_df.to_numpy(float)
X_t_arr     = df_model_data[state_factors].astype(float).reindex(rx_real_df.index).to_numpy(float)
X_tp1_arr   = df_model_data[state_factors].astype(float).shift(-1).reindex(rx_real_df.index).to_numpy(float)
r_t_arr     = df_model_data["short_rate_dec"].astype(float).reindex(rx_real_df.index).to_numpy()

max_n_chk = int(np.max(REAL_RET_MONTHS_FULLYEARS))
A_pi2 = np.zeros(max_n_chk + 1, float)
B_pi2 = np.zeros((max_n_chk + 1, len(state_factors)), float)
d0_R  = float(delta0_m) - pi0_hat
for n in range(1, max_n_chk + 1):
    Bp = B_pi2[n-1] + pi1_hat
    B_pi2[n] = phi_gls.T @ Bp - delta1_m.to_numpy()
    A_pi2[n] = A_pi2[n-1] + Bp @ mu_tilde_gls + 0.5*(Bp @ Sigma @ Bp) - d0_R

rx_hat_chk = np.empty((X_t_arr.shape[0], len(REAL_RET_MONTHS_FULLYEARS)), float)
for j, n in enumerate(REAL_RET_MONTHS_FULLYEARS):
    rx_hat_chk[:, j] = (A_pi2[n-1] + X_tp1_arr @ B_pi2[n-1]) - (A_pi2[n] + X_t_arr @ B_pi2[n]) - r_t_arr

err_df = pd.DataFrame(rx_real_arr - rx_hat_chk, index=rx_real_df.index,
                      columns=[f"{int(n)}m" for n in REAL_RET_MONTHS_FULLYEARS])

plt.figure(figsize=(11, 5))
for c in err_df.columns:
    plt.plot(err_df.index, err_df[c], linewidth=1, alpha=0.8, label=c)
plt.axhline(0, linestyle="--", linewidth=1)
plt.title("Real bond 1m excess return errors (Observed - Model)")
plt.ylabel("Decimal return")
plt.legend(title="Maturity", ncol=3, fontsize=9)
plt.tight_layout()
plt.show()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# PLOT: YIELD AND RETURN FIT
# ══════════════════════════════════════════════════════════════════════════════
 
mat_labels = {12: "1-year", 24: "2-year", 60: "5-year", 120: "10-year"}
 
def plot_3panel(mats, obs_df, hat_df, title, prefix):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for ax, m in zip(axes, mats):
        col = f"{prefix}{m}m"
        if col not in obs_df.columns:
            ax.set_visible(False); continue
        ax.plot(obs_df.index, obs_df[col] * 10_000, label="Observed", linewidth=1.8, color="#2a6ebb")
        ax.plot(hat_df.index, hat_df[col] * 10_000, label="Model",    linewidth=1.8, color="#e07b39")
        ax.set_title(mat_labels[m], fontsize=11)
        ax.set_ylabel("Basis points", fontsize=10)
        ax.legend(fontsize=9, loc="upper right")
        ax.axhline(0, color="black", lw=0.7, linestyle=":")
        ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        ax.set_axisbelow(True)
    plt.tight_layout()
    return fig

fig = plot_3panel([12, 60, 120], yobs_Q,       yhat_Q,       "Nominal Yield Fit",            "y_")
fig.savefig("nominal_yield_fit.png", dpi=150, bbox_inches="tight"); plt.show()

fig = plot_3panel([24, 60, 120], yobsR_Q,      yhatR_Q,      "Real Yield Fit",               "y_")
fig.savefig("real_yield_fit.png",    dpi=150, bbox_inches="tight"); plt.show()

fig = plot_3panel([12, 60, 120], rxobs_nom_Q,  rxhat_nom_Q,  "Nominal 1m Excess Return Fit", "rx_")
fig.savefig("nominal_ret_fit.png",   dpi=150, bbox_inches="tight"); plt.show()

fig = plot_3panel([24, 60, 120], rxobs_real_Q, rxhat_real_Q, "Real 1m Excess Return Fit",    "rx_")
fig.savefig("real_ret_fit.png",      dpi=150, bbox_inches="tight"); plt.show()
 
 
# ══════════════════════════════════════════════════════════════════════════════
# BEI DECOMPOSITION PLOTS
# ══════════════════════════════════════════════════════════════════════════════

colors = {"BEI": "#2c2c2c", "E_inf": "#e07b39", "IRP": "#2a6ebb", "LP": "#3aaa6e"}

for m in [24, 60, 120]:
    col = f"y_{m}m"
    if col not in E_inf.columns: continue
    idx    = (BEI_obs[col].dropna().index.intersection(IRP[col].dropna().index)
              .intersection(LP[col].dropna().index).intersection(E_inf[col].dropna().index))
    s_bei  = BEI_obs[col].loc[idx] * 10_000
    s_irp  = IRP[col].loc[idx]     * 10_000
    s_lp   = LP[col].loc[idx]      * 10_000
    s_einf = E_inf[col].loc[idx]   * 10_000

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(idx, s_bei,  color=colors["BEI"],   linewidth=2.0, linestyle="--", label="Breakeven inflation (raw)")
    ax.plot(idx, s_einf, color=colors["E_inf"], linewidth=2.0, label="Expected inflation ($E^P[\\pi]$)")
    ax.plot(idx, s_irp,  color=colors["IRP"],   linewidth=2.0, label="Inflation risk premium")
    ax.plot(idx, s_lp,   color=colors["LP"],    linewidth=2.0, label="Liquidity premium")
    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_title(f"BEI Decomposition — {mat_labels.get(m, f'{m}m')} maturity", fontsize=13, fontweight="bold")
    ax.set_ylabel("Basis points", fontsize=11)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.legend(loc="lower center", ncol=4, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.08))
    print(f"{mat_labels.get(m, f'{m}m')}: max decomposition residual = {(s_einf + s_irp + s_lp - s_bei).abs().max():.4f} bp")
    plt.tight_layout()
    fname_map = {24: "BEI_decomp_2y.png", 60: "BEI_decomp_5y.png", 120: "BEI_decomp_10y.png"}
    fig.savefig(fname_map[m], dpi=150, bbox_inches="tight")
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# FORWARD BEI DECOMPOSITION — 5y5y and 2y2y
# ═════════════════════════════════════════════════════════════════════════════

def to_fwd_5y5y(zc_df):
    """5y5y forward: [10*y120 - 5*y60] / 5 = 2*y120 - y60"""
    return 2.0 * zc_df["y_120m"] - zc_df["y_60m"]

def to_fwd_2y2y(zc_df):
    """2y2y forward: [4*y48 - 2*y24] / 2 = 2*y48 - y24"""
    return 2.0 * zc_df["y_48m"] - zc_df["y_24m"]

def fwd_liq_5y5y(liq_df):
    return 2.0 * liq_df["y_120m"] - liq_df["y_60m"]

def fwd_liq_2y2y(liq_df):
    return 2.0 * liq_df["y_48m"] - liq_df["y_24m"]

for fwd_label, fwd_nom, fwd_real, fwd_liq_nom, fwd_liq_real in [
    ("5y5y",
     to_fwd_5y5y(yhat_Q),  to_fwd_5y5y(yhatR_Q),
     fwd_liq_5y5y(liq_nom_Q), fwd_liq_5y5y(liq_real_Q),
    ),
    ("2y2y",
     to_fwd_2y2y(yhat_Q),  to_fwd_2y2y(yhatR_Q),
     fwd_liq_2y2y(liq_nom_Q), fwd_liq_2y2y(liq_real_Q),
    ),
]:
    fwd_nom_P      = to_fwd_5y5y(yhat_P)      if fwd_label == "5y5y" else to_fwd_2y2y(yhat_P)
    fwd_real_P     = to_fwd_5y5y(yhatR_P)     if fwd_label == "5y5y" else to_fwd_2y2y(yhatR_P)
    fwd_liq_nom_P  = fwd_liq_5y5y(liq_nom_P)  if fwd_label == "5y5y" else fwd_liq_2y2y(liq_nom_P)
    fwd_liq_real_P = fwd_liq_5y5y(liq_real_P) if fwd_label == "5y5y" else fwd_liq_2y2y(liq_real_P)

    fwd_bei_Q     = fwd_nom   - fwd_real
    fwd_bei_Q_adj = (fwd_nom  - fwd_liq_nom)   - (fwd_real  - fwd_liq_real)
    fwd_bei_P_adj = (fwd_nom_P - fwd_liq_nom_P) - (fwd_real_P - fwd_liq_real_P)
    fwd_LP        = fwd_bei_Q     - fwd_bei_Q_adj
    fwd_IRP       = fwd_bei_Q_adj - fwd_bei_P_adj
    fwd_E_inf     = fwd_bei_P_adj

    idx = (fwd_bei_Q.dropna().index.intersection(fwd_IRP.dropna().index)
           .intersection(fwd_LP.dropna().index).intersection(fwd_E_inf.dropna().index))
    s_bei  = fwd_bei_Q.loc[idx]  * 10_000
    s_irp  = fwd_IRP.loc[idx]    * 10_000
    s_lp   = fwd_LP.loc[idx]     * 10_000
    s_einf = fwd_E_inf.loc[idx]  * 10_000

    print(f"{fwd_label} forward: max decomposition residual = {(s_einf + s_irp + s_lp - s_bei).abs().max():.4f} bp")

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(idx, s_bei,  color=colors["BEI"],   linewidth=2.0, linestyle="--", label=f"{fwd_label} forward breakeven (raw)")
    ax.plot(idx, s_einf, color=colors["E_inf"], linewidth=2.0, label="Expected inflation ($E^P[\\pi]$)")
    ax.plot(idx, s_irp,  color=colors["IRP"],   linewidth=2.0, label="Inflation risk premium")
    ax.plot(idx, s_lp,   color=colors["LP"],    linewidth=2.0, label="Liquidity premium")
    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_title(f"BEI Decomposition — {fwd_label} forward", fontsize=13, fontweight="bold")
    ax.set_ylabel("Basis points", fontsize=11)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    fig.legend(loc="lower center", ncol=4, fontsize=10, frameon=True,
               bbox_to_anchor=(0.5, -0.08))
    plt.tight_layout()
    fig.savefig(f"BEI_decomp_fwd_{fwd_label}.png", dpi=150, bbox_inches="tight")
    plt.show()
 
# ══════════════════════════════════════════════════════════════════════════════
# FIT DIAGNOSTICS LATEX TABLES
# ══════════════════════════════════════════════════════════════════════════════
 
def pricing_errors(obs_df, hat_df, maturities, prefix, scale=10_000):
    rows = []
    for m in maturities:
        col = f"{prefix}{m}m"
        if col not in obs_df.columns or col not in hat_df.columns: continue
        err = (obs_df[col] - hat_df[col]).dropna() * scale
        rows.append({"n": m, "Mean": round(err.mean(), 3), "Std": round(err.std(), 3),
                     "Skew": round(skew(err), 3), "Kurt": round(kurtosis(err, fisher=False), 3)})
    return pd.DataFrame(rows).set_index("n")
 
def two_panel_latex(df_top, df_bot, panel_top, panel_bot, caption, label, note=None):
    col_fmt = "r" + "r" * len(df_top.columns)
    cols    = list(df_top.columns)
    header  = "$n$ (months) & " + " & ".join(cols) + r" \\"

    def rows_block(df):
        return "\n".join(
            f"        {idx} & " + " & ".join(f"{v:.3f}" for v in row.values) + r" \\"
            for idx, row in df.iterrows()
        )

    tex = (
        rf"\begin{{table}}[htbp]" + "\n"
        rf"    \centering" + "\n"
        rf"    \caption{{{caption}}}" + "\n"
        rf"    \label{{{label}}}" + "\n"
        rf"    \begin{{tabular}}{{{col_fmt}}}" + "\n"
        rf"    \toprule" + "\n"
        rf"        {header}" + "\n"
        rf"    \midrule" + "\n"
        rf"        \multicolumn{{{1 + len(cols)}}}{{l}}{{\small\textit{{Panel A: {panel_top}}}}} \\" + "\n"
        rf"    \midrule" + "\n"
        + rows_block(df_top) + "\n"
        rf"    \midrule" + "\n"
        rf"        \multicolumn{{{1 + len(cols)}}}{{l}}{{\small\textit{{Panel B: {panel_bot}}}}} \\" + "\n"
        rf"    \midrule" + "\n"
        + rows_block(df_bot) + "\n"
        rf"    \bottomrule" + "\n"
        rf"    \end{{tabular}}"
    )
    if note:
        tex += (
            "\n"
            r"    \begin{minipage}{\linewidth}" + "\n"
            r"        \vspace{4pt}" + "\n"
            rf"        \footnotesize \textit{{Note:}} {note}" + "\n"
            r"    \end{minipage}"
        )
    tex += "\n\\end{table}"
    return tex


def rmse_latex(is_df, oos_df, caption, label, note=None):
    cols    = list(is_df.columns)
    col_fmt = "l" + "r" * len(cols)
    header  = "Horizon & " + " & ".join(cols) + r" \\"

    def rows_block(df):
        return "\n".join(
            f"        {idx} & " + " & ".join(
                f"{v:.1f}" if not np.isnan(v) else "---"
                for v in row.values
            ) + r" \\"
            for idx, row in df.iterrows()
        )

    tex = (
        rf"\begin{{table}}[htbp]" + "\n"
        rf"    \centering" + "\n"
        rf"    \caption{{{caption}}}" + "\n"
        rf"    \label{{{label}}}" + "\n"
        rf"    \begin{{tabular}}{{{col_fmt}}}" + "\n"
        rf"    \toprule" + "\n"
        rf"        {header}" + "\n"
        rf"    \midrule" + "\n"
        rf"        \multicolumn{{{1 + len(cols)}}}{{l}}{{\small\textit{{Panel A: In-sample}}}} \\" + "\n"
        rf"    \midrule" + "\n"
        + rows_block(is_df) + "\n"
        rf"    \midrule" + "\n"
        rf"        \multicolumn{{{1 + len(cols)}}}{{l}}{{\small\textit{{Panel B: Out-of-sample (2009:01 onward)}}}} \\" + "\n"
        rf"    \midrule" + "\n"
        + rows_block(oos_df) + "\n"
        rf"    \bottomrule" + "\n"
        rf"    \end{{tabular}}"
    )
    if note:
        tex += (
            "\n"
            r"    \begin{minipage}{\linewidth}" + "\n"
            r"        \vspace{4pt}" + "\n"
            rf"        \footnotesize \textit{{Note:}} {note}" + "\n"
            r"    \end{minipage}"
        )
    tex += "\n\\end{table}"
    return tex
 
tbl_nom_y   = pricing_errors(yobs_Q,        yhat_Q,        [12, 24, 36, 60, 84, 120], "y_")
tbl_real_y  = pricing_errors(yobsR_Q,       yhatR_Q,       [24, 36, 60, 84, 120],     "y_")
tbl_nom_rx  = pricing_errors(rxobs_nom_Q,   rxhat_nom_Q,   [12, 24, 36, 60, 84, 120], "rx_")
tbl_real_rx = pricing_errors(rxobs_real_Q,  rxhat_real_Q,  [24, 36, 60, 84, 120],     "rx_")
 
note_y  = ("Mean, Std, Skew, and Kurt refer to the sample mean, standard deviation, "
           "skewness, and kurtosis of yield pricing errors in basis points. Sample: 2004:01--2025:12.")
note_rx = ("Mean, Std, Skew, and Kurt refer to the sample mean, standard deviation, "
           "skewness, and kurtosis of excess return pricing errors in basis points. Sample: 2004:01--2025:12.")
 
print(two_panel_latex(tbl_nom_y, tbl_real_y,
                      panel_top="Nominal SGB yield pricing errors",
                      panel_bot="Real SGBIL yield pricing errors",
                      caption="Yield fit diagnostics.",
                      label="tab:yield_fit", note=note_y))
 
print(two_panel_latex(tbl_nom_rx, tbl_real_rx,
                      panel_top="Nominal SGB excess return pricing errors",
                      panel_bot="Real SGBIL excess return pricing errors",
                      caption="Excess return fit diagnostics.",
                      label="tab:ret_fit", note=note_rx))
 
 
# ══════════════════════════════════════════════════════════════════════════════
# RECURSIVE OOS LOOP
# ══════════════════════════════════════════════════════════════════════════════
 
oos_start     = "2009-01-01"
reestim_dates = pd.date_range(start=oos_start, end="2025-12-31", freq="YS-JAN")
oos_E_inf_list = []
 
for reestim_date in reestim_dates:
    cutoff = reestim_date - pd.offsets.MonthEnd(1)
    print(f"\n>>> OOS estimation: training up to {cutoff.date()} ...")
 
    res_oos = estimate_model(
        cutoff_date               = cutoff,
        df_f                      = df_f,
        df_inf                    = df_inf,
        df_r                      = df_r,
        df_nom_y                  = df_nom_y,
        df_real_y                 = df_real_y,
        state_factors             = state_factors,
        NOM_RET_MONTHS_FULLYEARS  = NOM_RET_MONTHS_FULLYEARS,
        REAL_RET_MONTHS_FULLYEARS = REAL_RET_MONTHS_FULLYEARS,
        verbose                   = False,
    )
 
    E_inf_full  = rescore_E_inf(res_oos, df_f, state_factors)
    next_cutoff = reestim_date + pd.offsets.YearEnd(1)
    oos_E_inf_list.append(E_inf_full.loc[reestim_date:next_cutoff])
 
E_inf_oos = pd.concat(oos_E_inf_list)
E_inf_oos = E_inf_oos[~E_inf_oos.index.duplicated(keep="first")].sort_index()
print(f"\nE_inf_oos range: {E_inf_oos.index.min()} – {E_inf_oos.index.max()}, shape: {E_inf_oos.shape}")
 
 
# ══════════════════════════════════════════════════════════════════════════════
# RMSE — IN-SAMPLE AND OUT-OF-SAMPLE
# ══════════════════════════════════════════════════════════════════════════════
 
def realized_avg_inflation(infl_monthly, horizon_m):
    return infl_monthly.rolling(window=horizon_m).mean().shift(-horizon_m) * 12 * 10_000
 
def rmse(a, b):
    return np.sqrt(((a - b) ** 2).mean())
 
horizon_map = {"2 Year": 24, "5 Year": 60}
 
# ── In-sample ─────────────────────────────────────────────────────────────────
rows_is = []
for label, m in horizon_map.items():
    col       = f"y_{m}m"
    realized  = realized_avg_inflation(infl_m, m)
    model_fc  = E_inf[col]        * 10_000
    bei_fc    = BEI_obs[col]      * 10_000
    survey_fc = (df_survey[label] * 10_000).copy()
    survey_fc.index = survey_fc.index + pd.offsets.MonthEnd(0)
    rw_fc     = infl_m.rolling(m).mean() * 12 * 10_000
 
    common        = (realized.dropna().index.intersection(model_fc.dropna().index)
                     .intersection(rw_fc.dropna().index))
    survey_common = realized.dropna().index.intersection(survey_fc.dropna().index)
    bei_common    = common.intersection(bei_fc.dropna().index)
 
    rows_is.append({
        "Horizon":        label,
        "Model forecast": rmse(model_fc.loc[common],    realized.loc[common]),
        "Random walk":    rmse(rw_fc.loc[common],       realized.loc[common]),
        "Breakevens":     rmse(bei_fc.loc[bei_common],  realized.loc[bei_common]) if len(bei_common) > 10 else np.nan,
        "Survey":         rmse(survey_fc.loc[survey_common], realized.loc[survey_common]) if len(survey_common) > 10 else np.nan,
    })
 
is_rmse = pd.DataFrame(rows_is).set_index("Horizon")
# ── Out-of-sample ─────────────────────────────────────────────────────────────
rows_oos = []
for label, m in horizon_map.items():
    col       = f"y_{m}m"
    realized  = realized_avg_inflation(infl_m, m)           # full series
    model_fc  = E_inf_oos[col]    * 10_000
    bei_fc    = BEI_obs[col]      * 10_000
    survey_fc = (df_survey[label] * 10_000).copy()
    survey_fc.index = survey_fc.index + pd.offsets.MonthEnd(0)
    rw_fc     = infl_m.rolling(m).mean() * 12 * 10_000
 
    common        = (realized.dropna().index.intersection(model_fc.dropna().index)
                     .intersection(rw_fc.dropna().index))
    survey_common = (realized.dropna().index.intersection(survey_fc.dropna().index))
    survey_common = survey_common[survey_common >= pd.Timestamp(oos_start)]
    bei_common    = common.intersection(bei_fc.dropna().index)
 
    rows_oos.append({
        "Horizon":        label,
        "Model forecast": rmse(model_fc.loc[common],    realized.loc[common]),
        "Random walk":    rmse(rw_fc.loc[common],       realized.loc[common]),
        "Breakevens":     rmse(bei_fc.loc[bei_common],  realized.loc[bei_common]) if len(bei_common) > 10 else np.nan,
        "Survey":         rmse(survey_fc.loc[survey_common], realized.loc[survey_common]) if len(survey_common) > 10 else np.nan,
    })
 
oos_rmse = pd.DataFrame(rows_oos).set_index("Horizon")
 
def rmse_latex(is_df, oos_df, caption, label, note=None):
    cols     = list(is_df.columns)
    col_fmt  = "l" + "r" * len(cols)
    header   = "Horizon & " + " & ".join(cols) + r" \\"

    def rows_block(df):
        return "\n".join(
            f"        {idx} & " + " & ".join(
                f"{v:.1f}" if not np.isnan(v) else "---"
                for v in row.values
            ) + r" \\"
            for idx, row in df.iterrows()
        )

    tex = (
        rf"\begin{{table}}[htbp]" + "\n"
        rf"    \centering" + "\n"
        rf"    \caption{{{caption}}}" + "\n"
        rf"    \label{{{label}}}" + "\n"
        rf"    \begin{{tabular}}{{{col_fmt}}}" + "\n"
        rf"    \toprule" + "\n"
        rf"        {header}" + "\n"
        rf"    \midrule" + "\n"
        rf"        \multicolumn{{{1 + len(cols)}}}{{l}}{{\small\textit{{Panel A: In-sample}}}} \\" + "\n"
        rf"    \midrule" + "\n"
        + rows_block(is_df) + "\n"
        rf"    \midrule" + "\n"
        rf"        \multicolumn{{{1 + len(cols)}}}{{l}}{{\small\textit{{Panel B: Out-of-sample (2009:01 onward)}}}} \\" + "\n"
        rf"    \midrule" + "\n"
        + rows_block(oos_df) + "\n"
        rf"    \bottomrule" + "\n"
        rf"    \end{{tabular}}"
    )
    if note:
        tex += (
            "\n"
            rf"    \begin{{minipage}}{{\linewidth}}" + "\n"
            rf"        \vspace{{4pt}}" + "\n"
            rf"        \footnotesize \textit{{Note:}} {note}" + "\n"
            rf"    \end{{minipage}}"
        )
    tex += "\n\\end{table}"
    return tex

note_rmse = (
    "Root mean squared errors in basis points. "
    "Model forecast is the liquidity- and risk-adjusted expected inflation from the ATSM. "
    "Random walk takes the trailing $n$-month average CPI growth as the forecast. "
    "Breakevens are unadjusted nominal minus real zero-coupon yields. "
    "Survey RMSE computed over available survey dates only."
)

print(rmse_latex(
    is_rmse[["Model forecast", "Random walk", "Breakevens", "Survey"]],
    oos_rmse[["Model forecast", "Random walk", "Breakevens", "Survey"]],
    caption = "Inflation forecasting: root mean squared errors",
    label   = "tab:rmse_inflation",
    note    = note_rmse,
))
 
# ══════════════════════════════════════════════════════════════════════════════
# PLOT: EXPECTED VS REALIZED INFLATION
# ══════════════════════════════════════════════════════════════════════════════
 
for label, m in horizon_map.items():
    col        = f"y_{m}m"
    s_model    = E_inf[col].dropna()            * 10_000
    s_survey   = df_survey[label].dropna()      * 10_000
    s_bei      = BEI_obs[col].dropna()          * 10_000
    s_realized = realized_avg_inflation(infl_m, m).dropna()
 
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(s_bei.index,      s_bei,      color="#2c2c2c", linewidth=1.5, linestyle="--",
            label="Breakeven inflation (raw)")
    ax.plot(s_model.index,    s_model,    color="#e07b39", linewidth=2.0,
            label="Model-implied expected inflation")
    ax.plot(s_survey.index,   s_survey,   color="#9b2335", linewidth=1.8,
            linestyle=(0, (4, 2)), marker="o", markersize=2.5,
            label="Survey expected inflation")
    ax.plot(s_realized.index, s_realized, color="#3aaa6e", linewidth=1.8,
            label=f"Realized inflation ({label})")
    ax.axhline(0, color="black", linestyle=":", linewidth=0.8)
    ax.set_title(f"Expected vs Realized Inflation — {label} horizon", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel("Basis points", fontsize=11)
    ax.legend(frameon=True, fontsize=10, ncol=2)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    plt.tight_layout()
    fname_map_exp = {"2 Year": "exp_vs_realized_2y.png", "5 Year": "exp_vs_realized_5y.png"}
    fig.savefig(fname_map_exp[label], dpi=150, bbox_inches="tight")
    plt.show()

# ── Nominal and Real yield curves — all maturities ───────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Nominal yields
nom_plot_cols = [c for c in yobs_Q.columns if c.startswith("y_")]
nom_plot_cols = sorted(nom_plot_cols, key=lambda c: int(c.split("_")[1][:-1]))

for n in NOM_RET_MONTHS_FULLYEARS:
    col = f"y_{n}m"
    axes[0].plot(yobs_Q.index, yobs_Q[col] * 10_000,
                 linewidth=1.2, label=f"{n // 12}y")

axes[0].axhline(0, color="black", linestyle=":", linewidth=0.8)
axes[0].set_ylabel("Basis points", fontsize=11)
axes[0].set_title("Nominal SGB yields — all maturities", fontsize=12, fontweight="bold")
axes[0].yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
axes[0].set_axisbelow(True)

# Real yields
real_plot_cols = [c for c in yobsR_Q.columns if c.startswith("y_")]
real_plot_cols = sorted(real_plot_cols, key=lambda c: int(c.split("_")[1][:-1]))

for n in REAL_RET_MONTHS_FULLYEARS:
    col = f"y_{n}m"
    axes[1].plot(yobsR_Q.index, yobsR_Q[col] * 10_000,
                 linewidth=1.2, label=f"{n // 12}y")

axes[1].axhline(0, color="black", linestyle=":", linewidth=0.8)
axes[1].set_ylabel("Basis points", fontsize=11)
axes[1].set_title("Real SGBIL yields — all maturities", fontsize=12, fontweight="bold")
axes[1].yaxis.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
axes[1].set_axisbelow(True)

# Legends outside below each subplot
for ax in axes:
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=10,
        fontsize=8,
        frameon=True,
    )

plt.tight_layout()
plt.subplots_adjust(hspace=0.45)
fig.savefig("nominal_real_yields.png", dpi=150, bbox_inches="tight")
plt.show()
