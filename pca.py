import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt



# Paths

NOMINAL_PATH = "nominal_eom_zero_coupon_yields_monthly_grid.xlsx"
REAL_PATH    = "real_eom_zero_coupon_yields_monthly_grid.xlsx"
TURN_PATH    = "turnover_monthly_with_liquidity.xlsx"

OUT_NOMINAL  = "nominal_pcs_5m_120m.xlsx"
OUT_FACTORS  = "model_factors_nominal_real_liquidity.xlsx"



# 1) Load data

nom = pd.read_excel(NOMINAL_PATH, parse_dates=["date"]).sort_values("date").set_index("date")
real = pd.read_excel(REAL_PATH, parse_dates=["date"]).sort_values("date").set_index("date")

turnover = pd.read_excel(
    TURN_PATH,
    sheet_name="turnover_monthly",
    parse_dates=["Month"]).sort_values("Month").set_index("Month")
turnover.index.name = "date"



# Select maturity grids (Nominal: 5m..120m, Real: 23m..120m)
# (Assumes columns exist: y_5m..y_120m and y_23m..y_120m)

nom_cols  = [f"y_{m}m" for m in range(5, 121)]
real_cols = [f"y_{m}m" for m in range(23, 121)]


# Align sample (common index)

common_index = nom.index.intersection(real.index).intersection(turnover.index)
nom = nom.loc[common_index]
real = real.loc[common_index]
turnover = turnover.loc[common_index]

# PCA on nominal yields (demean across time, covariance PCA)

nom_sel = nom[nom_cols].copy()

Y = nom_sel.values
Y_demean = Y - Y.mean(axis=0, keepdims=True)

pca = PCA(n_components=3)
scores = pca.fit_transform(Y_demean)
loadings = pca.components_.T
explained = pca.explained_variance_ratio_

pc_df = pd.DataFrame(
    scores,
    index=nom_sel.index,
    columns=["PC1_level", "PC2_slope", "PC3_curvature"]
)

# Standardize PCAS (paper-like scale)

pc_df_z = (pc_df - pc_df.mean()) / pc_df.std(ddof=0)

load_df = pd.DataFrame(
    loadings,
    index=nom_cols,
    columns=["PC1_loading", "PC2_loading", "PC3_loading"]
)

explained_df = pd.DataFrame({
    "PC": ["PC1", "PC2", "PC3"],
    "explained_variance_share": explained
})

print("\nExplained variance (nominal):")
print(explained_df)

# Save nominal outputs (raw + standardized)

with pd.ExcelWriter(Path(OUT_NOMINAL), engine="openpyxl") as writer:
    nom_sel.to_excel(writer, sheet_name="nominal_yields_used")
    pc_df.to_excel(writer, sheet_name="nominal_PCs_raw")
    pc_df_z.to_excel(writer, sheet_name="nominal_PCs_z")
    load_df.to_excel(writer, sheet_name="PC_loadings")
    explained_df.to_excel(writer, sheet_name="explained_variance", index=False)

print(f"\nSaved: {OUT_NOMINAL}")

# Plot nominal factors (standardized)

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(pc_df_z.index, pc_df_z["PC1_level"])
axes[0].set_title("First Factor (Level)")

axes[1].plot(pc_df_z.index, pc_df_z["PC2_slope"])
axes[1].set_title("Second Factor (Slope)")

axes[2].plot(pc_df_z.index, pc_df_z["PC3_curvature"])
axes[2].set_title("Third Factor (Curvature)")

plt.xlabel("Year")
plt.tight_layout()
plt.show()

# 7) Real factors: regress real yields on (nominal PCs + liquidity), PCA on residuals

nominal_pcs = pc_df_z.copy()
real_sel = real[real_cols].copy()

common2 = nominal_pcs.index.intersection(real_sel.index).intersection(turnover.index)
nominal_pcs = nominal_pcs.loc[common2]
real_sel = real_sel.loc[common2]
turnover = turnover.loc[common2]

X = pd.concat(
    [
        nominal_pcs[["PC1_level", "PC2_slope", "PC3_curvature"]],
        turnover[["liquidity_ma3"]],
    ],
    axis=1
).dropna()

X = sm.add_constant(X, has_constant="add")

real_residuals = pd.DataFrame(index=X.index)

for col in real_sel.columns:
    y = real_sel[col].loc[X.index]
    model = sm.OLS(y, X).fit()
    real_residuals[col] = model.resid

Z = real_residuals.dropna()
Z_std = StandardScaler().fit_transform(Z)

pca_real = PCA(n_components=2)
real_factors = pca_real.fit_transform(Z_std)

real_pcs = pd.DataFrame(
    real_factors,
    index=Z.index,
    columns=["Real_PC1", "Real_PC2"]
)

# Standardize real factor scores too (paper-like scale)

real_pcs_z = (real_pcs - real_pcs.mean()) / real_pcs.std(ddof=0)

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(real_pcs_z.index, real_pcs_z["Real_PC1"])
axes[0].set_title("Real Factor 1")

axes[1].plot(real_pcs_z.index, real_pcs_z["Real_PC2"])
axes[1].set_title("Real Factor 2")

plt.xlabel("Year")
plt.tight_layout()
plt.show()


# Assemble final factor set & save

factors_df = pd.concat(
    [
        nominal_pcs[["PC1_level", "PC2_slope", "PC3_curvature"]],
        turnover[["liquidity_ma3"]].rename(columns={"liquidity_ma3": "Liquidity_MA3"}),
        real_pcs_z[["Real_PC1", "Real_PC2"]],
    ],
    axis=1
).dropna()

print("Factors shape:", factors_df.shape)

factors_df.to_excel(OUT_FACTORS, sheet_name="factors", index_label="date")
print(f"\nSaved: {OUT_FACTORS}")