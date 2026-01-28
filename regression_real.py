import pandas as pd
import pandas as pd
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# --- Turnover & liquidity ---
turnover = pd.read_excel(
    "turnover_monthly_with_liquidity.xlsx",
    sheet_name="turnover_monthly",
    parse_dates=["Month"]
).set_index("Month")

# --- Nominal PCs ---
nominal_pcs = pd.read_excel(
    "nominal_pcs_0.5y_10y.xlsx",
    sheet_name="nominal_PCs",
    parse_dates=["date"]
).set_index("date")

# --- Real zero-coupon yields ---
real_yields = pd.read_excel(
    "real_eom_zero_coupon_yields.xlsx",
    sheet_name="yields",
    parse_dates=["date"]
).set_index("date")
real_yields = real_yields[["y_2y","y_3y","y_4y","y_5y","y_6y","y_7y","y_8y","y_9y","y_10y"]]


print(turnover.shape)
print(nominal_pcs.shape)
print(real_yields.shape)

common_index = nominal_pcs.index.intersection(real_yields.index).intersection(turnover.index)

nominal_pcs = nominal_pcs.loc[common_index]
real_yields = real_yields.loc[common_index]
turnover= turnover.loc[common_index]

# 2) Build regressor matrix: const + nominal PCs + liquidity
X = pd.concat(
    [
        nominal_pcs[["PC1_level", "PC2_slope", "PC3_curvature"]],
        turnover["liquidity_ma3"]
    ],
    axis=1
).dropna()

X = sm.add_constant(X)

# 3) Regress each real yield maturity on X and collect residuals
real_residuals = pd.DataFrame(index=X.index)

for col in real_yields.columns:
    y = real_yields[col].loc[X.index]
    model = sm.OLS(y, X).fit()
    real_residuals[col] = model.resid

# 4) PCA on residuals -> Real factors (choose KR=2 as baseline)
Z = real_residuals.dropna()
Z_std = StandardScaler().fit_transform(Z)

pca_real = PCA(n_components=2)
real_factors = pca_real.fit_transform(Z_std)

real_pcs = pd.DataFrame(
    real_factors,
    index=Z.index,
    columns=["Real_PC1", "Real_PC2"]
)

# 5) Outputs / quick diagnostics
print("Residuals shape:", real_residuals.shape)
print("Real factors shape:", real_pcs.shape)
print("Explained variance (KR=2):", pca_real.explained_variance_ratio_)
print("Cumulative:", pca_real.explained_variance_ratio_.cumsum())

# plot factors
fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

axes[0].plot(real_pcs.index, real_pcs["Real_PC1"])
axes[0].set_title("Real Factor 1")

axes[1].plot(real_pcs.index, real_pcs["Real_PC2"])
axes[1].set_title("Real Factor 2")

plt.xlabel("Year")
plt.tight_layout()
plt.show()
