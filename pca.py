import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from pathlib import Path
import requests
import time


# Paths
NOMINAL_PATH = "nominal_eom_zero_coupon_yields.xlsx"
REAL_PATH    = "real_eom_zero_coupon_yields.xlsx"
OUT_PATH     = "nominal_pcs_0.5y_10y.xlsx"

# ----------------------------
# 1) Load data
# ----------------------------
nom = pd.read_excel(NOMINAL_PATH, sheet_name="yields")
real = pd.read_excel(REAL_PATH, sheet_name="yields")

nom["date"] = pd.to_datetime(nom["date"])
real["date"] = pd.to_datetime(real["date"])

nom = nom.sort_values("date").set_index("date")
real = real.sort_values("date").set_index("date")

# Maturity grids
nom_cols = ["y_0.5y"] + [f"y_{i}y" for i in range(1, 11)]  # 0.5y–10y
real_cols = [f"y_{i}y" for i in range(2, 11)]             # 2y–10y

# Select panels (no alignment / no dropping needed)
nom_sel = nom[nom_cols].copy()
real_sel = real[real_cols].copy()

print("Nominal panel:", nom_sel.shape)
print("Real panel:", real_sel.shape)
print("Sample:", nom_sel.index.min().date(), "to", nom_sel.index.max().date())

# ----------------------------
# 2) PCA on nominal yields
# ----------------------------
# Demean each maturity across time (no standardization)
Y = nom_sel.values
Y_demean = Y - Y.mean(axis=0, keepdims=True)

pca = PCA(n_components=3)
scores = pca.fit_transform(Y_demean)      # T x 3
loadings = pca.components_.T              # maturities x 3
explained = pca.explained_variance_ratio_

pc_df = pd.DataFrame(
    scores,
    index=nom_sel.index,
    columns=["PC1_level", "PC2_slope", "PC3_curvature"]
)

load_df = pd.DataFrame(
    loadings,
    index=nom_cols,
    columns=["PC1_loading", "PC2_loading", "PC3_loading"]
)

explained_df = pd.DataFrame({
    "PC": ["PC1", "PC2", "PC3"],
    "explained_variance_share": explained
})

print("\nExplained variance:")
print(explained_df)

# ----------------------------
# 3) Save outputs
# ----------------------------
out_path = Path(OUT_PATH)
with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
    nom_sel.to_excel(writer, sheet_name="nominal_yields_used")
    pc_df.to_excel(writer, sheet_name="nominal_PCs")
    load_df.to_excel(writer, sheet_name="PC_loadings")
    explained_df.to_excel(writer, sheet_name="explained_variance", index=False)

print(f"\nSaved: {out_path}")

# ----------------------------
# 4) Optional sanity plots
# ----------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    maturity_years = [0.5] + list(range(1, 11))
    plt.figure()
    plt.plot(maturity_years, load_df["PC1_loading"])
    plt.title("PC1 (Level) loadings")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Loading")
    plt.show()
    plt.figure()
    plt.plot(maturity_years, load_df["PC2_loading"])
    plt.title("PC2 (Slope) loadings")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Loading")
    plt.show()
    plt.figure()
    plt.plot(maturity_years, load_df["PC3_loading"])
    plt.title("PC3 (Curvature) loadings")
    plt.xlabel("Maturity (years)")
    plt.ylabel("Loading")
    plt.show()


fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(pc_df.index, pc_df["PC1_level"])
axes[0].set_title("First Factor (Level)")

axes[1].plot(pc_df.index, pc_df["PC2_slope"])
axes[1].set_title("Second Factor (Slope)")

axes[2].plot(pc_df.index, pc_df["PC3_curvature"])
axes[2].set_title("Third Factor (Curvature)")

plt.xlabel("Year")
plt.tight_layout()
plt.show()




