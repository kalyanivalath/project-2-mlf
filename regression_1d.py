
# Single linear regression on the 1D datasets (1D L, 1D M, 1D H) with scikit-learn.

# Model fitted:  y = b0 + b1*x + e,   e ~ N(0, sigma^2)

# Metrics: RSS, R^2, adjusted R^2, RSE, and standard errors of the coefficients.

# Graphics: fitted line with residual segments, predicted vs. actual,
# residuals vs. predicted, and normal Q-Q plots of the residuals.


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from pathlib import Path

DATASETS = {"Low noise (L)": "1D L.csv",
            "Moderate noise (M)": "1D M.csv",
            "High noise (H)": "1D H.csv"}

# CSV files live in a "Data" folder next to this script
DATA_DIR = Path(__file__).parent / "Data"


def load(path):
    # Files have no header row; 1D M.csv also ends with blank ',' rows.
    df = pd.read_csv(DATA_DIR / path, header=None, names=["x", "y"])
    df = df.dropna()
    return df[["x"]].to_numpy(), df["y"].to_numpy()


def coefficient_stats(X, resid, k):
    # Standard errors for [b0, b1]  
    n = X.shape[0]
    Xd = np.column_stack([np.ones(n), X])        # design matrix with intercept
    dof = n - k - 1
    sigma2 = np.sum(resid ** 2) / dof            # RSE^2, estimate of noise variance
    cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    return se


results, fits = [], {}
for label, path in DATASETS.items():
    X, y = load(path)
    n, k = X.shape

    model = LinearRegression().fit(X, y)
    y_hat = model.predict(X)
    resid = y - y_hat

    rss = np.sum(resid ** 2)
    r2 = r2_score(y, y_hat)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    rse = np.sqrt(rss / (n - k - 1))

    coefs = np.r_[model.intercept_, model.coef_]
    se = coefficient_stats(X, resid, k)

    results.append({
        "Dataset": path.replace(".csv", ""), "n": n,
        "b0 (intercept)": coefs[0], "SE b0": se[0],
        "b1 (slope)": coefs[1], "SE b1": se[1],
        "RSS": rss, "R2": r2, "Adj R2": adj_r2, "RSE": rse,
    })
    fits[label] = dict(X=X.ravel(), y=y, y_hat=y_hat, resid=resid,
                       b0=coefs[0], b1=coefs[1], r2=r2, rse=rse)

table = pd.DataFrame(results).set_index("Dataset")
table.to_csv("results_1d.csv", float_format="%.5f")

pd.set_option("display.width", 200, "display.max_columns", 30)
print(table.T.to_string(float_format=lambda v: f"{v:.5f}"))
print("\nFitted equations:")
for label, f in fits.items():
    print(f"  {label:20s} y_hat = {f['b0']:.4f} + {f['b1']:.4f} x")

##########################################################################################
# PLOTS
###########################################################################################

COLORS = {"Low noise (L)": "#2a6fb0", "Moderate noise (M)": "#d08a1f",
          "High noise (H)": "#b5332e"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

# 1. Data with fitted line and residual segments (the ISLR Fig 3.1 style)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
for ax, (label, f) in zip(axes, fits.items()):
    xs = np.linspace(0, 1, 100)
    ax.vlines(f["X"], f["y_hat"], f["y"], color="0.75", lw=0.8, zorder=1)
    ax.scatter(f["X"], f["y"], s=18, color=COLORS[label], zorder=2)
    ax.plot(xs, f["b0"] + f["b1"] * xs, color="black", lw=1.8, zorder=3)
    ax.set_title(f"1D {label}\n$\\hat y$ = {f['b0']:.3f} + {f['b1']:.3f}x"
                 f"   R² = {f['r2']:.3f}")
    ax.set_xlabel("x")
axes[0].set_ylabel("y")
fig.tight_layout()
fig.savefig("1d_fit_residuals.png", dpi=150)

# 2. Predicted vs actual
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
for ax, (label, f) in zip(axes, fits.items()):
    lo, hi = min(f["y"].min(), f["y_hat"].min()), max(f["y"].max(), f["y_hat"].max())
    ax.plot([lo, hi], [lo, hi], "--", color="0.4", lw=1)
    ax.scatter(f["y"], f["y_hat"], s=18, color=COLORS[label])
    ax.set_title(f"1D {label}")
    ax.set_xlabel("Actual y")
axes[0].set_ylabel("Predicted ŷ")
fig.tight_layout()
fig.savefig("1d_pred_vs_actual.png", dpi=150)

# 3. Residual diagnostics: residuals vs predicted (top) and Q-Q (bottom)
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
lim = max(np.abs(f["resid"]).max() for f in fits.values()) * 1.1
for j, (label, f) in enumerate(fits.items()):
    ax = axes[0, j]
    ax.axhline(0, color="0.4", lw=1)
    ax.scatter(f["y_hat"], f["resid"], s=18, color=COLORS[label])
    ax.set_ylim(-lim, lim)
    ax.set_title(f"1D {label}: residuals vs. predicted")
    ax.set_xlabel("Predicted ŷ"); ax.set_ylabel("Residual")
    ax = axes[1, j]
    (osm, osr), (slope, intercept, _) = stats.probplot(f["resid"], dist="norm")
    ax.scatter(osm, osr, s=18, color=COLORS[label])
    ax.plot(osm, slope * osm + intercept, color="black", lw=1.2)
    ax.set_title(f"1D {label}: normal Q-Q of residuals")
    ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
fig.tight_layout()
fig.savefig("1d_residual_diagnostics.png", dpi=150)

# 4. Metrics vs noise level
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
lv = ["L", "M", "H"]
axes[0].plot(lv, table["R2"], "o-", label="R²")
axes[0].plot(lv, table["Adj R2"], "s--", label="Adjusted R²")
axes[0].set_title("Explained variance vs. noise"); axes[0].set_ylim(0, 1)
axes[0].legend(frameon=False)
axes[1].plot(lv, table["RSE"], "o-", label="RSE")
axes[1].set_title("RSE (estimated noise σ) vs. noise"); axes[1].set_ylim(bottom=0)
axes[1].legend(frameon=False)
for ax in axes:
    ax.set_xlabel("Noise level")
fig.tight_layout()
fig.savefig("1d_metrics_vs_noise.png", dpi=150)
print("\nSaved results_1d.csv and 4 figures.")
