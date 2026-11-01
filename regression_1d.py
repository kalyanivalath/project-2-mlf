
# Single linear regression on the 1D datasets (1D L, 1D M, 1D H) with scikit-learn.

# Model fitted:  y = b0 + b1*x + e,   e ~ N(0, sigma^2)

# Metrics: RSS, R^2, adjusted R^2, RSE, and standard errors of the coefficients.

# Graphics: fitted line with residual segments, predicted vs. actual,
# residuals vs. predicted, and normal Q-Q plots of the residuals.
# Every graph is saved as its own PNG file in a "figures" folder.


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
# PLOTS  (one graph per file)
###########################################################################################

COLORS = {"Low noise (L)": "#2a6fb0", "Moderate noise (M)": "#d08a1f",
          "High noise (H)": "#b5332e"}
TAGS = {"Low noise (L)": "L", "Moderate noise (M)": "M", "High noise (H)": "H"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})

FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
saved = []


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    saved.append(name)


# Shared limits so the separate files stay directly comparable

all_y = np.concatenate([np.r_[f["y"], f["y_hat"]] for f in fits.values()])
pad = 0.05 * (all_y.max() - all_y.min())
y_lim = (all_y.min() - pad, all_y.max() + pad)
r_lim = max(np.abs(f["resid"]).max() for f in fits.values()) * 1.1

for label, f in fits.items():
    tag, color = TAGS[label], COLORS[label]

    # 1. Data with fitted line and residual segments
    fig, ax = plt.subplots(figsize=(6, 4.5))
    xs = np.linspace(0, 1, 100)
    ax.vlines(f["X"], f["y_hat"], f["y"], color="0.75", lw=0.8, zorder=1)
    ax.scatter(f["X"], f["y"], s=18, color=color, zorder=2)
    ax.plot(xs, f["b0"] + f["b1"] * xs, color="black", lw=1.8, zorder=3)
    ax.set_ylim(y_lim)
    ax.set_title(f"1D {label}\n$\\hat y$ = {f['b0']:.3f} + {f['b1']:.3f}x"
                 f"   R² = {f['r2']:.3f}")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    save(fig, f"1d_fit_residuals_{tag}.png")

    # 2. Predicted vs actual
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(y_lim, y_lim, "--", color="0.4", lw=1)
    ax.scatter(f["y"], f["y_hat"], s=18, color=color)
    ax.set_xlim(y_lim); ax.set_ylim(y_lim)
    ax.set_title(f"1D {label}: predicted vs. actual")
    ax.set_xlabel("Actual y"); ax.set_ylabel("Predicted ŷ")
    save(fig, f"1d_pred_vs_actual_{tag}.png")

    # 3a. Residuals vs predicted
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.axhline(0, color="0.4", lw=1)
    ax.scatter(f["y_hat"], f["resid"], s=18, color=color)
    ax.set_ylim(-r_lim, r_lim)
    ax.set_title(f"1D {label}: residuals vs. predicted")
    ax.set_xlabel("Predicted ŷ"); ax.set_ylabel("Residual")
    save(fig, f"1d_resid_vs_pred_{tag}.png")

    # 3b. Normal Q-Q of residuals
    fig, ax = plt.subplots(figsize=(6, 4.5))
    (osm, osr), (slope, intercept, _) = stats.probplot(f["resid"], dist="norm")
    ax.scatter(osm, osr, s=18, color=color)
    ax.plot(osm, slope * osm + intercept, color="black", lw=1.2)
    ax.set_title(f"1D {label}: normal Q-Q of residuals")
    ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
    save(fig, f"1d_qq_{tag}.png")

# 4. Metrics vs noise level
lv = ["L", "M", "H"]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(lv, table["R2"], "o-", label="R²")
ax.plot(lv, table["Adj R2"], "s--", label="Adjusted R²")
ax.set_title("Explained variance vs. noise"); ax.set_ylim(0, 1)
ax.set_xlabel("Noise level"); ax.legend(frameon=False)
save(fig, "1d_r2_vs_noise.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(lv, table["RSE"], "o-", label="RSE")
ax.set_title("RSE (estimated noise σ) vs. noise"); ax.set_ylim(bottom=0)
ax.set_xlabel("Noise level"); ax.legend(frameon=False)
save(fig, "1d_rse_vs_noise.png")

print(f"\nSaved results_1d.csv and {len(saved)} figures to '{FIG_DIR}/':")
for name in saved:
    print("  ", name)
