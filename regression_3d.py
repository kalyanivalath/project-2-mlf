# Multiple linear regression on the 3D datasets with scikit-learn.
# Model fitted: y = b0 + b1*x1 + b2*x2 + b3*x3 + e, e ~ N(0, sigma^2)
# Metrics: RSS, R^2, adjusted R^2, RSE, and standard errors of coefficients.
# Graphics: pairwise scatter plots, predicted vs actual, residuals vs predicted,
# normal Q-Q plots, and metrics vs noise level.
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

DATASETS = {"Low noise (L)": "3D L.csv",
            "Moderate noise (M)": "3D M.csv",
            "High noise (H)": "3D H.csv"}
FEATURES = ["x1", "x2", "x3"]
DATA_DIR = Path(__file__).parent / "Data"


def load(path):
    # files have no header row; remove blank rows.
    return pd.read_csv(DATA_DIR / path, header=None,
                       names=FEATURES + ["y"]).dropna()


def coefficient_stats(X, resid, k):
    # standard errors for [b0, b1, b2, b3].
    n = X.shape[0]
    Xd = np.column_stack([np.ones(n), X])
    dof = n - k - 1
    sigma2 = np.sum(resid ** 2) / dof

    # calculate standard errors for the coefficients
    cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
    se = np.sqrt(np.diag(cov))
    return se


results, fits = [], {}
for label, path in DATASETS.items():
    df = load(path)
    X, y = df[FEATURES].to_numpy(), df["y"].to_numpy()
    n, k = X.shape
    if n <= k + 1 or np.linalg.matrix_rank(np.column_stack([np.ones(n), X])) < k + 1:
        raise ValueError(f"{path}: not enough independent observations for coefficient statistics")

    # train the model
    model = LinearRegression().fit(X, y)

    # calculate predicted values and residuals
    y_hat = model.predict(X)
    resid = y - y_hat

    # calculate the regression metrics
    rss = np.sum(resid ** 2)
    r2 = r2_score(y, y_hat)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    rse = np.sqrt(rss / (n - k - 1))
    coefs = np.r_[model.intercept_, model.coef_]
    se = coefficient_stats(X, resid, k)

    row = {"Dataset": path.replace(".csv", ""), "n": n}
    for i, name in enumerate(["b0 (intercept)", "b1 (x1)", "b2 (x2)", "b3 (x3)"]):
        short = name.split()[0]
        row[name] = coefs[i]
        row[f"SE {short}"] = se[i]
    row.update({"RSS": rss, "R2": r2, "Adj R2": adj_r2, "RSE": rse})
    results.append(row)
    fits[label] = dict(df=df, X=X, y=y, y_hat=y_hat, resid=resid,
                       coefs=coefs, r2=r2, rse=rse)

table = pd.DataFrame(results).set_index("Dataset")
table.to_csv(Path(__file__).parent / "results_3d.csv", float_format="%.5f")
pd.set_option("display.width", 200, "display.max_columns", 30)
print(table.T.to_string(float_format=lambda v: f"{v:.5f}"))

# print the fitted equation for each dataset
print("\nFitted equations:")
for label, f in fits.items():
    coefs = f["coefs"]
    equation = f"y_hat = {coefs[0]:.4f}"
    for i, b in enumerate(coefs[1:], start=1):
        equation += f" {'+' if b >= 0 else '-'} {abs(b):.4f} x{i}"
    print(f"  {label:20s} {equation}")

# plotting
COLORS = {"Low noise (L)": "#2a6fb0", "Moderate noise (M)": "#d08a1f",
          "High noise (H)": "#b5332e"}
TAGS = {"Low noise (L)": "L", "Moderate noise (M)": "M", "High noise (H)": "H"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
FIG_DIR = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)
saved = []


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    plt.close(fig)
    saved.append(name)


all_y = np.concatenate([np.r_[f["y"], f["y_hat"]] for f in fits.values()])
pad = 0.05 * (all_y.max() - all_y.min())
y_lim = (all_y.min() - pad, all_y.max() + pad)
r_lim = max(np.abs(f["resid"]).max() for f in fits.values()) * 1.1

for label, f in fits.items():
    tag, color = TAGS[label], COLORS[label]
    d = f["df"]

    # pair plots: input vs output and correlations between inputs.
    for a, b in [("x1", "y"), ("x2", "y"), ("x3", "y"),
                 ("x1", "x2"), ("x1", "x3"), ("x2", "x3")]:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(d[a], d[b], s=14, color=color)
        ax.set_xlabel(a); ax.set_ylabel(b)
        if b == "y":
            ax.set_ylim(y_lim)
        r = np.corrcoef(d[a], d[b])[0, 1]
        ax.set_title(f"3D {label}: {b} vs {a}  (r = {r:.2f})")
        save(fig, f"3d_pair_{b}_vs_{a}_{tag}.png")

    # predicted vs actual
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(y_lim, y_lim, "--", color="0.4", lw=1)
    ax.scatter(f["y"], f["y_hat"], s=18, color=color)
    ax.set_xlim(y_lim); ax.set_ylim(y_lim)
    ax.set_title(f"3D {label}: predicted vs. actual")
    ax.set_xlabel("Actual y"); ax.set_ylabel("Predicted ŷ")
    save(fig, f"3d_pred_vs_actual_{tag}.png")

    # residuals vs predicted.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.axhline(0, color="0.4", lw=1)
    ax.scatter(f["y_hat"], f["resid"], s=18, color=color)
    ax.set_ylim(-r_lim, r_lim)
    ax.set_title(f"3D {label}: residuals vs. predicted")
    ax.set_xlabel("Predicted ŷ"); ax.set_ylabel("Residual")
    save(fig, f"3d_resid_vs_pred_{tag}.png")

    # normal Q-Q of residuals.
    fig, ax = plt.subplots(figsize=(6, 4.5))
    (osm, osr), (slope, intercept, _) = stats.probplot(f["resid"], dist="norm")
    ax.scatter(osm, osr, s=18, color=color)
    ax.plot(osm, slope * osm + intercept, color="black", lw=1.2)
    ax.set_title(f"3D {label}: normal Q-Q of residuals")
    ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
    save(fig, f"3d_qq_{tag}.png")

# metrics vs noise level.
lv = ["L", "M", "H"]
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(lv, table["R2"], "o-", label="R²")
ax.plot(lv, table["Adj R2"], "s--", label="Adjusted R²")
ax.set_title("Explained variance vs. noise"); ax.set_ylim(0, 1)
ax.set_xlabel("Noise level"); ax.legend(frameon=False)
save(fig, "3d_r2_vs_noise.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(lv, table["RSE"], "o-", label="RSE")
ax.set_title("RSE (estimated noise σ) vs. noise"); ax.set_ylim(bottom=0)
ax.set_xlabel("Noise level"); ax.legend(frameon=False)
save(fig, "3d_rse_vs_noise.png")

print(f"\nSaved results_3d.csv and {len(saved)} figures to '{FIG_DIR}/':")
for name in saved:
    print("  ", name)
