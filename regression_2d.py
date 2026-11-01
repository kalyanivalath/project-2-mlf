# Multiple linear regression on the 2D datasets (2D L, 2D M, 2D H) with scikit-learn.

# Model fitted:  y = b0 + b1*x1 + b2*x2 + e,   e ~ N(0, sigma^2)

# Metrics: RSS, R^2, adjusted R^2, RSE, and standard errors of
# the coefficients.

# Graphics: pair plot of the data, 3D scatter with the fitted plane and residual segments,
# predicted vs. actual, residuals vs. predicted, and normal Q-Q plots of the residuals.
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

DATASETS = {"Low noise (L)": "2D L.csv",
            "Moderate noise (M)": "2D M.csv",
            "High noise (H)": "2D H.csv"}
FEATURES = ["x1", "x2"]

# CSV files live in a "Data" folder next to this script
DATA_DIR = Path(__file__).parent / "Data"


def load(path):
    # Files have no header row; drop any blank ',' rows
    df = pd.read_csv(DATA_DIR / path, header=None, names=FEATURES + ["y"]).dropna()
    return df


def coefficient_stats(X, resid, k):
    # Standard errors for [b0, b1, b2]
    n = X.shape[0]
    Xd = np.column_stack([np.ones(n), X])        # design matrix with intercept
    dof = n - k - 1
    sigma2 = np.sum(resid ** 2) / dof            # RSE^2, estimate of noise variance
    cov = sigma2 * np.linalg.inv(Xd.T @ Xd)
    return np.sqrt(np.diag(cov))


results, fits = [], {}
for label, path in DATASETS.items():
    df = load(path)
    X, y = df[FEATURES].to_numpy(), df["y"].to_numpy()
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

    row = {"Dataset": path.replace(".csv", ""), "n": n}
    for i, name in enumerate(["b0 (intercept)", "b1 (x1)", "b2 (x2)"]):
        short = name.split()[0]
        row[name] = coefs[i]
        row[f"SE {short}"] = se[i]
    row.update({"RSS": rss, "R2": r2, "Adj R2": adj_r2, "RSE": rse})
    results.append(row)
    fits[label] = dict(df=df, X=X, y=y, y_hat=y_hat, resid=resid,
                       coefs=coefs, r2=r2, rse=rse)

table = pd.DataFrame(results).set_index("Dataset")
table.to_csv("results_2d.csv", float_format="%.5f")

pd.set_option("display.width", 200, "display.max_columns", 30)
print(table.T.to_string(float_format=lambda v: f"{v:.5f}"))

X_any = fits["Low noise (L)"]["X"]
print(f"\nCorrelation between x1 and x2: {np.corrcoef(X_any.T)[0, 1]:.3f}")
print("\nFitted equations:")
for label, f in fits.items():
    b0, b1, b2 = f["coefs"]
    sgn = "+" if b2 >= 0 else "-"
    print(f"  {label:20s} y_hat = {b0:.4f} + {b1:.4f} x1 {sgn} {abs(b2):.4f} x2")

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


# Shared limits so the separate files stay directly comparable across L/M/H
# (replaces sharex/sharey from the old side-by-side layout)
all_y = np.concatenate([np.r_[f["y"], f["y_hat"]] for f in fits.values()])
pad = 0.05 * (all_y.max() - all_y.min())
y_lim = (all_y.min() - pad, all_y.max() + pad)
r_lim = max(np.abs(f["resid"]).max() for f in fits.values()) * 1.1

g = np.linspace(0, 1, 20)
G1, G2 = np.meshgrid(g, g)

for label, f in fits.items():
    tag, color = TAGS[label], COLORS[label]
    d = f["df"]

    # 1. Pair plot: each input vs y, and x1 vs x2 (checks for correlated inputs)
    for a, b in [("x1", "y"), ("x2", "y"), ("x1", "x2")]:
        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(d[a], d[b], s=14, color=color)
        ax.set_xlabel(a); ax.set_ylabel(b)
        if b == "y":
            ax.set_ylim(y_lim)
        r = np.corrcoef(d[a], d[b])[0, 1]
        ax.set_title(f"2D {label}: {b} vs {a}  (r = {r:.2f})")
        save(fig, f"2d_pair_{b}_vs_{a}_{tag}.png")

    # 2. 3D scatter with fitted plane and residual segments
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    b0, b1, b2 = f["coefs"]
    ax.plot_surface(G1, G2, b0 + b1 * G1 + b2 * G2, color="0.6",
                    alpha=0.35, linewidth=0)
    x1, x2 = f["X"][:, 0], f["X"][:, 1]
    for xa, xb, yy, yh in zip(x1, x2, f["y"], f["y_hat"]):
        ax.plot([xa, xa], [xb, xb], [yh, yy], color="0.55", lw=0.6)
    ax.scatter(x1, x2, f["y"], s=14, color=color, depthshade=False)
    ax.set_zlim(y_lim)
    ax.set_xlabel("x1"); ax.set_ylabel("x2"); ax.set_zlabel("y")
    sgn = "+" if b2 >= 0 else "−"
    ax.set_title(f"2D {label}\n$\\hat y$ = {b0:.3f} + {b1:.3f}x1 {sgn} {abs(b2):.3f}x2"
                 f"   R² = {f['r2']:.3f}", fontsize=10)
    ax.view_init(elev=18, azim=-60)
    save(fig, f"2d_fit_plane_{tag}.png")

    # 3. Predicted vs actual
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(y_lim, y_lim, "--", color="0.4", lw=1)
    ax.scatter(f["y"], f["y_hat"], s=18, color=color)
    ax.set_xlim(y_lim); ax.set_ylim(y_lim)
    ax.set_title(f"2D {label}: predicted vs. actual")
    ax.set_xlabel("Actual y"); ax.set_ylabel("Predicted ŷ")
    save(fig, f"2d_pred_vs_actual_{tag}.png")

    # 4a. Residuals vs predicted
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.axhline(0, color="0.4", lw=1)
    ax.scatter(f["y_hat"], f["resid"], s=18, color=color)
    ax.set_ylim(-r_lim, r_lim)
    ax.set_title(f"2D {label}: residuals vs. predicted")
    ax.set_xlabel("Predicted ŷ"); ax.set_ylabel("Residual")
    save(fig, f"2d_resid_vs_pred_{tag}.png")

    # 4b. Normal Q-Q of residuals
    fig, ax = plt.subplots(figsize=(6, 4.5))
    (osm, osr), (slope, intercept, _) = stats.probplot(f["resid"], dist="norm")
    ax.scatter(osm, osr, s=18, color=color)
    ax.plot(osm, slope * osm + intercept, color="black", lw=1.2)
    ax.set_title(f"2D {label}: normal Q-Q of residuals")
    ax.set_xlabel("Theoretical quantiles"); ax.set_ylabel("Sample quantiles")
    save(fig, f"2d_qq_{tag}.png")

# 5. Metrics vs noise level
lv = ["L", "M", "H"]

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(lv, table["R2"], "o-", label="R²")
ax.plot(lv, table["Adj R2"], "s--", label="Adjusted R²")
ax.set_title("Explained variance vs. noise"); ax.set_ylim(0, 1)
ax.set_xlabel("Noise level"); ax.legend(frameon=False)
save(fig, "2d_r2_vs_noise.png")

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(lv, table["RSE"], "o-", label="RSE")
ax.set_title("RSE (estimated noise σ) vs. noise"); ax.set_ylim(bottom=0)
ax.set_xlabel("Noise level"); ax.legend(frameon=False)
save(fig, "2d_rse_vs_noise.png")

print(f"\nSaved results_2d.csv and {len(saved)} figures to '{FIG_DIR}/':")
for name in saved:
    print("  ", name)
