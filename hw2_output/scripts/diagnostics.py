"""PCA fitting diagnostics for the pivot_and_bootstrap package.

Provides statistical checks on a fitted PCA model including:
- Shapiro-Wilk normality test per component and pooled
- Homoscedasticity tests (Bartlett + Levene) across hours and PCs
- Vandermonde structural check on the loading matrix
- Mahalanobis distance per hour in whitened loading space
- Comprehensive 5-panel diagnostic figure

Usage
-----
    from pivot_and_bootstrap.diagnostics import run_diagnostics
    import pickle

    with open("hw2_output/pca_model.pkl", "rb") as f:
        model = pickle.load(f)

    run_diagnostics(model, output_dir="hw2_output/diagnostics")

CLI
---
    python -m pivot_and_bootstrap.diagnostics --pca-pkl hw2_output/pca_model.pkl --output-dir hw2_output/diagnostics
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.stats import shapiro, bartlett, levene, chi2
import warnings

warnings.filterwarnings("ignore")


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _shapiro_per_component(components: np.ndarray) -> Dict:
    """Run Shapiro-Wilk on each PC's loading vector (24 values)."""
    results = {}
    for j in range(components.shape[1]):
        w, p = shapiro(components[:, j])
        results[f"PC{j+1}"] = {"W": float(w), "p": float(p), "reject_h0": bool(p < 0.05)}
    pooled_w, pooled_p = shapiro(components.ravel())
    results["pooled"] = {"W": float(pooled_w), "p": float(pooled_p), "reject_h0": bool(pooled_p < 0.05)}
    return results


def _homoscedasticity(components: np.ndarray) -> Dict:
    """Bartlett + Levene tests across hours and across PCs."""
    hour_groups = [components[h, :] for h in range(components.shape[0])]
    comp_groups = [components[:, j] for j in range(components.shape[1])]

    b_stat_h, b_p_h = bartlett(*hour_groups)
    l_stat_h, l_p_h = levene(*hour_groups)
    b_stat_c, b_p_c = bartlett(*comp_groups)
    l_stat_c, l_p_c = levene(*comp_groups)

    hour_vars = [float(np.var(components[h, :])) for h in range(components.shape[0])]

    return {
        "across_hours": {
            "bartlett": {"stat": float(b_stat_h), "p": float(b_p_h), "reject_h0": bool(b_p_h < 0.05)},
            "levene":   {"stat": float(l_stat_h), "p": float(l_p_h), "reject_h0": bool(l_p_h < 0.05)},
        },
        "across_pcs": {
            "bartlett": {"stat": float(b_stat_c), "p": float(b_p_c), "reject_h0": bool(b_p_c < 0.05)},
            "levene":   {"stat": float(l_stat_c), "p": float(l_p_c), "reject_h0": bool(l_p_c < 0.05)},
        },
        "per_hour_variances": hour_vars,
        "verdict": "homoscedastic" if not (b_p_h < 0.05 or l_p_h < 0.05) else "heteroscedastic",
    }


def _vandermonde_check(components: np.ndarray) -> Dict:
    """
    Check whether the loading matrix has Vandermonde structure.

    A Vandermonde matrix V satisfies V[i,j] = x_i^j, so log|V[i,j]| is
    linear in j. We compute R² of that regression for each row and report
    mean/max. R² -> 1 would indicate Vandermonde structure.
    """
    r2_per_row = []
    for h in range(components.shape[0]):
        row = np.abs(components[h, :])
        row_nz = row[row > 1e-10]
        if len(row_nz) < 5:
            r2_per_row.append(0.0)
            continue
        x = np.arange(len(row_nz))
        _, _, r, *_ = stats.linregress(x, np.log(row_nz))
        r2_per_row.append(float(r ** 2))

    variances_arr = components  # just for eigenvalue ratio check
    return {
        "r2_per_row": r2_per_row,
        "mean_r2": float(np.mean(r2_per_row)),
        "max_r2":  float(np.max(r2_per_row)),
        "is_vandermonde": bool(np.mean(r2_per_row) > 0.95),
        "note": (
            "R² -> 1 for all rows would indicate Vandermonde structure. "
            "PCA eigenvector matrices are orthogonal, not polynomial, so low R² is expected."
        ),
    }


def _mahalanobis(components: np.ndarray, variances: np.ndarray, shrinkage: float = 0.1) -> Dict:
    """
    Mahalanobis distance for each hour in whitened loading space.

    Each hour h becomes a 24-dim point:
        x_h[j] = loading[h, j] * sqrt(lambda[j])

    This puts all components in original-data variance units.
    Covariance is regularised with Ledoit-Wolf shrinkage to handle n=p=24.

    Returns distances, D², p-values (chi²(p)), and outlier flags.
    """
    n_hours, n_pcs = components.shape
    X = components * np.sqrt(variances)[np.newaxis, :]   # (24, 24) whitened

    mu  = X.mean(axis=0)
    cov = np.cov(X.T)

    # Ledoit-Wolf shrinkage: blend with scaled identity
    cov_shrunk = (1 - shrinkage) * cov + shrinkage * (np.trace(cov) / n_pcs) * np.eye(n_pcs)
    cov_inv    = np.linalg.inv(cov_shrunk)

    d2 = np.array([float((X[h] - mu) @ cov_inv @ (X[h] - mu)) for h in range(n_hours)])
    d  = np.sqrt(d2)

    thresh_95  = float(chi2.ppf(0.95,  df=n_pcs))
    thresh_99  = float(chi2.ppf(0.99,  df=n_pcs))
    thresh_999 = float(chi2.ppf(0.999, df=n_pcs))

    per_hour = []
    for h in range(n_hours):
        pval = float(1 - chi2.cdf(d2[h], df=n_pcs))
        per_hour.append({
            "hour":      h,
            "D":         float(d[h]),
            "D2":        float(d2[h]),
            "p_value":   pval,
            "outlier_95":  bool(d2[h] > thresh_95),
            "outlier_99":  bool(d2[h] > thresh_99),
            "outlier_999": bool(d2[h] > thresh_999),
        })

    ranked = sorted(per_hour, key=lambda x: x["D2"], reverse=True)

    return {
        "shrinkage_alpha": shrinkage,
        "chi2_df": n_pcs,
        "thresholds": {"p0.05": thresh_95, "p0.01": thresh_99, "p0.001": thresh_999},
        "per_hour": per_hour,
        "ranked_by_D": [r["hour"] for r in ranked],
        "n_outliers_95":  sum(r["outlier_95"]  for r in per_hour),
        "n_outliers_99":  sum(r["outlier_99"]  for r in per_hour),
        "n_outliers_999": sum(r["outlier_999"] for r in per_hour),
        "note": (
            "Because V is orthogonal, naive Mahalanobis gives identical distances. "
            "Whitening (loading * sqrt(eigenvalue)) breaks symmetry and yields meaningful separation."
        ),
    }


def _total_contribution(components: np.ndarray, variances: np.ndarray) -> Dict:
    """Per-hour total variance contribution and Z-scores."""
    contrib = (components ** 2 * variances[np.newaxis, :]).sum(axis=1)
    z = (contrib - contrib.mean()) / contrib.std()
    return {
        "per_hour":   contrib.tolist(),
        "z_scores":   z.tolist(),
        "mean":       float(contrib.mean()),
        "std":        float(contrib.std()),
        "min_hour":   int(np.argmin(contrib)),
        "max_hour":   int(np.argmax(contrib)),
        "outliers_z2": [int(h) for h in np.where(np.abs(z) > 2)[0]],
    }


# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────

def _make_figure(
    components:   np.ndarray,
    variances:    np.ndarray,
    sw_results:   Dict,
    homo_results: Dict,
    vand_results: Dict,
    mah_results:  Dict,
    contrib_results: Dict,
    output_path:  str,
):
    hours     = np.arange(components.shape[0])
    n_pcs     = components.shape[1]
    sw_p      = [sw_results[f"PC{j+1}"]["p"] for j in range(n_pcs)]
    hour_vars = homo_results["per_hour_variances"]
    vand_r2   = vand_results["r2_per_row"]
    d_w       = np.array([r["D"]  for r in mah_results["per_hour"]])
    d2_w      = np.array([r["D2"] for r in mah_results["per_hour"]])
    z_scores  = np.array(contrib_results["z_scores"])
    thresh_95_d = mah_results["thresholds"]["p0.05"] ** 0.5

    fig = plt.figure(figsize=(20, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── Panel 1: Shapiro-Wilk p per PC ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    colors_sw = ["tomato" if p < 0.05 else "steelblue" for p in sw_p]
    ax1.bar(range(1, n_pcs + 1), sw_p, color=colors_sw, alpha=0.85)
    ax1.axhline(0.05, color="red", linestyle="--", lw=1.5, label="α=0.05")
    ax1.set_xlabel("PC index", fontsize=9)
    ax1.set_ylabel("Shapiro-Wilk p-value", fontsize=9)
    ax1.set_title("Shapiro-Wilk p-value per PC\n(red = reject normality at α=0.05)", fontsize=9)
    ax1.legend(fontsize=8)
    ax1.set_xticks(range(1, n_pcs + 1, 2))
    reject_count = sum(p < 0.05 for p in sw_p)
    ax1.text(0.97, 0.97, f"{reject_count}/{n_pcs} rejected",
             transform=ax1.transAxes, ha="right", va="top", fontsize=8, color="red",
             bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # ── Panel 2: Homoscedasticity ────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    hv   = np.array(hour_vars)
    b_p  = homo_results["across_hours"]["bartlett"]["p"]
    l_p  = homo_results["across_hours"]["levene"]["p"]
    ax2.bar(hours, hv, color="steelblue", alpha=0.8)
    ax2.axhline(hv.mean(), color="red", linestyle="--", lw=1.5, label=f"Mean={hv.mean():.4f}")
    ax2.fill_between([-0.5, hours[-1] + 0.5],
                     hv.mean() - 2 * hv.std(), hv.mean() + 2 * hv.std(),
                     alpha=0.12, color="red", label="±2σ band")
    ax2.set_xlabel("Hour of day", fontsize=9)
    ax2.set_ylabel("Var(loadings across PCs)", fontsize=9)
    ax2.set_title(
        f"Homoscedasticity: Loading Variance per Hour\n"
        f"(Bartlett p={b_p:.3f}, Levene p={l_p:.3f})", fontsize=9)
    ax2.legend(fontsize=8)
    ax2.set_xticks(hours)

    # ── Panel 3: Vandermonde check ───────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(hours, vand_r2, color="mediumpurple", alpha=0.85)
    ax3.axhline(1.0, color="red", linestyle="--", lw=1.5, label="R²=1 (perfect Vandermonde)")
    ax3.axhline(np.mean(vand_r2), color="orange", linestyle=":", lw=1.5,
                label=f"Mean R²={np.mean(vand_r2):.3f}")
    ax3.set_xlabel("Hour (row index)", fontsize=9)
    ax3.set_ylabel("R² of log|loading| ~ col index", fontsize=9)
    ax3.set_title("Vandermonde Structural Check\n(R²→1 would indicate Vandermonde structure)", fontsize=9)
    ax3.set_ylim(0, 1.1)
    ax3.legend(fontsize=8)
    ax3.set_xticks(hours)

    # ── Panel 4: Mahalanobis distances ──────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    bar_colors = ["tomato" if d > thresh_95_d else "steelblue" for d in d_w]
    ax4.bar(hours, d_w, color=bar_colors, alpha=0.85)
    ax4.axhline(thresh_95_d, color="red", linestyle="--", lw=1.5,
                label=f"χ²({n_pcs}) 95% → D={thresh_95_d:.2f}")
    ax4.set_xlabel("Hour of day", fontsize=9)
    ax4.set_ylabel("Mahalanobis D", fontsize=9)
    ax4.set_title(
        "Mahalanobis Distance per Hour\n(whitened loading space, Ledoit-Wolf shrinkage)", fontsize=9)
    ax4.legend(fontsize=8)
    ax4.set_xticks(hours)

    # ── Panel 5: Z-scores of total contribution ──────────────────────
    ax5 = fig.add_subplot(gs[1, 1:])
    bar_colors5 = ["tomato" if abs(z) > 2 else "steelblue" for z in z_scores]
    ax5.bar(hours, z_scores, color=bar_colors5, alpha=0.85)
    ax5.axhline( 2, color="red", linestyle="--", lw=1.2, label="|z|=2")
    ax5.axhline(-2, color="red", linestyle="--", lw=1.2)
    ax5.axhline( 0, color="black", lw=0.8)
    ax5.set_xlabel("Hour of day", fontsize=9)
    ax5.set_ylabel("Z-score", fontsize=9)
    ax5.set_title("Per-Hour Total Variance Contribution Z-Score\n(Σ loading²×λ, standardised)", fontsize=9)
    ax5.legend(fontsize=8)
    ax5.set_xticks(hours)

    plt.suptitle(
        "Extended PCA Fitting Diagnostics: "
        "Normality · Homoscedasticity · Vandermonde · Mahalanobis",
        fontsize=12, fontweight="bold", y=1.01)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def run_diagnostics(
    model:      Dict,
    output_dir: str = "./hw2_output/diagnostics",
    shrinkage:  float = 0.1,
) -> Dict:
    """Run all PCA fitting diagnostics and save outputs.

    Parameters
    ----------
    model : dict
        PCA model dict with keys: 'components' (24×24), 'variances' (24,),
        'mean' (24,), 'hour_cols' (list[str]).
    output_dir : str
        Directory where the JSON report and PNG figure will be saved.
    shrinkage : float
        Ledoit-Wolf shrinkage coefficient for Mahalanobis covariance (default 0.1).

    Returns
    -------
    dict
        Full diagnostic report including all test results and output file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    components = model["components"]   # (n_features, n_components)
    variances  = model["variances"]    # (n_components,)

    # ── Run all checks ────────────────────────────────────────────────
    sw_results      = _shapiro_per_component(components)
    homo_results    = _homoscedasticity(components)
    vand_results    = _vandermonde_check(components)
    mah_results     = _mahalanobis(components, variances, shrinkage=shrinkage)
    contrib_results = _total_contribution(components, variances)

    # ── Additional scalar summaries ───────────────────────────────────
    total_var   = float(variances.sum())
    var_ratio   = variances / total_var
    cumvar      = float(np.cumsum(var_ratio)[1])   # PC1+PC2

    orthogonality_max = float(np.abs(components.T @ components - np.eye(components.shape[1])).max())

    cond_number    = float(variances[0] / variances[-1])
    effective_rank = float(variances.sum() ** 2 / (variances ** 2).sum())
    kaiser_pass    = int(np.sum(variances / variances[0] > 1 / len(variances)))

    # ── Save figure ───────────────────────────────────────────────────
    fig_path = os.path.join(output_dir, "extended_diagnostics.png")
    _make_figure(
        components, variances,
        sw_results, homo_results, vand_results,
        mah_results, contrib_results,
        fig_path,
    )

    # ── Assemble report ───────────────────────────────────────────────
    report = {
        "summary": {
            "condition_number":  cond_number,
            "effective_rank":    effective_rank,
            "kaiser_passing":    f"{kaiser_pass}/{len(variances)}",
            "pc1_var_explained": float(var_ratio[0]),
            "pc1_pc2_var":       cumvar,
            "orthogonality_max_off_diag": orthogonality_max,
        },
        "shapiro_wilk":       sw_results,
        "homoscedasticity":   homo_results,
        "vandermonde":        vand_results,
        "mahalanobis":        mah_results,
        "total_contribution": contrib_results,
        "outputs": {
            "figure":      fig_path,
            "report_json": os.path.join(output_dir, "diagnostics_report.json"),
        },
    }

    json_path = report["outputs"]["report_json"]
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[diagnostics] Report saved to {json_path}")
    print(f"[diagnostics] Figure saved to {fig_path}")
    return report


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser(description="Run PCA fitting diagnostics.")
    parser.add_argument("--pca-pkl",    required=True, help="Path to pca_model.pkl")
    parser.add_argument("--output-dir", default="./hw2_output/diagnostics")
    parser.add_argument("--shrinkage",  type=float, default=0.1,
                        help="Ledoit-Wolf shrinkage alpha for Mahalanobis (default: 0.1)")
    args = parser.parse_args()

    with open(args.pca_pkl, "rb") as f:
        model = pickle.load(f)

    report = run_diagnostics(model, output_dir=args.output_dir, shrinkage=args.shrinkage)

    print("\n=== SUMMARY ===")
    for k, v in report["summary"].items():
        print(f"  {k}: {v}")
    print(f"\n  Shapiro-Wilk: {sum(v['reject_h0'] for k,v in report['shapiro_wilk'].items() if k != 'pooled')}/24 PCs reject normality")
    print(f"  Homoscedasticity verdict: {report['homoscedasticity']['verdict']}")
    print(f"  Vandermonde: {'YES' if report['vandermonde']['is_vandermonde'] else 'NO'} (mean R²={report['vandermonde']['mean_r2']:.3f})")
    print(f"  Mahalanobis outliers (p<0.05): {report['mahalanobis']['n_outliers_95']}")
