"""Tail analysis for PCA coefficient loadings.

Produces histogram, Q-Q plot, and log-log survival plot; fits power-law tail to top quantile and reports alpha and R^2.
"""
from __future__ import annotations

import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def analyze_coefficients(components: np.ndarray, output_dir: str, tail_fraction: float = 0.05) -> Dict:
    """Analyze distribution of PCA coefficients.

    Parameters
    ----------
    components : ndarray
        Array of shape (24, n_components) or (n_features, n_components).
    output_dir : str
        Directory to save plots and JSON report.
    tail_fraction : float
        Fraction of extreme values to consider for tail fit (default 0.05 → top 5%).

    Returns
    -------
    dict
        Report with classification, alpha, r2, xmin, and file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    vals = components.ravel()

    # Use signed coefficients for histogram and QQ, but tail fit on absolute magnitudes
    abs_vals = np.abs(vals)
    abs_vals = abs_vals[abs_vals > 0]  # drop zeros

    # Histogram and QQ plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(vals, bins=100, color="C0", alpha=0.8)
    axes[0].set_title("Coefficient histogram")

    stats.probplot(vals, dist="norm", plot=axes[1])
    axes[1].set_title("Q-Q plot vs Normal")

    # Log-log survival plot
    sorted_vals = np.sort(abs_vals)[::-1]
    n = len(sorted_vals)
    survival = np.arange(1, n + 1) / n  # P(X >= x) empirical
    axes[2].loglog(sorted_vals, survival, marker=".", linestyle="none")
    axes[2].set_xlabel("Coefficient magnitude")
    axes[2].set_ylabel("Survival (P(X >= x))")
    axes[2].set_title("Log-Log survival")

    plt.tight_layout()
    png_path = os.path.join(output_dir, "coefficient_distribution.png")
    plt.savefig(png_path, dpi=150)
    plt.close()

    # Fit power-law tail via linear regression on log-log survival for top tail_fraction
    tail_k = max(10, int(np.ceil(tail_fraction * n)))
    tail_vals = sorted_vals[:tail_k]
    tail_surv = survival[:tail_k]

    # Avoid zeros
    mask = (tail_vals > 0) & (tail_surv > 0)
    x = np.log10(tail_vals[mask])
    y = np.log10(tail_surv[mask])

    if len(x) < 5:
        alpha = None
        r2 = None
        classification = "insufficient_data"
    else:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        # For P(X>=x) ~ C x^{-alpha}, log10 S = -alpha * log10 x + c → slope = -alpha
        alpha = -slope
        r2 = r_value ** 2

        # Heuristic classification
        if r2 > 0.85 and alpha is not None and 0 < alpha < 20:
            classification = "heavy_tail"
        else:
            classification = "light_tail"

    report = {
        "classification": classification,
        "alpha": float(alpha) if alpha is not None else None,
        "r2": float(r2) if r2 is not None else None,
        "tail_fraction": float(tail_fraction),
        "tail_points": int(tail_k),
        "coefficient_distribution_png": png_path,
    }

    json_path = os.path.join(output_dir, "tail_analysis_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    report["report_path"] = json_path
    return report


if __name__ == "__main__":
    import argparse
    import pickle

    parser = argparse.ArgumentParser()
    parser.add_argument("--pca-pkl", required=True, help="PCA model pickle produced by fit_pca_dask")
    parser.add_argument("--output-dir", default="./hw2_output", help="Output directory")
    args = parser.parse_args()

    with open(args.pca_pkl, "rb") as f:
        model = pickle.load(f)
    comps = model["components"]
    res = analyze_coefficients(comps, args.output_dir)
    print(res)