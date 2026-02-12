"""Bootstrap stability analysis for PCA eigenvectors.

Produces bootstrap samples, fits PCA on each, and reports stability metrics and plots.
"""
from __future__ import annotations

import json
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes, svd
from scipy.stats import pearsonr


def _fit_pca_numpy(X: np.ndarray, n_components: int):
    # center
    X = X.astype(float)
    mean = np.mean(X, axis=0)
    Xc = X - mean
    n = Xc.shape[0]
    cov = (Xc.T @ Xc) / n
    eigvals, eigvecs = np.linalg.eigh(cov)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvecs[:, :n_components], eigvals[:n_components]


def _component_correlation(a: np.ndarray, b: np.ndarray) -> float:
    # align sign to maximize absolute correlation
    corr = pearsonr(a, b)[0]
    return float(corr)


def subspace_affinity(U: np.ndarray, V: np.ndarray) -> float:
    # U and V have shape (features, k). Compute singular values of U^T V
    s = svd(U.T @ V, compute_uv=False)
    # affinity as average squared cosines
    return float(np.mean(s ** 2))


def bootstrap_pca_stability(parquet_path: str, output_dir: str, B: int = 100, n_components: int = 2, storage_options: Dict = None) -> Dict:
    """Run bootstrap analysis.

    Parameters
    ----------
    parquet_path : str
        Parquet path to wide table (with hour_0..hour_23)
    output_dir : str
        Directory to save outputs
    B : int
        Number of bootstrap samples
    n_components : int
        Number of eigenvectors to extract

    Returns
    -------
    dict
        Paths and summary statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load full table into pandas (may be large; for heavy workloads use dask-based sampling)
    df = pd.read_parquet(parquet_path)
    hour_cols = [f"hour_{i}" for i in range(24)]
    X = df[hour_cols].fillna(df[hour_cols].mean()).values

    # Fit original PCA
    orig_vecs, orig_vars = _fit_pca_numpy(X, n_components)

    # Storage for metrics
    procrustes_dists: List[float] = []
    affinities: List[float] = []
    comp_corrs: List[List[float]] = []
    bootstrap_components: List[np.ndarray] = []

    n = X.shape[0]
    rng = np.random.default_rng(12345)

    for b in range(B):
        idx = rng.integers(0, n, size=n)
        Xb = X[idx]
        b_vecs, b_vars = _fit_pca_numpy(Xb, n_components)
        bootstrap_components.append(b_vecs)

        # Procrustes distance between matrices (align b_vecs to orig_vecs)
        R, scale = orthogonal_procrustes(b_vecs, orig_vecs)
        transformed = b_vecs @ R
        dist = np.linalg.norm(transformed - orig_vecs)
        procrustes_dists.append(float(dist))

        # subspace affinity
        aff = subspace_affinity(orig_vecs, b_vecs)
        affinities.append(aff)

        # component-wise correlations (align signs)
        corrs = []
        for j in range(n_components):
            v_orig = orig_vecs[:, j]
            v_boot = b_vecs[:, j]
            # align sign
            if np.corrcoef(v_orig, v_boot)[0, 1] < 0:
                v_boot = -v_boot
            corr = np.corrcoef(v_orig, v_boot)[0, 1]
            corrs.append(float(corr))
        comp_corrs.append(corrs)

    comp_corrs = np.array(comp_corrs)  # B x n_components

    # Save JSON report
    report = {
        "B": B,
        "n_components": n_components,
        "procrustes_dist_mean": float(np.mean(procrustes_dists)),
        "procrustes_dist_std": float(np.std(procrustes_dists)),
        "subspace_affinity_mean": float(np.mean(affinities)),
        "subspace_affinity_std": float(np.std(affinities)),
        "component_correlation_mean": comp_corrs.mean(axis=0).tolist(),
        "component_correlation_std": comp_corrs.std(axis=0).tolist(),
    }

    json_path = os.path.join(output_dir, "bootstrap_stability_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Plot eigenvector stability: boxplot of component-wise correlations
    plt.figure(figsize=(6, 4))
    plt.boxplot([comp_corrs[:, j] for j in range(n_components)], labels=[f"PC{j+1}" for j in range(n_components)])
    plt.ylabel("Correlation with original component")
    plt.title("Bootstrap component-wise correlations")
    plt.tight_layout()
    png1 = os.path.join(output_dir, "eigenvector_corr_boxplot.png")
    plt.savefig(png1, dpi=150)
    plt.close()

    # Plot coefficient bands for first component across bootstraps
    b1_all = np.stack([b[:, 0] for b in bootstrap_components], axis=1)  # (features, B)
    med = np.median(b1_all, axis=1)
    q1 = np.percentile(b1_all, 25, axis=1)
    q3 = np.percentile(b1_all, 75, axis=1)

    plt.figure(figsize=(8, 4))
    x = np.arange(len(med))
    plt.fill_between(x, q1, q3, color="C0", alpha=0.3)
    plt.plot(x, med, color="C0")
    plt.title("Bootstrap band for PC1 coefficients (median and IQR)")
    plt.xlabel("Coefficient index (hour)")
    plt.ylabel("Coefficient value")
    plt.tight_layout()
    png2 = os.path.join(output_dir, "bootstrap_pc1_band.png")
    plt.savefig(png2, dpi=150)
    plt.close()

    return {
        "report_json": json_path,
        "eigen_corr_boxplot": png1,
        "pc1_band": png2,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="./hw2_output")
    parser.add_argument("--B", type=int, default=100)
    parser.add_argument("--n-components", type=int, default=2)
    args = parser.parse_args()

    res = bootstrap_pca_stability(args.input, args.output_dir, B=args.B, n_components=args.n_components)
    print(res)