"""PCA utilities using Dask for large parquet tables.

Key functions:
- fit_pca_dask(parquet_path, output_dir, n_components=None, storage_options=None)
- save_pca_model(model_dict, output_path)

Notes:
- Expects input Parquet with columns hour_0 .. hour_23 and index (taxi_type, date, pickup_place)
- Replaces missing values with column means (computed excluding missing values)
- Computes covariance matrix as average of outer products once using dask arrays
"""
from __future__ import annotations

import os
import pickle
from typing import Dict, List, Optional

import dask.dataframe as dd
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt


def _hour_cols() -> List[str]:
    return [f"hour_{i}" for i in range(24)]


def fit_pca_dask(parquet_path: str, output_dir: str, n_components: Optional[int] = None, storage_options: Optional[dict] = None) -> Dict:
    """Fit PCA on unnormalized wide table stored as Parquet using Dask.

    Parameters
    ----------
    parquet_path: str
        Path to Parquet (file or directory). Supports S3 paths via dask with storage_options.
    output_dir: str
        Directory where outputs will be saved (pkl and png).
    n_components: int or None
        Number of components to keep. If None, keep all (24).
    storage_options: dict or None
        Passed to ``dd.read_parquet`` (e.g., {'anon': True} for public S3).

    Returns
    -------
    dict
        Contains 'components' (array shape (n_components, 24)), 'variances', 'explained_ratio' and file paths saved.
    """
    os.makedirs(output_dir, exist_ok=True)

    hour_cols = _hour_cols()

    df = dd.read_parquet(parquet_path, storage_options=storage_options)

    # Ensure hour columns exist
    missing = [c for c in hour_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected hour columns: {missing}")

    # Compute column means (excluding missing values)
    col_means = df[hour_cols].mean().compute()

    # Fill missing with column means
    df_filled = df[hour_cols].fillna(col_means.to_dict())

    # Convert to dask array (rows × 24)
    arr = df_filled.to_dask_array(lengths=True).astype(float)

    # Number of rows
    n = int(df_filled.shape[0].compute())

    # Compute mean across rows using dask
    mean = arr.mean(axis=0)
    mean = mean.compute()

    # Centered array
    centered = arr - mean

    # Compute covariance as (centered.T @ centered) / n
    # This performs the averaging of outer products once
    cov = da.matmul(centered.T, centered) / n
    cov = cov.compute()

    # Eigen decomposition
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    if n_components is None:
        n_components = eigvecs.shape[1]

    components = eigvecs[:, :n_components]  # shape (24, n_components)
    variances = eigvals[:n_components]

    # Save model
    model = {
        "components": components,  # columns are components
        "variances": variances,
        "hour_cols": hour_cols,
        "mean": mean,
    }

    pkl_path = os.path.join(output_dir, "pca_model.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    # Variance explained plot
    var_ratio = variances / np.sum(eigvals)
    cumvar = np.cumsum(var_ratio)
    png_path = os.path.join(output_dir, "variance_explained.png")
    plt.figure(figsize=(6, 4))
    plt.plot(np.arange(1, len(var_ratio) + 1), cumvar, marker="o")
    plt.xlabel("Number of components")
    plt.ylabel("Cumulative variance explained")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close()

    return {
        "model_path": pkl_path,
        "variance_plot": png_path,
        "components_shape": components.shape,
        "variances": variances,
    }


def save_pca_model(model: Dict, output_path: str):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Parquet path (local or s3://)")
    parser.add_argument("--output-dir", default="./hw2_output", help="Output directory")
    parser.add_argument("--anon-s3", action="store_true", help="Use anonymous S3 access")
    args = parser.parse_args()

    sopts = {"anon": True} if args.anon_s3 else None
    res = fit_pca_dask(args.input, args.output_dir, storage_options=sopts)
    print(res)