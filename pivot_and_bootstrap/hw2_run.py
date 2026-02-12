"""End-to-end runner for HW2 parts 1-4.

Usage example:
python -m pivot_and_bootstrap.hw2_run --input s3://dsc291-pprashant-results/taxi-wide/full --output-dir ./hw2_output --anon-s3
"""
from __future__ import annotations

import argparse
import os
import pandas as pd

from .pca_analysis import fit_pca_dask
from .tail_analysis import analyze_coefficients
from .mapping import create_pc1_pc2_map
from .bootstrap_stability import bootstrap_pca_stability


def aggregate_scores_by_pickup_place(pca_pkl_path: str, parquet_path: str, output_scores_csv: str):
    # Compute PC scores = X @ components for each row and aggregate mean by pickup_place
    import pickle
    import numpy as np

    with open(pca_pkl_path, "rb") as f:
        model = pickle.load(f)
    components = model["components"]  # shape (24, n_components)
    hour_cols = model.get("hour_cols")

    # Load parquet in chunks (pandas) to compute scores
    df = pd.read_parquet(parquet_path)
    X = df[hour_cols].fillna(df[hour_cols].mean())
    scores = X.values @ components  # (n_rows, n_components)
    n_comp = components.shape[1]
    cols = {i: f"pc{i+1}" for i in range(1, n_comp + 1)}

    scores_df = pd.DataFrame(scores, columns=[f"pc{i+1}" for i in range(n_comp)])
    scores_df["pickup_place"] = df.index.get_level_values("pickup_place") if hasattr(df.index, "names") else df["pickup_place"]

    agg = scores_df.groupby("pickup_place").mean().reset_index()
    agg.to_csv(output_scores_csv, index=False)
    return output_scores_csv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="s3://dsc291-pprashant-results/taxi-wide/full", help="Parquet wide table path")
    parser.add_argument("--output-dir", default="./hw2_output")
    parser.add_argument("--anon-s3", action="store_true")
    parser.add_argument("--zones-csv", default=None, help="CSV with pickup_place,latitude,longitude")
    parser.add_argument("--B", type=int, default=100)
    args = parser.parse_args()

    sopts = {"anon": True} if args.anon_s3 else None

    os.makedirs(args.output_dir, exist_ok=True)

    # Part 1: PCA
    print("Running PCA (Part 1)")
    pca_res = fit_pca_dask(args.input, args.output_dir, storage_options=sopts)
    pca_model_path = pca_res["model_path"]

    # Part 2: Tail analysis
    print("Running tail analysis (Part 2)")
    import pickle as _pickle
    with open(pca_model_path, "rb") as _f:
        _model = _pickle.load(_f)
    tail_res = analyze_coefficients(_model["components"], args.output_dir)

    # Part 3: Map
    print("Preparing PC scores and Folium map (Part 3)")
    scores_csv = os.path.join(args.output_dir, "pc_scores_by_pickup_place.csv")
    aggregate_scores_by_pickup_place(pca_model_path, args.input, scores_csv)

    if args.zones_csv is None:
        print("No zones CSV provided; skipping map creation. Provide --zones-csv to enable map.")
    else:
        zones_df = pd.read_csv(args.zones_csv)
        scores_df = pd.read_csv(scores_csv)
        map_path = os.path.join(args.output_dir, "pc1_pc2_folium_map.html")
        create_pc1_pc2_map(scores_df, zones_df, map_path)
        print("Saved map to", map_path)

    # Part 4: Bootstrap
    print("Running bootstrap stability analysis (Part 4)")
    boot_res = bootstrap_pca_stability(args.input, args.output_dir, B=args.B, n_components=2)
    print("Bootstrap results:", boot_res)


if __name__ == "__main__":
    main()