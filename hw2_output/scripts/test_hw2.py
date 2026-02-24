import os
import tempfile
import numpy as np
import pandas as pd

from pivot_and_bootstrap.pca_analysis import fit_pca_dask
from pivot_and_bootstrap.tail_analysis import analyze_coefficients
from pivot_and_bootstrap.bootstrap_stability import bootstrap_pca_stability


def _make_parquet(tmpdir, n=200):
    # Create synthetic wide table dataframe
    hours = {f"hour_{i}": np.sin(np.linspace(0, 2 * np.pi, n) + i) + np.random.normal(scale=0.1, size=n) for i in range(24)}
    df = pd.DataFrame(hours)
    # simple index
    df.index = pd.MultiIndex.from_product([["yellow"], pd.date_range("2023-01-01", periods=n).date, range(n)[:n]], names=["taxi_type", "date", "pickup_place"]) if n>0 else df.index
    path = os.path.join(tmpdir, "test_wide.parquet")
    df.to_parquet(path)
    return path


def test_fit_pca_and_tail(tmp_path):
    parquet = _make_parquet(str(tmp_path), n=120)
    outdir = os.path.join(str(tmp_path), "out")
    res = fit_pca_dask(parquet, outdir)
    assert os.path.exists(res["model_path"])
    # load and run tail analysis
    import pickle
    model = pickle.load(open(res["model_path"], "rb"))
    tail = analyze_coefficients(model["components"], outdir)
    assert "classification" in tail


def test_bootstrap_small(tmp_path):
    parquet = _make_parquet(str(tmp_path), n=80)
    outdir = os.path.join(str(tmp_path), "out2")
    res = bootstrap_pca_stability(parquet, outdir, B=10, n_components=2)
    assert os.path.exists(res["report_json"]) and os.path.exists(res["eigen_corr_boxplot"]) and os.path.exists(res["pc1_band"])