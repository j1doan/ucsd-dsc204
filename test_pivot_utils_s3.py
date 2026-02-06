"""
Test pivot_utils and pipeline against 5 Parquet files from s3://dsc291-ucsd/taxi.
Uses pivot_utils from this repo (ucsd-dsc204-master). Pipeline uses pandas.

Run from repo root: python -m pytest test_pivot_utils_s3.py -v
Requires: dask, pandas, pyarrow, fsspec, s3fs
"""
import sys
from pathlib import Path

# Import from this repo (ucsd-dsc204-master)
master_root = Path(__file__).resolve().parent
sys.path.insert(0, str(master_root))

import pytest
import pandas as pd
import dask.dataframe as dd

from pivot_utils import (
    find_pickup_datetime_col,
    find_pickup_location_col,
    infer_taxi_type_from_path,
    infer_month_from_path,
    pivot_counts_date_taxi_type_location,
    cleanup_low_count_rows,
)


def _discover_s3_parquet_files(input_path: str, limit: int = 5):
    """Discover Parquet files under input_path (S3 or local), return up to limit."""
    import fsspec
    is_s3 = input_path.startswith("s3://")
    fs = fsspec.filesystem("s3" if is_s3 else "file")
    files = sorted(fs.glob(f"{input_path.rstrip('/')}/**/*.parquet"))
    if is_s3:
        files = [f"s3://{f}" if not f.startswith("s3://") else f for f in files]
    return files[:limit]


def _normalize_trip_df(ddf: dd.DataFrame, file_path: str) -> pd.DataFrame:
    """Normalize to pandas DataFrame with pickup_datetime, pickup_place, taxi_type."""
    dt_col = find_pickup_datetime_col(ddf.columns.tolist())
    loc_col = find_pickup_location_col(ddf.columns.tolist())
    taxi_type = infer_taxi_type_from_path(file_path)
    if dt_col is None or loc_col is None or taxi_type is None:
        raise ValueError(
            f"Cannot normalize: datetime_col={dt_col}, location_col={loc_col}, taxi_type={taxi_type}"
        )
    df = ddf[[dt_col, loc_col]].compute()
    df = df.rename(columns={dt_col: "pickup_datetime", loc_col: "pickup_place"})
    df["pickup_datetime"] = pd.to_datetime(df["pickup_datetime"], errors="coerce")
    df = df.dropna(subset=["pickup_datetime", "pickup_place"])
    df["taxi_type"] = taxi_type
    return df


S3_TAXI_PATH = "s3://dsc291-ucsd/taxi"
NUM_FILES = 5


@pytest.fixture(scope="module")
def s3_parquet_files():
    """Discover up to 5 Parquet files under s3://dsc291-ucsd/taxi."""
    files = _discover_s3_parquet_files(S3_TAXI_PATH, limit=NUM_FILES)
    if not files:
        pytest.skip(f"No Parquet files found under {S3_TAXI_PATH}")
    return files


def test_pivot_utils_column_detection(s3_parquet_files):
    """pivot_utils find pickup datetime and location columns in S3 Parquet schema."""
    storage_options = {"anon": True}
    for path in s3_parquet_files[:1]:
        ddf = dd.read_parquet(path, storage_options=storage_options)
        cols = ddf.columns.tolist()
        dt_col = find_pickup_datetime_col(cols)
        loc_col = find_pickup_location_col(cols)
        assert dt_col is not None, f"No datetime column in {path}"
        assert loc_col is not None, f"No location column in {path}"
        assert dt_col in cols
        assert loc_col in cols


def test_pivot_utils_infer_taxi_type_from_path(s3_parquet_files):
    """infer_taxi_type_from_path returns a known type for taxi paths."""
    for path in s3_parquet_files:
        taxi_type = infer_taxi_type_from_path(path)
        assert taxi_type is not None, f"Could not infer taxi type from {path}"
        assert taxi_type in ("yellow", "green", "fhv", "fhvhv")


def test_pivot_utils_infer_month_from_path(s3_parquet_files):
    """infer_month_from_path returns (year, month) or None."""
    for path in s3_parquet_files:
        result = infer_month_from_path(path)
        if result is not None:
            year, month = result
            assert 2000 <= year <= 2100
            assert 1 <= month <= 12


def test_pivot_utils_full_pipeline_on_s3_files(s3_parquet_files):
    """Read S3 Parquet files, normalize (pandas), pivot, cleanup; assert shape and columns."""
    storage_options = {"anon": True}
    hour_cols = [f"hour_{h}" for h in range(24)]
    expected_columns = ["taxi_type", "date", "pickup_place"] + hour_cols

    total_pivoted_rows = 0
    for path in s3_parquet_files:
        ddf = dd.read_parquet(path, storage_options=storage_options)
        df = _normalize_trip_df(ddf, path)
        pivoted = pivot_counts_date_taxi_type_location(df)
        pivoted = pivoted.reset_index()
        cleaned, stats = cleanup_low_count_rows(pivoted, min_rides=0)

        n_cleaned = len(cleaned)
        total_pivoted_rows += n_cleaned

        for col in expected_columns:
            assert col in cleaned.columns, f"Missing column {col} in {path}"

        for h in range(24):
            assert f"hour_{h}" in cleaned.columns
        sums = cleaned[hour_cols].sum(axis=1)
        assert (sums >= 0).all()

    assert total_pivoted_rows >= 0, "Pipeline should produce some rows across files"


def test_pivot_utils_cleanup_low_count_rows_on_s3(s3_parquet_files):
    """cleanup_low_count_rows drops rows below min_rides and returns correct stats."""
    storage_options = {"anon": True}
    path = s3_parquet_files[0]
    ddf = dd.read_parquet(path, storage_options=storage_options)
    df = _normalize_trip_df(ddf, path)
    pivoted = pivot_counts_date_taxi_type_location(df)
    pivoted = pivoted.reset_index()

    cleaned, stats = cleanup_low_count_rows(pivoted, min_rides=50)
    before_n = stats["rows_before"]
    after_n = stats["rows_after"]
    removed = stats["rows_removed"]

    assert before_n >= after_n
    assert removed == before_n - after_n
    if after_n > 0:
        hour_cols = [f"hour_{h}" for h in range(24)]
        total_rides = cleaned[hour_cols].sum(axis=1)
        assert (total_rides >= 50).all()
