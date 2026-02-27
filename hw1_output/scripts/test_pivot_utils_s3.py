"""
Demo: Check pivot_utils functions on 5 random parquet files from s3://dsc291-ucsd/taxi.
Run from repo: python demo_pivot_utils_s3.py
Requires: dask, pandas, pyarrow, fsspec, s3fs
"""
import random
import sys
from pathlib import Path

# Ensure repo root on path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root))

from pivot_utils import (
    discover_parquet_files,
    get_storage_options,
    get_filesystem,
    is_s3_path,
    read_parquet_with_dask,
    find_pickup_datetime_col,
    find_pickup_location_col,
    infer_taxi_type_from_path,
    infer_month_from_path,
    pivot_counts_date_taxi_type_location,
    cleanup_low_count_rows,
    get_common_schema,
)

S3_TAXI_PATH = "s3://dsc291-ucsd/taxi"
NUM_FILES = 5
RANDOM_SEED = 42

def get_column_names_per_file(file_paths, storage_options=None):
    """
    Get column names for each chosen parquet file (S3 or local).

    Args:
        file_paths: List of parquet file paths (e.g. S3 URIs or local paths).
        storage_options: Optional dict for storage (e.g. {"anon": True} for public S3).
                         If None and any path is S3, uses get_storage_options(first_s3_path).

    Returns:
        dict: Mapping file_path -> list of column names (order preserved).
    """
    if not file_paths:
        return {}
    if storage_options is None and any(
        p.lower().startswith("s3://") for p in file_paths
    ):
        storage_options = get_storage_options(file_paths[0])
    elif storage_options is None:
        storage_options = {}

    result = {}
    for path in file_paths:
        try:
            ddf = read_parquet_with_dask(path, storage_options=storage_options)
            result[path] = ddf.columns.tolist()
        except Exception as e:
            result[path] = []  # or raise / log
    return result

def _normalize_trip_df(ddf, file_path: str):
    """Normalize to pandas DataFrame with pickup_datetime, pickup_place, taxi_type."""
    import pandas as pd
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


def main():
    random.seed(RANDOM_SEED)
    print("=== pivot_utils demo on S3 taxi data (10 random files) ===\n")

    # 1) Discover parquet files
    print("1. Discovering parquet files...")
    all_files = discover_parquet_files(S3_TAXI_PATH)
    if not all_files:
        print("No parquet files found. Exiting.")
        return
    print(f"   Found {len(all_files)} files. Sampling {NUM_FILES} random.")
    files = random.sample(all_files, min(NUM_FILES, len(all_files)))

    columns_per_file = get_column_names_per_file(files, storage_options=get_storage_options(S3_TAXI_PATH))
    print("\nColumn names per file:")
    for path, cols in columns_per_file.items():
        print(f"   {path.split('/')[-1]}: {len(cols)} cols -> {cols}")

    for i, f in enumerate(files):
        print(f"   [{i+1}] {f}")

    storage_options = get_storage_options(S3_TAXI_PATH)
    print(f"\n2. Storage options for S3: {storage_options}")

    # 3) Column detection + path inference on first file
    print("\n3. Column detection & path inference (first file)...")
    path0 = files[0]
    ddf0 = read_parquet_with_dask(path0, storage_options=storage_options)
    cols0 = ddf0.columns.tolist()
    dt_col = find_pickup_datetime_col(cols0)
    loc_col = find_pickup_location_col(cols0)
    taxi_type = infer_taxi_type_from_path(path0)
    month_info = infer_month_from_path(path0)
    print(f"   Datetime col: {dt_col}, Location col: {loc_col}")
    print(f"   Taxi type: {taxi_type}, (year, month): {month_info}")
    if loc_col is None:
        print(f"   [DEBUG] Columns ({len(cols0)}): {cols0}")
        print(f"   [DEBUG] Column types: {[type(c).__name__ for c in cols0]}")

    # 4) Common schema across the 10 files (optional: use first 3 for speed)
    print("\n4. Common schema (first 3 files)...")
    try:
        # get_common_schema can take file paths; pyarrow may need storage_options
        ddfs = [read_parquet_with_dask(p, storage_options=storage_options) for p in files[:3]]
        schema = get_common_schema([ddf.columns.tolist() for ddf in ddfs])
        # get_common_schema expects df_list as list of dfs or paths; with paths it uses pq
        # So we pass dataframes
        schema = get_common_schema(ddfs)
        print(f"   Common schema: {schema}")
    except Exception as e:
        print(f"   (Schema check skipped or failed: {e})")

    # 5) Full pipeline: read -> normalize -> pivot -> cleanup for each file
    print("\n5. Full pipeline (normalize -> pivot -> cleanup) per file:")
    hour_cols = [f"hour_{h}" for h in range(24)]
    total_cleaned_rows = 0
    for i, path in enumerate(files):
        try:
            ddf = read_parquet_with_dask(path, storage_options=storage_options)
            df = _normalize_trip_df(ddf, path)
            pivoted = pivot_counts_date_taxi_type_location(df)
            pivoted = pivoted.reset_index()
            cleaned, stats = cleanup_low_count_rows(pivoted, min_rides=0)
            n = len(cleaned)
            total_cleaned_rows += n
            assert all(c in cleaned.columns for c in ["taxi_type", "date", "pickup_place"] + hour_cols)
            print(f"   [{i+1}] {path.split('/')[-1]}: pivoted -> {stats['rows_before']} rows, cleaned -> {n} rows")
        except Exception as e:
            print(f"   [{i+1}] FAILED: {path}: {e}")
            try:
                ddf_fail = read_parquet_with_dask(path, storage_options=storage_options)
                cols_fail = ddf_fail.columns.tolist()
                loc_fail = find_pickup_location_col(cols_fail)
                if loc_fail is None:
                    print(f"        [DEBUG] No location col. Columns: {cols_fail}")
                    print(f"        [DEBUG] Types: {[type(c).__name__ for c in cols_fail]}")
            except Exception:
                pass

    print(f"\n   Total cleaned rows across {len(files)} files: {total_cleaned_rows}")
    print("\n=== Demo finished. ===\n")


if __name__ == "__main__":
    main()