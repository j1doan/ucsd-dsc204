"""
pivot_utils.py

Core utility functions for taxi data pivoting pipeline.
Handles column detection, month inference, pivoting, and data cleaning.
Uses dask and pyarrow for efficient large-file processing.
"""

import re
import os
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import logging
import dask.dataframe as dd
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# S3 and filesystem imports
try:
    import fsspec
    HAS_FSSPEC = True
except ImportError:
    HAS_FSSPEC = False
    logger.warning("fsspec not installed; S3 support will be limited")

try:
    import s3fs
    HAS_S3FS = True
except ImportError:
    HAS_S3FS = False
    logger.warning("s3fs not installed; S3 support requires fsspec")


# ============================================================================
# Column Detection Functions
# ============================================================================


def find_pickup_datetime_col(columns: list) -> Optional[str]:
    """
    Find the pickup datetime column name, handling common variants.
    
    Args:
        columns: List of column names
        
    Returns:
        Column name if found, None otherwise
        
    Examples:
        - 'pickup_datetime', 'tpep_pickup_datetime', 'pickupDatetime'
    """
    pattern = re.compile(r'.*pickup.*datetime.*', re.IGNORECASE)
    matches = [col for col in columns if pattern.match(col)]
    
    if matches:
        return matches[0]
    return None


def _normalize_col_for_match(col) -> Optional[str]:
    """Coerce column to str for comparison; return None if empty. Handles bytes/other types from Parquet."""
    if col is None:
        return None
    s = str(col).strip()
    return s if s else None


def find_pickup_location_col(columns: list) -> Optional[str]:
    """
    Find the pickup location column name, handling common variants.
    Prefers zone-ID columns (e.g. PULocationID); falls back to lat/lon for older schemas.
    Coerces column names to str so bytes or other Parquet/Dask types match correctly.
    
    Args:
        columns: List of column names (str, bytes, or other; will be coerced to str for matching)
        
    Returns:
        Column name if found (original from list, for use in df[col]), None otherwise
        
    Examples:
        - 'PULocationID', 'PUlocationID', 'pickup_location_id', 'pulocationid', 'pickup_zone'
        - 'pickup_latitude' (fallback for older NYC TLC files without zone ID)
    """
    # 1) Exact match for NYC TLC column names (case-insensitive) - preserve actual casing from file
    for col in columns:
        col_str = _normalize_col_for_match(col)
        if not col_str:
            continue
        normalized = col_str.lower().replace('_', '').replace(' ', '')
        if normalized == 'pulocationid':
            return col

    # 2) Other known zone/location names (include 'pickup_place' = normalized name used after schema normalization)
    known_zone = (
        'pickup_place', 'pickup_location_id', 'pickup_zone', 'pu_zone', 'start_location_id',
    )
    for col in columns:
        col_str = _normalize_col_for_match(col)
        if col_str and col_str.lower() in known_zone:
            return col

    # 3) Regex patterns for pickup location
    patterns = [
        r'.*pickup.*location.*',
        r'.*pulocationid.*',
        r'.*pu_location.*',
        r'.*pu\s*location.*',
    ]
    for pattern_str in patterns:
        pattern = re.compile(pattern_str, re.IGNORECASE)
        for col in columns:
            col_str = _normalize_col_for_match(col)
            if col_str and pattern.match(col_str):
                return col

    # 4) Fallback: older NYC TLC files (e.g. 2010 yellow) have pickup_latitude / pickup_longitude only
    fallback = ('pickup_latitude', 'pickup_longitude', 'pickup_lat', 'pickup_lon')
    for col in columns:
        col_str = _normalize_col_for_match(col)
        if col_str and col_str.lower() in fallback:
            return col

    return None


def infer_taxi_type_from_path(file_path: str) -> Optional[str]:
    """
    Infer taxi type from file path.
    
    Args:
        file_path: Path to parquet file
        
    Returns:
        Taxi type ('yellow', 'green', 'fhv', etc.) or None if not inferrable
        
    Examples:
        - '/path/yellow_tripdata_2023-01.parquet' → 'yellow'
        - 's3://bucket/green/data.parquet' → 'green'
        - 'fhv_tripdata.parquet' → 'fhv'
    """
    # Common taxi type names
    taxi_types = ['yellow', 'green', 'fhv', 'fhvhv']
    
    path_lower = file_path.lower()
    for taxi_type in taxi_types:
        if taxi_type in path_lower:
            return taxi_type
    
    return None


def infer_month_from_path(file_path: str) -> Optional[Tuple[int, int]]:
    """
    Infer (year, month) from file path.
    
    Args:
        file_path: Path to parquet file
        
    Returns:
        Tuple of (year, month) or None if not inferrable
        
    Examples:
        - 'yellow_tripdata_2023-01.parquet' → (2023, 1)
        - 's3://bucket/year=2023/month=05/data.parquet' → (2023, 5)
        - 'data_202312.parquet' → (2023, 12)
    """
    patterns = [
        # Match YYYY-MM pattern
        r'(\d{4})-(\d{2})',
        # Match year=YYYY/month=MM pattern
        r'year=(\d{4}).*month=(\d{2})',
        # Match YYYYMM pattern (e.g., 202301)
        r'(\d{4})(\d{2})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, file_path)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            if 1 <= month <= 12:
                return (year, month)
    
    return None


# ============================================================================
# Pivoting Functions
# ============================================================================


def pivot_counts_date_taxi_type_location(
    df,
    datetime_col: Optional[str] = None,
    location_col: Optional[str] = None,
    taxi_type_col: Optional[str] = None,
):
    """
    Pivot trip-level records into (date × taxi_type × pickup_place × hour) counts.
    
    Works with both pandas DataFrames and dask DataFrames.
    
    The resulting table will have:
    - Index: (taxi_type, date, pickup_place)
    - Columns: hour_0, hour_1, ..., hour_23 (counts for each hour)
    - Missing hour values are filled with 0
    
    Args:
        df: Pandas or Dask DataFrame with trip records
        datetime_col: Name of pickup datetime column (auto-detected if None)
        location_col: Name of pickup location column (auto-detected if None)
        taxi_type_col: Name of taxi type column (auto-detected if None)
        
    Returns:
        Pivoted pandas DataFrame indexed by (taxi_type, date, pickup_place)
        
    Raises:
        ValueError: If required columns cannot be found or inferred
    """
    
    # Auto-detect columns if not provided
    columns = df.columns.tolist()
    
    if datetime_col is None:
        datetime_col = find_pickup_datetime_col(columns)
    if location_col is None:
        location_col = find_pickup_location_col(columns)
    
    if datetime_col is None:
        raise ValueError("Could not find pickup datetime column")
    if location_col is None:
        raise ValueError("Could not find pickup location column")
    
    # If taxi_type_col not provided, require a taxi_type column
    if taxi_type_col is None:
        if 'taxi_type' in columns:
            taxi_type_col = 'taxi_type'
        else:
            raise ValueError("Could not find taxi type column")
    
    # Convert dask to pandas if needed
    is_dask = isinstance(df, dd.DataFrame)
    if is_dask:
        df = df.compute()
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Ensure datetime column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    
    # Extract date and hour
    df['date'] = df[datetime_col].dt.date
    df['hour'] = df[datetime_col].dt.hour
    
    # Ensure taxi_type_col exists
    if taxi_type_col not in df.columns:
        raise ValueError(f"Could not find taxi type column: {taxi_type_col}")
    
    # Group and count
    counts = df.groupby(
        [taxi_type_col, 'date', location_col, 'hour']
    ).size().reset_index(name='count')
    
    # Pivot: hours become columns
    pivoted = counts.pivot_table(
        index=[taxi_type_col, 'date', location_col],
        columns='hour',
        values='count',
        fill_value=0,
        aggfunc='sum'
    )
    
    # Ensure all hours 0-23 are present
    for hour in range(24):
        if hour not in pivoted.columns:
            pivoted[f'hour_{hour}'] = 0
        else:
            # Rename hour column to hour_X format
            pivoted = pivoted.rename(columns={hour: f'hour_{hour}'})
    
    # Ensure correct column naming
    pivoted.columns = [f'hour_{col}' if isinstance(col, int) else f'hour_{int(col.split("_")[1])}' 
                      if 'hour_' in str(col) else f'hour_{col}' 
                      for col in pivoted.columns]
    
    # Sort columns by hour number
    hour_cols = sorted([col for col in pivoted.columns if 'hour_' in col],
                      key=lambda x: int(x.split('_')[1]))
    pivoted = pivoted[hour_cols]
    
    # Rename index to standardized names
    pivoted.index.names = ['taxi_type', 'date', 'pickup_place']
    
    return pivoted


def cleanup_low_count_rows(
    df,
    min_rides: int = 50,
) -> Tuple:
    """
    Discard rows with fewer than min_rides (sum across hour columns).
    
    Works with both pandas DataFrames and dask DataFrames.
    
    Args:
        df: Pandas or Dask DataFrame with hour_0, hour_1, ..., hour_23 columns
        min_rides: Minimum rides per row to keep (default 50)
        
    Returns:
        Tuple of (cleaned_df, stats_dict)
        
    Stats dict contains:
        - 'rows_before': Number of rows before cleanup
        - 'rows_after': Number of rows after cleanup
        - 'rows_removed': Number of rows removed
        - 'min_rides': Threshold used
    """
    
    hour_cols = [col for col in df.columns if col.startswith('hour_')]
    
    if not hour_cols:
        logger.warning("No hour columns found in dataframe")
        is_dask = isinstance(df, dd.DataFrame)
        before = len(df) if not is_dask else df.compute().shape[0]
        return df, {
            'rows_before': before,
            'rows_after': before,
            'rows_removed': 0,
            'min_rides': min_rides,
        }
    
    # Convert dask to pandas if needed
    is_dask = isinstance(df, dd.DataFrame)
    if is_dask:
        df = df.compute()
    
    # Calculate total rides per row
    df = df.copy()
    df['total_rides'] = df[hour_cols].sum(axis=1)
    
    # Filter
    before = len(df)
    df_cleaned = df[df['total_rides'] >= min_rides].drop(columns=['total_rides'])
    after = len(df_cleaned)
    
    stats = {
        'rows_before': before,
        'rows_after': after,
        'rows_removed': before - after,
        'min_rides': min_rides,
    }
    
    logger.info(f"Cleanup: {before} → {after} rows (removed {before - after})")
    
    return df_cleaned, stats


# ============================================================================
# Schema Validation
# ============================================================================


def get_common_schema(df_list: list) -> Dict[str, str]:
    """
    Find common required columns across multiple dataframes or parquet files.
    
    Args:
        df_list: List of pandas/dask DataFrames or file paths to parquet files
        
    Returns:
        Dict mapping column name to column role (datetime, location, taxi_type)
    """
    required_cols = {
        'datetime': find_pickup_datetime_col,
        'location': find_pickup_location_col,
        'taxi_type': lambda cols: 'taxi_type' if 'taxi_type' in cols else None,
    }
    
    common_schema = {}
    
    for item in df_list:
        if isinstance(item, str):
            # It's a file path - read parquet schema
            try:
                pf = pq.ParquetFile(item)
                cols = pf.schema_arrow.names
            except Exception as e:
                logger.error(f"Could not read schema from {item}: {e}")
                continue
        elif isinstance(item, list) and item and isinstance(item[0], str):
            # It's a list of column names
            cols = item
        else:
            # It's a dataframe
            cols = item.columns.tolist()


        
        for key, detector in required_cols.items():
            col = detector(cols)
            if col and key not in common_schema:
                common_schema[key] = col
    
    return common_schema


def normalize_schema(df, common_schema: Dict[str, str]):
    """
    Normalize a dataframe to a common schema.
    
    Works with pandas and dask DataFrames.
    Ensures required columns exist and renames them to standard names.
    
    Args:
        df: Pandas or Dask DataFrame
        common_schema: Mapping of standard names to actual column names
        
    Returns:
        Dataframe with normalized column names
    """
    # Convert dask to pandas if needed for preprocessing
    is_dask = isinstance(df, dd.DataFrame)
    if is_dask:
        df_work = df.compute()
    else:
        df_work = df.copy()
    
    # Map actual columns to standard names
    rename_map = {}
    columns = df_work.columns.tolist()
    
    if 'datetime' in common_schema:
        datetime_col = find_pickup_datetime_col(columns)
        if datetime_col:
            rename_map[datetime_col] = 'pickup_datetime'
    
    if 'location' in common_schema:
        location_col = find_pickup_location_col(columns)
        if location_col:
            rename_map[location_col] = 'pickup_location'
    
    if 'taxi_type' in common_schema:
        if 'taxi_type' not in columns:
            raise ValueError("taxi_type column not found")
    
    if rename_map:
        if is_dask:
            df = df.rename(columns=rename_map)
        else:
            df = df_work.rename(columns=rename_map)
    
    return df


# ============================================================================
# Helper Functions
# ============================================================================


def get_hour_columns(df) -> list:
    """
    Get all hour_* columns from a dataframe, sorted by hour.
    
    Works with pandas and dask DataFrames.
    
    Args:
        df: Pandas or Dask DataFrame to inspect
        
    Returns:
        Sorted list of hour column names
    """
    hour_cols = [col for col in df.columns if col.startswith('hour_')]
    return sorted(hour_cols, key=lambda x: int(x.split('_')[1]))


def get_total_rides(df):
    """
    Calculate total rides (sum across hours) for each row.
    
    Works with pandas and dask DataFrames.
    
    Args:
        df: Pandas or Dask DataFrame with hour_* columns
        
    Returns:
        Series of total rides per row
    """
    hour_cols = get_hour_columns(df)
    return df[hour_cols].sum(axis=1)


def read_parquet_with_dask(file_path: str, **kwargs):
    """
    Read a parquet file using dask.
    
    Args:
        file_path: Path to parquet file (local or S3)
        **kwargs: Additional arguments to pass to dask.dataframe.read_parquet
        
    Returns:
        Dask DataFrame
    """
    return dd.read_parquet(file_path, **kwargs)


def read_parquet_schema(file_path: str) -> pa.Schema:
    """
    Read the schema of a parquet file using pyarrow.
    
    Args:
        file_path: Path to parquet file (local or S3)
        
    Returns:
        PyArrow Schema object
    """
    pf = pq.ParquetFile(file_path)
    return pf.schema_arrow


# ============================================================================
# Part 2: S3 & File Discovery Functions
# ============================================================================


def is_s3_path(path: str) -> bool:
    """
    Check if a path is an S3 URI.
    
    Args:
        path: Path or URI string
        
    Returns:
        True if path starts with 's3://', False otherwise
    """
    return path.lower().startswith('s3://')


def get_storage_options(path: str) -> Dict[str, Any]:
    """
    Get storage options for fsspec based on the path.
    
    For S3 paths, returns anonymous=True by default to support
    public S3 buckets. Can be customized based on credentials.
    
    Args:
        path: Local or S3 path
        
    Returns:
        Dictionary of storage options for fsspec
    """
    if is_s3_path(path):
        # Default to anonymous access for S3; can be overridden
        return {
            'anon': True,  # Anonymous access
            'requester_pays': False,
        }
    else:
        # Local filesystem doesn't need special options
        return {}


def get_filesystem(path: str):
    """
    Get the appropriate fsspec filesystem for the given path.
    
    Args:
        path: Local or S3 path
        
    Returns:
        fsspec filesystem object (LocalFileSystem for local paths, 
        S3FileSystem for S3 paths)
        
    Raises:
        ImportError: If fsspec is not installed
    """
    if not HAS_FSSPEC:
        raise ImportError(
            "fsspec is required for file operations. Install with: pip install fsspec s3fs"
        )
    
    if is_s3_path(path):
        if not HAS_S3FS:
            raise ImportError(
                "s3fs is required for S3 operations. Install with: pip install s3fs"
            )
        storage_opts = get_storage_options(path)
        return fsspec.filesystem('s3', **storage_opts)
    else:
        # Use local filesystem
        return fsspec.filesystem('file')


def discover_parquet_files(input_path: str) -> List[str]:
    """
    Discover all parquet files recursively in a directory or S3 path.
    
    Args:
        input_path: Directory path (local) or S3 URI (s3://bucket/prefix)
        
    Returns:
        Sorted list of absolute paths to .parquet files
        
    Raises:
        ValueError: If input path does not exist
        ImportError: If required packages are not installed
    """
    if not input_path:
        raise ValueError("input_path cannot be empty")
    
    parquet_files = []
    
    if is_s3_path(input_path):
        # S3 path discovery
        try:
            fs = get_filesystem(input_path)
        except ImportError as e:
            logger.error(f"Cannot discover S3 files: {e}")
            raise
        
        try:
            # Use glob to find all parquet files recursively
            # S3 path format: s3://bucket/prefix
            pattern = input_path.rstrip('/') + '/**/*.parquet'
            files = fs.glob(pattern)
            
            if not files:
                logger.warning(f"No parquet files found at {input_path}")
                return []
            
            # Convert to s3:// URIs if not already
            parquet_files = [
                f's3://{f}' if not f.startswith('s3://') else f
                for f in files
            ]
            
        except Exception as e:
            logger.error(f"Error discovering S3 files at {input_path}: {e}")
            raise ValueError(f"Could not discover files at {input_path}: {e}")
    
    else:
        # Local path discovery
        path = Path(input_path)
        
        if not path.exists():
            raise ValueError(f"Path does not exist: {input_path}")
        
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {input_path}")
        
        try:
            # Recursively find all .parquet files
            parquet_files = [
                str(f.resolve()) for f in path.rglob('*.parquet')
            ]
        except Exception as e:
            logger.error(f"Error discovering local files at {input_path}: {e}")
            raise ValueError(f"Could not discover files at {input_path}: {e}")
    
    if not parquet_files:
        logger.warning(f"No parquet files found at {input_path}")
        return []
    
    # Return sorted list
    parquet_files = sorted(parquet_files)
    logger.info(f"Discovered {len(parquet_files)} parquet files at {input_path}")
    
    return parquet_files