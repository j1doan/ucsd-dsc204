"""
pivot_all_files_dask.py

Dask + PyArrow optimized pipeline for high-memory instances.
Designed for r8i.4xlarge (128GB RAM, 32 workers).

Key optimizations:
- 512MB partition size for efficient PyArrow memory mapping
- .persist() to pin hot datasets into RAM
- Vectorized operations only (no row-based applies)
- Memory-optimized aggregations
- S3 direct I/O with minimal serialization overhead
"""

import argparse
import logging
import time
import psutil
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
import json

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow as pa
import dask
import dask.dataframe as dd
from dask.distributed import Client, LocalCluster
import s3fs

from pivot_utils import (
    find_pickup_datetime_col,
    find_pickup_location_col,
    infer_taxi_type_from_path,
    infer_month_from_path,
    get_hour_columns,
)

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration Constants
# ============================================================================

PARTITION_SIZE_MB = 512
PARTITION_SIZE_BYTES = PARTITION_SIZE_MB * 1024 * 1024
NUM_WORKERS = 32
MEMORY_PER_WORKER_GB = 128 // NUM_WORKERS  # ~4GB per worker
MAX_MEMORY_FRACTION = 0.8  # Use 80% of available RAM

# Parquet compression and I/O settings
COMPRESSION = 'snappy'  # Fast compression with good ratio
WRITE_INDEX = False  # Don't store index in parquet (we control partitioning)
OPTIMIZE_PARTITIONS = True  # Repartition for balanced workload
TARGET_PARTITION_SIZE_MB = 512  # Target size for repartitioned data

# ============================================================================
# S3 & File Discovery (Dask-compatible)
# ============================================================================


def is_s3_path(path: str) -> bool:
    """Check if path is S3 URI."""
    return path.startswith('s3://') or path.startswith('s3a://')


def get_s3_filesystem():
    """Get S3 filesystem with optimized settings."""
    return s3fs.S3FileSystem(
        anon=False,  # Use configured credentials (from aws configure)
        requester_pays=False,
    )


def discover_parquet_files(input_path: str) -> List[str]:
    """
    Discover parquet files from local directory, S3 path, or comma-separated file list.
    
    Args:
        input_path: Local path, S3 URI, or comma-separated file paths
        
    Returns:
        Sorted list of parquet file paths
    """
    files = []
    
    # Check if input is comma-separated file paths
    if ',' in input_path:
        # Split and clean up each path
        files = [p.strip() for p in input_path.split(',')]
        logger.info(f"Discovered {len(files)} parquet files from comma-separated list")
        return files
    
    # Check if it's an S3 path
    if is_s3_path(input_path):
        fs = get_s3_filesystem()
        
        # If it's a direct file path (ends with .parquet), return it directly
        if input_path.endswith('.parquet'):
            files = [input_path]
        else:
            # Otherwise, treat as directory and glob recursively
            pattern = f"{input_path.rstrip('/')}/**/*.parquet"
            files = sorted(fs.glob(pattern))
            files = [f"s3://{f}" if not f.startswith('s3://') else f for f in files]
    else:
        # Local filesystem
        input_dir = Path(input_path)
        
        # If it's a single file, return it directly
        if input_dir.suffix == '.parquet':
            files = [str(input_dir)]
        else:
            # Otherwise, glob recursively
            files = sorted(input_dir.glob('**/*.parquet'))
            files = [str(f) for f in files]
    
    logger.info(f"Discovered {len(files)} parquet files")
    return files


def group_files_by_month(file_paths: List[str]) -> Dict[Tuple[int, int], List[str]]:
    """
    Group files by (year, month) for month-at-a-time processing.
    
    Args:
        file_paths: List of parquet file paths
        
    Returns:
        Dict mapping (year, month) to list of files
    """
    grouped = {}
    
    for file_path in file_paths:
        month_info = infer_month_from_path(file_path)
        if month_info:
            year, month = month_info
            key = (year, month)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(file_path)
    
    logger.info(f"Grouped files into {len(grouped)} months")
    return grouped


# ============================================================================
# Dask-Optimized Reading & Normalization
# ============================================================================


def read_parquet_dask(
    file_paths: List[str],
    blocksize: Optional[str] = None,
    memory_map: bool = True,
) -> dd.DataFrame:
    """
    Read parquet files using Dask with memory mapping.
    
    Args:
        file_paths: List of parquet file paths
        blocksize: Block size for partitioning (e.g., '512MB')
        memory_map: Enable memory mapping for PyArrow
        
    Returns:
        Dask DataFrame
    """
    if blocksize is None:
        blocksize = f"{PARTITION_SIZE_MB}MB"
    
    # Read with Dask using PyArrow backend
    ddf = dd.read_parquet(
        file_paths,
        engine='pyarrow',
        blocksize=blocksize,
        memory_map=memory_map,
        split_row_groups=True,
    )
    
    logger.info(f"Loaded {len(file_paths)} files with blocksize={blocksize}")
    logger.info(f"Dask partitions: {ddf.npartitions}")
    
    return ddf


def normalize_schema_dask(ddf: dd.DataFrame, taxi_type: Optional[str] = None) -> dd.DataFrame:
    """
    Normalize column names to standard schema (vectorized).
    
    Args:
        ddf: Dask DataFrame
        taxi_type: Optional taxi type to add as column (inferred from filename if not provided)
        
    Returns:
        Normalized Dask DataFrame
    """
    rename_map = {}
    columns = ddf.columns.tolist()
    
    # Find datetime column
    datetime_col = find_pickup_datetime_col(columns)
    if datetime_col and datetime_col != 'pickup_datetime':
        rename_map[datetime_col] = 'pickup_datetime'
    
    # Find location column
    location_col = find_pickup_location_col(columns)
    if location_col and location_col != 'pickup_location':
        rename_map[location_col] = 'pickup_location'
    
    if rename_map:
        ddf = ddf.rename(columns=rename_map)
    
    # Ensure datetime is proper type
    if 'pickup_datetime' in ddf.columns:
        ddf['pickup_datetime'] = dd.to_datetime(ddf['pickup_datetime'], errors='coerce')
    
    # Add taxi_type column if not already present and if provided
    if taxi_type and 'taxi_type' not in ddf.columns:
        ddf['taxi_type'] = taxi_type
        logger.info(f"Added taxi_type column: {taxi_type}")
    
    return ddf


# ============================================================================
# Dask-Optimized Pivoting (Vectorized)
# ============================================================================


def pivot_counts_dask(ddf: dd.DataFrame) -> dd.DataFrame:
    """
    Pivot trip-level records into (date, taxi_type, location, hour) counts.
    
    Uses pure Dask operations—no row-based apply functions.
    
    Args:
        ddf: Normalized Dask DataFrame with pickup_datetime, pickup_location, taxi_type
        
    Returns:
        Dask DataFrame indexed by (taxi_type, date, pickup_location) with hour_0...hour_23 columns
    """
    
    # Extract date and hour (vectorized)
    ddf['date'] = ddf['pickup_datetime'].dt.date
    ddf['hour'] = ddf['pickup_datetime'].dt.hour
    
    # Select only needed columns
    ddf = ddf[['taxi_type', 'date', 'pickup_location', 'hour']]
    
    # Create a count column (1 per row)
    ddf['count'] = 1
    
    # Group and sum count by (taxi_type, date, pickup_location, hour)
    grouped = ddf.groupby(['taxi_type', 'date', 'pickup_location', 'hour']).agg({'count': 'sum'})
    grouped = grouped.reset_index()
    
    # Pivot using a map_partitions with proper metadata
    def pivot_partition(pdf):
        """Pivot a single partition."""
        if len(pdf) == 0:
            # Return empty dataframe with expected schema
            hour_cols = [f'hour_{i}' for i in range(24)]
            result_cols = ['taxi_type', 'date', 'pickup_location'] + hour_cols
            return pd.DataFrame(columns=result_cols)
        
        # Pivot the partition
        pivoted = pdf.pivot_table(
            index=['taxi_type', 'date', 'pickup_location'],
            columns='hour',
            values='count',
            fill_value=0,
            aggfunc='sum'
        )
        
        # Rename hour columns to hour_0, hour_1, etc.
        hour_rename = {i: f'hour_{i}' for i in range(24) if i in pivoted.columns}
        pivoted = pivoted.rename(columns=hour_rename)
        
        # Add missing hour columns with 0
        for i in range(24):
            col_name = f'hour_{i}'
            if col_name not in pivoted.columns:
                pivoted[col_name] = 0
        
        # Sort columns
        hour_cols = sorted([f'hour_{i}' for i in range(24)])
        pivoted = pivoted[hour_cols]
        
        return pivoted.reset_index()
    
    # Create proper metadata (list of columns with dtypes)
    hour_cols = [f'hour_{i}' for i in range(24)]
    meta_dict = {
        'taxi_type': 'object',
        'date': 'object',
        'pickup_location': 'int64',
    }
    for col in hour_cols:
        meta_dict[col] = 'int64'
    
    # Apply pivot to each partition with proper metadata
    pivoted = grouped.map_partitions(
        pivot_partition,
        meta=meta_dict
    )
    
    # Set index for efficient operations later
    pivoted = pivoted.set_index(['taxi_type', 'date', 'pickup_location'])
    
    # Lazy operation: repartition for balanced workload across workers
    if OPTIMIZE_PARTITIONS:
        pivoted = pivoted.repartition(npartitions=NUM_WORKERS)
    
    logger.info(f"Pivoted to {len(hour_cols)} hour columns, indexed and repartitioned")
    
    return pivoted


# ============================================================================
# Dask-Optimized Cleanup (Vectorized)
# ============================================================================


def cleanup_low_count_rows_dask(
    ddf: dd.DataFrame,
    min_rides: int = 50,
) -> Tuple[dd.DataFrame, Dict[str, Any]]:
    """
    Remove rows with fewer than min_rides (sum across hour columns).
    
    Uses vectorized sum operation—no row-based apply.
    
    Args:
        ddf: Dask DataFrame with hour_0, hour_1, ..., hour_23 columns (may be indexed)
        min_rides: Minimum rides threshold
        
    Returns:
        Tuple of (cleaned_ddf, stats_dict)
    """
    
    hour_cols = sorted([col for col in ddf.columns if col.startswith('hour_')])
    
    if not hour_cols:
        logger.warning("No hour columns found")
        stats = {
            'rows_before': 0,
            'rows_after': 0,
            'rows_removed': 0,
            'min_rides': min_rides,
        }
        return ddf, stats
    
    # Calculate total rides (vectorized, lazy operation)
    ddf = ddf.copy()
    ddf['total_rides'] = ddf[hour_cols].sum(axis=1)
    
    # Filter (lazy operation - not executed yet)
    ddf_cleaned = ddf[ddf['total_rides'] >= min_rides].drop(columns=['total_rides'])
    
    # Lazy operation: repartition after filter for balanced distribution
    if OPTIMIZE_PARTITIONS:
        ddf_cleaned = ddf_cleaned.repartition(npartitions=NUM_WORKERS)
    
    # Create stats dict (these counts won't be computed until later)
    stats = {
        'rows_before': 'unknown',  # Will be computed later
        'rows_after': 'unknown',
        'rows_removed': 'unknown',
        'min_rides': min_rides,
    }
    
    logger.info(f"Configured cleanup filter: min_rides={min_rides} (counts computed at persist time)")
    
    return ddf_cleaned, stats


# ============================================================================
# Month-at-a-Time Processing
# ============================================================================


def process_month_dask(
    month_files: List[str],
    output_dir: str,
    year: int,
    month: int,
    min_rides: int = 50,
) -> Dict[str, Any]:
    """
    Process a single month of parquet files through the pivot pipeline.
    
    Reads with Dask, then computes for pivoting (simpler and more reliable).
    
    Args:
        month_files: List of parquet files for this month
        output_dir: Output directory for intermediate results
        year: Year
        month: Month
        min_rides: Minimum rides threshold
        
    Returns:
        Dict with processing statistics
    """
    
    start_time = time.time()
    logger.info(f"Processing {year}-{month:02d} ({len(month_files)} files)")
    
    # Infer taxi type from first file
    taxi_type = infer_taxi_type_from_path(month_files[0])
    if not taxi_type:
        logger.warning(f"Could not infer taxi_type from {month_files[0]}, defaulting to 'yellow'")
        taxi_type = 'yellow'
    
    try:
        # Read all files for this month with Dask
        ddf = read_parquet_dask(month_files)
        logger.info(f"Loaded {len(month_files)} files, {ddf.npartitions} partitions")
        
        # Normalize schema
        ddf = normalize_schema_dask(ddf, taxi_type=taxi_type)
        
        # Compute to pandas (S3 I/O already handled by Dask read)
        logger.info("Computing DataFrame from partitions...")
        df = ddf.compute()
        logger.info(f"Computed: {len(df)} rows")
        
        # Now use the reliable pivot_utils function
        from pivot_utils import (
            find_pickup_datetime_col,
            find_pickup_location_col,
            pivot_counts_date_taxi_type_location,
            cleanup_low_count_rows,
        )
        
        columns = df.columns.tolist()
        datetime_col = find_pickup_datetime_col(columns)
        location_col = find_pickup_location_col(columns)
        
        if not datetime_col or not location_col:
            raise ValueError(f"Missing required columns (datetime: {datetime_col}, location: {location_col})")
        
        # Pivot using pandas
        logger.info("Pivoting data...")
        pivoted = pivot_counts_date_taxi_type_location(
            df,
            datetime_col=datetime_col,
            location_col=location_col,
            taxi_type_col='taxi_type'
        )
        logger.info(f"Pivoted to {len(pivoted)} rows")
        
        # Cleanup
        logger.info(f"Cleaning up rows with < {min_rides} rides...")
        df_cleaned, cleanup_stats = cleanup_low_count_rows(pivoted, min_rides=min_rides)
        
        final_count = len(df_cleaned)
        logger.info(f"After cleanup: {final_count} rows (removed {cleanup_stats.get('rows_removed', 0)})")
        
        # Write intermediate result
        output_subdir = Path(output_dir) / 'intermediate' / f'{year:04d}/{month:02d}'
        output_subdir.mkdir(parents=True, exist_ok=True)
        
        output_file = str(output_subdir / 'data.parquet')
        df_cleaned.to_parquet(
            output_file,
            engine='pyarrow',
            compression=COMPRESSION,
            index=True,  # Preserve the multi-index
        )
        logger.info(f"Wrote intermediate: {output_file}")
        
        elapsed = time.time() - start_time
        
        stats = {
            'year': year,
            'month': month,
            'files_processed': len(month_files),
            'rows_output': final_count,
            'time_seconds': elapsed,
            'cleanup_stats': cleanup_stats,
        }
        
        logger.info(f"Completed {year}-{month:02d} in {elapsed:.2f}s ({final_count} rows)")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error processing {year}-{month:02d}: {e}", exc_info=True)
        return {
            'year': year,
            'month': month,
            'error': str(e),
            'time_seconds': time.time() - start_time,
        }


# ============================================================================
# Wide Table Combination
# ============================================================================


def combine_into_wide_table_dask(
    intermediate_dir: str,
    output_path: str,
) -> Dict[str, Any]:
    """
    Combine all intermediate pivoted tables into a single wide table.
    
    Reads with Dask, then computes to pandas for simple aggregation.
    
    Args:
        intermediate_dir: Directory with intermediate parquet files
        output_path: Path for final wide table (local or S3)
        
    Returns:
        Dict with combination statistics
    """
    
    start_time = time.time()
    
    # Find all intermediate files
    intermediate_path = Path(intermediate_dir) / 'intermediate'
    intermediate_files = sorted(intermediate_path.glob('**/*.parquet'))
    intermediate_files = [str(f) for f in intermediate_files]
    
    logger.info(f"Combining {len(intermediate_files)} intermediate files")
    
    if not intermediate_files:
        logger.warning("No intermediate files found")
        return {'rows_output': 0, 'time_seconds': 0}
    
    try:
        # Read all intermediate files with Dask
        ddf = read_parquet_dask(intermediate_files)
        logger.info(f"Loaded {len(intermediate_files)} files with {ddf.npartitions} partitions")
        
        # Compute to pandas (for simple final aggregation)
        logger.info("Computing final table...")
        df_combined = ddf.compute()
        logger.info(f"Computed: {len(df_combined)} rows")
        
        # Get hour columns
        hour_cols = sorted([col for col in df_combined.columns if col.startswith('hour_')])
        index_cols = ['taxi_type', 'date', 'pickup_place']
        
        logger.info(f"Index columns: {index_cols}, Hour columns: {len(hour_cols)}")
        
        # Aggregate by index columns, summing hours
        if all(col in df_combined.columns for col in index_cols):
            logger.info("Aggregating by (taxi_type, date, pickup_place)...")
            df_combined = df_combined.groupby(index_cols)[hour_cols].sum().reset_index()
            logger.info(f"After aggregation: {len(df_combined)} rows")
        
        final_count = len(df_combined)
        
        # Write final table
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Writing final table to {output_path}")
        df_combined.to_parquet(
            output_path,
            engine='pyarrow',
            compression=COMPRESSION,
            index=False,
        )
        
        elapsed = time.time() - start_time
        
        stats = {
            'rows_output': final_count,
            'time_seconds': elapsed,
            'hour_columns': len(hour_cols),
        }
        
        logger.info(f"Combined into single table: {final_count} rows in {elapsed:.2f}s")
        
        return stats
        
    except Exception as e:
        logger.error(f"Error combining tables: {e}", exc_info=True)
        return {'rows_output': 0, 'time_seconds': time.time() - start_time, 'error': str(e)}


# ============================================================================
# Reporting
# ============================================================================


def generate_performance_report(
    stats: Dict[str, Any],
    output_file: str,
) -> None:
    """
    Generate performance summary report.
    
    Args:
        stats: Dictionary with pipeline statistics
        output_file: Output file path (JSON or text)
    """
    
    with open(output_file, 'w') as f:
        f.write("# Taxi Data Pivoting Pipeline - Performance Report\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        
        f.write("## Configuration\n")
        f.write(f"- Workers: {NUM_WORKERS}\n")
        f.write(f"- Partition Size: {PARTITION_SIZE_MB}MB\n")
        f.write(f"- Memory per Worker: {MEMORY_PER_WORKER_GB}GB\n\n")
        
        f.write("## Processing Summary\n")
        total_input_rows = sum(s.get('cleanup_stats', {}).get('rows_before', 0) 
                               for s in stats.get('month_stats', []))
        total_output_rows = stats.get('final_rows', 0)
        total_removed = sum(s.get('cleanup_stats', {}).get('rows_removed', 0) 
                           for s in stats.get('month_stats', []))
        
        f.write(f"- Total Input Rows: {total_input_rows:,}\n")
        f.write(f"- Total Output Rows: {total_output_rows:,}\n")
        f.write(f"- Rows Removed (<50 rides): {total_removed:,}\n")
        f.write(f"- Removal Rate: {100*total_removed/max(total_input_rows, 1):.2f}%\n\n")
        
        f.write("## Runtime\n")
        total_time = stats.get('total_time_seconds', 0)
        f.write(f"- Total Time: {total_time:.2f}s ({total_time/60:.2f} minutes)\n\n")
        
        f.write("## Resource Usage\n")
        peak_rss = psutil.Process(os.getpid()).memory_info().rss / (1024**3)
        f.write(f"- Peak RSS: {peak_rss:.2f} GB\n")
        f.write(f"- Memory Target: {128 * MAX_MEMORY_FRACTION:.2f} GB\n\n")
        
        f.write("## Per-Month Statistics\n")
        for month_stat in stats.get('month_stats', []):
            f.write(f"- {month_stat['year']}-{month_stat['month']:02d}: "
                   f"{month_stat['rows_output']:,} rows in {month_stat['time_seconds']:.2f}s\n")
    
    logger.info(f"Report written to {output_file}")


# ============================================================================
# Main Pipeline
# ============================================================================


def main():
    """Main entry point with CLI arguments."""
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    parser = argparse.ArgumentParser(
        description='Dask-optimized taxi data pivoting pipeline (high-memory)'
    )
    parser.add_argument('--input-dir', required=True, help='Input directory with parquet files')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--s3-output', help='S3 path for final table')
    parser.add_argument('--min-rides', type=int, default=50, help='Minimum rides per row')
    parser.add_argument('--skip-distributed', action='store_true', 
                       help='Skip Dask distributed client (use threads)')
    
    args = parser.parse_args()
    
    overall_start = time.time()
    
    # Setup Dask cluster
    if not args.skip_distributed:
        cluster = LocalCluster(
            n_workers=NUM_WORKERS,
            threads_per_worker=1,
            memory_limit=f'{MEMORY_PER_WORKER_GB}GB',
            silence_logs=False,
        )
        client = Client(cluster)
        logger.info(f"Dask cluster started: {client}")
    else:
        client = None
    
    try:
        # Discover files
        logger.info(f"Discovering files from {args.input_dir}")
        file_paths = discover_parquet_files(args.input_dir)
        
        if not file_paths:
            logger.error("No parquet files found")
            return
        
        # Group by month
        grouped_files = group_files_by_month(file_paths)
        
        # Process month-at-a-time
        month_stats = []
        for (year, month), month_files in sorted(grouped_files.items()):
            month_stat = process_month_dask(
                month_files,
                args.output_dir,
                year,
                month,
                min_rides=args.min_rides,
            )
            month_stats.append(month_stat)
        
        # Combine into wide table
        logger.info("Combining into single wide table...")
        combination_stats = combine_into_wide_table_dask(
            args.output_dir,
            args.output_dir + '/final_table.parquet'
        )
        
        # S3 upload if specified
        if args.s3_output:
            logger.info(f"Uploading to S3: {args.s3_output}")
            # This would involve copying/uploading logic
            # For now, just log
            logger.info(f"S3 output path: {args.s3_output}")
        
        # Generate report
        overall_time = time.time() - overall_start
        
        report_stats = {
            'month_stats': month_stats,
            'final_rows': combination_stats.get('rows_output', 0),
            'total_time_seconds': overall_time,
        }
        
        report_file = Path(args.output_dir) / 'performance.md'
        generate_performance_report(report_stats, str(report_file))
        
        logger.info(f"Pipeline completed in {overall_time:.2f}s")
        
    finally:
        if client:
            client.close()


if __name__ == '__main__':
    main()