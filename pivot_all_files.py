"""
pivot_all_files.py

Main pipeline orchestration for taxi data pivoting.
Processes NYC TLC taxi data month-at-a-time with dask and pyarrow.
"""

import argparse
import logging
import json
import time
import os
import shutil
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
from multiprocessing import Pool
import psutil

import pandas as pd
import dask.dataframe as dd
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm

import pivot_utils as pu
import partition_optimization as popt

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def _process_single_file_star(args: Tuple[Any, ...]) -> Dict[str, Any]:
    return process_single_file(*args)


def process_single_file(
    file_path: str,
    output_dir: str,
    min_rides: int = 50,
    partition_size: Optional[int] = None,
    taxi_type: Optional[str] = None,
    common_schema: Optional[Dict[str, str]] = None,
    storage_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Process a single parquet file through the pivot pipeline.
    
    Steps:
    1. Read Parquet file (local or S3)
    2. Infer expected month from path
    3. Normalize schema
    4. Aggregate by (date, taxi_type, pickup_place, hour)
    5. Pivot to wide format (hour_0...hour_23)
    6. Cleanup rows with < min_rides
    7. Write intermediate parquet file
    8. Count and drop month-mismatch rows
    9. Drop rows with invalid datetime
    
    Args:
        file_path: Path to input parquet file (local or S3)
        output_dir: Directory for intermediate output files
        min_rides: Minimum rides per row threshold (default 50)
        partition_size: Optional blocksize for dask reading
        taxi_type: Override taxi type detection from path
        common_schema: Optional schema mapping for normalization
        storage_options: Optional storage options for S3 access
        
    Returns:
        Dict with results:
            - 'file_path': Input file path
            - 'output_path': Path to written intermediate file
            - 'input_rows': Rows read from input
            - 'output_rows': Rows in pivoted output
            - 'removed_rows': Rows removed due to min_rides filtering
            - 'month_mismatch_rows': Rows where row month != file month (dropped)
            - 'parse_fail_rows': Rows dropped due to invalid datetime parsing
            - 'month_mismatch_files': Count of files with mismatches
            - 'expected_month': (year, month) from path
            - 'processing_time_sec': Elapsed time
            - 'success': Whether processing succeeded
            - 'error': Error message if failed
    """
    
    result = {
        'file_path': file_path,
        'success': False,
        'error': None,
    }
    
    start_time = time.time()
    
    try:
        logger.info(f"Processing file: {file_path}")
        
        # Create output directory if needed
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Infer expected month from path
        expected_month = pu.infer_month_from_path(file_path)
        if expected_month is None:
            logger.warning(f"Could not infer month from path: {file_path}")
            expected_month = (None, None)
        
        result['expected_month'] = expected_month
        
        # Infer taxi type if not provided
        if taxi_type is None:
            inferred_taxi = pu.infer_taxi_type_from_path(file_path)
            if inferred_taxi is None:
                logger.warning(f"Could not infer taxi type from path: {file_path}")
                inferred_taxi = 'unknown'
        else:
            inferred_taxi = taxi_type
        
        result['taxi_type'] = inferred_taxi
        
        # Read parquet file using dask
        read_kwargs: Dict[str, Any] = {}
        if partition_size is not None:
            read_kwargs['blocksize'] = partition_size
        if storage_options:
            read_kwargs['storage_options'] = storage_options
        
        logger.info(f"Reading parquet file: {file_path}")
        ddf = pu.read_parquet_with_dask(file_path, **read_kwargs)
        
        # Convert to pandas for processing
        df = ddf.compute()
        input_rows = len(df)
        result['input_rows'] = input_rows
        logger.info(f"Read {input_rows} rows from {file_path}")

        # Normalize schema if available
        if common_schema:
            df = pu.normalize_schema(df, common_schema)
        
        # Detect columns (prefer standardized names if present)
        columns = df.columns.tolist()
        datetime_col = 'pickup_datetime' if 'pickup_datetime' in columns else pu.find_pickup_datetime_col(columns)
        location_col = 'pickup_location' if 'pickup_location' in columns else pu.find_pickup_location_col(columns)
        
        if datetime_col is None:
            raise ValueError("Could not find pickup datetime column")
        if location_col is None:
            raise ValueError("Could not find pickup location column")
        
        # Add taxi_type column if not present
        if 'taxi_type' not in df.columns:
            df['taxi_type'] = inferred_taxi
        
        # Convert datetime column to ensure it's datetime type
        if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
            df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')

        # Drop rows with invalid datetime
        parse_fail_rows = int(df[datetime_col].isna().sum())
        if parse_fail_rows:
            logger.warning(f"Dropping {parse_fail_rows} rows with invalid datetime")
            df = df[df[datetime_col].notna()]
        result['parse_fail_rows'] = parse_fail_rows
        
        # Extract date for month-mismatch detection
        df['_row_date'] = df[datetime_col].dt.date
        df['_row_month'] = df[datetime_col].dt.month
        df['_row_year'] = df[datetime_col].dt.year
        
        # Count month mismatches and drop them
        month_mismatch_count = 0
        if expected_month[0] is not None and expected_month[1] is not None:
            mismatches = df[
                (df['_row_year'] != expected_month[0]) |
                (df['_row_month'] != expected_month[1])
            ]
            month_mismatch_count = len(mismatches)
            if month_mismatch_count > 0:
                logger.warning(
                    f"Found {month_mismatch_count} rows with month mismatch "
                    f"(expected {expected_month[0]}-{expected_month[1]:02d})"
                )
                # Drop mismatched rows from further processing
                df = df.drop(mismatches.index)
        
        result['month_mismatch_rows'] = month_mismatch_count
        
        # Pivot the data
        logger.info("Pivoting data...")
        pivoted = pu.pivot_counts_date_taxi_type_location(
            df,
            datetime_col=datetime_col,
            location_col=location_col,
            taxi_type_col='taxi_type'
        )
        logger.info(f"Pivoted to {len(pivoted)} rows with {len(pivoted.columns)} columns")
        
        # Cleanup rows with low counts
        logger.info(f"Cleaning up rows with < {min_rides} rides...")
        df_cleaned, cleanup_stats = pu.cleanup_low_count_rows(pivoted, min_rides=min_rides)
        
        result['output_rows'] = len(df_cleaned)
        result['removed_rows'] = cleanup_stats['rows_removed']
        
        logger.info(
            f"After cleanup: {cleanup_stats['rows_after']} rows "
            f"(removed {cleanup_stats['rows_removed']})"
        )
        
        # Generate output filename
        file_stem = Path(file_path).stem
        output_path = str(Path(output_dir) / f"{file_stem}_pivoted.parquet")
        
        # Write to parquet
        logger.info(f"Writing output to {output_path}")
        write_kwargs: Dict[str, Any] = {'engine': 'pyarrow'}
        if storage_options and pu.is_s3_path(output_path):
            write_kwargs['storage_options'] = storage_options
        df_cleaned.to_parquet(output_path, **write_kwargs)
        
        result['output_path'] = output_path
        result['success'] = True
        
    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}", exc_info=True)
        result['error'] = str(e)
    
    finally:
        result['processing_time_sec'] = time.time() - start_time
    
    return result


def combine_into_wide_table(
    intermediate_dir: str,
    output_path: str,
    storage_options: Optional[Dict[str, Any]] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    Combine all intermediate pivoted tables into a single wide table.
    
    Steps:
    1. Discover all intermediate parquet files
    2. Read each intermediate file
    3. Aggregate by (taxi_type, date, pickup_place) - sum hour columns
    4. Store as single parquet file (or partitioned)
    5. Return row counts and statistics
    
    Args:
        intermediate_dir: Directory containing intermediate parquet files
        output_path: Path for final wide table (local or S3)
        storage_options: Optional storage options for S3 access
        
    Returns:
        Tuple of (output_row_count, stats_dict)
        
    Stats dict contains:
        - 'output_rows': Number of rows in final table
        - 'output_path': Path where final table was stored
        - 'num_intermediate_files': Number of intermediate files combined
        - 'num_partitions': Number of partitions in input (if partitioned)
        - 'total_input_rows': Total rows from all intermediate files
        - 'num_hour_columns': Number of hour columns (should be 24)
        - 'output_columns': List of output column names
    """
    
    logger.info(f"Combining intermediate files from {intermediate_dir}")
    
    # Discover intermediate files
    parquet_files: List[str] = []
    if pu.is_s3_path(intermediate_dir):
        fs = pu.get_filesystem(intermediate_dir)
        pattern = intermediate_dir.rstrip('/') + '/*_pivoted.parquet'
        parquet_files = [
            f's3://{f}' if not f.startswith('s3://') else f
            for f in fs.glob(pattern)
        ]
        parquet_files = sorted(parquet_files)
    else:
        intermediate_path = Path(intermediate_dir)
        parquet_files = sorted(str(p) for p in intermediate_path.glob('*_pivoted.parquet'))
    
    if not parquet_files:
        logger.warning(f"No intermediate parquet files found in {intermediate_dir}")
        return 0, {'error': 'No intermediate files found'}
    
    logger.info(f"Found {len(parquet_files)} intermediate files")
    
    try:
        # Read all intermediate files using dask
        ddf_list = []
        total_input_rows = 0
        
        for pf in parquet_files:
            logger.info(f"Reading intermediate file: {pf}")
            if storage_options and pu.is_s3_path(pf):
                ddf = dd.read_parquet(str(pf), storage_options=storage_options)
            else:
                ddf = dd.read_parquet(str(pf))
            
            # Get row count
            n_rows = len(ddf)
            total_input_rows += n_rows
            logger.info(f"  {n_rows} rows")
            
            ddf_list.append(ddf)
        
        # Concatenate all dataframes
        logger.info("Concatenating all intermediate dataframes...")
        ddf_combined = dd.concat(ddf_list, axis=0)
        
        # Reset index to make (taxi_type, date, pickup_place) regular columns
        ddf_combined = ddf_combined.reset_index()
        
        # Get hour columns
        hour_cols = [col for col in ddf_combined.columns if col.startswith('hour_')]
        hour_cols = sorted(hour_cols, key=lambda x: int(x.split('_')[1]))
        
        logger.info(f"Aggregating by (taxi_type, date, pickup_place) with {len(hour_cols)} hour columns...")
        
        # Aggregate: group by (taxi_type, date, pickup_place) and sum hour columns
        ddf_wide = ddf_combined.groupby(['taxi_type', 'date', 'pickup_place'])[hour_cols].sum()
        
        # Convert to pandas for final operations
        df_wide = ddf_wide.compute()
        
        # Reset index back to multiindex
        df_wide = df_wide.reset_index()
        df_wide = df_wide.set_index(['taxi_type', 'date', 'pickup_place'])
        
        output_rows = len(df_wide)
        logger.info(f"Final wide table: {output_rows} rows with {len(df_wide.columns)} columns")
        
        # Create output directory if needed
        if not pu.is_s3_path(output_path):
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write to parquet
        logger.info(f"Writing wide table to {output_path}")
        write_kwargs: Dict[str, Any] = {'engine': 'pyarrow'}
        if storage_options and pu.is_s3_path(output_path):
            write_kwargs['storage_options'] = storage_options
        df_wide.to_parquet(output_path, **write_kwargs)
        
        stats = {
            'output_rows': output_rows,
            'output_path': output_path,
            'num_intermediate_files': len(parquet_files),
            'total_input_rows': total_input_rows,
            'num_hour_columns': len(hour_cols),
            'output_columns': df_wide.columns.tolist(),
        }
        
        logger.info(f"Successfully created wide table: {output_rows} rows")
        
        return output_rows, stats
        
    except Exception as e:
        logger.error(f"Error combining intermediate files: {e}", exc_info=True)
        return 0, {'error': str(e)}


def generate_report(
    stats: Dict[str, Any],
    output_file: str,
) -> None:
    """
    Generate performance and processing report.
    
    Outputs a small .tex report by default (or JSON if output_file ends with .json).
    Includes:
    - Total runtime (wall-clock)
    - Peak memory usage
    - Input/output row counts
    - Breakdown of discarded rows
    - Month-mismatch statistics
    
    Args:
        stats: Dictionary with pipeline statistics
        output_file: Target file for report (.tex or .json)
    """
    logger.info(f"Generating report to {output_file}")
    
    try:
        # Create output directory if needed
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # Add system info
        process = psutil.Process(os.getpid())
        stats['process_info'] = {
            'memory_rss_mb': process.memory_info().rss / (1024**2),
            'peak_memory_mb': stats.get('peak_memory_mb', 'unknown'),
        }
        
        if output_file.lower().endswith('.json'):
            with open(output_file, 'w') as f:
                json.dump(stats, f, indent=2, default=str)
        else:
            total_runtime = stats.get('total_runtime_sec', 'N/A')
            total_input = stats.get('total_input_rows', 'N/A')
            total_output = stats.get('wide_table_rows', stats.get('total_output_rows', 'N/A'))
            total_discarded = stats.get('total_discarded_rows', 'N/A')
            total_discarded_raw = stats.get('total_discarded_raw_rows', 'N/A')
            total_discarded_pivot = stats.get('total_discarded_pivot_rows', 'N/A')
            total_mismatches = stats.get('total_month_mismatches', 'N/A')
            peak_memory = stats.get('peak_memory_mb', 'N/A')
            parse_fails = stats.get('total_parse_fail_rows', 0)
            low_count = stats.get('total_removed_rows', 0)
            
            tex_lines = [
                r"\section*{Pipeline Report}",
                r"\begin{itemize}",
                f"  \\item Total runtime (sec): {total_runtime}",
                f"  \\item Peak memory (MB): {peak_memory}",
                f"  \\item Total input rows: {total_input}",
                f"  \\item Total output rows: {total_output}",
                f"  \\item Total discarded rows: {total_discarded}",
                f"  \\item Discarded raw rows: {total_discarded_raw}",
                f"  \\item Discarded pivot rows: {total_discarded_pivot}",
                f"  \\item Month-mismatch rows: {total_mismatches}",
                f"  \\item Parse failures (datetime): {parse_fails}",
                f"  \\item Low-count rows removed: {low_count}",
                r"\end{itemize}",
            ]
            
            with open(output_file, 'w') as f:
                f.write("\n".join(tex_lines))
        
        logger.info(f"Report written to {output_file}")
        
        # Also print summary
        print("\n" + "="*60)
        print("PIPELINE SUMMARY REPORT")
        print("="*60)
        total_runtime = stats.get('total_runtime_sec', None)
        if isinstance(total_runtime, (int, float)):
            runtime_str = f"{total_runtime:.2f}s"
        else:
            runtime_str = "N/A"
        print(f"Total runtime: {runtime_str}")
        print(f"Input rows: {stats.get('total_input_rows', 'N/A')}")
        print(f"Output rows: {stats.get('wide_table_rows', stats.get('total_output_rows', 'N/A'))}")
        print(f"Discarded raw rows: {stats.get('total_discarded_raw_rows', 'N/A')}")
        print(f"Discarded pivot rows: {stats.get('total_discarded_pivot_rows', 'N/A')}")
        print(f"Month-mismatch rows: {stats.get('total_month_mismatches', 'N/A')}")
        print("="*60 + "\n")
        
    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)


def group_files_by_month(file_list: List[str]) -> Dict[Tuple[int, int], List[str]]:
    """
    Group discovered files by (year, month) from their paths.
    
    Args:
        file_list: List of file paths
        
    Returns:
        Dict mapping (year, month) to list of file paths
    """
    grouped = defaultdict(list)
    
    for file_path in file_list:
        month_info = pu.infer_month_from_path(file_path)
        if month_info is not None:
            grouped[month_info].append(file_path)
        else:
            # Group files with unknown month under (None, None)
            grouped[(None, None)].append(file_path)
    
    return grouped


def process_month_files(
    files: List[str],
    output_dir: str,
    min_rides: int = 50,
    partition_size: Optional[int] = None,
    num_workers: int = 1,
    common_schema: Optional[Dict[str, str]] = None,
    storage_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Process all files for a specific month.
    
    Uses multiprocessing to parallelize across files within a month.
    
    Args:
        files: List of file paths for this month
        output_dir: Directory for intermediate outputs
        min_rides: Minimum rides threshold
        partition_size: Optional blocksize for dask
        num_workers: Number of parallel workers
        common_schema: Optional schema mapping for normalization
        storage_options: Optional storage options for S3 access
        
    Returns:
        Dict with aggregated stats for the month
    """
    
    logger.info(f"Processing {len(files)} files for this month with {num_workers} workers...")
    
    month_stats = {
        'total_input_rows': 0,
        'total_output_rows': 0,
        'total_removed_rows': 0,
        'total_month_mismatches': 0,
        'total_parse_fail_rows': 0,
        'file_results': [],
        'errors': 0,
    }
    
    # Prepare arguments for multiprocessing
    task_args = [
        (f, output_dir, min_rides, partition_size, None, common_schema, storage_options)
        for f in files
    ]
    
    # Process files in parallel
    if num_workers > 1:
        with Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_process_single_file_star, task_args),
                    total=len(task_args),
                    desc="Processing files",
                )
            )
    else:
        # Sequential processing
        results = [
            process_single_file(*args)
            for args in tqdm(task_args, desc="Processing files")
        ]
    
    # Aggregate results
    for result in results:
        month_stats['file_results'].append(result)
        
        if result['success']:
            month_stats['total_input_rows'] += result.get('input_rows', 0)
            month_stats['total_output_rows'] += result.get('output_rows', 0)
            month_stats['total_removed_rows'] += result.get('removed_rows', 0)
            month_stats['total_month_mismatches'] += result.get('month_mismatch_rows', 0)
            month_stats['total_parse_fail_rows'] += result.get('parse_fail_rows', 0)
        else:
            month_stats['errors'] += 1
            logger.error(f"Failed to process {result['file_path']}: {result['error']}")
    
    return month_stats


def main():
    """
    Main entry point: orchestrate the full pipeline.
    
    Pipeline steps:
    1. Discover parquet files
    2. Check schema consistency
    3. Group files by month
    4. Optionally optimize partition size
    5. Process each month (files in parallel)
    6. Report month-mismatch statistics
    7. Combine into single wide table
    8. Upload to S3 (if specified)
    9. Generate final report
    """
    
    parser = argparse.ArgumentParser(
        description='Taxi data pivoting pipeline - process month-at-a-time'
    )
    parser.add_argument(
        '--input-dir', required=True,
        help='Input directory with parquet files (local or S3://...)'
    )
    parser.add_argument(
        '--output-dir', required=True,
        help='Output directory for pipeline (local or S3://...)'
    )
    parser.add_argument(
        '--s3-output', default=None,
        help='S3 path for final wide table (e.g., s3://bucket/path/table/)'
    )
    parser.add_argument(
        '--min-rides', type=int, default=50,
        help='Minimum rides per row to keep (default 50)'
    )
    parser.add_argument(
        '--workers', type=int, default=4,
        help='Number of parallel workers for file processing (default 4)'
    )
    parser.add_argument(
        '--partition-size', type=str, default=None,
        help='Partition size for dask (e.g., "200MB", "1GB")'
    )
    parser.add_argument(
        '--skip-partition-optimization', action='store_true',
        help='Skip partition size optimization'
    )
    parser.add_argument(
        '--keep-intermediate', action='store_true',
        help='Keep intermediate pivot files after combining'
    )
    parser.add_argument(
        '--report-file', type=str, default='pipeline_report.tex',
        help='Output file for pipeline report (.tex or .json)'
    )
    
    args = parser.parse_args()
    
    pipeline_start = time.time()
    
    try:
        # Initialize stats
        pipeline_stats = {
            'input_dir': args.input_dir,
            'output_dir': args.output_dir,
            's3_output': args.s3_output,
            'min_rides': args.min_rides,
            'workers': args.workers,
            'total_input_rows': 0,
            'total_output_rows': 0,
            'total_discarded_rows': 0,
            'total_discarded_raw_rows': 0,
            'total_discarded_pivot_rows': 0,
            'total_month_mismatches': 0,
            'total_parse_fail_rows': 0,
            'total_removed_rows': 0,
            'num_files_processed': 0,
            'num_errors': 0,
            'month_stats': {},
        }
        
        # Step 1: Discover files
        logger.info("Step 1: Discovering parquet files...")
        discover_start = time.time()
        file_list = pu.discover_parquet_files(args.input_dir)
        
        if not file_list:
            logger.error("No parquet files found!")
            return
        
        logger.info(f"Discovered {len(file_list)} parquet files")
        pipeline_stats['num_files_discovered'] = len(file_list)
        pipeline_stats['discovery_time_sec'] = time.time() - discover_start
        
        # Step 2: Check schema consistency
        logger.info("Step 2: Checking schema consistency...")
        common_schema: Optional[Dict[str, str]] = None
        try:
            common_schema = pu.get_common_schema(file_list)
            logger.info(f"Common schema detected: {common_schema}")
            pipeline_stats['common_schema'] = common_schema
        except Exception as e:
            logger.warning(f"Could not detect common schema: {e}")
        
        # Step 3: Group files by month
        logger.info("Step 3: Grouping files by month...")
        files_by_month = group_files_by_month(file_list)
        logger.info(f"Grouped into {len(files_by_month)} months")
        
        # Determine storage options for S3 (if needed)
        storage_options: Optional[Dict[str, Any]] = None
        if pu.is_s3_path(args.input_dir) or (args.s3_output and pu.is_s3_path(args.s3_output)):
            storage_options = pu.get_storage_options(args.input_dir if pu.is_s3_path(args.input_dir) else args.s3_output)

        # Optional: Parse/optimize partition size
        partition_size = None
        if args.partition_size:
            partition_size = popt.parse_size(args.partition_size)
            logger.info(f"Using partition size: {popt.format_size(partition_size)}")
        elif not args.skip_partition_optimization:
            sample_file = file_list[0]
            if pu.is_s3_path(sample_file):
                logger.warning(
                    "Skipping partition optimization for S3 inputs. "
                    "Provide --partition-size to set manually."
                )
            else:
                logger.info("Running partition size optimization on a sample file...")
                opt = popt.find_optimal_partition_size(sample_file, size_range=("50MB", "1GB"), num_sizes=5)
                partition_size = opt.get('optimal_size')
                if partition_size:
                    logger.info(f"Selected partition size: {popt.format_size(partition_size)}")
        
        # Step 4: Create intermediate output directory
        intermediate_dir = Path(args.output_dir) / 'intermediate'
        intermediate_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 5: Process each month (month-at-a-time)
        logger.info("Step 4: Processing files month-at-a-time...")
        
        month_items = sorted(files_by_month.items())
        for month_key, month_files in tqdm(month_items, desc="Processing months"):
            if month_key == (None, None):
                month_str = "unknown_month"
            else:
                month_str = f"{month_key[0]}-{month_key[1]:02d}"
            
            logger.info(f"\nProcessing month: {month_str} ({len(month_files)} files)")
            
            month_process_start = time.time()
            month_result = process_month_files(
                month_files,
                str(intermediate_dir),
                min_rides=args.min_rides,
                partition_size=partition_size,
                num_workers=args.workers,
                common_schema=common_schema,
                storage_options=storage_options,
            )
            month_result['processing_time_sec'] = time.time() - month_process_start
            
            pipeline_stats['month_stats'][month_str] = month_result
            pipeline_stats['total_input_rows'] += month_result['total_input_rows']
            pipeline_stats['total_output_rows'] += month_result['total_output_rows']
            pipeline_stats['total_removed_rows'] += month_result['total_removed_rows']
            pipeline_stats['total_month_mismatches'] += month_result['total_month_mismatches']
            pipeline_stats['total_parse_fail_rows'] += month_result['total_parse_fail_rows']
            pipeline_stats['num_files_processed'] += len(month_files)
            pipeline_stats['num_errors'] += month_result['errors']
            
            # Report month-mismatch
            if month_result['total_month_mismatches'] > 0:
                logger.warning(
                    f"Month {month_str}: {month_result['total_month_mismatches']} "
                    f"rows had date mismatches"
                )
        
        # Month mismatch breakdown
        pipeline_stats['month_mismatch_by_month'] = {
            month: stats.get('total_month_mismatches', 0)
            for month, stats in pipeline_stats['month_stats'].items()
        }
        
        # Step 6: Combine into wide table
        logger.info("\nStep 5: Combining intermediate files into single wide table...")
        combine_start = time.time()
        
        # Determine output path for wide table
        if args.s3_output:
            final_output_path = args.s3_output
        else:
            final_output_path = str(Path(args.output_dir) / 'final_wide_table.parquet')
        
        wide_rows, combine_stats = combine_into_wide_table(
            str(intermediate_dir),
            final_output_path,
            storage_options=storage_options,
        )
        
        pipeline_stats['wide_table_rows'] = wide_rows
        pipeline_stats['wide_table_path'] = final_output_path
        pipeline_stats['combine_time_sec'] = time.time() - combine_start
        pipeline_stats.update(combine_stats)
        
        # Step 7: Cleanup intermediates if not keeping
        if not args.keep_intermediate:
            logger.info("Cleaning up intermediate files...")
            shutil.rmtree(intermediate_dir, ignore_errors=True)
        
        # Step 8: Generate report
        logger.info("\nStep 6: Generating pipeline report...")
        pipeline_stats['total_runtime_sec'] = time.time() - pipeline_start
        pipeline_stats['total_discarded_raw_rows'] = (
            pipeline_stats['total_month_mismatches']
            + pipeline_stats['total_parse_fail_rows']
        )
        pipeline_stats['total_discarded_pivot_rows'] = pipeline_stats['total_removed_rows']
        pipeline_stats['total_discarded_rows'] = (
            pipeline_stats['total_removed_rows']
            + pipeline_stats['total_month_mismatches']
            + pipeline_stats['total_parse_fail_rows']
        )
        
        # Add memory info
        process = psutil.Process(os.getpid())
        pipeline_stats['peak_memory_mb'] = process.memory_info().rss / (1024**2)
        
        generate_report(pipeline_stats, args.report_file)
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("="*60)
        logger.info(f"Total runtime: {pipeline_stats['total_runtime_sec']:.2f}s")
        logger.info(f"Input rows: {pipeline_stats['total_input_rows']}")
        logger.info(f"Output rows: {pipeline_stats['wide_table_rows']}")
        logger.info(f"Final table: {final_output_path}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()