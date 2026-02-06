# Taxi Data Pivoting Pipeline

A production-grade pipeline for processing NYC TLC taxi trip Parquet data into aggregated time-series tables.

## Overview

This pipeline:
1. Discovers parquet files from local or S3 paths
2. Validates and normalizes schemas across files
3. Processes files month-by-month
4. Pivots trip-level records into (date × taxi_type × pickup_place × hour) counts
5. Removes low-count rows (< 50 rides)
6. Combines results into a single wide table
7. Stores output as Parquet on S3
8. Generates performance reports

## Setup

```bash
pip install pandas pyarrow s3fs dask
```

## Usage
# Taxi Data Pivoting Pipeline

A production-grade pipeline for processing NYC TLC taxi trip Parquet data into aggregated time-series tables.

## Overview

This pipeline:
1. Discovers parquet files from local or S3 paths
2. Validates and normalizes schemas across files
3. Processes files month-by-month
4. Pivots trip-level records into (date × taxi_type × pickup_place × hour) counts
5. Removes low-count rows (< 50 rides)
6. Combines results into a single wide table
7. Stores output as Parquet on S3
8. Generates performance reports

## Setup

```bash
pip install pandas pyarrow s3fs dask
```

## Usage

python pivot_all_files.py \
    --input-dir /path/to/parquet/files \
    --output-dir ./output \
    --s3-output s3://your-bucket/taxi-pivot \
    --workers 8 \
    --min-rides 50

## CLI Arguments
--input-dir: Local directory or S3 path with parquet files (required)
--output-dir: Local directory for intermediate outputs (required)
--s3-output: S3 URI for final wide table
--min-rides: Minimum rides threshold (default: 50)
--workers: Number of parallel workers (default: 4)
--skip-partition-optimization: Skip partition size optimization
--keep-intermediate: Keep intermediate files after combining

## Output
The pipeline produces:

Intermediate files: Pivoted Parquet files (one per month) in output/intermediate/
Final wide table: Single Parquet file (or partitioned set) with structure:
Index: (taxi_type, date, pickup_place)
Columns: hour_0, hour_1, ..., hour_23 (ride counts)
Reports: performance.md with processing statistics
S3 Output
Store the final wide table on S3 using:

The resulting Parquet can be queried with:

## Testing
### Documentation
pivot_utils.py: Core utility functions
pivot_all_files.py: Main pipeline and CLI
partition_optimization.py: Partition sizing helpers
Performance
Expected runtime on r8i.4xlarge (8 vCPU, 128GB RAM): < 30 minutes for full dataset