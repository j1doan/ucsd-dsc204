# Taxi Data Pivoting Pipeline

A production-grade pipeline for processing NYC TLC taxi trip Parquet data into aggregated time-series tables.

**Assignment:** DSC 204 Homework 1 (100 points)  
**Status:** Complete with Parts 1–5 implemented and tested

---

## Overview

This pipeline processes raw NYC TLC taxi trip data (Parquet format) into an analytics-ready wide table:

1. **Discovers** parquet files from local or S3 paths (recursive)
2. **Validates** schemas across all files; normalizes to common schema
3. **Groups** files by month (year–month) for memory-efficient processing
4. **Processes** each file: reads → normalizes → pivots by (date, taxi_type, pickup_place, hour) → cleans (removes rows with < 50 rides) → writes intermediate Parquet
5. **Tracks** month-mismatch rows (rows where row date doesn't match file's inferred month)
6. **Combines** all intermediates into a single wide table indexed by (taxi_type, date, pickup_place) with hour_0…hour_23 columns
7. **Uploads** final table to S3 as Parquet
8. **Generates** JSON performance report with timing, memory, and row statistics

**Key Features:**
- ✅ Month-at-a-time processing (controls memory footprint)
- ✅ Parallel file processing within each month (multiprocessing)
- ✅ S3 support (reads from S3, writes results to S3)
- ✅ Adaptive partition optimization (via PyArrow blocksize)
- ✅ Comprehensive error handling and logging
- ✅ Full test coverage (58 tests across all modules)

---

## Installation

### Prerequisites
- Python 3.7+
- pip or conda

### Required Packages

```bash
pip install pandas numpy pyarrow dask[dataframe] fsspec s3fs psutil
```

**Note:** Install all packages together to ensure compatibility. If running on AWS:

```bash
pip install --upgrade pandas numpy pyarrow dask[dataframe] fsspec s3fs psutil
```

### Optional: Development & Testing

```bash
pip install pytest pytest-cov
```

---

## Quick Start

### Local Example

Process Parquet files in a local directory:

```bash
python pivot_all_files.py \
    --input-dir ./data/parquet_files \
    --output-dir ./output \
    --workers 4 \
    --min-rides 50
```

This will:
1. Discover all `.parquet` files in `./data/parquet_files/`
2. Process month-by-month (parallelized within each month)
3. Write intermediate pivoted tables to `./output/intermediate/`
4. Combine into final wide table: `./output/taxi_pivot_wide.parquet`
5. Generate performance report: `./output/report.json`

### S3 Example

Process from S3 and write results back to S3:

```bash
python pivot_all_files.py \
    --input-dir s3://my-bucket/nyc-taxi-data \
    --output-dir ./output \
    --s3-output s3://my-bucket/taxi-results/wide-table \
    --workers 8 \
    --keep-intermediate
```

This will:
1. Discover `.parquet` files from S3
2. Process locally (writes intermediates to `./output/intermediate/`)
3. Upload final wide table to `s3://my-bucket/taxi-results/wide-table/`
4. Optionally keep intermediate files for debugging

---

## CLI Arguments

### Required Arguments

| Argument | Description | Example |
|----------|-------------|---------|
| `--input-dir` | Local directory or S3 path containing Parquet files | `./data` or `s3://bucket/taxi-data` |
| `--output-dir` | Local directory for intermediate and final outputs | `./output` |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--s3-output` | (none) | S3 path where final wide table will be uploaded (e.g., `s3://bucket/results/wide-table`) |
| `--min-rides` | 50 | Minimum total rides across hours to keep a row (rows with fewer rides are discarded) |
| `--workers` | 4 | Number of parallel workers for per-month file processing |
| `--partition-size` | auto | Block size for Parquet reading (e.g., `"200MB"`, `"1GB"`); auto-optimizes if not set |
| `--skip-partition-optimization` | False | Skip automatic partition size optimization |
| `--keep-intermediate` | False | Keep intermediate Parquet files after combining (useful for debugging) |
| `--report-file` | `{output-dir}/report.json` | Path to JSON performance report |

### Example with All Options

```bash
python pivot_all_files.py \
    --input-dir /mnt/data/taxi \
    --output-dir ./pipeline_output \
    --s3-output s3://my-analytics/taxi-pivot-table \
    --min-rides 50 \
    --workers 8 \
    --partition-size "500MB" \
    --keep-intermediate \
    --report-file ./pipeline_output/performance_report.json
```

---

## Output Structure

### Intermediate Files

Located in `{output-dir}/intermediate/`:

```
intermediate/
├── yellow_2023_01.parquet   # Month 2023-01, taxi type yellow
├── yellow_2023_02.parquet   # Month 2023-02, taxi type yellow
├── green_2023_01.parquet    # Month 2023-01, taxi type green
└── ...
```

**Schema:**
- **Index:** (taxi_type, date, pickup_place) – MultiIndex
- **Columns:** hour_0, hour_1, ..., hour_23 (integer ride counts)

### Final Wide Table

**Location:** `{output-dir}/taxi_pivot_wide.parquet`  
**Also uploaded to:** `{s3-output}/` (if specified)

**Schema:**
- **Index:** (taxi_type, date, pickup_place)
- **Columns:** hour_0, hour_1, ..., hour_23
- **Data type:** Numeric (counts)

**Example query:**
```python
import pandas as pd

df = pd.read_parquet('./output/taxi_pivot_wide.parquet')
print(df.head())
# Output:
#                                    hour_0  hour_1  ...  hour_23
# taxi_type date       pickup_place
# yellow    2023-01-01 40            5       3       ...  2
#           2023-01-01 41            8       6       ...  4
#           2023-01-01 42            3       2       ...  1
# green     2023-01-01 40            2       1       ...  1
```

### Performance Report

**Location:** `{output-dir}/report.json`

**Contents:**
```json
{
  "total_runtime_seconds": 1234.5,
  "peak_memory_mb": 2048,
  "total_input_rows": 10000000,
  "total_output_rows": 50000,
  "intermediate_table_rows": 75000,
  "rows_discarded_low_count": 25000,
  "rows_with_month_mismatch": 150,
  "month_mismatch_by_month": {
    "2023-01": 75,
    "2023-02": 75
  },
  "files_processed": 24,
  "files_with_errors": 0
}
```

---

## S3 Output Path

**Default Behavior:**
If `--s3-output` is specified, the final wide table is uploaded to the given S3 path. The upload uses **anonymous credentials** by default.

**S3 Path Format:**
```
s3://your-bucket/path/to/wide-table/
```

**Content:**
- The final wide table Parquet file(s) (may be partitioned by year/month for very large datasets)
- Readable by Athena, Spark, pandas, etc.

**Example S3 Output:**
```
s3://my-analytics-bucket/taxi-pipeline/wide_table/
├── part-0.parquet
├── _metadata
└── _common_metadata
```

**Accessing the S3 table:**
```python
import pandas as pd
import s3fs

# With anonymous access
fs = s3fs.S3FileSystem(anon=True)
df = pd.read_parquet('s3://my-bucket/path/wide_table/', filesystem=fs)
```

**Note:** If your S3 bucket requires authentication, set AWS credentials in environment variables or configure `.aws/credentials`:
```bash
export AWS_ACCESS_KEY_ID=your_key
export AWS_SECRET_ACCESS_KEY=your_secret
export AWS_DEFAULT_REGION=us-east-1
```

---

## Module Documentation

### `pivot_utils.py`

Core utility functions for column detection, pivoting, and S3 operations.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `find_pickup_datetime_col(columns)` | Detect pickup datetime column (handles variant names: tpep_pickup_datetime, pickup_datetime, etc.) |
| `find_pickup_location_col(columns)` | Detect pickup location column (PULocationID, PULocation, etc.) |
| `infer_month_from_path(file_path)` | Extract (year, month) from file path (e.g., "2023-01" or "year=2023/month=01") |
| `infer_taxi_type_from_path(file_path)` | Detect taxi type: yellow, green, fhv, fhvhv |
| `pivot_counts_date_taxi_type_location(df)` | Pivot trips into (date, taxi_type, pickup_place) × hour_0…hour_23 |
| `cleanup_low_count_rows(df, min_rides=50)` | Remove rows with < min_rides total; return (cleaned_df, stats_dict) |
| `is_s3_path(path)` | Check if path is S3 (starts with s3://) |
| `get_storage_options(path)` | Return fsspec options dict (anon=True for S3) |
| `get_filesystem(path)` | Get fsspec filesystem (s3fs for S3, local for paths) |
| `discover_parquet_files(path)` | Recursively find all .parquet files (local or S3), sorted |
| `read_parquet_with_dask(path, blocksize=None)` | Read Parquet with Dask (supports S3, lazy evaluation) |

**Type Hints & Docstrings:** All functions fully documented with comprehensive docstrings.

### `partition_optimization.py`

Partition sizing utilities for efficient Parquet reading.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `parse_size(size_str)` | Convert "200MB", "1.5GB", etc. → bytes (int) |
| `format_size(num_bytes)` | Convert bytes → "1.5GB", "200MB", etc. |
| `get_process_memory_mb()` | Get current process memory (MB) |
| `get_available_memory_mb()` | Get available system memory (MB) |
| `find_optimal_partition_size(parquet_path, max_memory_usage=...)` | Test candidate block sizes, pick best for speed/memory |
| `test_partition_size(parquet_path, blocksize, iterations=3)` | Measure read performance for a specific block size |

**Type Hints & Docstrings:** All functions fully documented with comprehensive docstrings.

### `pivot_all_files.py`

Main pipeline orchestration.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `process_single_file(file_path, output_dir, expected_month, ...)` | Read, pivot, clean single Parquet; detect month mismatches; return stats |
| `combine_into_wide_table(intermediate_dir, output_path, ...)` | Read all intermediate Parquets, aggregate by (taxi_type, date, pickup_place), sum hours |
| `group_files_by_month(file_list)` | Group discovered files into (year, month) buckets |
| `process_month_files(month, file_list, ...)` | Process all files for a single month (parallel within month) |
| `generate_report(stats, output_path)` | Write JSON report with timing, memory, row counts |
| `main()` | CLI entry point; orchestrates full pipeline |

**Type Hints & Docstrings:** All functions fully documented with comprehensive docstrings.

---

## Testing

### Run All Tests

```bash
python -m pytest test_pivot_comprehensive.py -v
```

### Test Coverage

- **Column Detection:** 10 tests (variants, case-insensitivity)
- **Month Inference:** 8 tests (YYYY-MM, YYYYMM, S3 patterns)
- **Taxi Type Inference:** 6 tests
- **Pivot Output Shape:** 5 tests (index structure, hour columns)
- **Error Handling:** 5 tests (missing columns, bad data)
- **Cleanup Low Counts:** 3 tests
- **S3 Path Handling:** 4 tests
- **File Discovery:** 6 tests
- **Partition Size Parsing:** 10 tests
- **Month-Mismatch Detection:** 1 test
- **End-to-End Integration:** 1 test

**Total:** 58+ tests covering all modules.

### Run Specific Test Class

```bash
python -m pytest test_pivot_comprehensive.py::TestColumnDetectionVariants -v
```

---

## Troubleshooting

### 1. **ImportError: No module named 'dask' / 's3fs' / 'fsspec'**

**Solution:**
```bash
pip install --upgrade dask[dataframe] s3fs fsspec
```

Ensure all packages are installed in your current Python environment.

### 2. **S3 Access Denied or Connection Refused**

**If bucket is public (anonymous):**
- Ensure S3 path is correct: `s3://bucket-name/path/to/files/`
- Verify files are actually present: `aws s3 ls s3://bucket-name/path/to/files/`

**If bucket requires authentication:**
- Set AWS credentials:
  ```bash
  export AWS_ACCESS_KEY_ID=your_key
  export AWS_SECRET_ACCESS_KEY=your_secret
  export AWS_DEFAULT_REGION=us-east-1
  ```
- Or configure `~/.aws/credentials`
- Then run pipeline with `--s3-output s3://your-bucket/path/`

### 3. **MemoryError or Out of Memory**

**Causes:** Month contains too many files; partition size too large.

**Solutions:**
- Reduce `--workers` (process fewer files in parallel): `--workers 2`
- Decrease `--partition-size` (e.g., `--partition-size "100MB"`)
- Increase available RAM on instance
- Monitor memory: `watch -n 1 'ps aux | grep pivot'`

### 4. **No Parquet Files Found**

**Check:**
```bash
# List local files
ls -la /path/to/input/dir/

# List S3 files
aws s3 ls s3://bucket/path/ --recursive | grep parquet
```

**Ensure:**
- Path is correct and files exist
- Files have `.parquet` extension (case-sensitive)
- Recursive search includes subdirectories

### 5. **Month Mismatch Warnings**

If the report shows high `rows_with_month_mismatch`, it may indicate:
- Files named incorrectly (e.g., wrong date in filename)
- Data spanning month boundaries
- Schema inconsistencies

**Check:** Review logs and the full report JSON for which months/files are affected.

### 6. **Slow Performance**

**Optimization checklist:**
- Increase `--workers` (if CPU cores available): `--workers 8`
- Optimize `--partition-size` (default auto-detects; or set manually to ~500MB)
- Use SSD for local I/O (faster than spinning disk)
- Ensure network connectivity for S3 (if pulling from S3)
- Monitor memory usage; reduce workers if approaching instance limits

### 7. **File Mixed Schema Errors**

If files have different schemas:
- Pipeline auto-detects common schema across all files
- Missing columns are filled with NaN in intermediate tables
- Check logs for schema mismatch warnings

---

## Performance Notes

**Expected Runtime (r8i.4xlarge, 8 vCPU, 128 GB RAM):**
- Full year of NYC taxi data: ~20–30 minutes
- File discovery: 1–2 minutes
- Per-month processing: 2–5 minutes each
- Final combination: 5–10 minutes
- S3 upload: Network-dependent (typically 5–15 minutes for full dataset)

**Memory Usage:**
- Intermediate tables: ~100–500 MB per month
- Peak usage during combination phase: 5–20 GB (depends on month size)
- S3 operations: <100 MB overhead

**CPU Utilization:**
- With `--workers 8`: ~6–8 cores utilized during per-month processing
- Partition optimization uses 1 core during initial testing phase
- Dask parallel reads on multiple cores during combine phase

---

## Code Quality

- ✅ **Type Hints:** All functions include type annotations (`Optional`, `Dict`, `Tuple`, etc.)
- ✅ **Docstrings:** Comprehensive docstrings for all public functions (Args, Returns, Raises)
- ✅ **Error Handling:** Try/except blocks with informative error messages and logging
- ✅ **Logging:** All major operations logged at INFO level; DEBUG available for verbosity
- ✅ **Testing:** 58+ pytest tests covering all functions, error cases, and integration scenarios
- ✅ **PEP 8 Compliance:** Code follows PEP 8 style guidelines (line length, naming, imports)

---

## File Structure

```
DSC 204/
├── pivot_utils.py                 # Core utilities (Parts 1-2)
├── partition_optimization.py       # Partition sizing (Part 3)
├── pivot_all_files.py             # Main pipeline (Part 4)
├── test_pivot_comprehensive.py    # Test suite (Part 5)
├── README.md                       # Documentation (Part 6)
└── performance.md                  # Performance report (generated after running)
```

---

## References

- **NYC TLC Data:** https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **Parquet Format:** https://parquet.apache.org/
- **Dask Documentation:** https://dask.readthedocs.io/
- **fsspec:** https://filesystem-spec.readthedocs.io/
- **S3fs:** https://s3fs.readthedocs.io/
- **PyArrow:** https://arrow.apache.org/docs/python/

---

## License & Attribution

**Course:** Data Science / Big Data Analytics (DSC 204)  
**Semester:** Spring 2024  
**Assignment:** Homework 1 – Taxi Data Pivoting  
**Instructor:** Course Staff  
