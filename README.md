# Taxi Data Pivoting Pipeline

A production-grade pipeline for processing NYC TLC taxi trip Parquet data into aggregated time-series tables.

**Assignments:** DSC 291 Homework 1, 2, and 3  
**Status:** HW1 complete (Parts 1–5 implemented and tested) · HW2 complete (PCA, tail analysis, Folium map, bootstrap) · HW3 complete (GAM fare prediction notebook)

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
- ✅ PCA (Dask covariance) + tail analysis + Folium map + bootstrap stability (HW2)
- ✅ GAM fare prediction with partial dependence, bootstrap CIs, and location enrichment (HW3)

---

## Installation

### Prerequisites
- Python 3.7+
- pip or conda

### Required Packages

```bash
pip install pandas numpy pyarrow dask[dataframe] fsspec s3fs psutil tqdm scipy matplotlib folium scikit-learn pygam
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
python3 pivot_all_files.py \
    --input-dir ./data/parquet_files \
    --output-dir ./output \
    --workers 48 \
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
python3 pivot_all_files.py \
  --input-dir s3://dsc291-ucsd/taxi/Dataset \
  --output-dir ./output-full \
  --s3-output s3://dsc291-pprashant-results/taxi-wide/full \
  --intermediate-dir s3://dsc291-pprashant-results/taxi-intermediate/full \
  --workers 48 \
  --partition-size "500MB" \
  --report-file report.json \
  --keep-intermediate
```

This will:
1. Discover `.parquet` files from S3
2. Process locally (writes intermediates to `./output/intermediate/`)
3. Upload final wide table to `s3://dsc291-ucsd//wide-table/`
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
python3 pivot_all_files.py \
    --input-dir /mnt/data/taxi \
    --output-dir ./pipeline_output \
    --s3-output s3://s3://dsc291-ucsd/ \
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
s3://dsc291-ucsd/wide-table/
```

**Content:**
- The final wide table Parquet file(s) (may be partitioned by year/month for very large datasets)
- Readable by Athena, Spark, pandas, etc.

**Example S3 Output:**
```
s3://dsc291-ucsd/wide_table/
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
df = pd.read_parquet('s3://dsc291-ucsd/wide_table/', filesystem=fs)
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

Partition optimization utilities for efficient large-file processing. Tests different partition sizes to find an optimal balance between memory use and performance using PyArrow batching and memory profiling.

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `parse_size(size_str)` | Parse a human-readable size string to bytes. Accepts e.g. `"200MB"`, `"1.5GB"`, `"1024KB"`, `"2TB"`. Returns size in bytes (int). Raises `ValueError` if format is invalid. |
| `format_size(size_bytes)` | Convert bytes to human-readable format (e.g. `"1.5GB"`, `"200MB"`). |
| `get_process_memory_mb()` | Get current process memory usage in MB (RSS). |
| `get_available_memory_mb()` | Get available system memory in MB. |
| `find_optimal_partition_size(file_path, candidate_sizes=None, size_range=None, num_sizes=5, max_memory_usage=None, warmup=True)` | Find optimal partition size for a parquet file using PyArrow batching. Tests different partition sizes (default 50MB–1GB if `size_range` not set), measures time and memory for each, and selects the best size within `max_memory_usage` (default 80% of available memory). Returns a dict with `optimal_size`, `optimal_size_str`, `results`, `max_memory_usage`, and `file_info`. |
| `test_partition_size(file_path, partition_size, num_iterations=3)` | Test performance of a specific partition size with multiple iterations. Reads parquet with Dask using the given `blocksize` (bytes), measures elapsed time and memory delta per run. Returns a dict with `avg_time_sec`, `min_time_sec`, `max_time_sec`, `avg_memory_mb`, `max_memory_mb`, and iteration count. |

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
python3 -m pytest test_pivot_comprehensive.py -v
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
python3 -m pytest test_pivot_comprehensive.py::TestColumnDetectionVariants -v
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
- Then run pipeline with `--s3-output s3://dsc291-ucsd/`

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
aws s3 ls s3://dsc291-ucsd/ --recursive | grep parquet
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
ucsd-dsc204/
├── pivot_utils.py                 # Core utilities: column detection, pivoting, S3 helpers
├── partition_optimization.py      # Adaptive Parquet partition sizing
├── pivot_all_files.py             # Main HW1 pipeline (pandas + multiprocessing)
├── pivot_all_files_dask.py        # Alternative HW1 pipeline (pure Dask)
├── pivot_and_bootstrap/           # HW2 package: PCA, tail analysis, Folium map, bootstrap
│   ├── pca_analysis.py            # Part 1: Dask covariance PCA
│   ├── tail_analysis.py           # Part 2: Tail classification + power-law fit
│   ├── mapping.py                 # Part 3: Folium choropleth map
│   ├── bootstrap_stability.py     # Part 4: Bootstrap eigenvector stability
│   └── hw2_run.py                 # HW2 CLI entry point
├── hw3_output/
│   └── taxi_fare_gam.ipynb        # HW3: GAM fare prediction notebook
├── hw2_output/                    # HW2 produced outputs (see HW2 section above)
├── output/                        # HW1 dask pipeline output (final_table.parquet)
├── data/
│   └── taxi_zones.csv             # NYC TLC zone → coordinate mapping (used by HW2 map)
├── sample_data/
│   └── sample_wide.parquet        # Small sample of wide table for testing
├── test_pivot_comprehensive.py    # Full test suite (58+ tests)
├── pa/
│   ├── hw2.md                     # HW2 specification
│   └── hw3.md                     # HW3 specification
├── README.md                      # This file
└── performance.md                 # HW1 performance report
```

---

## HW2: PCA, Tail Analysis, Folium Map, & Bootstrap Stability

A companion package lives under `pivot_and_bootstrap/` and implements the four parts of HW2.
All outputs are written to `hw2_output/`.

### Parts

| Part | Module | Description | Output |
|------|--------|-------------|--------|
| 1 | `pca_analysis` | PCA on unnormalized wide table; Dask covariance; missing values filled with column mean | `pca_model.pkl`, `variance_explained.png` |
| 2 | `tail_analysis` | Distribution of eigenvector loadings; light/heavy-tail classification; power-law α fit | `coefficient_distribution.png`, `tail_analysis_report.json` |
| 3 | `mapping` | Aggregate PC1/PC2 scores by pickup zone; interactive Folium choropleth | `pc1_pc2_folium_map.html`, `pc_scores_by_pickup_place.csv` |
| 4 | `bootstrap_stability` | Resample rows B=100 times; subspace affinity, Procrustes distance, component correlations | `bootstrap_stability_report.json`, `bootstrap_pc1_band.png`, `eigenvector_corr_boxplot.png` |

Diagnostics (condition number, Shapiro-Wilk, homoscedasticity, etc.) are written to `hw2_output/diagnostics/`.

### Key Results

**PCA (Part 1)**
- PC1 explains **84.1 %** of variance; PC1 + PC2 explain **91.9 %** combined.
- Effective rank ≈ 1.40 (2 PCs pass the Kaiser criterion out of 24).
- Eigenvectors are perfectly orthogonal (max off-diagonal < 2.5 × 10⁻¹⁵).

**Tail Analysis (Part 2)**
- Classification: **light-tailed** (Gaussian-like, not power-law).
- Best-fit power-law exponent α ≈ **5.93** (steep falloff), R² ≈ 0.982 on top 5 % of loadings.

**Bootstrap Stability (Part 4, B = 100)**

| Metric | Mean | Std |
|--------|------|-----|
| Subspace affinity | 0.99997 | 5.4 × 10⁻⁵ |
| Procrustes distance | 0.00657 | 0.00443 |
| PC1 correlation | 0.99999 | 2.0 × 10⁻⁵ |
| PC2 correlation | 0.99997 | 5.5 × 10⁻⁵ |

Eigenvectors are highly stable across bootstrap resamples.

### Quick Run

```bash
python -m pivot_and_bootstrap.hw2_run \
  --input s3://dsc291-pprashant-results/taxi-wide/full \
  --output-dir ./hw2_output --anon-s3 --zones-csv ./data/taxi_zones.csv --B 100
```

Notes:
- Pass `--anon-s3` for anonymous access to public S3 buckets.
- Provide `--zones-csv` (columns: `pickup_place`, `latitude`, `longitude`) to enable the Folium map.
- Default input: `s3://dsc291-pprashant-results/taxi-wide/full`.

### HW2 Output Structure

```
hw2_output/
├── pca_model.pkl                    # Saved PCA model (eigenvectors + variances)
├── variance_explained.png           # Scree / cumulative variance plot
├── coefficient_distribution.png     # Eigenvector loading histogram + Q-Q + log-log survival
├── tail_analysis_report.json        # classification, alpha, R², tail_fraction
├── pc_scores_by_pickup_place.csv    # Aggregated PC1–PC24 scores per zone (used by HW3)
├── pc1_pc2_folium_map.html          # Interactive Folium choropleth (PC1 color, PC2 size)
├── bootstrap_pc1_band.png           # Bootstrap PC1 band plot
├── eigenvector_corr_boxplot.png     # Per-component correlation boxplot across B resamples
├── bootstrap_stability_report.json  # Subspace affinity, Procrustes, component correlations
└── diagnostics/
    ├── diagnostics_report.json      # Condition number, Shapiro-Wilk, homoscedasticity, etc.
    └── extended_diagnostics.png     # Diagnostic figures
```

---

## HW3: Taxi Fare Prediction with a GAM

**Notebook:** `hw3_output/taxi_fare_gam.ipynb`

Predicts NYC yellow-taxi `fare_amount` from *non-fare* trip features using a Gaussian Generalized
Additive Model (`pygam.LinearGAM`) with identity link. The notebook runs top-to-bottom without
errors and produces all required figures.

### Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | Title & Intro | Goal statement: predict fare from trip features via GAM |
| 2 | Imports & Compat shim | SciPy `csr_matrix.A` compatibility fix for pygam + all imports |
| 3 | Configuration | Configurable `DATA_PATH`, `MAX_ROWS`, `RANDOM_SEED`, `TRAIN_FRAC`, extra-credit toggles |
| 4 | Load & Prepare | Anonymous S3 Parquet read; schema normalization (handles both TLC column-name variants via regex); sample `MAX_ROWS` rows |
| 5 | Feature Engineering | Derive `trip_duration_min`, `hour_of_day`, `day_of_week`; clean invalid rows; cap at 99th percentile |
| 6 | Location PCA (EC) | Join HW2 `pc_scores_by_pickup_place.csv` on `pickup_location` → adds `pc1_score`, `pc2_score` as smooth terms |
| 7 | Train/Val Split | 80 % train / 20 % validation; reports sizes |
| 8 | GAM Fit | `LinearGAM(s(0)+s(1)+s(2)+s(3)+l(4)[+s(5)+s(6)])` with λ grid search; prints `gam.summary()` |
| 9 | Evaluate | Validation RMSE, MAE, R²; actual-vs-predicted scatter plot with y = x reference line |
| 10 | Partial Dependence | Term-effect plots with 95 % CI bands for all predictors (distance, duration, hour, weekday, passenger count, PC1/PC2) |
| 11 | Bootstrap CIs (EC) | 50 bootstrap resamples → compare bootstrap 95 % percentile bands vs. pygam analytic CIs for top 3 terms |
| 12 | Fare Breakdown (EC) | Mean `|partial effect|` per term plotted as a horizontal bar chart |
| 13 | Discussion | Limitations: no surcharges/tolls, no GPS coordinates, single-month data, noisy passenger count |

### Data

- **Default source:** `s3://dsc291-ucsd/taxi/Dataset/2021/yellow_taxi/yellow_tripdata_2021-01.parquet` (anonymous S3 read)
- **Predictors:** `trip_distance`, `trip_duration_min`, `hour_of_day`, `day_of_week`, `passenger_count` (+ `pc1_score`, `pc2_score` if location PCA enabled)
- **Excluded (forbidden fare-related columns):** `total_amount`, `tip_amount`, `extra`, `mta_tax`, `tolls_amount`, surcharges, airport fees

### Extra Credit Included

1. **Location PCA enrichment** — HW2 `pc_scores_by_pickup_place.csv` joined on `PULocationID`; PC1 and PC2 added as smooth GAM terms, absorbing spatial demand structure.
2. **Bootstrap CI comparison** — For `trip_distance`, `trip_duration_min`, and `hour_of_day`, bootstrap 95 % percentile CIs (B = 50) are overlaid against pygam's analytic CIs to assess whether the model-derived uncertainty is realistic.
3. **Fare component breakdown** — Mean absolute partial effect per term visualized as a sorted horizontal bar chart, showing which features drive the most fare variation.

### GAM Model Terms

| Index | Feature | Term type | Notes |
|-------|---------|-----------|-------|
| 0 | `trip_distance` | `s()` smooth | Primary fare driver |
| 1 | `trip_duration_min` | `s()` smooth | Correlated with distance |
| 2 | `hour_of_day` | `s()` smooth, 24 knots | Captures rush-hour surge |
| 3 | `day_of_week` | `s()` smooth, 7 knots | Weekend vs weekday patterns |
| 4 | `passenger_count` | `l()` linear | Discrete; linear sufficient |
| 5 | `pc1_score` | `s()` smooth | Zone demand intensity (EC) |
| 6 | `pc2_score` | `s()` smooth | Zone demand contrast (EC) |

### Quick Run

```bash
# Install extra dependency (not in HW1/HW2 requirements)
pip install pygam

# Execute notebook
cd hw3_output
jupyter nbconvert --to notebook --execute --inplace taxi_fare_gam.ipynb \
    --ExecutePreprocessor.timeout=600
```

Or open `hw3_output/taxi_fare_gam.ipynb` in VS Code / JupyterLab and run all cells.

### HW3 Output Structure

```
hw3_output/
├── taxi_fare_gam.ipynb              # Main deliverable notebook
├── actual_vs_predicted.png          # Scatter plot: actual vs predicted fare
├── partial_dependence.png           # Term-effect plots with 95 % CI bands
├── bootstrap_ci_comparison.png      # Bootstrap vs. pygam CI comparison (EC)
└── fare_component_breakdown.png     # Per-term fare contribution bar chart (EC)
```

---

## References

- **NYC TLC Data:** https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page
- **Parquet Format:** https://parquet.apache.org/
- **Dask Documentation:** https://dask.readthedocs.io/
- **fsspec:** https://filesystem-spec.readthedocs.io/
- **S3fs:** https://s3fs.readthedocs.io/
- **PyArrow:** https://arrow.apache.org/docs/python/
- **pygam:** https://pygam.readthedocs.io/
- **scikit-learn:** https://scikit-learn.org/
- **Folium:** https://python-visualization.github.io/folium/

---

## License & Attribution

**Course:** DSC 291: Big Data Analytics (WI26)
**Instructor:** UC San Diego
**Assignments:** Homework 1 (Taxi Data Pivoting), Homework 2 (PCA + Tail + Map + Bootstrap), Homework 3 (GAM Fare Prediction)  
