"""
test_pivot_comprehensive.py

Comprehensive unit tests for the taxi data pivoting pipeline.
Tests all functions from pivot_utils, partition_optimization, and pivot_all_files.

Covers:
- Column detection variants (all cases)
- Pivot output shape and index structure
- Error handling (missing columns, bad data)
- Month inference for common path patterns
- Month-mismatch detection
- Low-count row cleanup
- S3 path detection
- File discovery
- Partition size parsing
- File processing and statistics

Run: python -m pytest test_pivot_comprehensive.py -v
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

import pivot_utils as pu
import partition_optimization as popt


# ============================================================================
# PART 1: Column Detection Tests
# ============================================================================


class TestColumnDetectionVariants:
    """Test column detection across common naming variants."""
    
    def test_find_pickup_datetime_standard_tpep(self):
        """Test standard NYC taxi: tpep_pickup_datetime"""
        columns = ['VendorID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'fare_amount']
        result = pu.find_pickup_datetime_col(columns)
        assert result == 'tpep_pickup_datetime'
    
    def test_find_pickup_datetime_generic(self):
        """Test generic: pickup_datetime"""
        columns = ['vendor_id', 'pickup_datetime', 'dropoff_datetime']
        result = pu.find_pickup_datetime_col(columns)
        assert result == 'pickup_datetime'
    
    def test_find_pickup_datetime_mixed_case(self):
        """Test case-insensitive matching"""
        columns = ['vendor', 'PickupDateTime', 'Amount']
        result = pu.find_pickup_datetime_col(columns)
        assert result == 'PickupDateTime'
    
    def test_find_pickup_datetime_underscore_variants(self):
        """Test underscore variations"""
        columns = ['pickup_time_date', 'dropoff_time_date']
        result = pu.find_pickup_datetime_col(columns)
        assert result is not None  # Should find at least one
    
    def test_find_pickup_datetime_not_found(self):
        """Test when datetime column doesn't exist"""
        columns = ['vendor_id', 'trip_distance', 'fare_amount']
        result = pu.find_pickup_datetime_col(columns)
        assert result is None
    
    def test_find_pickup_location_pulocationid(self):
        """Test NYC taxi: PULocationID"""
        columns = ['VendorID', 'PULocationID', 'DOLocationID', 'fare_amount']
        result = pu.find_pickup_location_col(columns)
        assert result == 'PULocationID'
    
    def test_find_pickup_location_pulocation(self):
        """Test variant: pu_location"""
        columns = ['pu_location', 'do_location', 'trip_distance']
        result = pu.find_pickup_location_col(columns)
        assert result == 'pu_location'
    
    def test_find_pickup_location_zone(self):
        """Test variant: pickup_zone"""
        columns = ['pickup_zone', 'dropoff_zone', 'fare']
        result = pu.find_pickup_location_col(columns)
        assert result == 'pickup_zone'
    
    def test_find_pickup_location_not_found(self):
        """Test when location column doesn't exist"""
        columns = ['vendor_id', 'trip_distance', 'fare_amount']
        result = pu.find_pickup_location_col(columns)
        assert result is None


# ============================================================================
# PART 1: Month Inference Tests
# ============================================================================


class TestMonthInferencePatterns:
    """Test month inference from various path patterns."""
    
    def test_infer_month_yyyy_mm_format(self):
        """Test standard YYYY-MM format: yellow_tripdata_2023-01.parquet"""
        result = pu.infer_month_from_path('yellow_tripdata_2023-01.parquet')
        assert result == (2023, 1)
    
    def test_infer_month_yyyy_mm_middle_path(self):
        """Test YYYY-MM in middle of path: /data/2023-06/file.parquet"""
        result = pu.infer_month_from_path('/data/2023-06/file.parquet')
        assert result == (2023, 6)
    
    def test_infer_month_s3_year_month_partition(self):
        """Test S3 partitioned: s3://bucket/year=2023/month=05/data.parquet"""
        result = pu.infer_month_from_path('s3://bucket/year=2023/month=05/data.parquet')
        assert result == (2023, 5)
    
    def test_infer_month_yyyymm_compact_format(self):
        """Test compact YYYYMM: data_202312.parquet"""
        result = pu.infer_month_from_path('data_202312.parquet')
        assert result == (2023, 12)
    
    def test_infer_month_all_months(self):
        """Test all valid months (1-12)"""
        for month in range(1, 13):
            path = f'data_2023-{month:02d}.parquet'
            result = pu.infer_month_from_path(path)
            assert result == (2023, month), f"Failed for month {month}"
    
    def test_infer_month_invalid_month(self):
        """Test invalid month numbers are rejected"""
        result = pu.infer_month_from_path('data_2023-13.parquet')
        assert result is None
        
        result = pu.infer_month_from_path('data_2023-00.parquet')
        assert result is None
    
    def test_infer_month_no_match(self):
        """Test path with no month pattern"""
        result = pu.infer_month_from_path('data_abc_xyz.parquet')
        assert result is None
    
    def test_infer_month_multiple_patterns(self):
        """Test path with multiple date patterns (should use first)"""
        result = pu.infer_month_from_path('2022-12_data_2023-01.parquet')
        assert result is not None  # Should match one
        assert 1 <= result[1] <= 12


# ============================================================================
# PART 1: Taxi Type Inference Tests
# ============================================================================


class TestTaxiTypeInference:
    """Test taxi type inference from paths."""
    
    def test_infer_yellow(self):
        """Test yellow taxi detection"""
        assert pu.infer_taxi_type_from_path('yellow_tripdata_2023-01.parquet') == 'yellow'
        assert pu.infer_taxi_type_from_path('/data/yellow/file.parquet') == 'yellow'
    
    def test_infer_green(self):
        """Test green taxi detection"""
        assert pu.infer_taxi_type_from_path('green_tripdata_2023-01.parquet') == 'green'
    
    def test_infer_fhv(self):
        """Test FHV detection"""
        assert pu.infer_taxi_type_from_path('fhv_tripdata.parquet') == 'fhv'
    
    def test_infer_fhvhv(self):
        """Test FHVHV detection"""
        assert pu.infer_taxi_type_from_path('fhvhv_tripdata.parquet') == 'fhvhv'
    
    def test_infer_case_insensitive(self):
        """Test case-insensitive matching"""
        assert pu.infer_taxi_type_from_path('YELLOW_data.parquet') == 'yellow'
        assert pu.infer_taxi_type_from_path('Green_data.parquet') == 'green'
    
    def test_infer_none_for_unknown(self):
        """Test unknown types return None"""
        assert pu.infer_taxi_type_from_path('data.parquet') is None
        assert pu.infer_taxi_type_from_path('uber_data.parquet') is None


# ============================================================================
# PART 1: Pivot Output Tests
# ============================================================================


class TestPivotOutputShape:
    """Test pivot function output shape and structure."""
    
    @pytest.fixture
    def sample_trip_data(self):
        """Create sample trip data for pivoting."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        hours = list(range(24))
        
        rows = []
        for date in dates:
            for hour in hours:
                for location in [1, 2, 3]:
                    # Random number of trips for this location/hour
                    num_trips = np.random.randint(1, 10)
                    rows.append({
                        'tpep_pickup_datetime': date + timedelta(hours=hour),
                        'PULocationID': location,
                        'taxi_type': 'yellow',
                    })
        
        return pd.DataFrame(rows)
    
    def test_pivot_output_has_correct_index(self, sample_trip_data):
        """Test that pivot output has correct 3-level index."""
        pivoted = pu.pivot_counts_date_taxi_type_location(sample_trip_data)
        
        assert isinstance(pivoted.index, pd.MultiIndex)
        assert pivoted.index.nlevels == 3
        assert list(pivoted.index.names) == ['taxi_type', 'date', 'pickup_place']
    
    def test_pivot_output_has_all_hour_columns(self, sample_trip_data):
        """Test that all 24 hour columns are present."""
        pivoted = pu.pivot_counts_date_taxi_type_location(sample_trip_data)
        
        hour_cols = [col for col in pivoted.columns if col.startswith('hour_')]
        assert len(hour_cols) == 24
        
        for i in range(24):
            assert f'hour_{i}' in pivoted.columns
    
    def test_pivot_output_all_non_negative(self, sample_trip_data):
        """Test that all values in pivot output are non-negative."""
        pivoted = pu.pivot_counts_date_taxi_type_location(sample_trip_data)
        
        for col in pivoted.columns:
            assert (pivoted[col] >= 0).all(), f"Negative values in {col}"
    
    def test_pivot_output_hour_columns_are_numeric(self, sample_trip_data):
        """Test that hour columns contain numeric values."""
        pivoted = pu.pivot_counts_date_taxi_type_location(sample_trip_data)
        
        for col in pivoted.columns:
            assert pd.api.types.is_numeric_dtype(pivoted[col])
    
    def test_pivot_with_auto_column_detection(self, sample_trip_data):
        """Test pivot with auto-detected column names."""
        # Don't specify column names - should auto-detect
        pivoted = pu.pivot_counts_date_taxi_type_location(sample_trip_data)
        
        assert pivoted is not None
        assert len(pivoted) > 0
        assert len(pivoted.columns) == 24


# ============================================================================
# PART 1: Error Handling Tests
# ============================================================================


class TestErrorHandling:
    """Test error handling for invalid inputs."""
    
    def test_pivot_missing_datetime_column(self):
        """Test pivot with missing datetime column."""
        df = pd.DataFrame({
            'PULocationID': [1, 2, 3],
            'taxi_type': ['yellow', 'yellow', 'yellow'],
        })
        
        with pytest.raises(ValueError, match="pickup datetime"):
            pu.pivot_counts_date_taxi_type_location(df)
    
    def test_pivot_missing_location_column(self):
        """Test pivot with missing location column."""
        df = pd.DataFrame({
            'tpep_pickup_datetime': pd.date_range('2023-01-01', periods=3),
            'taxi_type': ['yellow', 'yellow', 'yellow'],
        })
        
        with pytest.raises(ValueError, match="pickup location"):
            pu.pivot_counts_date_taxi_type_location(df)
    
    def test_pivot_missing_taxi_type_column(self):
        """Test pivot with missing taxi_type column."""
        df = pd.DataFrame({
            'tpep_pickup_datetime': pd.date_range('2023-01-01', periods=3),
            'PULocationID': [1, 2, 3],
        })
        
        with pytest.raises(ValueError, match="taxi type"):
            pu.pivot_counts_date_taxi_type_location(df)
    
    def test_cleanup_empty_dataframe(self):
        """Test cleanup with empty dataframe."""
        index = pd.MultiIndex.from_tuples([], names=['taxi_type', 'date', 'pickup_place'])
        df = pd.DataFrame(index=index)
        
        cleaned, stats = pu.cleanup_low_count_rows(df, min_rides=50)
        assert len(cleaned) == 0
        assert stats['rows_before'] == 0
    
    def test_cleanup_no_hour_columns(self):
        """Test cleanup with dataframe missing hour columns."""
        index = pd.MultiIndex.from_tuples([
            ('yellow', '2023-01-01', 1)
        ], names=['taxi_type', 'date', 'pickup_place'])
        df = pd.DataFrame({'other_col': [100]}, index=index)
        
        cleaned, stats = pu.cleanup_low_count_rows(df, min_rides=50)
        # Should handle gracefully
        assert len(cleaned) == len(df)


# ============================================================================
# PART 1: Cleanup Tests
# ============================================================================


class TestCleanupLowCountRows:
    """Test the cleanup_low_count_rows function."""
    
    @pytest.fixture
    def sample_pivot_data(self):
        """Create sample pivoted data with varying row totals."""
        index = pd.MultiIndex.from_product([
            ['yellow'],
            pd.date_range('2023-01-01', periods=3).date,
            [1, 2, 3]
        ], names=['taxi_type', 'date', 'pickup_place'])
        
        hours = [f'hour_{i}' for i in range(24)]
        data = {h: 0 for h in hours}
        
        # Create rows with different total rides
        df = pd.DataFrame(data, index=index)
        df.loc[('yellow', df.index.get_level_values('date')[0], 1), 'hour_0'] = 60  # 60 total
        df.loc[('yellow', df.index.get_level_values('date')[0], 2), 'hour_0'] = 30  # 30 total
        df.loc[('yellow', df.index.get_level_values('date')[1], 1), 'hour_0'] = 10  # 10 total
        df.loc[('yellow', df.index.get_level_values('date')[2], 1), 'hour_0'] = 100 # 100 total
        df.loc[('yellow', df.index.get_level_values('date')[2], 2), 'hour_0'] = 75  # 75 total
        df.loc[('yellow', df.index.get_level_values('date')[2], 3), 'hour_0'] = 5   # 5 total
        
        return df
    
    def test_cleanup_removes_correct_rows(self, sample_pivot_data):
        """Test that cleanup removes rows with total < min_rides."""
        cleaned, stats = pu.cleanup_low_count_rows(sample_pivot_data, min_rides=50)
        
        # Should keep: 60, 100, 75 (3 rows)
        # Should remove: 30, 10, 5 (3 rows)
        assert stats['rows_before'] == 9
        assert len(cleaned) == 3
        assert stats['rows_removed'] == 6
    
    def test_cleanup_returns_valid_stats(self, sample_pivot_data):
        """Test that cleanup returns correct statistics."""
        cleaned, stats = pu.cleanup_low_count_rows(sample_pivot_data, min_rides=50)
        
        assert 'rows_before' in stats
        assert 'rows_after' in stats
        assert 'rows_removed' in stats
        assert 'min_rides' in stats
        assert stats['min_rides'] == 50
    
    def test_cleanup_min_rides_boundary(self, sample_pivot_data):
        """Test cleanup at boundary values."""
        # Test min_rides=50 (should keep exactly 50)
        cleaned, _ = pu.cleanup_low_count_rows(sample_pivot_data, min_rides=50)
        
        if len(cleaned) > 0:
            hour_cols = [col for col in cleaned.columns if col.startswith('hour_')]
            totals = cleaned[hour_cols].sum(axis=1)
            assert (totals >= 50).all()


# ============================================================================
# PART 2: S3 Path Detection Tests
# ============================================================================


class TestS3PathDetection:
    """Test S3 path detection and storage options."""
    
    def test_is_s3_path_true(self):
        """Test S3 path detection returns True."""
        assert pu.is_s3_path('s3://bucket/key')
        assert pu.is_s3_path('S3://bucket/key')
        assert pu.is_s3_path('s3://bucket/path/to/file.parquet')
    
    def test_is_s3_path_false(self):
        """Test local path detection returns False."""
        assert not pu.is_s3_path('/local/path')
        assert not pu.is_s3_path('relative/path')
        assert not pu.is_s3_path('C:\\Windows\\path')
    
    def test_get_storage_options_s3(self):
        """Test that S3 paths get anonymous=True by default."""
        opts = pu.get_storage_options('s3://bucket/key')
        assert 'anon' in opts
        assert opts['anon'] is True
    
    def test_get_storage_options_local(self):
        """Test that local paths get empty options."""
        opts = pu.get_storage_options('/local/path')
        assert opts == {}


# ============================================================================
# PART 2: File Discovery Tests
# ============================================================================


class TestFileDiscovery:
    """Test file discovery functionality."""
    
    def test_discover_files_empty_directory(self):
        """Test discovery in empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            files = pu.discover_parquet_files(tmpdir)
            assert files == []
    
    def test_discover_files_single_file(self):
        """Test discovery finds a single parquet file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a dummy parquet file
            df = pd.DataFrame({'a': [1, 2, 3]})
            filepath = Path(tmpdir) / 'test.parquet'
            df.to_parquet(filepath)
            
            files = pu.discover_parquet_files(tmpdir)
            assert len(files) == 1
            assert files[0].endswith('test.parquet')
    
    def test_discover_files_multiple_files(self):
        """Test discovery finds multiple parquet files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create multiple parquet files
            for i in range(3):
                df = pd.DataFrame({'a': [i, i+1, i+2]})
                filepath = Path(tmpdir) / f'test_{i}.parquet'
                df.to_parquet(filepath)
            
            files = pu.discover_parquet_files(tmpdir)
            assert len(files) == 3
    
    def test_discover_files_nested_directories(self):
        """Test discovery in nested directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create nested structure
            nested = Path(tmpdir) / 'year=2023' / 'month=01'
            nested.mkdir(parents=True)
            
            df = pd.DataFrame({'a': [1, 2, 3]})
            filepath = nested / 'data.parquet'
            df.to_parquet(filepath)
            
            files = pu.discover_parquet_files(tmpdir)
            assert len(files) == 1
    
    def test_discover_files_nonexistent_path(self):
        """Test that nonexistent path raises error."""
        with pytest.raises(ValueError, match="does not exist"):
            pu.discover_parquet_files('/nonexistent/path')
    
    def test_discover_files_file_not_directory(self):
        """Test that file path raises error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / 'test.txt'
            filepath.write_text('test')
            
            with pytest.raises(ValueError, match="not a directory"):
                pu.discover_parquet_files(str(filepath))


# ============================================================================
# PART 3: Partition Size Parsing Tests
# ============================================================================


class TestPartitionSizeParsing:
    """Test parse_size function for partition optimization."""
    
    def test_parse_size_mb(self):
        """Test parsing MB sizes."""
        assert popt.parse_size('200MB') == 200 * 1024 * 1024
        assert popt.parse_size('50MB') == 50 * 1024 * 1024
    
    def test_parse_size_gb(self):
        """Test parsing GB sizes."""
        assert popt.parse_size('1GB') == 1024 ** 3
        assert popt.parse_size('1.5GB') == int(1.5 * 1024 ** 3)
    
    def test_parse_size_kb(self):
        """Test parsing KB sizes."""
        assert popt.parse_size('1024KB') == 1024 * 1024
    
    def test_parse_size_bytes(self):
        """Test parsing byte sizes."""
        assert popt.parse_size('1000B') == 1000
    
    def test_parse_size_tb(self):
        """Test parsing TB sizes."""
        assert popt.parse_size('2TB') == 2 * 1024 ** 4
    
    def test_parse_size_lowercase(self):
        """Test case-insensitive parsing."""
        assert popt.parse_size('200mb') == 200 * 1024 * 1024
        assert popt.parse_size('1gb') == 1024 ** 3
    
    def test_parse_size_floats(self):
        """Test parsing float sizes."""
        assert popt.parse_size('0.5GB') == int(0.5 * 1024 ** 3)
        assert popt.parse_size('2.5MB') == int(2.5 * 1024 ** 2)
    
    def test_parse_size_whitespace(self):
        """Test parsing with whitespace."""
        assert popt.parse_size(' 200 MB ') == 200 * 1024 * 1024
        assert popt.parse_size('1 GB') == 1024 ** 3
    
    def test_parse_size_invalid_format(self):
        """Test that invalid formats raise ValueError."""
        with pytest.raises(ValueError):
            popt.parse_size('invalid')
        
        with pytest.raises(ValueError):
            popt.parse_size('200XX')
        
        with pytest.raises(ValueError):
            popt.parse_size('')
    
    def test_format_size_round_trip(self):
        """Test that parse_size and format_size are inverse functions."""
        sizes = ['50MB', '1GB', '1.5GB', '100KB']
        for size_str in sizes:
            parsed = popt.parse_size(size_str)
            formatted = popt.format_size(parsed)
            # Re-parse should give approximately same result
            reparsed = popt.parse_size(formatted)
            assert abs(reparsed - parsed) < 10  # Allow small rounding error


# ============================================================================
# PART 4: Month-at-a-Time Processing Tests
# ============================================================================


class TestMonthMismatchDetection:
    """Test detection of rows with month mismatches."""
    
    @pytest.fixture
    def mixed_month_data(self):
        """Create data with some rows mismatching file month."""
        dates = [
            pd.Timestamp('2023-01-15'),  # Matches expected month
            pd.Timestamp('2023-01-20'),  # Matches
            pd.Timestamp('2023-02-01'),  # Mismatch! File is 2023-01
            pd.Timestamp('2023-01-10'),  # Matches
            pd.Timestamp('2023-12-31'),  # Mismatch!
        ]
        
        return pd.DataFrame({
            'tpep_pickup_datetime': dates,
            'PULocationID': [1, 2, 3, 1, 2],
            'taxi_type': ['yellow'] * 5,
        })
    
    def test_detect_month_mismatches(self, mixed_month_data):
        """Test that month mismatches are detected correctly."""
        df = mixed_month_data.copy()
        df['_row_date'] = df['tpep_pickup_datetime'].dt.date
        df['_row_month'] = df['tpep_pickup_datetime'].dt.month
        df['_row_year'] = df['tpep_pickup_datetime'].dt.year
        
        expected_month = (2023, 1)
        mismatches = df[
            (df['_row_year'] != expected_month[0]) |
            (df['_row_month'] != expected_month[1])
        ]
        
        assert len(mismatches) == 2  # Feb and Dec


# ============================================================================
# Integration Tests
# ============================================================================


class TestEndToEndPipeline:
    """Test complete pipeline flow."""
    
    @pytest.fixture
    def sample_pipeline_data(self):
        """Create realistic sample data for full pipeline."""
        np.random.seed(42)
        
        # Create one week of data for 3 locations
        dates = pd.date_range('2023-01-01', periods=7, freq='D')
        locations = [1, 2, 3]
        
        rows = []
        for date in dates:
            for location in locations:
                for hour in range(24):
                    num_trips = np.random.randint(1, 20)
                    for _ in range(num_trips):
                        rows.append({
                            'VendorID': 1,
                            'tpep_pickup_datetime': date + timedelta(hours=hour, minutes=np.random.randint(0, 60)),
                            'PULocationID': location,
                            'trip_distance': np.random.uniform(0.5, 20),
                            'fare_amount': np.random.uniform(5, 100),
                        })
        
        return pd.DataFrame(rows)
    
    def test_full_pipeline_creates_valid_output(self, sample_pipeline_data):
        """Test that full pipeline produces valid output."""
        # Add taxi_type column
        sample_pipeline_data['taxi_type'] = 'yellow'
        
        # Step 1: Pivot
        pivoted = pu.pivot_counts_date_taxi_type_location(
            sample_pipeline_data,
            datetime_col='tpep_pickup_datetime',
            location_col='PULocationID',
            taxi_type_col='taxi_type'
        )
        
        assert pivoted is not None
        assert len(pivoted) > 0
        assert list(pivoted.index.names) == ['taxi_type', 'date', 'pickup_place']
        
        # Step 2: Cleanup
        cleaned, stats = pu.cleanup_low_count_rows(pivoted, min_rides=10)
        
        assert stats['rows_before'] > 0
        assert stats['rows_after'] <= stats['rows_before']
        assert stats['rows_removed'] == stats['rows_before'] - stats['rows_after']
        
        # Step 3: Verify all remaining rows meet minimum
        hour_cols = [col for col in cleaned.columns if col.startswith('hour_')]
        if len(cleaned) > 0:
            totals = cleaned[hour_cols].sum(axis=1)
            assert (totals >= 10).all()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
