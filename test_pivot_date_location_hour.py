"""
test_pivot_date_location_hour.py

Unit tests for pivot utilities.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from pivot_utils import (
    find_pickup_datetime_col,
    find_pickup_location_col,
    infer_taxi_type_from_path,
    infer_month_from_path,
    pivot_counts_date_taxi_type_location,
    cleanup_low_count_rows,
)


class TestColumnDetection(unittest.TestCase):
    """Test column detection functions."""
    
    def test_find_pickup_datetime_col_standard(self):
        """Test standard datetime column name."""
        columns = ['VendorID', 'tpep_pickup_datetime', 'fare_amount']
        result = find_pickup_datetime_col(columns)
        self.assertEqual(result, 'tpep_pickup_datetime')
    
    def test_find_pickup_datetime_col_variant(self):
        """Test variant datetime column name."""
        columns = ['vendor_id', 'pickup_datetime', 'trip_distance']
        result = find_pickup_datetime_col(columns)
        self.assertEqual(result, 'pickup_datetime')
    
    def test_find_pickup_location_col(self):
        """Test pickup location column detection."""
        columns = ['VendorID', 'tpep_pickup_datetime', 'PUlocationID', 'fare_amount']
        result = find_pickup_location_col(columns)
        self.assertIn(result, ['PUlocationID', 'pulocationid'])


class TestMonthInference(unittest.TestCase):
    """Test month inference from paths."""
    
    def test_infer_month_standard_format(self):
        """Test standard YYYY-MM format."""
        result = infer_month_from_path('yellow_tripdata_2023-01.parquet')
        self.assertEqual(result, (2023, 1))
    
    def test_infer_month_s3_partitioned(self):
        """Test S3 partitioned format."""
        result = infer_month_from_path('s3://bucket/year=2023/month=05/data.parquet')
        self.assertEqual(result, (2023, 5))
    
    def test_infer_month_yyyymm_format(self):
        """Test YYYYMM format."""
        result = infer_month_from_path('data_202312.parquet')
        self.assertEqual(result, (2023, 12))
    
    def test_infer_month_none(self):
        """Test non-matching path."""
        result = infer_month_from_path('data_xyz.parquet')
        self.assertIsNone(result)


class TestTaxiTypeInference(unittest.TestCase):
    """Test taxi type inference from paths."""
    
    def test_infer_yellow(self):
        """Test yellow taxi inference."""
        result = infer_taxi_type_from_path('yellow_tripdata_2023-01.parquet')
        self.assertEqual(result, 'yellow')
    
    def test_infer_green(self):
        """Test green taxi inference."""
        result = infer_taxi_type_from_path('/data/green/tripdata.parquet')
        self.assertEqual(result, 'green')
    
    def test_infer_none(self):
        """Test no match."""
        result = infer_taxi_type_from_path('data.parquet')
        self.assertIsNone(result)


class TestPivoting(unittest.TestCase):
    """Test pivoting functionality."""
    
    def setUp(self):
        """Create sample data for testing."""
        dates = pd.date_range('2023-01-01', periods=10, freq='H')
        self.df = pd.DataFrame({
            'tpep_pickup_datetime': dates * 2,  # Repeat to get 20 rows
            'PULocationID': [1, 2] * 10,
            'taxi_type': ['yellow'] * 20,
        })
        self.df = self.df.head(20)
    
    def test_pivot_shape_and_index(self):
        """Test that pivot produces correct shape and index."""
        result = pivot_counts_date_taxi_type_location(
            self.df,
            datetime_col='tpep_pickup_datetime',
            location_col='PULocationID',
            taxi_type_col='taxi_type'
        )
        
        # Check index names
        self.assertEqual(list(result.index.names), ['taxi_type', 'date', 'pickup_place'])
        
        # Check hour columns exist
        hour_cols = [col for col in result.columns if col.startswith('hour_')]
        self.assertEqual(len(hour_cols), 24)
    
    def test_pivot_values_non_negative(self):
        """Test that all pivot values are non-negative."""
        result = pivot_counts_date_taxi_type_location(
            self.df,
            datetime_col='tpep_pickup_datetime',
            location_col='PULocationID',
            taxi_type_col='taxi_type'
        )
        
        for col in result.columns:
            self.assertTrue((result[col] >= 0).all())


class TestCleanup(unittest.TestCase):
    """Test low-count row cleanup."""
    
    def setUp(self):
        """Create sample pivoted data."""
        index = pd.MultiIndex.from_tuples([
            ('yellow', '2023-01-01', 1),
            ('yellow', '2023-01-01', 2),
            ('yellow', '2023-01-02', 1),
        ], names=['taxi_type', 'date', 'pickup_place'])
        
        hours = [f'hour_{i}' for i in range(24)]
        data = {h: [0] * 3 for h in hours}
        data['hour_0'] = [60, 30, 10]  # 60, 30, 10 total rides
        
        self.df = pd.DataFrame(data, index=index)
    
    def test_cleanup_removes_low_count_rows(self):
        """Test that rows with fewer rides are removed."""
        cleaned, stats = cleanup_low_count_rows(self.df, min_rides=50)
        
        self.assertEqual(stats['rows_before'], 3)
        self.assertEqual(stats['rows_after'], 1)
        self.assertEqual(stats['rows_removed'], 2)


if __name__ == '__main__':
    unittest.main()