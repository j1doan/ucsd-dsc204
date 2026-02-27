"""
test_pivot_date_location_hour.py

Unit tests for Part 5 (Testing): column detection, pivot shape/values,
error handling, month inference, and low-count cleanup.
"""

import unittest
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Add repo root to path
sys.path.insert(0, str(Path(__file__).parent))

from pivot_utils import (
    cleanup_low_count_rows,
    find_pickup_datetime_col,
    find_pickup_location_col,
    infer_month_from_path,
    infer_taxi_type_from_path,
    pivot_counts_date_taxi_type_location,
)


class TestColumnDetectionVariants(unittest.TestCase):
    """Test column detection for common naming variants."""

    def test_find_pickup_datetime_variants(self):
        columns = ["VendorID", "tpep_pickup_datetime", "fare_amount"]
        self.assertEqual(find_pickup_datetime_col(columns), "tpep_pickup_datetime")

        columns = ["vendor_id", "pickup_datetime", "trip_distance"]
        self.assertEqual(find_pickup_datetime_col(columns), "pickup_datetime")

        columns = ["vendor", "PickupDateTime", "Amount"]
        self.assertEqual(find_pickup_datetime_col(columns), "PickupDateTime")

    def test_find_pickup_location_variants(self):
        columns = ["VendorID", "PULocationID", "fare_amount"]
        self.assertEqual(find_pickup_location_col(columns), "PULocationID")

        columns = ["pu_location", "do_location", "trip_distance"]
        self.assertEqual(find_pickup_location_col(columns), "pu_location")

        columns = ["pickup_zone", "dropoff_zone", "fare_amount"]
        self.assertEqual(find_pickup_location_col(columns), "pickup_zone")


class TestMonthInference(unittest.TestCase):
    """Test month inference from common path patterns."""

    def test_infer_month_patterns(self):
        self.assertEqual(
            infer_month_from_path("yellow_tripdata_2023-01.parquet"), (2023, 1)
        )
        self.assertEqual(
            infer_month_from_path("s3://bucket/year=2023/month=05/data.parquet"),
            (2023, 5),
        )
        self.assertEqual(infer_month_from_path("data_202312.parquet"), (2023, 12))

    def test_infer_month_none(self):
        self.assertIsNone(infer_month_from_path("data_xyz.parquet"))


class TestTaxiTypeInference(unittest.TestCase):
    """Test taxi type inference from paths."""

    def test_infer_taxi_type(self):
        self.assertEqual(
            infer_taxi_type_from_path("yellow_tripdata_2023-01.parquet"), "yellow"
        )
        self.assertEqual(
            infer_taxi_type_from_path("/data/green/tripdata.parquet"), "green"
        )
        self.assertIsNone(infer_taxi_type_from_path("data.parquet"))


class TestPivotOutput(unittest.TestCase):
    """Test pivot output shape, index, and values."""

    def setUp(self):
        base = datetime(2023, 1, 1, 0, 0, 0)
        self.df = pd.DataFrame(
            {
                "tpep_pickup_datetime": [
                    base,
                    base + timedelta(hours=0),
                    base + timedelta(hours=5),
                    base + timedelta(hours=5),
                ],
                "PULocationID": [1, 1, 1, 2],
                "taxi_type": ["yellow"] * 4,
            }
        )

    def test_pivot_shape_and_index(self):
        pivoted = pivot_counts_date_taxi_type_location(
            self.df,
            datetime_col="tpep_pickup_datetime",
            location_col="PULocationID",
            taxi_type_col="taxi_type",
        )
        self.assertEqual(list(pivoted.index.names), ["taxi_type", "date", "pickup_place"])
        hour_cols = [col for col in pivoted.columns if col.startswith("hour_")]
        self.assertEqual(len(hour_cols), 24)

    def test_pivot_counts(self):
        pivoted = pivot_counts_date_taxi_type_location(self.df)
        key = ("yellow", self.df["tpep_pickup_datetime"].dt.date.iloc[0], 1)
        self.assertEqual(int(pivoted.loc[key, "hour_0"]), 2)
        self.assertEqual(int(pivoted.loc[key, "hour_5"]), 1)

    def test_pivot_non_negative_values(self):
        pivoted = pivot_counts_date_taxi_type_location(self.df)
        for col in pivoted.columns:
            self.assertTrue((pivoted[col] >= 0).all())


class TestErrorHandling(unittest.TestCase):
    """Test error handling for missing columns and bad data."""

    def test_missing_datetime_column(self):
        df = pd.DataFrame(
            {
                "PULocationID": [1, 2, 3],
                "taxi_type": ["yellow", "yellow", "yellow"],
            }
        )
        with self.assertRaises(ValueError):
            pivot_counts_date_taxi_type_location(df)

    def test_missing_location_column(self):
        df = pd.DataFrame(
            {
                "tpep_pickup_datetime": pd.date_range("2023-01-01", periods=3),
                "taxi_type": ["yellow", "yellow", "yellow"],
            }
        )
        with self.assertRaises(ValueError):
            pivot_counts_date_taxi_type_location(df)

    def test_missing_taxi_type_column(self):
        df = pd.DataFrame(
            {
                "tpep_pickup_datetime": pd.date_range("2023-01-01", periods=3),
                "PULocationID": [1, 2, 3],
            }
        )
        with self.assertRaises(ValueError):
            pivot_counts_date_taxi_type_location(df)

    def test_bad_datetime_rows_ignored(self):
        df = pd.DataFrame(
            {
                "tpep_pickup_datetime": ["2023-01-01 00:00:00", "not_a_date"],
                "PULocationID": [1, 1],
                "taxi_type": ["yellow", "yellow"],
            }
        )
        pivoted = pivot_counts_date_taxi_type_location(df)
        total = int(pivoted.sum(axis=1).sum())
        self.assertEqual(total, 1)


class TestCleanupLowCountRows(unittest.TestCase):
    """Test cleanup for rows with fewer than min_rides."""

    def test_cleanup_removes_low_count_rows(self):
        index = pd.MultiIndex.from_tuples(
            [
                ("yellow", "2023-01-01", 1),
                ("yellow", "2023-01-01", 2),
            ],
            names=["taxi_type", "date", "pickup_place"],
        )
        hours = [f"hour_{i}" for i in range(24)]
        data = {h: [0, 0] for h in hours}
        data["hour_0"] = [60, 10]
        df = pd.DataFrame(data, index=index)

        cleaned, stats = cleanup_low_count_rows(df, min_rides=50)
        self.assertEqual(stats["rows_before"], 2)
        self.assertEqual(stats["rows_after"], 1)
        self.assertEqual(stats["rows_removed"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)