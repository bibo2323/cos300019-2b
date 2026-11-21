"""
COS30019 - Introduction to AI
Assignment 2B: Traffic-Based Route Guidance System

File: tests/test_random_forest.py
Owner: Siu Han

Description:
    Contains automated unit tests for the Random Forest
    traffic prediction model.

    Test Cases:
        T36 - Valid traffic flow prediction
        T37 - Hourly traffic flow conversion
        T38 - Insufficient historical traffic data
        T39 - Invalid SCATS site

    Uses Python's unittest framework to verify model
    prediction behaviour and input handling.
"""
import os
import sys
import unittest
import math

# Allow tests to import files from the project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


from random_forest_model import predict_flow_15min, predict_flow_per_hour


class TestRandomForest(unittest.TestCase):

    # T36: Test prediction with valid input
    def test_T36_valid_prediction(self):

        prediction = predict_flow_15min(
            2000,
            "2006-10-28 08:30"
        )

        self.assertTrue(math.isfinite(prediction))
        self.assertGreaterEqual(prediction, 0)

        self.assertAlmostEqual(
            prediction,
            683.56,
            delta=0.02
        )

    
    # T37: Test hourly traffic flow conversion
    def test_T37_hourly_conversion(self):

        flow_15min = predict_flow_15min(
            2000,
            "2006-10-28 08:30"
        )

        flow_hourly = predict_flow_per_hour(
            2000,
            "2006-10-28 08:30"
        )

        self.assertAlmostEqual(
            flow_hourly,
            flow_15min * 4,
            places=4
        )


    # T38: Test prediction with insufficient historical data
    def test_T38_insufficient_historical_data(self):

        with self.assertRaises((ValueError, KeyError)):

            predict_flow_15min(
                2000,
                "2026-11-12 08:30"
            )


    # T39: Test prediction with invalid SCATS site
    def test_T39_invalid_scats_site(self):

        with self.assertRaises((ValueError, KeyError)):

            predict_flow_15min(
                999999,
                "2006-10-28 08:30"
            )



if __name__ == "__main__":
    unittest.main(verbosity=2)
