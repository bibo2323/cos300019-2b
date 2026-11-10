import os
import sys
import unittest
 
import numpy as np
import pandas as pd
 
# let the tests import the files that sit in the repo root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)                                     # data/ paths are relative to the root
 
import evaluation as ev
import ts_common as tc
 
 
class TestLstmPipeline(unittest.TestCase):
    """T19-T20: the data fed to the LSTM and the prediction it returns."""
 
    @classmethod
    def setUpClass(cls):
        series = tc.load_series()
        scalers = tc.fit_site_scalers(series)
        cls.data = tc.build_dataset(series, scalers)
 
    def test_T19_train_val_test_do_not_overlap_in_time(self):
        train_end, val_end = tc.split_dates()
        tr = self.data["train"][2]["timestamp"]
        va = self.data["val"][2]["timestamp"]
        te = self.data["test"][2]["timestamp"]
        self.assertTrue((tr <= train_end).all())
        self.assertTrue(((va > train_end) & (va <= val_end)).all())
        self.assertTrue((te > val_end).all())
        # every sample is 12 past steps (3 h) with 5 features, one target value
        for X, y, _ in self.data.values():
            self.assertEqual(X.shape[1:], (tc.LOOKBACK, 5))
            self.assertEqual(len(X), len(y))
 
    def test_T20_prediction_is_valid_and_hourly_is_four_times_15min(self):
        try:
            import tensorflow  # noqa: F401
        except ImportError:
            self.skipTest("TensorFlow not installed")
        import lstm_model as lm
        if not (os.path.exists(lm.MODEL_FILE) and os.path.exists(lm.SCALER_FILE)):
            self.skipTest("Train the model first: python lstm_model.py")
        when = "2006-10-27 08:00"                 # a Friday morning peak in the data
        q = lm.predict_flow_15min(2000, when)
        self.assertTrue(np.isfinite(q))
        self.assertGreaterEqual(q, 0.0)
        self.assertLess(q, 3000)                   # far above any real 15-min count
        self.assertAlmostEqual(lm.predict_flow_per_hour(2000, when), 4.0 * q, places=4)
 
 
class TestEvaluation(unittest.TestCase):
    """T21-T22: the comparison code gives correct, fair numbers."""
 
    def test_T21_metrics_match_hand_calculation(self):
        # actual = [1, 2, 3, 4], predicted = [2, 2, 4, 4] -> errors -1, 0, -1, 0
        m = ev.metrics([1, 2, 3, 4], [2, 2, 4, 4])
        self.assertAlmostEqual(m["RMSE"], np.sqrt(2 / 4), places=6)      # 0.7071
        self.assertAlmostEqual(m["MAE"], 0.5, places=6)
        self.assertAlmostEqual(m["WAPE_%"], 100 * 2 / 10, places=6)      # 20 %
        self.assertAlmostEqual(m["NRMSE"], np.sqrt(0.5) / 2.5, places=6)
        self.assertAlmostEqual(m["R2"], 1 - 2 / 5, places=6)             # SST = 5
        perfect = ev.metrics([5, 7, 9], [5, 7, 9])
        self.assertEqual(perfect["RMSE"], 0.0)
        self.assertEqual(perfect["R2"], 1.0)
 
    def test_T22_only_rows_predicted_by_every_model_are_compared(self):
        t = pd.date_range("2006-10-25", periods=3, freq="15min")
        rows = []
        for i, ts in enumerate(t):                         # lstm predicts 3 rows
            rows.append(("lstm", 2000, ts, 100.0 + i, 101.0 + i))
        for i, ts in enumerate(t[:2]):                     # gru predicts only 2 rows
            rows.append(("gru", 2000, ts, 100.0 + i, 99.0 + i))
        df = pd.DataFrame(rows, columns=["model", "site", "timestamp", "y_true", "y_pred"])
        out = ev.align(df)
        self.assertEqual(len(out), 4)                      # 2 timestamps x 2 models
        self.assertEqual(out.groupby("model").size().tolist(), [2, 2])
        self.assertNotIn(t[2], set(out["timestamp"]))      # the unmatched row is dropped
 
 
if __name__ == "__main__":
    unittest.main(verbosity=2)