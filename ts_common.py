import os
 
import numpy as np
import pandas as pd
 
from data_processing import load_15min
 
# ---- settings (change here only, tell the whole team) ----------------------
LOOKBACK = 12                    # 12 x 15 min = the previous 3 hours are the input
HORIZON_MIN = 15                 # we predict the NEXT 15-minute interval
TRAIN_END = "2006-10-21"         # train: 1 Oct - 21 Oct   (21 days)
VAL_END = "2006-10-24"           # validation: 22 - 24 Oct (3 days, early stopping)
                                 # test: 25 Oct - 31 Oct   (7 days, never used for training)
RESULTS_DIR = "results"
SEED = 42
# -----------------------------------------------------------------------------
 
STEP = pd.Timedelta(minutes=15)
 
 
def split_dates():
    """Return (train_end, val_end) as Timestamps (end of that day, inclusive)."""
    end = pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return pd.Timestamp(TRAIN_END) + end, pd.Timestamp(VAL_END) + end
 
 
def load_series(out_dir=None):
    """Dict {site: pandas Series of flow_15min indexed by timestamp, sorted}.
    A few sites are missing some days in the raw file, so the series are NOT
    always continuous - make_windows() skips windows that cross a gap."""
    df = load_15min() if out_dir is None else load_15min(out_dir)
    return {int(s): g.set_index("timestamp")["flow_15min"].astype(float).sort_index()
            for s, g in df.groupby("site")}
 
 
def time_features(index):
    """4 columns: sin/cos of time of day, sin/cos of day of week.
    Sin/cos keeps 23:45 close to 00:00 and Sunday close to Monday."""
    idx = pd.DatetimeIndex(index)
    tod = (idx.hour * 60 + idx.minute) / (24 * 60)
    dow = idx.dayofweek / 7.0
    return np.column_stack([np.sin(2 * np.pi * tod), np.cos(2 * np.pi * tod),
                            np.sin(2 * np.pi * dow), np.cos(2 * np.pi * dow)])
 
 
def make_windows(series, lookback=LOOKBACK):
    """Sliding windows over ONE site's series.
 
    Returns
        X_idx   (n, lookback) integer positions of the input steps
        y_idx   (n,)          integer position of the target step
        t_target (n,)         Timestamp of the target step
    Only windows whose lookback+1 timestamps are exactly 15 minutes apart are
    kept (so a missing day never gets glued together)."""
    ts = series.index.values.astype("datetime64[m]").astype(np.int64)
    n = len(ts)
    if n <= lookback:
        return (np.empty((0, lookback), int), np.empty(0, int),
                pd.DatetimeIndex([]))
    ends = np.arange(lookback, n)                       # target positions
    ok = (ts[ends] - ts[ends - lookback]) == lookback * 15   # contiguous span
    ends = ends[ok]
    x_idx = ends[:, None] - np.arange(lookback, 0, -1)[None, :]
    return x_idx, ends, series.index[ends]
 
 
def subset_by_date(t_target, start=None, end=None):
    """Boolean mask of targets with start < t <= end (None = open)."""
    m = np.ones(len(t_target), dtype=bool)
    if start is not None:
        m &= np.asarray(t_target > start)
    if end is not None:
        m &= np.asarray(t_target <= end)
    return m
 
 
def fit_site_scalers(series_by_site):
    """Min-max scaler per site, fitted on TRAIN rows only (no test leakage).
    Returns {site: (min, max)}."""
    train_end, _ = split_dates()
    out = {}
    for s, ser in series_by_site.items():
        tr = ser[ser.index <= train_end]
        lo, hi = float(tr.min()), float(tr.max())
        out[s] = (lo, hi if hi > lo else lo + 1.0)
    return out
 
 
def build_dataset(series_by_site, scalers, lookback=LOOKBACK):
    """Make X/y for every site and split into train / val / test.
 
    X has shape (samples, lookback, 5): scaled flow + 4 time features.
    y is the scaled flow of the next interval.
    Also returns meta (site, timestamp) per sample for test, so predictions can
    be written out with the right labels.
 
    Returns dict with keys 'train', 'val', 'test', each = (X, y, meta_df)."""
    train_end, val_end = split_dates()
    buckets = {k: ([], [], []) for k in ("train", "val", "test")}
    for site, ser in series_by_site.items():
        lo, hi = scalers[site]
        scaled = ((ser.values - lo) / (hi - lo)).astype(np.float32)
        tf = time_features(ser.index).astype(np.float32)
        feats = np.column_stack([scaled, tf])                   # (T, 5)
        x_idx, y_idx, t_tgt = make_windows(ser, lookback)
        if len(y_idx) == 0:
            continue
        X = feats[x_idx]                                        # (n, lookback, 5)
        y = scaled[y_idx]
        masks = {"train": subset_by_date(t_tgt, None, train_end),
                 "val": subset_by_date(t_tgt, train_end, val_end),
                 "test": subset_by_date(t_tgt, val_end, None)}
        for k, m in masks.items():
            buckets[k][0].append(X[m])
            buckets[k][1].append(y[m])
            buckets[k][2].append(pd.DataFrame({"site": site, "timestamp": t_tgt[m]}))
    out = {}
    for k, (xs, ys, ms) in buckets.items():
        out[k] = (np.concatenate(xs), np.concatenate(ys),
                  pd.concat(ms, ignore_index=True))
    return out
 
 
def inverse_scale(y_scaled, sites, scalers):
    """Back to vehicles per 15 min. `sites` is the per-sample site column."""
    lo = np.array([scalers[int(s)][0] for s in sites])
    hi = np.array([scalers[int(s)][1] for s in sites])
    return np.asarray(y_scaled, dtype=float).ravel() * (hi - lo) + lo
 
 
def save_predictions(model_name, meta, y_true, y_pred, results_dir=RESULTS_DIR):
    """Write results/predictions_<model_name>.csv (the file evaluation.py reads)."""
    os.makedirs(results_dir, exist_ok=True)
    out = meta.copy()
    out.insert(0, "model", model_name)
    out["y_true"] = np.asarray(y_true, dtype=float).ravel()
    out["y_pred"] = np.clip(np.asarray(y_pred, dtype=float).ravel(), 0, None)
    path = os.path.join(results_dir, "predictions_%s.csv" % model_name)
    out.to_csv(path, index=False)
    return path