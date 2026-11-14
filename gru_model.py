"""
gru_model.py - COS30019 Assignment 2B (TBRGS).  Owner: Emily Sheryllina Aren Standly

"""
import argparse
import json
import os
import sys
 
import numpy as np
import pandas as pd
 
import ts_common as tc
 
MODEL_NAME = "gru"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "gru.keras")
SCALER_FILE = os.path.join(MODEL_DIR, "gru_scalers.json")
HISTORY_FILE = os.path.join(tc.RESULTS_DIR, "gru_history.csv")
 
UNITS = 64
DROPOUT = 0.2
DENSE_UNITS = 32
LEARNING_RATE = 1e-3
BATCH_SIZE = 128
EPOCHS = 60
PATIENCE = 8                    

 
 
def _keras():
    """Import TensorFlow only when needed, with a clear message if missing."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        return tf, keras
    except ImportError:
        sys.exit("TensorFlow is not installed. Run:  pip install tensorflow")
 
 
def build_model(n_features, units=UNITS, dropout=DROPOUT, dense_units=DENSE_UNITS,
                lr=LEARNING_RATE, lookback=tc.LOOKBACK):
    """Input (lookback, n_features) -> GRU -> Dropout -> Dense -> 1 value."""
    _, keras = _keras()
    layers = keras.layers
    model = keras.Sequential([
        keras.Input(shape=(lookback, n_features)),
        layers.GRU(units),
        layers.Dropout(dropout), #avoid having to memorise
        layers.Dense(dense_units, activation="relu"),
        layers.Dense(1),     #scaled flow of next interval              
    ], name="gru_flow")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse",
                  metrics=[keras.metrics.RootMeanSquaredError(name="rmse")])
    return model
 
 
def train(epochs=EPOCHS, units=UNITS, batch_size=BATCH_SIZE):
    tf, keras = _keras()
    keras.utils.set_random_seed(tc.SEED)
 
    series = tc.load_series()
    scalers = tc.fit_site_scalers(series)
    data = tc.build_dataset(series, scalers)
    X_tr, y_tr, _ = data["train"]
    X_va, y_va, _ = data["val"]
    X_te, y_te, meta_te = data["test"]
    print("samples  train %d | val %d | test %d | input shape %s"
          % (len(X_tr), len(X_va), len(X_te), X_tr.shape[1:]))
 
    model = build_model(X_tr.shape[2], units=units)
    model.summary()
  #60 epochs maximum
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                      restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                          patience=3, min_lr=1e-5),
    ]
    hist = model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs,
                     batch_size=batch_size, shuffle=True, callbacks=callbacks,
                     verbose=2)
 
    # save model, scalers, training history
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(tc.RESULTS_DIR, exist_ok=True)
    model.save(MODEL_FILE)
    with open(SCALER_FILE, "w") as f:
        json.dump({str(s): list(v) for s, v in scalers.items()}, f)
    pd.DataFrame(hist.history).rename_axis("epoch").reset_index().to_csv(
        HISTORY_FILE, index=False)
 
    # test predictions in real units :vehicles in 15 mins
    pred_scaled = model.predict(X_te, batch_size=1024, verbose=0).ravel()
    y_pred = tc.inverse_scale(pred_scaled, meta_te["site"], scalers)
    y_true = tc.inverse_scale(y_te, meta_te["site"], scalers)
    path = tc.save_predictions(MODEL_NAME, meta_te, y_true, y_pred)
 
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    print("\nGRU test RMSE = %.2f vehicles/15 min | MAE = %.2f" % (rmse, mae))
    print("Best epoch: %d of %d run" % (int(np.argmin(hist.history["val_loss"])) + 1,
                                         len(hist.history["loss"])))
    print("Saved: %s, %s, %s, %s" % (MODEL_FILE, SCALER_FILE, HISTORY_FILE, path))
    return model
 
 
# prediction for rest of system 
_cache = {}
 
 
def _load_for_prediction():
    if not _cache:
        _, keras = _keras()
        if not (os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE)):
            raise FileNotFoundError("Train first:  python gru_model.py")
        _cache["model"] = keras.models.load_model(MODEL_FILE)
        with open(SCALER_FILE) as f:
            _cache["scalers"] = {int(k): tuple(v) for k, v in json.load(f).items()}
        _cache["series"] = tc.load_series()
    return _cache
 
 
def predict_flow_15min(site, when):
    #predict vehicles in 15 minutes int
    
    c = _load_for_prediction()
    site = int(site)
    if site not in c["series"]:
        raise ValueError("Unknown SCATS site %s" % site)
    ser = c["series"][site]
    when = pd.Timestamp(when).floor("15min")
    steps = pd.date_range(end=when - tc.STEP, periods=tc.LOOKBACK, freq="15min")
    if not steps.isin(ser.index).all():
        raise ValueError("No data for site %s in the 3 hours before %s" % (site, when))
    lo, hi = c["scalers"][site]
    past = ser.loc[steps]
    # features of each input step: scaled flow + time features of that step
    feats = np.column_stack([(past.values - lo) / (hi - lo),
                             tc.time_features(steps)]).astype(np.float32)
    pred = float(c["model"].predict(feats[None, :, :], verbose=0)[0, 0])
    return max(0.0, pred * (hi - lo) + lo)
 
 
def predict_flow_per_hour(site, when):
   #4 multiply 15 mins prediction to get vehicles per hour 
    return 4.0 * predict_flow_15min(site, when)
 
 
def main():
    ap = argparse.ArgumentParser(description="Train the GRU traffic-flow model")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--units", type=int, default=UNITS)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE)
    args = ap.parse_args()
    train(epochs=args.epochs, units=args.units, batch_size=args.batch)
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())