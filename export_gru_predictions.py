import json
import tensorflow as tf
import ts_common as tc

# Load the trained GRU model
model = tf.keras.models.load_model("models/gru.keras")

# Load the saved scalers
with open("models/gru_scalers.json") as f:
    scalers = {
        int(site): tuple(values)
        for site, values in json.load(f).items()
    }

# Prepare the same testing dataset
series = tc.load_series()
dataset = tc.build_dataset(series, scalers)

X_test, y_test, meta_test = dataset["test"]

# Predict using the existing model (no retraining)
pred_scaled = model.predict(
    X_test,
    batch_size=1024,
    verbose=0
).ravel()

# Convert predictions back to vehicles/15 min
y_pred = tc.inverse_scale(
    pred_scaled, meta_test["site"], scalers
)

y_true = tc.inverse_scale(
    y_test, meta_test["site"], scalers
)

# Save GRU testing predictions
path = tc.save_predictions(
    "gru", meta_test, y_true, y_pred
)

print("GRU predictions saved to:", path)
print("Total test samples:", len(y_true))