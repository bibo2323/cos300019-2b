
import numpy as np
import os
import json
import joblib

from sklearn.ensemble import RandomForestRegressor

import ts_common as tc

import pandas as pd


# Train and evaluate the Random Forest model
def train_model():

    # Load the traffic data
    series = tc.load_series()

    # Prepare the scalers
    scalers = tc.fit_site_scalers(series)

    # Create the datasets
    data = tc.build_dataset(series, scalers)

    # Split the datasets
    X_train, y_train, _ = data["train"]
    X_val, y_val, meta_val = data["val"]
    X_test, y_test, meta_test = data["test"]

    # Check the data
    print("Training data:", X_train.shape)
    print("Validation data:", X_val.shape)
    print("Testing data:", X_test.shape)


    # Reshape 3D data into 2D for Random Forest
    X_train_rf = X_train.reshape(X_train.shape[0], -1)
    X_val_rf = X_val.reshape(X_val.shape[0], -1)
    X_test_rf = X_test.reshape(X_test.shape[0], -1)

    # Check the new data shapes
    print("Random Forest Training:", X_train_rf.shape)
    print("Random Forest Validation:", X_val_rf.shape)
    print("Random Forest Testing:", X_test_rf.shape)

    # Create the Random Forest model
    model = RandomForestRegressor(
        n_estimators=60,
        max_depth=16,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )

    # Train the Random Forest model
    print("Training Random Forest...")

    model.fit(X_train_rf, y_train)

    print("Random Forest training completed!")


    # Predict using validation data
    y_val_pred = model.predict(X_val_rf)

    # Display the first five predictions
    print("First 5 predictions:")
    print(y_val_pred[:5])

    # Display the first five actual values
    print("First 5 actual values:")
    print(y_val[:5])


    # Convert scaled values back to actual traffic flow
    y_val_actual = tc.inverse_scale(
        y_val, meta_val["site"], scalers
    )

    y_val_pred_actual = tc.inverse_scale(
        y_val_pred, meta_val["site"], scalers
    )

    # Traffic flow cannot be negative
    y_val_pred_actual = np.clip(y_val_pred_actual, 0, None)

    # Display actual traffic flow
    print("Actual traffic flow:", y_val_actual[:5])
    print("Predicted traffic flow:", y_val_pred_actual[:5])


    # Calculate evaluation metrics
    rmse = np.sqrt(np.mean((y_val_actual - y_val_pred_actual) ** 2))

    mae = np.mean(np.abs(y_val_actual - y_val_pred_actual))

    nrmse = rmse / np.mean(y_val_actual)

    print("Validation RMSE:", round(rmse, 3))
    print("Validation MAE:", round(mae, 3))
    print("Validation NRMSE:", round(nrmse, 4))


    # Predict using testing data
    y_test_pred = model.predict(X_test_rf)

    # Convert scaled values back to actual traffic flow
    y_test_actual = tc.inverse_scale(
        y_test, meta_test["site"], scalers
    )

    y_test_pred_actual = tc.inverse_scale(
        y_test_pred, meta_test["site"], scalers
    )

    # Prevent negative traffic flow predictions
    y_test_pred_actual = np.clip(y_test_pred_actual, 0, None)


    # Calculate testing metrics
    test_rmse = np.sqrt(
        np.mean((y_test_actual - y_test_pred_actual) ** 2)
    )

    test_mae = np.mean(
        np.abs(y_test_actual - y_test_pred_actual)
    )

    test_nrmse = test_rmse / np.mean(y_test_actual)

    print("\n=== Random Forest Test Results ===")
    print("Test RMSE:", round(test_rmse, 3))
    print("Test MAE:", round(test_mae, 3))
    print("Test NRMSE:", round(test_nrmse, 4))


    # Save predictions for model comparison
    output_path = tc.save_predictions(
        "random_forest",
        meta_test,
        y_test_actual,
        y_test_pred_actual
    )

    print("Predictions saved to:", output_path)

    # Create a folder to save the trained model
    os.makedirs("models", exist_ok=True)

    # Save the trained Random Forest model
    joblib.dump(model, "models/random_forest.joblib")

    print("Random Forest model saved successfully!")

    # Save the scaling information for each SCATS site
    scaler_data = {
        str(site): [float(lo), float(hi)]
        for site, (lo, hi) in scalers.items()
    }

    with open("models/random_forest_scalers.json", "w") as file:
        json.dump(scaler_data, file, indent=4)

    print("Random Forest scalers saved successfully!")

# Predict traffic flow for one SCATS site
def predict_flow_15min(site, when):

    # Load the trained Random Forest model
    model = joblib.load("models/random_forest.joblib")

    # Load the saved scalers
    with open("models/random_forest_scalers.json", "r") as file:
        scalers = json.load(file)

    # Load historical traffic data
    series = tc.load_series()

    # Check the SCATS site
    site = int(site)

    if site not in series:
        raise ValueError("Unknown SCATS site")

    # Convert the input into a timestamp
    when = pd.Timestamp(when).floor("15min")

    # Get the previous 12 time intervals
    steps = pd.date_range(
        end=when - tc.STEP,
        periods=tc.LOOKBACK,
        freq="15min"
    )

    traffic = series[site]

    if not steps.isin(traffic.index).all():
        raise ValueError("Not enough historical traffic data")

    # Get the site's scaling values
    lo, hi = scalers[str(site)]

    # Prepare the model input
    past_flow = traffic.loc[steps].values

    scaled_flow = (past_flow - lo) / (hi - lo)

    time_features = tc.time_features(steps)

    features = np.column_stack([
        scaled_flow,
        time_features
    ]).astype(np.float32)

    X_new = features.reshape(1, -1)

    # Predict traffic flow
    prediction_scaled = model.predict(X_new)[0]

    # Convert back to actual vehicle count
    prediction = prediction_scaled * (hi - lo) + lo

    return max(0.0, float(prediction))


# Convert the predicted traffic flow to vehicles per hour
def predict_flow_per_hour(site, when):

    flow_15min = predict_flow_15min(site, when)

    flow_per_hour = flow_15min * 4

    return flow_per_hour

if __name__ == "__main__":
    train_model()
