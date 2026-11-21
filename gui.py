"""
COS30019 - Introduction to AI
Assignment 2B: Traffic-Based Route Guidance System

File: gui.py
Owner: Siu Han

Description:
    Implements the graphical user interface (GUI) for the
    Traffic-Based Route Guidance System (TBRGS).

    Allows users to select SCATS sites, departure date and time,
    prediction models, and the number of recommended routes.

    Integrates the team's machine learning models and route
    search module to display predicted traffic flow, Top-K
    routes, estimated travel times, and distances.

    Includes input validation, error handling, background
    threading, and a scrollable results display.
"""
import tkinter as tk

from tkinter import ttk

from tkinter import messagebox
from datetime import datetime
from data_processing import load_sites

from random_forest_model import predict_flow_15min
from lstm_model import predict_flow_15min as predict_lstm_flow
from gru_model import predict_flow_15min as predict_gru_flow

from tbrgs_routing import find_routes, format_routes
from map_routes import generate_route_map

import json
from pathlib import Path

from road_map import build_adjacency, connected_components
import threading
import webbrowser


# Load default settings from configuration file
config_path = Path(__file__).resolve().parent / "config.json"

with open(config_path, "r", encoding="utf-8") as file:
    config = json.load(file)

# Create the main window
root = tk.Tk()

# Set the window title
root.title("Traffic-Based Route Guidance System")

# Set the window size
root.geometry("800x700")

# Store the latest successful route result
last_route_result = None

# Add a title label
title_label = tk.Label(
    root,
    text="Traffic-Based Route Guidance System",
    font=("Arial", 18, "bold")
)

title_label.pack(pady=20)

# Create a frame for user inputs
input_frame = tk.Frame(root)
input_frame.pack(pady=20)

# Origin label
origin_label = tk.Label(
    input_frame,
    text="Origin SCATS Site:",
    font=("Arial", 12)
)
origin_label.grid(row=0, column=0, padx=10, pady=10)

# Origin input box
origin_entry = tk.Entry(input_frame, font=("Arial", 12))
origin_entry.grid(row=0, column=1, padx=10, pady=10)
origin_entry.insert(0, str(config["origin"]))

# Destination label
destination_label = tk.Label(
    input_frame,
    text="Destination SCATS Site:",
    font=("Arial", 12)
)
destination_label.grid(row=1, column=0, padx=10, pady=10)

# Destination input box
destination_entry = tk.Entry(input_frame, font=("Arial", 12))
destination_entry.grid(row=1, column=1, padx=10, pady=10)
destination_entry.insert(0, str(config["destination"]))

# Departure Date label
date_label = tk.Label(
    input_frame,
    text="Departure Date (YYYY-MM-DD):",
    font=("Arial", 12)
)
date_label.grid(row=2, column=0, padx=10, pady=10)

# Departure Date input box
date_entry = tk.Entry(input_frame, font=("Arial", 12))
date_entry.grid(row=2, column=1, padx=10, pady=10)
date_entry.insert(0, config["departure_date"])

# Departure Time label
time_label = tk.Label(
    input_frame,
    text="Departure Time (HH:MM):",
    font=("Arial", 12)
)
time_label.grid(row=3, column=0, padx=10, pady=10)

# Departure Time input box
time_entry = tk.Entry(input_frame, font=("Arial", 12))
time_entry.grid(row=3, column=1, padx=10, pady=10)
time_entry.insert(0, config["departure_time"])

# Prediction Model label
model_label = tk.Label(
    input_frame,
    text="Prediction Model:",
    font=("Arial", 12)
)
model_label.grid(row=4, column=0, padx=10, pady=10)

# Prediction Model dropdown menu
model_combo = ttk.Combobox(
    input_frame,
    values=["LSTM", "GRU", "Random Forest"],
    font=("Arial", 12),
    state="readonly"
)

model_combo.grid(row=4, column=1, padx=10, pady=10)

# Set Random Forest as the default model
model_combo.set(config["prediction_model"])

# Top K Routes label
routes_label = tk.Label(
    input_frame,
    text="Number of Routes (Top K):",
    font=("Arial", 12)
)
routes_label.grid(row=5, column=0, padx=10, pady=10)

# Top K Routes dropdown
routes_combo = ttk.Combobox(
    input_frame,
    values=[1, 2, 3, 4, 5],
    font=("Arial", 12),
    state="readonly"
)
routes_combo.grid(row=5, column=1, padx=10, pady=10)

# Set default number of routes
routes_combo.set(config["top_k"])

# Function to read user inputs
def get_user_inputs():

    # Clear old results before checking new inputs
    clear_prediction_result()

    # Get origin and destination
    origin = origin_entry.get()
    destination = destination_entry.get()

    # Get departure date and time
    date = date_entry.get()
    time = time_entry.get()

    # Get selected model and number of routes
    selected_model = model_combo.get()
    top_k = routes_combo.get()

    # Validate origin and destination
    try:
        origin = int(origin)
        destination = int(destination)
    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Origin and Destination must be numbers."
        )
        return

    # Check whether the SCATS sites exist
    sites = load_sites()
    valid_sites = set(sites["site"].astype(int))

    if origin not in valid_sites or destination not in valid_sites:
        messagebox.showerror(
            "Invalid SCATS Site",
            "One or both SCATS sites do not exist."
        )
        return

    # Validate departure date and time
    try:
        departure = datetime.strptime(
            date + " " + time,
            "%Y-%m-%d %H:%M"
        )
    except ValueError:
        messagebox.showerror(
            "Invalid Date or Time",
            "Use YYYY-MM-DD for date and HH:MM for time."
        )
        return

    # Check that time matches a 15-minute interval
    if departure.minute % 15 != 0:
        messagebox.showerror(
            "Invalid Time",
            "Please select a 15-minute interval."
        )
        return

    # Validate Top K Routes
    try:
        top_k = int(top_k)
    except ValueError:
        messagebox.showerror(
            "Invalid Top K",
            "Number of routes must be between 1 and 5."
        )
        return

    if top_k < 1 or top_k > 5:
        messagebox.showerror(
            "Invalid Top K",
            "Number of routes must be between 1 and 5."
        )
        return

    # Display inputs in the terminal
    print("\n=== User Inputs ===")
    print("Origin:", origin)
    print("Destination:", destination)
    print("Departure Date:", date)
    print("Departure Time:", time)
    print("Prediction Model:", selected_model)
    print("Top K Routes:", top_k)


    # Select prediction model
    if selected_model == "Random Forest":
        model_key = "random_forest"
        predictor = predict_flow_15min

    elif selected_model == "LSTM":
        model_key = "lstm"
        predictor = predict_lstm_flow

    elif selected_model == "GRU":
        model_key = "gru"
        predictor = predict_gru_flow

    else:
        messagebox.showerror(
            "Invalid Model",
            "Please select a valid prediction model."
        )
        return

    # Combine departure date and time
    when = departure.strftime("%Y-%m-%d %H:%M")

    # Disable the button during prediction
    check_button.config(state="disabled", text="Predicting...")

    # Show loading message
    result_box.config(state="normal")
    result_box.delete("1.0", tk.END)
    result_box.insert("1.0", "Calculating traffic and Top-K routes...")
    result_box.config(state="disabled")


    # Run prediction in a background thread
    def prediction_task():
        try:
            # Predict traffic flow
            predicted_flow = predictor(origin, when)

            # Find Top-K routes
            route_result = find_routes(
                origin=origin,
                destination=destination,
                when=when,
                k=top_k,
                model=model_key
            )

            # Prepare result text
            result_text = (
                f"Origin: {origin}\n"
                f"Destination: {destination}\n"
                f"Departure Time: {when}\n"
                f"Prediction Model: {selected_model}\n"
                f"Predicted Traffic Flow: {round(predicted_flow)} vehicles/15 min\n\n"
                f"=== Top {top_k} Routes ===\n"
                f"{format_routes(route_result)}"
            )

            root.after(
            0,
            lambda: show_prediction_result(result_text, route_result)
        )

        except Exception as error:
            error_message = str(error)
            root.after(
                0,
                lambda: show_prediction_error(error_message)
            )


    # Start background thread
    threading.Thread(
        target=prediction_task,
        daemon=True
    ).start()

def show_prediction_result(result_text, route_result):
    global last_route_result

    # Save the latest successful Top-K route result
    last_route_result = route_result

    # Display prediction results
    result_box.config(state="normal")
    result_box.delete("1.0", tk.END)
    result_box.insert("1.0", result_text)
    result_box.config(state="disabled")

    # Restore the Predict Traffic button
    check_button.config(state="normal", text="Predict Traffic")

        # Enable View Map when routes are available
    if route_result.get("routes"):
        view_map_button.config(state="normal")
    else:
        view_map_button.config(state="disabled")


def show_prediction_error(error_message):
    # Clear the loading message
    clear_prediction_result()

    # Restore the predict button
    check_button.config(
        state="normal",
        text="Predict Traffic"
    )

    # Display the error message
    messagebox.showerror(
        "Prediction Error",
        error_message
    )

# Clear previous prediction results
def clear_prediction_result():
    global last_route_result

    # Clear the previous route data
    last_route_result = None

    # Disable View Map
    view_map_button.config(state="disabled")

    # Clear previous prediction results
    result_box.config(state="normal")
    result_box.delete("1.0", tk.END)
    result_box.insert("1.0", "No valid prediction available.")
    result_box.config(state="disabled")

# Open the latest predicted routes on an interactive map
def open_route_map():
    if last_route_result is None or not last_route_result.get("routes"):
        messagebox.showwarning(
            "No Routes",
            "Please predict a route first."
        )
        return

    try:
        # Generate the interactive HTML map
        html_file = generate_route_map(last_route_result)

        # Convert the file path to a browser URL
        map_url = Path(html_file).resolve().as_uri()

        # Open the map in the default browser
        webbrowser.open(map_url)

    except Exception as error:
        messagebox.showerror(
            "Map Error",
            str(error)
        )

# Check whether two SCATS sites are connected
def check_route_connection():
    try:
        origin = int(origin_entry.get())
        destination = int(destination_entry.get())

    except ValueError:
        messagebox.showerror(
            "Invalid Input",
            "Origin and Destination must be numbers."
        )
        return

    # Load the road network
    adjacency = build_adjacency()

    # Check whether the sites exist
    if origin not in adjacency or destination not in adjacency:
        messagebox.showerror(
            "Invalid SCATS Site",
            "One or both sites do not exist in the road network."
        )
        return

    # Check whether origin and destination are the same
    if origin == destination:
        messagebox.showinfo(
            "Route Check",
            "Origin and Destination are the same site."
        )
        return

    # Find connected groups of SCATS sites
    components = connected_components(adjacency)

    # Check if both sites belong to the same group
    is_connected = any(
        origin in group and destination in group
        for group in components
    )

    if is_connected:
        messagebox.showinfo(
            "Route Check",
            f"SCATS {origin} and {destination} are connected "
            "in the current road network."
        )
    else:
        messagebox.showwarning(
            "No Connection",
            "No route exists between these sites "
            "in the current road network."
        )

# Create button to check user inputs
check_button = tk.Button(
    root,
    text="Predict Traffic",
    font=("Arial", 12, "bold"),
    command=get_user_inputs,
    width=20
)

check_button.pack(pady=20)

# Button to check road connection
route_button = tk.Button(
    root,
    text="Check Route Connection",
    font=("Arial", 12, "bold"),
    command=check_route_connection,
    width=22
)

route_button.pack(pady=5)

# Button to open the interactive route map
view_map_button = tk.Button(
    root,
    text="View Map",
    font=("Arial", 12, "bold"),
    command=open_route_map,
    width=22,
    state="disabled"
)

view_map_button.pack(pady=5)

# Create a frame for displaying prediction results
result_frame = tk.LabelFrame(
    root,
    text="Prediction Results",
    font=("Arial", 12, "bold"),
    padx=15,
    pady=15
)

result_frame.pack(
    fill="both",
    expand=True,
    padx=30,
    pady=(10, 20)
)

# Create a scrollable text box for prediction results
result_box = tk.Text(
    result_frame,
    font=("Consolas", 10),
    height=12,
    wrap="word"
)

result_box.pack(side="left", fill="both", expand=True)

# Add a vertical scrollbar
scrollbar = ttk.Scrollbar(
    result_frame,
    orient="vertical",
    command=result_box.yview
)

scrollbar.pack(side="right", fill="y")

result_box.config(yscrollcommand=scrollbar.set)

# Default message
result_box.insert("1.0", "No prediction yet.")
result_box.config(state="disabled")

# Keep the window running
root.mainloop()
