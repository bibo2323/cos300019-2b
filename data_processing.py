"""
data_processing.py - COS30019 Assignment 2B (TBRGS)

Reads the raw VicRoads SCATS file (Boroondara, October 2006) and turns it into
clean tables that every ML model (LSTM, GRU, third model) and the GUI can share.

Run it once:
    python data_processing.py

It creates three files inside data/processed/:
    scats_15min.csv   one row per (site, 15-minute interval)   <- models train on this
    scats_hourly.csv  one row per (site, hour), vehicles/hour  <- travel-time formula uses this
    sites.csv         one row per SCATS site (id, name, latitude, longitude)

Other files can use it like this:
    from data_processing import load_15min, load_sites, series_for_site
"""
import os
import sys

import pandas as pd

# ---- settings (change here only) -----------------------------------------
RAW_FILE = os.path.join("data", "Scats_Data_October_2006.xls")
OUT_DIR = os.path.join("data", "processed")
INTERVALS_PER_DAY = 96                      # 24 hours x 4 (every 15 minutes)
VOLUME_COLUMNS = ["V%02d" % i for i in range(INTERVALS_PER_DAY)]   # V00 ... V95
# Each SCATS site has several detectors (one per approach road). A site's
# traffic is the SUM over its approaches, i.e. all cars passing the intersection.
SITE_AGGREGATION = "sum"
# ---------------------------------------------------------------------------


def read_raw(path=RAW_FILE):
    """Read the 'Data' sheet. Row 0 of the sheet is a time row, the real header
    is on row 1, so header=1."""
    df = pd.read_excel(path, sheet_name="Data", header=1)
    df = df.rename(columns={"SCATS Number": "site", "Location": "location",
                            "NB_LATITUDE": "lat", "NB_LONGITUDE": "lon",
                            "Date": "date"})
    missing = [c for c in VOLUME_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError("Raw file is missing columns: %s" % missing[:5])
    return df


def clean_raw(df):
    """Basic cleaning. Returns (clean_df, report_dict) so we can quote the
    numbers in the report."""
    report = {"raw_rows": len(df)}

    df = df.dropna(subset=["site", "date"]).copy()
    df["site"] = df["site"].astype(int)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()   # keep the day only

    before = len(df)
    df = df.drop_duplicates(subset=["site", "location", "date"], keep="first")
    report["duplicate_rows_removed"] = before - len(df)

    # a count can never be negative or missing
    report["missing_values"] = int(df[VOLUME_COLUMNS].isna().sum().sum())
    report["negative_values"] = int((df[VOLUME_COLUMNS] < 0).sum().sum())
    df[VOLUME_COLUMNS] = df[VOLUME_COLUMNS].fillna(0).clip(lower=0)

    report["clean_rows"] = len(df)
    report["sites"] = df["site"].nunique()
    report["days"] = df["date"].nunique()
    return df, report


def build_sites(df):
    """One row per site with a representative name and coordinates (mean of its
    detectors, because each detector has a slightly different lat/lon)."""
    sites = (df.groupby("site")
               .agg(lat=("lat", "mean"), lon=("lon", "mean"),
                    n_approaches=("location", "nunique"),
                    name=("location", "first"))
               .reset_index())
    return sites


def build_15min(df):
    """Wide table (V00..V95 columns) -> long table with one row per site and
    15-minute time stamp. flow_15min = cars in that 15-minute interval."""
    per_site_day = df.groupby(["site", "date"])[VOLUME_COLUMNS].agg(SITE_AGGREGATION)
    long = per_site_day.reset_index().melt(
        id_vars=["site", "date"], value_vars=VOLUME_COLUMNS,
        var_name="slot", value_name="flow_15min")
    long["slot"] = long["slot"].str[1:].astype(int)           # 'V07' -> 7
    long["timestamp"] = long["date"] + pd.to_timedelta(long["slot"] * 15, unit="m")
    long = long.sort_values(["site", "timestamp"]).reset_index(drop=True)
    return long[["site", "timestamp", "date", "slot", "flow_15min"]]


def build_hourly(long15):
    """Vehicles per hour = sum of the four 15-minute counts in that hour.
    The travel-time conversion document works in vehicles/hour."""
    tmp = long15.copy()
    tmp["hour_start"] = tmp["timestamp"].dt.floor("h")
    hourly = (tmp.groupby(["site", "hour_start"])["flow_15min"]
                 .sum().reset_index().rename(columns={"flow_15min": "flow_per_hour"}))
    return hourly


def process(raw_path=RAW_FILE, out_dir=OUT_DIR):
    """Full pipeline. Returns (long15, hourly, sites, report)."""
    df, report = clean_raw(read_raw(raw_path))
    sites = build_sites(df)
    long15 = build_15min(df)
    hourly = build_hourly(long15)

    os.makedirs(out_dir, exist_ok=True)
    long15.to_csv(os.path.join(out_dir, "scats_15min.csv"), index=False)
    hourly.to_csv(os.path.join(out_dir, "scats_hourly.csv"), index=False)
    sites.to_csv(os.path.join(out_dir, "sites.csv"), index=False)
    return long15, hourly, sites, report


# ---- helpers other team members import ------------------------------------
def load_15min(out_dir=OUT_DIR):
    return pd.read_csv(os.path.join(out_dir, "scats_15min.csv"),
                       parse_dates=["timestamp", "date"])


def load_sites(out_dir=OUT_DIR):
    return pd.read_csv(os.path.join(out_dir, "sites.csv"))


def series_for_site(site, out_dir=OUT_DIR):
    """The 15-minute flow of ONE site as a time-ordered pandas Series
    (what LSTM/GRU need as input)."""
    df = load_15min(out_dir)
    s = df[df["site"] == site].set_index("timestamp")["flow_15min"]
    return s.sort_index()


def main():
    if not os.path.exists(RAW_FILE):
        print("Cannot find %s. Put the SCATS .xls file inside the data folder." % RAW_FILE)
        return 1
    long15, hourly, sites, report = process()
    print("Data processing finished.")
    for key, value in report.items():
        print("  %-24s %s" % (key, value))
    print("  15-minute rows written  %d" % len(long15))
    print("  hourly rows written     %d" % len(hourly))
    print("  sites written           %d" % len(sites))
    print("Files saved in %s" % OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
