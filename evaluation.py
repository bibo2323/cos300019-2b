import argparse
import glob
import os
import sys
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
import ts_common as tc
 
EVAL_DIR = os.path.join(tc.RESULTS_DIR, "evaluation")
PRED_GLOB = os.path.join(tc.RESULTS_DIR, "predictions_*.csv")
ORDER_HINT = ["lstm", "gru"]           # shown first, in this order; the rest follow
BASELINES = ["naive_last", "hist_avg"]
 
 
# ---- metrics ---------------------------------------------------------------
def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    err = y_true - y_pred
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mean_y = float(np.mean(y_true))
    sst = float(np.sum((y_true - mean_y) ** 2))
    return {
        "RMSE": rmse,
        "MAE": float(np.mean(np.abs(err))),
        "NRMSE": rmse / mean_y if mean_y else float("nan"),
        "WAPE_%": 100.0 * float(np.sum(np.abs(err)) / np.sum(y_true)) if np.sum(y_true) else float("nan"),
        "R2": 1.0 - float(np.sum(err ** 2)) / sst if sst else float("nan"),
    }
 
 
def group_rmse(df, by):
    """RMSE of every model for each value of `by` ('site', 'hour', 'daytype')."""
    rows = []
    for (model, key), g in df.groupby(["model", by]):
        rows.append({"model": model, by: key,
                     "RMSE": float(np.sqrt(np.mean((g.y_true - g.y_pred) ** 2)))})
    return pd.DataFrame(rows).pivot(index=by, columns="model", values="RMSE")
 
 
# ---- loading / baselines ---------------------------------------------------
def load_predictions():
    files = sorted(glob.glob(PRED_GLOB))
    if not files:
        sys.exit("No prediction files found (%s). Train the models first, e.g. "
                 "python lstm_model.py" % PRED_GLOB)
    frames = []
    for f in files:
        d = pd.read_csv(f, parse_dates=["timestamp"])
        need = {"model", "site", "timestamp", "y_true", "y_pred"}
        if not need.issubset(d.columns):
            sys.exit("%s is missing columns %s" % (f, sorted(need - set(d.columns))))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
 
 
def make_baselines():
    """Return a long DataFrame (same columns as the prediction files) for the two
    baselines, on exactly the same test rows as the ML models."""
    series = tc.load_series()
    scalers = tc.fit_site_scalers(series)
    data = tc.build_dataset(series, scalers)
    X_te, y_te, meta = data["test"]
    sites = meta["site"]
    y_true = tc.inverse_scale(y_te, sites, scalers)
    naive = tc.inverse_scale(X_te[:, -1, 0], sites, scalers)      # last observed flow
 
    # historical average from TRAIN + VAL days only (never the test days)
    train_end, val_end = tc.split_dates()
    long = pd.concat([s.rename("flow").to_frame().assign(site=k) for k, s in series.items()])
    long = long[long.index <= val_end].reset_index(names="timestamp")
    long["slot"] = long.timestamp.dt.hour * 4 + long.timestamp.dt.minute // 15
    long["weekend"] = long.timestamp.dt.dayofweek >= 5
    avg = long.groupby(["site", "slot", "weekend"])["flow"].mean().rename("hist").reset_index()
    m = meta.copy()
    m["slot"] = m.timestamp.dt.hour * 4 + m.timestamp.dt.minute // 15
    m["weekend"] = m.timestamp.dt.dayofweek >= 5
    hist = m.merge(avg, on=["site", "slot", "weekend"], how="left")["hist"].values
 
    out = []
    for name, pred in (("naive_last", naive), ("hist_avg", hist)):
        d = meta.copy()
        d.insert(0, "model", name)
        d["y_true"] = y_true
        d["y_pred"] = pred
        out.append(d)
    return pd.concat(out, ignore_index=True)
 
 
def align(df):
    """Keep only (site, timestamp) rows that EVERY model predicted, so no model
    is scored on an easier subset. Also checks the y_true columns agree."""
    keys = df.groupby(["site", "timestamp"])["model"].nunique()
    n_models = df["model"].nunique()
    common = keys[keys == n_models].index
    dropped = len(keys) - len(common)
    if dropped:
        print("WARNING: %d (site, timestamp) rows were not predicted by every model "
              "and were dropped." % dropped)
    df = df.set_index(["site", "timestamp"]).loc[common].reset_index()
    spread = df.groupby(["site", "timestamp"])["y_true"].agg(lambda s: s.max() - s.min())
    if (spread > 1e-6).any():
        print("WARNING: models disagree on y_true for some rows - did they use "
              "different data splits? Use ts_common.py for the split.")
    return df
 
 
def order_models(models):
    first = [m for m in ORDER_HINT if m in models]
    base = [m for m in BASELINES if m in models]
    rest = sorted(m for m in models if m not in first and m not in base)
    return first + rest + base
 
 
# ---- charts ----------------------------------------------------------------
def _save(fig, name):
    path = os.path.join(EVAL_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
 
 
def chart_overall(overall, models):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, col in zip(axes, ["RMSE", "MAE", "R2"]):
        vals = overall.loc[models, col]
        bars = ax.bar(models, vals, color=plt.cm.tab10(np.arange(len(models))))
        ax.set_title(col + (" (lower is better)" if col != "R2" else " (higher is better)"))
        ax.bar_label(bars, fmt="%.2f", fontsize=8)
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Overall test-set performance (vehicles per 15 min)")
    return _save(fig, "fig_overall_metrics.png")
 
 
def chart_by_site(by_site, models):
    fig, ax = plt.subplots(figsize=(15, 5))
    w = 0.8 / len(models)
    x = np.arange(len(by_site))
    for i, m in enumerate(models):
        ax.bar(x + i * w, by_site[m], width=w, label=m)
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels([str(s) for s in by_site.index], rotation=90, fontsize=8)
    ax.set_xlabel("SCATS site")
    ax.set_ylabel("RMSE (vehicles / 15 min)")
    ax.set_title("RMSE per SCATS site")
    ax.legend()
    return _save(fig, "fig_rmse_by_site.png")
 
 
def chart_by_hour(by_hour, models):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for m in models:
        ax.plot(by_hour.index, by_hour[m], marker="o", ms=3, label=m)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("RMSE (vehicles / 15 min)")
    ax.set_title("Error by time of day (peaks are harder to predict)")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "fig_rmse_by_hour.png")
 
 
def chart_actual_vs_pred(df, models, site, days=3):
    g = df[df.site == site]
    first_ts = g.timestamp.min()
    g = g[g.timestamp < first_ts + pd.Timedelta(days=days)]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    truth = g[g.model == models[0]].sort_values("timestamp")
    ax.plot(truth.timestamp, truth.y_true, color="black", lw=2, label="actual")
    for m in models:
        p = g[g.model == m].sort_values("timestamp")
        ax.plot(p.timestamp, p.y_pred, lw=1.2, alpha=0.85, label=m)
    ax.set_title("Site %s: actual vs predicted flow, first %d test days" % (site, days))
    ax.set_ylabel("vehicles / 15 min")
    ax.legend(ncol=len(models) + 1, fontsize=8)
    ax.grid(alpha=0.3)
    return _save(fig, "fig_actual_vs_pred.png")
 
 
def chart_scatter(df, models):
    n = len(models)
    cols = min(n, 3)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.3 * cols, 4.2 * rows), squeeze=False)
    top = df.y_true.max() * 1.05
    for ax, m in zip(axes.ravel(), models):
        g = df[df.model == m]
        ax.scatter(g.y_true, g.y_pred, s=2, alpha=0.2)
        ax.plot([0, top], [0, top], "r--", lw=1)
        ax.set_title(m)
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted")
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    return _save(fig, "fig_scatter.png")
 
 
def chart_training_curves():
    files = sorted(glob.glob(os.path.join(tc.RESULTS_DIR, "*_history.csv")))
    if not files:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for f in files:
        name = os.path.basename(f).replace("_history.csv", "")
        h = pd.read_csv(f)
        ax.plot(h["epoch"] + 1, h["loss"], label=name + " train")
        if "val_loss" in h:
            ax.plot(h["epoch"] + 1, h["val_loss"], "--", label=name + " validation")
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE loss (scaled flow)")
    ax.set_title("Training curves")
    ax.legend()
    ax.grid(alpha=0.3)
    return _save(fig, "fig_training_curves.png")
 
 
# ---- markdown summary ------------------------------------------------------
def write_summary(overall, models, by_site, by_daytype, n_rows, n_sites, best_counts):
    ml = [m for m in models if m not in BASELINES]
    lines = ["# Model evaluation summary", "",
             "Test set: %s to %s, %d sites, %d predictions per model (next 15-minute flow)." % (
                 pd.Timestamp(tc.VAL_END).date() + pd.Timedelta(days=1), "31 Oct 2006",
                 n_sites, n_rows // len(models)),
             "", "## Overall metrics", "", overall.loc[models].round(3).to_markdown(), "",
             "## RMSE: weekday vs weekend", "", by_daytype[models].round(2).to_markdown(), "",
             "## Number of sites where each model has the lowest RMSE", "",
             best_counts.to_frame("sites").to_markdown(), ""]
    if ml:
        best = overall.loc[ml, "RMSE"].idxmin()
        worst = overall.loc[ml, "RMSE"].idxmax()
        gain = 100 * (overall.loc[worst, "RMSE"] - overall.loc[best, "RMSE"]) / overall.loc[worst, "RMSE"]
        lines += ["## Findings (check against the charts before using)", "",
                  "- Lowest overall RMSE among the ML models: **%s** (%.2f vehicles/15 min); "
                  "%.1f%% lower than %s." % (best, overall.loc[best, "RMSE"], gain, worst)]
        for b in BASELINES:
            if b in models:
                lines.append("- Baseline `%s` RMSE = %.2f; best ML model is %.1f%% %s." % (
                    b, overall.loc[b, "RMSE"],
                    abs(100 * (overall.loc[b, "RMSE"] - overall.loc[best, "RMSE"]) / overall.loc[b, "RMSE"]),
                    "better" if overall.loc[best, "RMSE"] < overall.loc[b, "RMSE"] else "WORSE"))
        hard = by_site[best].idxmax()
        lines.append("- Hardest site for %s: %s (RMSE %.2f); easiest: %s (RMSE %.2f)." % (
            best, hard, by_site[best].max(), by_site[best].idxmin(), by_site[best].min()))
    path = os.path.join(EVAL_DIR, "summary.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return path
 
 
def main():
    ap = argparse.ArgumentParser(description="Compare all traffic-flow models")
    ap.add_argument("--site", type=int, default=None,
                    help="site for the actual-vs-predicted chart (default: busiest site)")
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()
 
    df = load_predictions()
    if not args.no_baselines:
        df = pd.concat([df, make_baselines()], ignore_index=True)
    df = align(df)
    os.makedirs(EVAL_DIR, exist_ok=True)
 
    models = order_models(df["model"].unique().tolist())
    df["hour"] = df.timestamp.dt.hour
    df["daytype"] = np.where(df.timestamp.dt.dayofweek >= 5, "weekend", "weekday")
 
    overall = pd.DataFrame({m: metrics(g.y_true, g.y_pred)
                            for m, g in df.groupby("model")}).T.loc[models]
    by_site = group_rmse(df, "site")[models]
    by_hour = group_rmse(df, "hour")[models]
    by_daytype = group_rmse(df, "daytype")[models]
    best_counts = by_site[[m for m in models if m not in BASELINES] or models].idxmin(axis=1) \
        .value_counts().reindex([m for m in models if m not in BASELINES] or models, fill_value=0)
 
    overall.round(4).rename_axis("model").to_csv(os.path.join(EVAL_DIR, "metrics_overall.csv"))
    by_site.round(3).to_csv(os.path.join(EVAL_DIR, "metrics_by_site.csv"))
    by_hour.round(3).to_csv(os.path.join(EVAL_DIR, "metrics_by_hour.csv"))
    by_daytype.round(3).to_csv(os.path.join(EVAL_DIR, "metrics_by_daytype.csv"))
 
    site = args.site or int(df[df.model == models[0]].groupby("site")["y_true"].mean().idxmax())
    figs = [chart_overall(overall, models), chart_by_site(by_site, models),
            chart_by_hour(by_hour, models), chart_actual_vs_pred(df, models, site),
            chart_scatter(df, models), chart_training_curves()]
    n_sites = df.site.nunique()
    summ = write_summary(overall, models, by_site, by_daytype, len(df), n_sites, best_counts)
 
    print(overall.round(3).to_string())
    print("\nSaved in %s:" % EVAL_DIR)
    for p in [summ] + [f for f in figs if f]:
        print("  " + os.path.basename(p))
    return 0
 
 
if __name__ == "__main__":
    sys.exit(main())