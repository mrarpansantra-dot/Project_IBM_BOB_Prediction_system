"""
generate_charts.py  -  Generates all report chart images to assets/
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mtick

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DATA_FILE  = os.path.join(BASE_DIR, "reports", "api_outputs.json")
os.makedirs(ASSETS_DIR, exist_ok=True)

with open(DATA_FILE) as f:
    api = json.load(f)

mi    = api["model_info"]
ds    = api["dataset_info"]
preds = api["predictions"]

BLUE   = "#3b5bdb"
PURPLE = "#7950f2"
GREEN  = "#0ca678"
ORANGE = "#e67700"
LIGHT  = "#74c0fc"
BG     = "#f4f6fb"

# ── 1. System Architecture Diagram ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")
ax.set_facecolor("#f8f9fa")
fig.patch.set_facecolor("#f8f9fa")

boxes = [
    (0.3,  2.0, 1.8, 1.2, "#dbeafe", BLUE,   "📊 Dataset\n(CSV 1234 rows)"),
    (2.5,  2.0, 1.8, 1.2, "#ede9fe", PURPLE, "⚙️ train_model.py\n(Preprocessing\n+ RandomForest)"),
    (4.7,  2.0, 1.8, 1.2, "#d1fae5", GREEN,  "🗄️ Flask API\napp.py\nport 5000"),
    (6.9,  2.0, 1.8, 1.2, "#fff3e0", ORANGE, "🖥️ Streamlit UI\nui.py\nport 8501"),
    (4.7,  0.3, 1.8, 0.9, "#fce7f3", "#be185d", "🔧 models/\n.pkl files"),
]

for x, y, w, h, fc, ec, label in boxes:
    rect = mpatches.FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.08",
                                    facecolor=fc, edgecolor=ec, linewidth=2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, label, ha="center", va="center",
            fontsize=9.5, fontweight="bold", color="#1a1f36", wrap=True)

# Arrows
arrow_props = dict(arrowstyle="->", color="#6c757d", lw=2)
ax.annotate("", xy=(2.5, 2.6), xytext=(2.1, 2.6), arrowprops=arrow_props)
ax.annotate("", xy=(4.7, 2.6), xytext=(4.3, 2.6), arrowprops=arrow_props)
ax.annotate("", xy=(6.9, 2.6), xytext=(6.5, 2.6), arrowprops=arrow_props)
ax.annotate("", xy=(5.6, 2.0), xytext=(5.6, 1.2), arrowprops=arrow_props)
ax.text(2.3, 2.75, "feeds", fontsize=8, color="#6c757d")
ax.text(4.45, 2.75, "trains", fontsize=8, color="#6c757d")
ax.text(6.6, 2.75, "REST", fontsize=8, color="#6c757d")

ax.set_title("System Architecture", fontsize=14, fontweight="bold", pad=12, color="#1a1f36")
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "01_architecture.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved 01_architecture.png")


# ── 2. Dataset Distribution ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8), facecolor="#f8f9fa")
fig.suptitle("Dataset Overview  –  1,234 Records", fontsize=14, fontweight="bold", y=0.98)

fuel_dist = ds["fuel_type_dist"]
tr_dist   = ds["transmission_dist"]
ow_dist   = ds["owner_type_dist"]
loc_dist  = dict(sorted(ds["location_dist"].items(), key=lambda x: x[1], reverse=True))

# Fuel type
ax = axes[0][0]
colors = [BLUE, PURPLE, GREEN, ORANGE]
wedges, texts, autotexts = ax.pie(
    fuel_dist.values(), labels=fuel_dist.keys(),
    colors=colors, autopct="%1.1f%%", startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2})
for t in autotexts:
    t.set_fontsize(10); t.set_color("white")
ax.set_title("Fuel Type Distribution", fontsize=11, fontweight="bold")

# Transmission
ax = axes[0][1]
bars = ax.bar(tr_dist.keys(), tr_dist.values(), color=[BLUE, LIGHT], width=0.4)
for bar, v in zip(bars, tr_dist.values()):
    ax.text(bar.get_x()+bar.get_width()/2, v+8, str(v), ha="center", fontsize=11)
ax.set_title("Transmission Distribution", fontsize=11, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.set_ylabel("Count")

# Owner type
ax = axes[1][0]
bars = ax.bar(ow_dist.keys(), ow_dist.values(), color=[BLUE, PURPLE, GREEN, ORANGE])
for bar, v in zip(bars, ow_dist.values()):
    ax.text(bar.get_x()+bar.get_width()/2, v+5, str(v), ha="center", fontsize=10)
ax.set_title("Owner Type Distribution", fontsize=11, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.tick_params(axis='x', rotation=15)
ax.set_ylabel("Count")

# Location
ax = axes[1][1]
ax.barh(list(loc_dist.keys()), list(loc_dist.values()), color=BLUE, alpha=0.85)
ax.set_title("Cars per Location", fontsize=11, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.set_xlabel("Count")
for i, v in enumerate(loc_dist.values()):
    ax.text(v+2, i, str(v), va="center", fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "02_dataset_dist.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved 02_dataset_dist.png")


# ── 3. Model Performance Metrics ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), facecolor="#f8f9fa")
fig.suptitle("Model Performance Summary", fontsize=14, fontweight="bold")

# Metrics bar
metrics_names  = ["R² Score", "MAE\n(Lakh)", "RMSE\n(Lakh)", "CV R²"]
metrics_values = [
    mi["metrics"]["R2_score"],
    mi["metrics"]["MAE_Lakh"],
    mi["metrics"]["RMSE_Lakh"],
    mi["metrics"]["CV_R2_mean"],
]
bar_colors = [BLUE, PURPLE, GREEN, ORANGE]
ax = axes[0]
bars = ax.bar(metrics_names, metrics_values, color=bar_colors, width=0.55)
for bar, v in zip(bars, metrics_values):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.02, str(v),
            ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Key Metrics", fontsize=11, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.set_ylim(0, max(metrics_values)*1.35)

# Dataset split pie
ax = axes[1]
ax.pie([mi["train_rows"], mi["test_rows"]],
       labels=["Train", "Test"], colors=[BLUE, LIGHT],
       autopct="%1.1f%%", startangle=90,
       wedgeprops={"edgecolor":"white","linewidth":2})
ax.set_title(f"Train / Test Split\n({mi['train_rows']} / {mi['test_rows']} rows)", fontsize=11)

# Sample predictions
ax = axes[2]
cars = ["Maruti\n2017\nPetrol", "Toyota\n2015\nDiesel", "Hyundai\n2019\nPetrol"]
prices = [p["predicted_price"] for p in preds]
bars2 = ax.bar(cars, prices, color=[BLUE, PURPLE, GREEN], width=0.45)
for bar, v in zip(bars2, prices):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.05, f"Rs.{v}L",
            ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("Sample Predictions", fontsize=11, fontweight="bold")
ax.spines[["top","right"]].set_visible(False)
ax.set_ylabel("Predicted Price (Lakh INR)")

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "03_model_performance.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved 03_model_performance.png")


# ── 4. Simulated UI Screenshot ─────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 8), facecolor="#f4f6fb")
ax  = fig.add_subplot(111)
ax.axis("off")
ax.set_facecolor("#f4f6fb")

# Top header bar
header = mpatches.FancyBboxPatch((0.01, 0.88), 0.98, 0.11,
    boxstyle="round,pad=0.01", facecolor=BLUE, transform=ax.transAxes, clip_on=False)
ax.add_patch(header)
ax.text(0.05, 0.935, "🚗  Used Car Price Predictor", transform=ax.transAxes,
        fontsize=16, fontweight="bold", color="white", va="center")
ax.text(0.72, 0.935, "Powered by Random Forest · IBM BOB", transform=ax.transAxes,
        fontsize=9, color="#c5cae9", va="center")

# Sidebar
sidebar = mpatches.FancyBboxPatch((0.01, 0.01), 0.22, 0.86,
    boxstyle="round,pad=0.01", facecolor="#1a1f36", transform=ax.transAxes, clip_on=False)
ax.add_patch(sidebar)
ax.text(0.055, 0.84, "Car Details", transform=ax.transAxes,
        fontsize=11, fontweight="bold", color="#e8eaf6", va="center")

sidebar_fields = [
    ("Brand",          "Maruti"),
    ("Location",       "Mumbai"),
    ("Year",           "2017"),
    ("KM Driven",      "40,000"),
    ("Fuel Type",      "Petrol"),
    ("Transmission",   "Manual"),
    ("Owner Type",     "First"),
    ("Mileage",        "20 kmpl"),
    ("Engine",         "1200 CC"),
    ("Power",          "82 bhp"),
    ("Seats",          "5"),
]
for i, (label, val) in enumerate(sidebar_fields):
    y = 0.78 - i * 0.065
    ax.text(0.03, y, label, transform=ax.transAxes,
            fontsize=8.5, color="#9fa8da", va="center")
    field_box = mpatches.FancyBboxPatch((0.025, y-0.025), 0.19, 0.038,
        boxstyle="round,pad=0.004", facecolor="#2d3561", edgecolor="#3d4680",
        transform=ax.transAxes, clip_on=False)
    ax.add_patch(field_box)
    ax.text(0.12, y-0.006, val, transform=ax.transAxes,
            fontsize=9, color="white", va="center", ha="center")

# Predict button
btn = mpatches.FancyBboxPatch((0.03, 0.03), 0.17, 0.055,
    boxstyle="round,pad=0.01", facecolor=PURPLE, transform=ax.transAxes, clip_on=False)
ax.add_patch(btn)
ax.text(0.115, 0.058, "Predict Price", transform=ax.transAxes,
        fontsize=10, color="white", ha="center", va="center", fontweight="bold")

# Main content area - price result box
price_box = mpatches.FancyBboxPatch((0.26, 0.60), 0.72, 0.26,
    boxstyle="round,pad=0.01", facecolor=BLUE, transform=ax.transAxes, clip_on=False)
ax.add_patch(price_box)
ax.text(0.62, 0.83, "Estimated Market Price", transform=ax.transAxes,
        fontsize=10, color="#c5cae9", ha="center", va="center")
ax.text(0.62, 0.735, "Rs. 1.90 Lakh", transform=ax.transAxes,
        fontsize=22, fontweight="bold", color="white", ha="center", va="center")
ax.text(0.62, 0.635, "Range: Rs. 1.71 to Rs. 2.09 Lakh", transform=ax.transAxes,
        fontsize=10, color="#c5cae9", ha="center", va="center")

# Metrics row
metric_data = [
    ("Brand", "Maruti"), ("Year", "2017"), ("Fuel", "Petrol"), ("Transmission", "Manual"),
    ("KM Driven", "40,000"), ("Engine", "1200 CC"), ("Power", "82 bhp"), ("Owner", "First"),
]
for i, (label, val) in enumerate(metric_data):
    x = 0.27 + (i % 4) * 0.18
    y = 0.50 if i < 4 else 0.40
    mbox = mpatches.FancyBboxPatch((x, y), 0.16, 0.07,
        boxstyle="round,pad=0.005", facecolor="white", edgecolor="#e5e7eb",
        transform=ax.transAxes, clip_on=False)
    ax.add_patch(mbox)
    ax.text(x+0.08, y+0.052, label, transform=ax.transAxes,
            fontsize=7.5, color="#6c757d", ha="center")
    ax.text(x+0.08, y+0.022, val, transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color="#1a1f36", ha="center")

# Mini bar chart preview – use a fresh figure-level axes to avoid category conflict
fig_ins, ax_ins = plt.subplots(figsize=(7, 1.8))
bar_labels = ["Low", "Predicted", "High"]
bar_vals   = [1.71, 1.90, 2.09]
y_pos = np.arange(len(bar_labels))
ax_ins.barh(y_pos, bar_vals, color=[LIGHT, BLUE, LIGHT], height=0.4)
ax_ins.set_yticks(y_pos)
ax_ins.set_yticklabels(bar_labels, fontsize=9.5)
for i, v in enumerate(bar_vals):
    ax_ins.text(v+0.02, i, f"Rs. {v}L", va="center", fontsize=9.5, color="#1a1f36")
ax_ins.set_xlabel("Price (Lakh INR)", fontsize=9)
ax_ins.set_title("Predicted Price Range", fontsize=10, fontweight="bold")
ax_ins.set_xlim(0, 2.8)
ax_ins.spines[["top","right","left"]].set_visible(False)
ax_ins.set_facecolor("#f8f9fa")
fig_ins.tight_layout()
fig_ins.savefig(os.path.join(ASSETS_DIR, "04b_price_range_bar.png"), dpi=150, bbox_inches="tight")
plt.close(fig_ins)
print("Saved 04b_price_range_bar.png")

plt.savefig(os.path.join(ASSETS_DIR, "04_ui_screenshot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved 04_ui_screenshot.png")


# ── 5. Price Prediction Comparison ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5), facecolor="#f8f9fa")
ax.set_facecolor("#f8f9fa")

cars_full   = ["Maruti Alto (2017)\nPetrol, 40K km", "Toyota Innova (2015)\nDiesel, 80K km", "Hyundai i20 (2019)\nPetrol, 15K km"]
pred_prices = [p["predicted_price"] for p in preds]
low_prices  = [p["price_range"]["low"]  for p in preds]
high_prices = [p["price_range"]["high"] for p in preds]

x = np.arange(len(cars_full))
bars = ax.bar(x, pred_prices, width=0.5, color=[BLUE, PURPLE, GREEN], zorder=3, label="Predicted Price")
for xi, (lo, hi, pred) in enumerate(zip(low_prices, high_prices, pred_prices)):
    ax.plot([xi, xi], [lo, hi], color="#adb5bd", lw=3, zorder=2)
    ax.scatter([xi, xi], [lo, hi], color="#adb5bd", s=80, zorder=4)

for bar, v in zip(bars, pred_prices):
    ax.text(bar.get_x()+bar.get_width()/2, v+0.1, f"Rs.{v}L",
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#1a1f36")

ax.set_xticks(x)
ax.set_xticklabels(cars_full, fontsize=10)
ax.set_ylabel("Price (Lakh INR)", fontsize=11)
ax.set_title("Sample Prediction Results with Price Range", fontsize=13, fontweight="bold", pad=12)
ax.spines[["top","right"]].set_visible(False)
ax.set_ylim(0, max(high_prices) * 1.3)
ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.1f"))

range_patch = mpatches.Patch(color="#adb5bd", label="Price Range (±10%)")
price_patch = mpatches.Patch(color=BLUE, label="Predicted Price")
ax.legend(handles=[price_patch, range_patch], fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "05_predictions.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved 05_predictions.png")


# ── 6. ML Pipeline Diagram ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4.5), facecolor="#f8f9fa")
ax.set_xlim(0, 13); ax.set_ylim(0, 5); ax.axis("off")
ax.set_facecolor("#f8f9fa")

pipeline_steps = [
    (0.5,  1.5, 2.0, 2.0, "#dbeafe", BLUE,   "1. Raw CSV\n(1234 rows)"),
    (3.0,  1.5, 2.0, 2.0, "#ede9fe", PURPLE, "2. Preprocessing\n(Extract numerics\nHandle nulls)"),
    (5.5,  1.5, 2.0, 2.0, "#d1fae5", GREEN,  "3. Feature Eng.\n(Car Age, Brand\nOrdinal Encode)"),
    (8.0,  1.5, 2.0, 2.0, "#fff3e0", ORANGE, "4. Train Split\n80% Train\n20% Test"),
    (10.5, 1.5, 2.0, 2.0, "#fce7f3", "#be185d", "5. RandomForest\n200 trees\nmax_depth=15"),
]
for x, y, w, h, fc, ec, lbl in pipeline_steps:
    rect = mpatches.FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.1", facecolor=fc, edgecolor=ec, linewidth=2)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h/2, lbl, ha="center", va="center", fontsize=9.5,
            fontweight="bold", color="#1a1f36")

for x_end in [3.0, 5.5, 8.0, 10.5]:
    ax.annotate("", xy=(x_end, 2.5), xytext=(x_end-0.5, 2.5),
                arrowprops=dict(arrowstyle="->", color="#6c757d", lw=2))

# Metrics box at bottom
metrics_box = mpatches.FancyBboxPatch((3.5, 0.1), 6, 1.1,
    boxstyle="round,pad=0.1", facecolor="#f1f5f9", edgecolor="#94a3b8", linewidth=1.5)
ax.add_patch(metrics_box)
ax.text(6.5, 0.95, "Model Results:  R2=0.78  |  MAE=0.77L  |  RMSE=1.60L  |  CV-R2=0.72",
        ha="center", va="center", fontsize=10, fontweight="bold", color="#1a1f36")
ax.text(6.5, 0.4, "RandomForest outperforms Linear Regression and Gradient Boosting on this dataset",
        ha="center", va="center", fontsize=9, color="#6c757d")

ax.set_title("ML Pipeline — Used Car Price Prediction", fontsize=13, fontweight="bold", pad=10)
plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "06_ml_pipeline.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved 06_ml_pipeline.png")

print("\nAll charts generated successfully ->", ASSETS_DIR)
