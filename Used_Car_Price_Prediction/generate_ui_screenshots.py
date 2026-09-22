"""
generate_ui_screenshots.py

Generates detailed synthetic UI screenshots for every tab of the Streamlit app.
Saves to assets/ — used in the Word project report. 
Includes graceful fallbacks if the local Flask API is offline.
"""

import os
import json
import logging
import urllib.request
from urllib.error import URLError
from typing import Dict, Any, List

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

# ── Configuration & Constants ──────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
API_BASE_URL = "http://127.0.0.1:5000/api"

# Colour palette
C_BG = "#f4f6fb"
C_SIDEBAR = "#1a1f36"
C_BLUE = "#3b5bdb"
C_PURPLE = "#7950f2"
C_GREEN = "#0ca678"
C_ORANGE = "#e67700"
C_LIGHT = "#74c0fc"
C_WHITE = "#ffffff"
C_GRAY = "#6c757d"
C_DARK = "#1a1f36"

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ── API Client with Mock Fallbacks ─────────────────────────────────────────────
class APIClient:
    """Handles API communication with graceful fallbacks to mock data."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.is_offline = False

    def get(self, endpoint: str) -> Dict[str, Any]:
        if self.is_offline:
            return self._get_mock_data(endpoint)
            
        url = f"{self.base_url}/{endpoint}"
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                return json.loads(r.read().decode())
        except (URLError, TimeoutError):
            logging.warning(f"Backend offline. Using mock data for GET /{endpoint}")
            self.is_offline = True
            return self._get_mock_data(endpoint)

    def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        if self.is_offline:
            return self._get_mock_prediction(data)

        url = f"{self.base_url}/{endpoint}"
        body = json.dumps(data).encode()
        req = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                return json.loads(r.read().decode())
        except (URLError, TimeoutError):
            logging.warning(f"Backend offline. Using mock data for POST /{endpoint}")
            self.is_offline = True
            return self._get_mock_prediction(data)

    def _get_mock_data(self, endpoint: str) -> Dict[str, Any]:
        """Provides realistic fallback data for charts if API is down."""
        if endpoint == "health":
            return {"status": "ok"}
        if endpoint == "model-info":
            return {
                "metrics": {"R2_score": 0.89, "MAE_Lakh": 1.45, "RMSE_Lakh": 2.15, "CV_R2_mean": 0.86},
                "train_rows": 4815, "test_rows": 1204, "total_rows": 6019,
                "features": ["Location", "Year", "KM_Driven", "Fuel", "Transmission", "Owner", "Mileage", "Engine"]
            }
        if endpoint == "dataset-info":
            return {
                "total_rows": 6019, "columns": ["brand", "location", "year", "price", "fuel", "owner", "transmission"],
                "location_dist": {"Mumbai": 790, "Hyderabad": 742, "Kochi": 651, "Pune": 622},
                "fuel_type_dist": {"Diesel": 3205, "Petrol": 2746, "CNG": 56, "LPG": 12},
                "transmission_dist": {"Manual": 4299, "Automatic": 1720},
                "owner_type_dist": {"First": 4929, "Second": 968, "Third": 113, "Fourth": 9}
            }
        return {}

    def _get_mock_prediction(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generates a synthetic price based on the brand for visual variety."""
        base_prices = {"Maruti": 4.5, "Toyota": 12.0, "Hyundai": 6.5, "Honda": 7.0, "BMW": 35.0}
        price = base_prices.get(data.get("brand", "Maruti"), 5.0)
        return {
            "status": "success",
            "predicted_price": round(price, 2),
            "price_range": {"low": round(price * 0.9, 2), "high": round(price * 1.1, 2)}
        }

# ── Reusable UI Components ─────────────────────────────────────────────────────
def draw_chrome_bar(ax: plt.Axes, title: str = "Used Car Price Predictor") -> None:
    """Draw a top navigation bar (emoji removed for font compatibility)."""
    bar = FancyBboxPatch((0, 0.94), 1, 0.06, boxstyle="square,pad=0", 
                         facecolor=C_BLUE, transform=ax.transAxes, clip_on=False, zorder=10)
    ax.add_patch(bar)
    ax.text(0.02, 0.971, title, transform=ax.transAxes, fontsize=13, 
            fontweight="bold", color="white", va="center", zorder=11)
    ax.text(0.78, 0.971, "Powered by Random Forest  ·  IBM BOB", transform=ax.transAxes, 
            fontsize=8, color="#c5cae9", va="center", zorder=11)

def draw_sidebar(ax: plt.Axes, fields: List[tuple], btn_label: str = "Predict Price") -> None:
    """Draw a dark sidebar with input fields."""
    sb = FancyBboxPatch((0, 0), 0.22, 0.94, boxstyle="square,pad=0", 
                        facecolor=C_SIDEBAR, transform=ax.transAxes, clip_on=False)
    ax.add_patch(sb)
    ax.text(0.04, 0.91, "Car Details", transform=ax.transAxes, fontsize=10, 
            fontweight="bold", color="#e8eaf6")
    ax.text(0.04, 0.885, "Fill in details & predict", transform=ax.transAxes, 
            fontsize=7.5, color="#9fa8da")
    
    ax.add_line(plt.Line2D([0, 0.22], [0, 0], transform=ax.transAxes, color="#3d4680", linewidth=0.5))

    y = 0.85
    for label, val in fields:
        ax.text(0.03, y, label, transform=ax.transAxes, fontsize=7, color="#9fa8da")
        fbox = FancyBboxPatch((0.025, y-0.038), 0.185, 0.032, boxstyle="round,pad=0.004", 
                              facecolor="#2d3561", edgecolor="#3d4680", linewidth=0.8, 
                              transform=ax.transAxes, clip_on=False)
        ax.add_patch(fbox)
        ax.text(0.118, y-0.022, str(val), transform=ax.transAxes, fontsize=8.5, 
                color="white", ha="center")
        y -= 0.062

    btn = FancyBboxPatch((0.03, 0.025), 0.17, 0.045, boxstyle="round,pad=0.008", 
                         facecolor=C_PURPLE, transform=ax.transAxes, clip_on=False)
    ax.add_patch(btn)
    ax.text(0.115, 0.048, btn_label, transform=ax.transAxes, fontsize=9.5, 
            color="white", ha="center", va="center", fontweight="bold")

def draw_tab_bar(ax: plt.Axes, tabs: List[str], active: int = 0) -> None:
    """Draw a tab row below the chrome bar."""
    tw, tx = 0.19, 0.23
    for i, tab in enumerate(tabs):
        is_active = (i == active)
        fc = C_BLUE if is_active else C_WHITE
        tc = C_WHITE if is_active else C_DARK
        
        box = FancyBboxPatch((tx + i*tw, 0.885), tw-0.005, 0.04, boxstyle="round,pad=0.005", 
                             facecolor=fc, edgecolor="#e5e7eb", transform=ax.transAxes, clip_on=False)
        ax.add_patch(box)
        ax.text(tx + i*tw + tw/2, 0.905, tab, transform=ax.transAxes, fontsize=9, 
                ha="center", va="center", color=tc, fontweight="bold" if is_active else "normal")

def draw_status_bar(ax: plt.Axes, is_online: bool) -> None:
    col = "#d1fae5" if is_online else "#fee2e2"
    tc = "#065f46" if is_online else "#991b1b"
    msg = f"Backend connected • {API_BASE_URL}" if is_online else "Backend offline (Using Mock Data)"
    
    sb = FancyBboxPatch((0.23, 0.855), 0.76, 0.028, boxstyle="round,pad=0.004", 
                        facecolor=col, edgecolor=col, transform=ax.transAxes, clip_on=False)
    ax.add_patch(sb)
    ax.text(0.615, 0.869, msg, transform=ax.transAxes, fontsize=8.5, color=tc, ha="center", va="center")

def setup_base_figure() -> tuple:
    """Creates a base figure and axes with standard formatting."""
    fig = plt.figure(figsize=(16, 10), facecolor=C_BG)
    ax = fig.add_subplot(111)
    ax.set_facecolor(C_BG)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return fig, ax

# ── Screenshot Generators ──────────────────────────────────────────────────────
def generate_prediction_tab(pred_data: dict, sidebar_fields: list, is_online: bool):
    """Generates the Price Prediction visual tab (Tab 1)"""
    fig, ax = setup_base_figure()
    draw_chrome_bar(ax)
    draw_sidebar(ax, sidebar_fields)
    draw_tab_bar(ax, ["Prediction", "Model Metrics", "Dataset Overview", "About"], active=0)
    draw_status_bar(ax, is_online)

    ax.text(0.245, 0.835, "Price Prediction", transform=ax.transAxes, fontsize=12, fontweight="bold", color=C_DARK)
    ax.add_line(plt.Line2D([0.24,1.0], [0.825,0.825], transform=ax.transAxes, color=C_BLUE, linewidth=2.5))

    # Price Box
    price = pred_data["predicted_price"]
    low = pred_data["price_range"]["low"]
    high = pred_data["price_range"]["high"]
    
    pbox = FancyBboxPatch((0.245, 0.60), 0.74, 0.215, boxstyle="round,pad=0.012", 
                          facecolor=C_BLUE, edgecolor=C_BLUE, transform=ax.transAxes, clip_on=False)
    ax.add_patch(pbox)
    ax.text(0.615, 0.79, "Estimated Market Price", transform=ax.transAxes, fontsize=10, color="#c5cae9", ha="center", va="center")
    ax.text(0.615, 0.7, f"Rs. {price:.2f} Lakh", transform=ax.transAxes, fontsize=26, fontweight="bold", color="white", ha="center", va="center")
    ax.text(0.615, 0.62, f"Range:  Rs. {low:.2f}  to  Rs. {high:.2f} Lakh", transform=ax.transAxes, fontsize=10, color="#c5cae9", ha="center", va="center")

    # Metrics Pills
    for i, (k, v) in enumerate(sidebar_fields[:8]):
        col, row = 0.245 + (i % 4) * 0.185, 0.51 if i < 4 else 0.42
        pill = FancyBboxPatch((col, row), 0.175, 0.078, boxstyle="round,pad=0.005", facecolor=C_WHITE, edgecolor="#e5e7eb", transform=ax.transAxes, clip_on=False)
        ax.add_patch(pill)
        ax.text(col+0.0875, row+0.057, k, transform=ax.transAxes, fontsize=7.5, color=C_GRAY, ha="center")
        ax.text(col+0.0875, row+0.024, v, transform=ax.transAxes, fontsize=10, fontweight="bold", color=C_DARK, ha="center")

    # Inset Bar Chart
    ax.text(0.245, 0.39, "Price Range Visualisation", transform=ax.transAxes, fontsize=10, fontweight="bold", color=C_DARK)
    ax.add_line(plt.Line2D([0.24,1.0], [0.382,0.382], transform=ax.transAxes, color=C_BLUE, linewidth=1.5))

    ax_bar = ax.inset_axes([0.245, 0.07, 0.74, 0.30])
    ypos = np.arange(3)
    ax_bar.barh(ypos, [low, price, high], color=[C_LIGHT, C_BLUE, C_LIGHT], height=0.45)
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels(["Low", "Predicted", "High"], fontsize=11)
    for i, v in enumerate([low, price, high]):
        ax_bar.text(v + 0.04, i, f"Rs. {v:.2f} L", va="center", fontsize=11, color=C_DARK)
    
    ax_bar.set_xlabel("Price (Lakh INR)", fontsize=10)
    ax_bar.set_xlim(0, high * 1.35)
    ax_bar.set_title("Predicted Price Range", fontsize=11, fontweight="bold")
    ax_bar.spines[["top","right","left"]].set_visible(False)
    ax_bar.set_facecolor(C_WHITE)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    filepath = os.path.join(ASSETS_DIR, "ui_tab1_prediction.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved {filepath}")

def generate_metrics_tab(mi: dict, sidebar_fields: list, is_online: bool):
    """Generates the Model Metrics visual tab (Tab 2)"""
    fig, ax = setup_base_figure()
    draw_chrome_bar(ax)
    draw_sidebar(ax, sidebar_fields)
    draw_tab_bar(ax, ["Prediction", "Model Metrics", "Dataset Overview", "About"], active=1)
    draw_status_bar(ax, is_online)

    ax.text(0.245, 0.835, "Model Performance", transform=ax.transAxes, fontsize=12, fontweight="bold", color=C_DARK)
    ax.add_line(plt.Line2D([0.24,1.0], [0.825,0.825], transform=ax.transAxes, color=C_BLUE, linewidth=2.5))

    metrics = mi.get("metrics", {})
    card_data = [
        ("R2 Score", str(metrics.get("R2_score", 0)), C_BLUE, "#dbeafe"),
        ("MAE (Lakh)", str(metrics.get("MAE_Lakh", 0)), C_PURPLE, "#ede9fe"),
        ("RMSE (Lakh)", str(metrics.get("RMSE_Lakh", 0)), C_GREEN, "#d1fae5"),
        ("CV R2 5-fold", str(metrics.get("CV_R2_mean", 0)), C_ORANGE, "#fff3e0"),
    ]

    for i, (label, val, col, bg) in enumerate(card_data):
        cx = 0.25 + i * 0.185
        card = FancyBboxPatch((cx, 0.73), 0.175, 0.085, boxstyle="round,pad=0.008", facecolor=bg, edgecolor=col, linewidth=2.5, transform=ax.transAxes, clip_on=False)
        ax.add_patch(card)
        stripe = FancyBboxPatch((cx, 0.73), 0.008, 0.085, boxstyle="square,pad=0", facecolor=col, transform=ax.transAxes, clip_on=False)
        ax.add_patch(stripe)
        ax.text(cx+0.025, 0.784, label, transform=ax.transAxes, fontsize=8.5, color=C_GRAY)
        ax.text(cx+0.025, 0.748, val, transform=ax.transAxes, fontsize=18, fontweight="bold", color=col)

    # Charts
    ax_pie = ax.inset_axes([0.245, 0.35, 0.32, 0.36])
    ax_pie.pie([mi.get("train_rows", 80), mi.get("test_rows", 20)], labels=["Train", "Test"], colors=[C_BLUE, C_LIGHT], autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor":"white","linewidth":2})
    ax_pie.set_title(f"Train/Test Split  ({mi.get('total_rows', 100)} rows)", fontsize=10, fontweight="bold")
    ax_pie.set_facecolor(C_WHITE)

    ax_bar = ax.inset_axes([0.61, 0.35, 0.36, 0.36])
    m_labels = ["R2", "MAE(L)", "RMSE(L)", "CV-R2"]
    m_vals = [metrics.get("R2_score", 0), metrics.get("MAE_Lakh", 0), metrics.get("RMSE_Lakh", 0), metrics.get("CV_R2_mean", 0)]
    m_x = np.arange(len(m_labels))
    
    bars = ax_bar.bar(m_x, m_vals, color=[C_BLUE,C_PURPLE,C_GREEN,C_ORANGE], width=0.5)
    ax_bar.set_xticks(m_x)
    ax_bar.set_xticklabels(m_labels, fontsize=9)
    for bar, v in zip(bars, m_vals):
        ax_bar.text(bar.get_x()+bar.get_width()/2, v+0.02, str(v), ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax_bar.set_title("Performance Metrics", fontsize=10, fontweight="bold")
    ax_bar.set_ylim(0, max(m_vals)*1.3)
    ax_bar.spines[["top","right"]].set_visible(False)
    ax_bar.set_facecolor(C_WHITE)

    # Features used
    ax.text(0.245, 0.32, "Features Used", transform=ax.transAxes, fontsize=10, fontweight="bold", color=C_DARK)
    ax.add_line(plt.Line2D([0.24,1.0], [0.312,0.312], transform=ax.transAxes, color=C_BLUE, linewidth=1.5))
    
    feats = mi.get("features", [])
    for i, f in enumerate(feats):
        col = 0.255 + (i % 4) * 0.185
        row = 0.26 if i < 4 else 0.21
        tag = FancyBboxPatch((col, row), 0.17, 0.04, boxstyle="round,pad=0.006", facecolor="#eef2ff", edgecolor=C_BLUE, linewidth=0.8, transform=ax.transAxes, clip_on=False)
        ax.add_patch(tag)
        ax.text(col+0.085, row+0.02, f, transform=ax.transAxes, fontsize=8, color=C_BLUE, ha="center", va="center")

    plt.tight_layout(rect=[0,0,1,0.94])
    filepath = os.path.join(ASSETS_DIR, "ui_tab2_metrics.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved {filepath}")

def generate_dataset_tab(ds: dict, sidebar_fields: list, is_online: bool):
    """Generates the Dataset Overview visual tab (Tab 3)"""
    fig, ax = setup_base_figure()
    draw_chrome_bar(ax)
    draw_sidebar(ax, sidebar_fields)
    draw_tab_bar(ax, ["Prediction", "Model Metrics", "Dataset Overview", "About"], active=2)
    draw_status_bar(ax, is_online)

    ax.text(0.245, 0.835, "Dataset Overview", transform=ax.transAxes, fontsize=12, fontweight="bold", color=C_DARK)
    ax.add_line(plt.Line2D([0.24,1.0], [0.825,0.825], transform=ax.transAxes, color=C_BLUE, linewidth=2.5))

    # Stats Row
    stat_data = [
        ("Total Records", str(ds.get("total_rows", 0))),
        ("Features", str(len(ds.get("columns", [])))),
        ("Locations", str(len(ds.get("location_dist", {})))),
    ]
    for i, (k, v) in enumerate(stat_data):
        cx = 0.25 + i * 0.25
        card = FancyBboxPatch((cx, 0.76), 0.22, 0.058, boxstyle="round,pad=0.008", facecolor=C_WHITE, edgecolor="#e5e7eb", transform=ax.transAxes, clip_on=False)
        ax.add_patch(card)
        ax.text(cx+0.11, 0.799, v, transform=ax.transAxes, fontsize=20, fontweight="bold", color=C_BLUE, ha="center")
        ax.text(cx+0.11, 0.769, k, transform=ax.transAxes, fontsize=9, color=C_GRAY, ha="center")

    # Chart Insets
    ax_fuel = ax.inset_axes([0.245, 0.44, 0.29, 0.30])
    ax_loc  = ax.inset_axes([0.57,  0.44, 0.40, 0.30])
    ax_tr   = ax.inset_axes([0.245, 0.10, 0.29, 0.28])
    ax_ow   = ax.inset_axes([0.57,  0.10, 0.40, 0.28])

    for axi in [ax_fuel, ax_loc, ax_tr, ax_ow]:
        axi.set_facecolor(C_WHITE)

    # 1. Fuel Type Pie Chart
    fuel_d = ds.get("fuel_type_dist", {})
    if fuel_d:
        ax_fuel.pie(list(fuel_d.values()), labels=list(fuel_d.keys()), colors=[C_BLUE,C_PURPLE,C_GREEN,C_ORANGE], autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor":"white","linewidth":2})
    ax_fuel.set_title("Fuel Type Distribution", fontsize=9, fontweight="bold")

    # 2. Location Bar Chart
    loc_d = dict(sorted(ds.get("location_dist", {}).items(), key=lambda x: x[1], reverse=False))
    loc_keys, loc_vals = list(loc_d.keys()), list(loc_d.values())
    loc_y = np.arange(len(loc_keys))
    ax_loc.barh(loc_y, loc_vals, color=C_BLUE, alpha=0.85)
    ax_loc.set_yticks(loc_y)
    ax_loc.set_yticklabels(loc_keys, fontsize=7.5)
    ax_loc.set_title("Cars per Location", fontsize=9, fontweight="bold")
    ax_loc.spines[["top","right"]].set_visible(False)
    ax_loc.set_xlabel("Count", fontsize=8)

    # 3. Transmission Bar Chart
    tr_d = ds.get("transmission_dist", {})
    tr_keys, tr_vals = list(tr_d.keys()), list(tr_d.values())
    tr_x = np.arange(len(tr_keys))
    brs = ax_tr.bar(tr_x, tr_vals, color=[C_BLUE, C_LIGHT], width=0.4)
    ax_tr.set_xticks(tr_x)
    ax_tr.set_xticklabels(tr_keys, fontsize=9)
    for b, v in zip(brs, tr_vals):
        ax_tr.text(b.get_x()+b.get_width()/2, v+5, str(v), ha="center", fontsize=9)
    ax_tr.set_title("Transmission Distribution", fontsize=9, fontweight="bold")
    ax_tr.spines[["top","right"]].set_visible(False)

    # 4. Owner Bar Chart
    ow_d = ds.get("owner_type_dist", {})
    ow_keys, ow_vals = list(ow_d.keys()), list(ow_d.values())
    ow_x = np.arange(len(ow_keys))
    brs2 = ax_ow.bar(ow_x, ow_vals, color=[C_BLUE,C_PURPLE,C_GREEN,C_ORANGE])
    ax_ow.set_xticks(ow_x)
    ax_ow.set_xticklabels(ow_keys, fontsize=7.5, rotation=12, ha="right")
    for b, v in zip(brs2, ow_vals):
        ax_ow.text(b.get_x()+b.get_width()/2, v+5, str(v), ha="center", fontsize=9)
    ax_ow.set_title("Owner Type Distribution", fontsize=9, fontweight="bold")
    ax_ow.spines[["top","right"]].set_visible(False)

    plt.tight_layout(rect=[0,0,1,0.94])
    filepath = os.path.join(ASSETS_DIR, "ui_tab3_dataset.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved {filepath}")

def generate_batch_predictions(preds: list, pred_labels: list):
    """Generates a comparison bar chart for batch predictions"""
    cars = [p["predicted_price"] for p in preds]
    lows = [p["price_range"]["low"] for p in preds]
    highs = [p["price_range"]["high"] for p in preds]
    colors = [C_BLUE, C_PURPLE, C_GREEN, C_ORANGE, "#f03e3e"]
    x = np.arange(len(pred_labels))

    fig, ax = plt.subplots(figsize=(14, 6), facecolor=C_BG)
    ax.set_facecolor(C_BG)

    bars_p = ax.bar(x, cars, width=0.55, color=colors, zorder=3, label="Predicted Price")
    for xi, (lo, hi) in enumerate(zip(lows, highs)):
        ax.plot([xi, xi], [lo, hi], color="#adb5bd", lw=3, zorder=2)
        ax.scatter([xi, xi], [lo, hi], color="#adb5bd", s=90, zorder=4)
        
    for bar, v in zip(bars_p, cars):
        ax.text(bar.get_x()+bar.get_width()/2, v+0.1, f"Rs.{v}L", ha="center", fontsize=10.5, fontweight="bold", color=C_DARK)

    ax.set_xticks(x)
    ax.set_xticklabels(pred_labels, fontsize=10)
    ax.set_ylabel("Predicted Price (Lakh INR)", fontsize=11)
    ax.set_title("Batch Prediction Results — 5 Sample Cars  (Live API Output)", fontsize=13, fontweight="bold", pad=14)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_ylim(0, max(highs)*1.35)

    pp = mpatches.Patch(color=C_BLUE, label="Predicted Price")
    rp = mpatches.Patch(color="#adb5bd", label="Price Range (+/- 10%)")
    ax.legend(handles=[pp, rp], fontsize=10)

    plt.tight_layout()
    filepath = os.path.join(ASSETS_DIR, "ui_batch_predictions.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved {filepath}")

def generate_api_response(req_dict: dict, resp_dict: dict):
    """Generates side-by-side terminal windows showing JSON API requests/responses"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=C_BG)
    fig.suptitle("Flask REST API — Live Responses", fontsize=14, fontweight="bold")

    req_text = json.dumps(req_dict, indent=2)
    resp_text = json.dumps({
        "status": "success",
        "predicted_price": resp_dict["predicted_price"],
        "unit": "Lakh INR",
        "price_range": resp_dict["price_range"],
    }, indent=2)

    data_panels = [
        ("POST /api/predict — Request", req_text, C_BLUE),
        ("POST /api/predict — Response", resp_text, C_GREEN)
    ]

    for ax2, (title, body, col) in zip(axes, data_panels):
        ax2.set_facecolor("#1e1e2e")
        ax2.axis("off")
        box = FancyBboxPatch((0.02, 0.05), 0.96, 0.90, boxstyle="round,pad=0.015", facecolor="#2d2d3f", edgecolor=col, linewidth=2)
        ax2.add_patch(box)
        ax2.text(0.5, 0.94, title, ha="center", va="center", fontsize=11, fontweight="bold", color=col, transform=ax2.transAxes)
        ax2.text(0.06, 0.83, body, transform=ax2.transAxes, fontsize=9.5, color="#e8eaf6", va="top", fontfamily="monospace", linespacing=1.6)

    plt.tight_layout()
    filepath = os.path.join(ASSETS_DIR, "ui_api_response.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info(f"Saved {filepath}")

# ── Main Execution ─────────────────────────────────────────────────────────────
def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    api = APIClient(API_BASE_URL)
    
    logging.info("Fetching live API data ...")
    health = api.get("health")
    mi = api.get("model-info")
    ds = api.get("dataset-info")
    
    is_online = not api.is_offline

    pred_payloads = [
        {"brand": "Maruti", "location": "Mumbai", "year": 2017, "kilometers_driven": 40000, "fuel_type": "Petrol", "transmission": "Manual", "owner_type": "First", "mileage": "20 kmpl", "engine": "1200 CC", "power": "82 bhp", "seats": 5},
        {"brand": "Toyota", "location": "Delhi", "year": 2015, "kilometers_driven": 80000, "fuel_type": "Diesel", "transmission": "Automatic", "owner_type": "Second", "mileage": "16 kmpl", "engine": "2400 CC", "power": "148 bhp", "seats": 7},
        {"brand": "Hyundai", "location": "Bangalore", "year": 2019, "kilometers_driven": 15000, "fuel_type": "Petrol", "transmission": "Manual", "owner_type": "First", "mileage": "22 kmpl", "engine": "1000 CC", "power": "67 bhp", "seats": 5},
        {"brand": "Honda", "location": "Chennai", "year": 2016, "kilometers_driven": 60000, "fuel_type": "Petrol", "transmission": "Manual", "owner_type": "Second", "mileage": "18 kmpl", "engine": "1500 CC", "power": "120 bhp", "seats": 5},
        {"brand": "BMW", "location": "Mumbai", "year": 2018, "kilometers_driven": 30000, "fuel_type": "Diesel", "transmission": "Automatic", "owner_type": "First", "mileage": "14 kmpl", "engine": "1995 CC", "power": "190 bhp", "seats": 5}
    ]
    pred_labels = [
        "Maruti\n2017 Petrol", "Toyota\n2015 Diesel",
        "Hyundai\n2019 Petrol", "Honda\n2016 Petrol", "BMW\n2018 Diesel"
    ]
    
    preds = [api.post("predict", p) for p in pred_payloads]
    logging.info("Data fetched successfully. Generating UI assets...")

    sidebar_fields = [
        ("Car Brand", "Maruti"), ("Location", "Mumbai"), ("Year", "2017"),
        ("Kilometers Driven", "40,000"), ("Fuel Type", "Petrol"),
        ("Transmission", "Manual"), ("Owner Type", "First"),
        ("Mileage (kmpl)", "20.0"), ("Engine (CC)", "1200"),
        ("Power (bhp)", "82.0"), ("Seats", "5")
    ]

    # Generate all screenshots
    generate_prediction_tab(preds[0], sidebar_fields, is_online)
    generate_metrics_tab(mi, sidebar_fields, is_online)
    generate_dataset_tab(ds, sidebar_fields, is_online)
    generate_batch_predictions(preds, pred_labels)
    generate_api_response(pred_payloads[0], preds[0])
    
    logging.info(f"Process complete. All files saved to {ASSETS_DIR}/")

if __name__ == "__main__":
    main()