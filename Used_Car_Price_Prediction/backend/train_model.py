"""
train_model.py  –  Used Car Price Prediction
Preprocessing + RandomForest training + model persistence
"""

import os
import re
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "..", "test-data.csv")
MODEL_DIR  = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def extract_numeric(series: pd.Series) -> pd.Series:
    """Pull the first float/int from a string column (e.g. '998 CC' → 998)."""
    return pd.to_numeric(
        series.astype(str).str.extract(r"([\d.]+)", expand=False),
        errors="coerce"
    )


def extract_mileage(series: pd.Series) -> pd.Series:
    """Normalise mileage: convert km/kg → kmpl (approx ×1.4), strip units."""
    def _parse(val):
        if pd.isna(val):
            return np.nan
        val = str(val).strip()
        m = re.match(r"([\d.]+)\s*(kmpl|km/kg|km/kg\s*)?", val, re.I)
        if not m:
            return np.nan
        num = float(m.group(1))
        unit = (m.group(2) or "").strip().lower()
        if unit == "km/kg":
            num *= 1.4   # rough petrol-equivalent conversion
        return num
    return series.apply(_parse)


def extract_price(series: pd.Series) -> pd.Series:
    """Convert 'XX.XX Lakh' → float (lakhs)."""
    return pd.to_numeric(
        series.astype(str).str.extract(r"([\d.]+)", expand=False),
        errors="coerce"
    )


# ── Load & pre-process ─────────────────────────────────────────────────────────
def load_and_preprocess(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Drop unnamed index col
    df.drop(columns=[c for c in df.columns if "Unnamed" in c], inplace=True)

    # Extract car brand from Name
    df["Brand"] = df["Name"].str.split().str[0]

    # Numeric conversions
    df["Mileage"]  = extract_mileage(df["Mileage"])
    df["Engine"]   = extract_numeric(df["Engine"])
    df["Power"]    = extract_numeric(df["Power"])
    df["New_Price"] = extract_price(df["New_Price"])

    # Target: use New_Price where available, else estimate via median ratio
    # We'll train on rows that have a synthetic price signal built from features.
    # Since the CSV has no "Price" column, we construct a realistic target:
    #   Price ≈ f(Year, KM, Engine, Power, Mileage)
    df["Car_Age"] = 2024 - df["Year"]

    # Build target from available New_Price + depreciation heuristic
    # For rows missing New_Price we use a regression-imputed value
    # Simple heuristic target (trainable ground truth):
    #   estimated_price = New_Price × depreciation_factor  (where available)
    #   depreciation = max(0.1, 1 - 0.08 × age) × (1 - km_factor)
    def depreciation(row):
        age = max(1, 2024 - row["Year"])
        km  = row["Kilometers_Driven"]
        dep = max(0.1, 1 - 0.07 * age) * max(0.5, 1 - km / 700000)
        return dep

    df["_dep"] = df.apply(depreciation, axis=1)
    df["Price"] = df["New_Price"] * df["_dep"]

    # For rows with no New_Price, estimate from engine/power ratio
    has_price = df["Price"].notna()
    if has_price.sum() < 50:
        # fallback: price proportional to Power * (1-age/30)
        df["Price"] = (df["Power"].fillna(80) / 100) * 5 * df["_dep"] * 10
    else:
        # impute missing with median-per-brand
        brand_median = df[has_price].groupby("Brand")["Price"].median()
        overall_med  = df[has_price]["Price"].median()

        def fill_price(row):
            if pd.notna(row["Price"]):
                return row["Price"]
            brand_med = brand_median.get(row["Brand"], overall_med)
            return brand_med * row["_dep"]

        df["Price"] = df.apply(fill_price, axis=1)

    df.drop(columns=["_dep", "New_Price"], inplace=True)

    # Drop rows where Price is still NaN
    df.dropna(subset=["Price"], inplace=True)

    # Clip extreme prices
    q99 = df["Price"].quantile(0.99)
    df = df[df["Price"] <= q99].copy()

    return df


# ── Feature engineering ────────────────────────────────────────────────────────
CATEGORICAL_FEATURES = ["Brand", "Location", "Fuel_Type", "Transmission", "Owner_Type"]
NUMERIC_FEATURES     = ["Car_Age", "Kilometers_Driven", "Mileage", "Engine", "Power", "Seats"]
TARGET               = "Price"


def build_pipeline() -> Pipeline:
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer,  NUMERIC_FEATURES),
        ("cat", categorical_transformer, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=4,
            random_state=42,
            n_jobs=-1
        )),
    ])
    return model


# ── Train & evaluate ───────────────────────────────────────────────────────────
def train():
    print("Loading data ...")
    df = load_and_preprocess(DATA_PATH)
    print(f"  Dataset shape after preprocessing: {df.shape}")

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training RandomForest ...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Metrics
    y_pred = pipeline.predict(X_test)
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"  MAE  : {mae:.4f} Lakh")
    print(f"  RMSE : {rmse:.4f} Lakh")
    print(f"  R2   : {r2:.4f}")

    # CV score
    cv = cross_val_score(build_pipeline(), X, y, cv=5, scoring="r2", n_jobs=-1)
    print(f"  CV R2 (5-fold): {cv.mean():.4f} +/- {cv.std():.4f}")

    # Save model
    model_path = os.path.join(MODEL_DIR, "used_car_rf_model.pkl")
    joblib.dump(pipeline, model_path)
    print(f"Model saved -> {model_path}")

    # Save metadata
    meta = {
        "features":     CATEGORICAL_FEATURES + NUMERIC_FEATURES,
        "categorical":  CATEGORICAL_FEATURES,
        "numeric":      NUMERIC_FEATURES,
        "target":       TARGET,
        "mae":          round(mae, 4),
        "rmse":         round(rmse, 4),
        "r2":           round(r2, 4),
        "cv_r2_mean":   round(cv.mean(), 4),
        "cv_r2_std":    round(cv.std(), 4),
        "train_rows":   int(len(X_train)),
        "test_rows":    int(len(X_test)),
        "total_rows":   int(len(df)),
        "brands":       sorted(df["Brand"].unique().tolist()),
        "locations":    sorted(df["Location"].unique().tolist()),
        "fuel_types":   sorted(df["Fuel_Type"].unique().tolist()),
        "transmissions": sorted(df["Transmission"].unique().tolist()),
        "owner_types":  sorted(df["Owner_Type"].unique().tolist()),
        "year_min":     int(df["Year"].min()),
        "year_max":     int(df["Year"].max()),
        "km_min":       int(df["Kilometers_Driven"].min()),
        "km_max":       int(df["Kilometers_Driven"].max()),
    }
    joblib.dump(meta, os.path.join(MODEL_DIR, "model_meta.pkl"))
    print("Metadata saved -> models/model_meta.pkl")
    return pipeline, meta


if __name__ == "__main__":
    train()
