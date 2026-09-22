# 🚗 Used Car Price Prediction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-2563eb?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-REST%20API-059669?style=for-the-badge&logo=flask&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-7c3aed?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-d97706?style=for-the-badge&logo=scikit-learn&logoColor=white)

**Author:** Arpan Santra &nbsp;|&nbsp; **Project:** IBM BOB AI/ML Project &nbsp;|&nbsp; **Year:** 2026

</div>

---

## 📋 Project Description

An end-to-end Machine Learning system that predicts the **resale price of used cars in India** using a **Random Forest Regressor** trained on 1,234 real-world automotive listings.

The system exposes predictions via a **Flask REST API** and presents an interactive **Streamlit web application** — both served from a **single Python file** (`backend/app.py`).

> **Single-file design:**
> - `python app.py` → starts Flask API on port **5000**
> - `streamlit run app.py` → starts Streamlit UI on port **8501**

---

## 🎨 Design Token Colour System

All UI, charts, and visualisations follow a consistent colour palette:

| Token | Hex | Usage |
|-------|-----|-------|
| 🔵 **Primary Blue** | `#2563eb` | Main actions, headers, primary charts |
| 🟣 **Purple** | `#7c3aed` | MAE metric, secondary charts |
| 🟢 **Green** | `#059669` | RMSE metric, success states |
| 🟡 **Amber** | `#d97706` | CV-R² metric, warnings |
| 🔴 **Red** | `#dc2626` | Errors, fifth chart series |
| ⚫ **Dark** | `#0f172a` | Body text, titles |
| 🩶 **Muted** | `#64748b` | Labels, axis text |
| ⬜ **Background** | `#f1f5f9` | App background |

---

## 📊 Dataset

| Property       | Value |
|----------------|-------|
| **File**       | `test-data.csv` |
| **Source**     | Used car listings — online automotive platforms (India) |
| **Link**       | https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction |
| **Records**    | 1,234 rows |
| **Columns**    | 12 (after dropping unnamed index column) |
| **Cities**     | Ahmedabad, Bangalore, Chennai, Coimbatore, Delhi, Hyderabad, Jaipur, Kochi, Kolkata, Mumbai, Pune |
| **Year range** | 1996 – 2019 |
| **Fuel types** | Petrol, Diesel, CNG, LPG |

### Dataset Columns

| Column            | Type        | Description                                   |
|-------------------|-------------|-----------------------------------------------|
| Name              | Text        | Full car model name                           |
| Location          | Categorical | City where the car is listed                  |
| Year              | Integer     | Manufacturing year                            |
| Kilometers_Driven | Integer     | Total kilometres driven                       |
| Fuel_Type         | Categorical | Petrol / Diesel / CNG / LPG                  |
| Transmission      | Categorical | Manual / Automatic                            |
| Owner_Type        | Categorical | First / Second / Third / Fourth & Above       |
| Mileage           | Text        | Fuel efficiency (kmpl or km/kg — mixed units) |
| Engine            | Text        | Engine displacement (e.g. 998 CC)             |
| Power             | Text        | Engine power (e.g. 82.85 bhp)                 |
| Seats             | Float       | Passenger capacity                            |
| New_Price         | Text        | Original ex-showroom price — 182 non-null     |

---

## 🛠 Technologies Used

| Layer              | Technology                              |
|--------------------|-----------------------------------------|
| Language           | Python 3.10+                            |
| ML Model           | scikit-learn — RandomForestRegressor    |
| Backend API        | Flask 3.0 + Flask-CORS                  |
| Frontend UI        | Streamlit 1.36                          |
| Data Processing    | Pandas 2.2, NumPy 1.26                  |
| Visualisation      | Matplotlib 3.9, Seaborn 0.13            |
| Model Persistence  | joblib 1.4                              |
| Report Generation  | python-docx 1.1                         |
| Notebook           | Jupyter / ipykernel                     |

---

## 📁 Project Structure

```
Used_Car_Price_Prediction/
├── test-data.csv                                    Raw dataset (1,234 rows, 12 columns)
├── requirements.txt                                 Python dependencies
├── README.md                                        This file
├── HOW_TO_RUN.txt                                   Step-by-step run guide
├── ArpanSantra_UsedCarPricePrediction.ipynb         Complete Jupyter Notebook (41 cells)
├── ArpanSantra_UsedCarPricePredictionReport.docx    Full project report with screenshots
│
├── backend/
│   ├── train_model.py    ML training script — run once to train & save model
│   └── app.py            Flask REST API  +  Streamlit UI  (ONE combined file)
│
├── models/               Auto-created when train_model.py is run
│   ├── used_car_rf_model.pkl    Trained pipeline (preprocessing + RandomForest)
│   └── model_meta.pkl           Metadata: metrics, option lists, feature names
│
├── assets/               Auto-generated chart PNGs for the report
└── reports/              API output JSON and intermediate report files
```

---

## ⚡ Quick Start

### Step 1 — Install Dependencies
```bash
python -m pip install -r requirements.txt
```

### Step 2 — Train the Model
Run from inside `Used_Car_Price_Prediction/backend/`:
```bash
cd Used_Car_Price_Prediction/backend
python train_model.py
```

Expected output:
```
Loading data ...
  Dataset shape after preprocessing: (1221, 14)
Training RandomForest ...
  MAE  : 0.7717 Lakh
  RMSE : 1.6014 Lakh
  R2   : 0.7800
  CV R2 (5-fold): 0.7164 +/- 0.1022
Model saved -> ../models/used_car_rf_model.pkl
Metadata saved -> models/model_meta.pkl
```

### Step 3 — Start the Flask Backend *(Terminal 1)*
```bash
cd Used_Car_Price_Prediction/backend
python app.py
```
Flask API runs at: **http://127.0.0.1:5000**

### Step 4 — Launch the Streamlit UI *(Terminal 2 — keep Terminal 1 open)*
```bash
cd Used_Car_Price_Prediction/backend
python -m streamlit run app.py
```
Streamlit opens at: **http://localhost:8501**

### Step 5 — Open the Notebook *(Optional)*
```bash
jupyter notebook ArpanSantra_UsedCarPricePrediction.ipynb
```

---

## 🌐 API Reference

Base URL: `http://127.0.0.1:5000`

| Method | Endpoint             | Description                                         |
|--------|----------------------|-----------------------------------------------------|
| `GET`  | `/api/health`        | Liveness check — returns status and model_loaded    |
| `GET`  | `/api/model-info`    | Model metrics + feature list + dropdown option lists|
| `GET`  | `/api/dataset-info`  | Dataset statistics and distribution counts          |
| `POST` | `/api/predict`       | Single-car price prediction                         |
| `POST` | `/api/batch-predict` | Batch predictions — accepts a JSON array            |

### POST `/api/predict` — Request Body
```json
{
  "brand": "Maruti",
  "location": "Mumbai",
  "year": 2017,
  "kilometers_driven": 40000,
  "fuel_type": "Petrol",
  "transmission": "Manual",
  "owner_type": "First",
  "mileage": "20 kmpl",
  "engine": "1200 CC",
  "power": "82 bhp",
  "seats": 5
}
```

### Response
```json
{
  "status": "success",
  "predicted_price": 1.9,
  "unit": "Lakh INR",
  "price_range": {
    "low": 1.71,
    "high": 2.09
  }
}
```

### Supported Input Values

| Field          | Accepted Values |
|----------------|-----------------|
| `brand`        | Audi, BMW, Bentley, Chevrolet, Datsun, Fiat, Ford, Honda, Hyundai, Jaguar, Jeep, Land, Mahindra, Maruti, Mercedes-Benz, Mini, Mitsubishi, Nissan, Porsche, Renault, Skoda, Tata, Toyota, Volkswagen, Volvo |
| `location`     | Ahmedabad, Bangalore, Chennai, Coimbatore, Delhi, Hyderabad, Jaipur, Kochi, Kolkata, Mumbai, Pune |
| `year`         | 1996 – 2019 |
| `fuel_type`    | CNG, Diesel, LPG, Petrol |
| `transmission` | Automatic, Manual |
| `owner_type`   | First, Second, Third, Fourth & Above |

---

## 📈 Model Performance

| Metric              | Value                  |
|---------------------|------------------------|
| **R² Score**        | **0.78**               |
| **MAE**             | **0.7717 Lakh INR**    |
| **RMSE**            | **1.6014 Lakh INR**    |
| **CV R² (5-fold)**  | **0.7164 ± 0.1022**    |
| Train rows          | 976 (80%)              |
| Test rows           | 245 (20%)              |
| Total (post-preprocessing) | 1,221         |

### sklearn Pipeline Architecture

```
Raw CSV (1,234 rows)
  → Drop unnamed index column
  → Extract Brand (first word of Name)
  → Parse: Mileage (kmpl), Engine (CC), Power (bhp)  — strip text units
  → Derive Car_Age = 2024 − Year
  → Build Price target: New_Price × depreciation factor
     → Impute missing via brand-level median × depreciation
  → Clip top 1% price outliers
  → ColumnTransformer:
       Numeric (6)      → Median Imputer → Standard Scaler
       Categorical (5)  → Mode Imputer  → Ordinal Encoder (unknown = -1)
  → RandomForestRegressor
       n_estimators = 200 | max_depth = 15 | min_samples_split = 4
  → Save pipeline + metadata via joblib (.pkl)
```

---

## 🖥 Streamlit UI Tabs

| Tab                    | Colour  | Content |
|------------------------|---------|---------|
| 🎯 **Prediction**      | Blue    | 11-field sidebar form; styled price result box; price range bar chart |
| 📊 **Model Metrics**   | Purple  | R², MAE, RMSE, CV-R² cards; train/test split pie; metrics bar chart  |
| 🗂 **Dataset Overview**| Green   | Fuel type pie; city bar chart; transmission & owner-type bars; numeric stats |
| ℹ **About**            | Amber   | Architecture table; API reference; model performance; run instructions |

---

## 📦 Submission Files

| File | Format | Description |
|------|--------|-------------|
| `ArpanSantra_UsedCarPricePrediction.ipynb` | `.ipynb` | Complete Jupyter Notebook (41 cells) |
| `requirements.txt` | `.txt` | Python library dependencies |
| `ArpanSantra_UsedCarPricePredictionReport.docx` | `.docx` | Full project report with UI screenshots |
| `README.md` | `.md` | This file |
| `HOW_TO_RUN.txt` | `.txt` | Step-by-step run instructions |

---

<div align="center">

*IBM BOB AI/ML Project — Used Car Price Prediction — Arpan Santra — 2024*

</div>
