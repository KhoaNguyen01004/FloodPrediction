# Flood Prediction System: Comprehensive Learning Algorithm and Decision-Making Summary

## Overview
The FloodPrediction system is a machine learning-based application for forecasting flood risks in Vietnamese cities (34 first-level subdivisions as per 2026 administrative structure) using weather data, regional factors, and historical disaster patterns. The core algorithm is XGBoost (eXtreme Gradient Boosting), an ensemble method that iteratively builds a predictive model through boosting. However, the system integrates multiple components: data preprocessing from disaster records, weather API fetching, ML model training/evaluation, rule-based expert overrides for safety, interactive UI with simulations, and PDF reporting. This document summarizes the entire system, from data generation to deployment, explaining how it learns, processes inputs, and draws conclusions.

## Data Generation and Preprocessing
The training dataset is not directly collected but synthetically generated from historical disaster data to simulate realistic scenarios.

### Raw Data Sources
- **Disaster Records**: `data/raw/disaster_in_vietnam.csv` (EM-DAT floods in Vietnam, 2005-2023). Filtered for floods with damages/deaths/affected counts.
- **City Coordinates**: `data/raw/vietnam_city_coords.json` (latitude/longitude for 34 cities, parsed from DMS to decimal degrees).
- **Configurations**: `src/config.py` defines API keys (OpenWeatherMap), flood thresholds (e.g., Low: <20mm rain), and city lists.

### Training Data Creation (`src/process_disaster_data.py`)
- **Location Mapping**: Maps disaster location strings (e.g., "Mekong Delta") to affected cities using predefined dictionaries for regions (North, Central, South) and subdivisions.
- **Risk Labeling**: Assigns flood risk levels based on disaster impacts:
  - High: Damages >$500k or deaths >0.
  - Medium: Damages >$0 or affected >5 or injured >0.
  - Low: Otherwise.
- **Synthetic Weather Generation**: For each city/month/year combo, generates weather features via Gaussian distributions overlapping by risk level (e.g., High: rain ~150mm mean, humidity ~95%; Medium: overlapping; Low: rain ~10mm).
- **Augmentation**: Adds Gaussian noise (2-5% std) and oversamples minority classes (High: 5x, Medium: 50x) with noise injection to balance the dataset.
- **Output**: `data/processed/flood_training.csv` with thousands of high-quality samples (after balancing), including chaos noise (5% random overrides) for robustness.

This creates a labeled dataset mimicking real flood-weather correlations without needing extensive historical weather logs.

## Weather Data Fetching (`src/weather_api.py`)
Predictions require current/past weather inputs, fetched dynamically:

- **Live Data** (dates ≤7 days ahead): Uses OpenWeatherMap API for real-time forecasts (current + 24h). Computes 24h rain, past cumulatives via historical API calls.
- **Historical Data** (older dates): Uses Open-Meteo API for past weather (up to 7 days back) or archive for deeper history. Aggregates hourly data to daily snapshots (e.g., max temp, total rain).
- **Fallbacks**: If no data, uses monthly averages from training data or hardcoded defaults.
- **Cumulatives**: Calculates `rain_last_3_days` (sum over 72h) and `rain_last_7_days` (exponential decay approximation for predictions).

This ensures the system uses accurate, up-to-date weather for real-time predictions.

## Model Training and Learning (`src/train_model.py`)
The ML core trains an XGBoost classifier on the processed dataset.

### Feature Engineering
- **Rolling/Cumulative**: `rain_last_3_days` (sum of recent rain), `rain_last_7_days` (soil saturation proxy).
- **Cyclical Encoding**: `month_sin`/`month_cos` for seasonal cycles.
- **Interactions**: `rain_north`/`central`/`south` (region-specific rain effects).
- **Encoding**: One-hot for regions; city encoding (if used).
- **Balancing**: Synthetic augmentation (as above) + class weights (e.g., weight High risk 3x more).
- **Final Features**: Weather + regions (e.g., `['max', 'rain', 'humidi', 'month_sin', 'month_cos', 'rain_north', 'rain_central', 'rain_south']` + region one-hots).

### XGBoost Boosting Algorithm
1. **Ensemble Structure**: Builds 100 shallow trees (`max_depth=3`) sequentially.
2. **Sequential Correction**: Each tree fits residuals (errors) from the previous ensemble using gradient descent on multi-class softmax loss (3 classes: Low=0, Medium=1, High=2).
3. **Regularization**: L1 (`reg_alpha=2.0`) and L2 (`reg_lambda=3.0`) to shrink weights, implicitly selecting features by reducing "unimportant" ones (but retaining all for stability).
4. **Randomness**: Subsampling (80% data/tree), column sampling (80% features/tree, 50%/node) to prevent overfitting and promote diversity.
5. **Training Enhancements**:
   - Stratified 80/20 split.
   - Class weights to upweight Medium/High risks.
   - 3-fold CV for hyperparameter tuning/validation.
   - Calibration with `CalibratedClassifierCV` (sigmoid) to adjust probabilities for realism (e.g., reduce overconfidence).
6. **Evaluation**:
   - Metrics: Accuracy, macro F1, precision/recall per class.
   - Confusion matrix, Brier score (probability calibration).
   - Model comparison: XGBoost outperforms Random Forest, SVM, MLP in imbalanced handling.
   - Misclassification analysis: Reviews errors (e.g., extreme weather misclassifications).
   - Feature importance: Gain-based scores (top: rain, regions); visualized in `models/feature_importance.png`.

### Why Certain Features Are Selected
- **Implicit Selection**: Boosting evaluates all features at splits; regularization shrinks less useful ones (e.g., via L1). Column subsampling forces exploration of subsets.
- **No Hard Removal**: "Unimportant" features (low gain) are kept for interactions or edge cases, as XGBoost favors stability over strict pruning.
- **Importance Insights**: Helps explain predictions but doesn't alter the model.

## Prediction Pipeline (`src/prediction.py`)
For new queries (city + date):

1. **Data Source Selection**:
   - Live: If date ≤7 days ahead.
   - Seasonal Historical: Monthly averages from training data, filtered by city/region.
   - Fallback: Hardcoded defaults if no matches.

2. **Feature Engineering**: Mirrors training (cyclical months, `rain_last_7_days` approximation, interactions, encodings).

3. **ML Prediction**: Loads calibrated model (`models/flood_model.pkl`), predicts probabilities.

4. **Expert Rule Overrides**:
   - Boosts high-risk probability by 0.1-0.4 based on thresholds (rain >50mm, 3-day rain >100mm, humidity >70%, wind >8m/s).
   - Redistributes probabilities (scales down Low/Medium proportionally) for "Safety First".
   - Thresholds: High if prob[2] >0.20, Medium if prob[1] >0.30, else Low (or safety-first with 0.35 for High risk).

5. **Output**: Risk label, confidence (max prob %), base/ML probs, weather data.

This combines ML with rules for conservative, actionable predictions.

## User Interface and Interactions (`streamlit_app.py`)
A Streamlit app provides interactive access:

- **Tabs**: "Flood Prediction" (main) and "Model Performance" (comparison charts/tables).
- **Inputs**: City selectbox, date picker (seasonal context).
- **Analysis Button**: Triggers prediction, displays:
  - Weather metrics (temp, humidity, rain, pressure).
  - Data source info.
  - Risk gauge (progress bar for saturation), pie chart (probabilities), gauge (overall risk).
  - Map (city location).
  - Feature importance bar chart, table (values + notes).
  - Historical context (if matching past floods).
- **PDF Export**: Generates reports with soil saturation analysis, recommendations.
- **Session State**: Persists data across reruns.

### Scenario Simulator (`src/scenario_simulator.py`)
- **Interactive Sliders**: Adjust weather variables (rain, temp, etc.) to simulate "what-if" scenarios.
- **Presets**: Clear Sky, Heavy Monsoon, Approaching Typhoon.
- **Prediction**: Uses model + expert rules, shows updated risks/probabilities.
- **Breakdown**: Displays calculation steps (ML + rules), feature importance.

## Other Components
- **Utils (`src/utils.py`)**: Cached model loaders, display functions (calculation breakdown with icons, bar charts), CSS styles.
- **Coordinates (`src/coordinates.py`)**: Loads/parses city coords, DMS conversion, caching.
- **PDF Generator (`src/pdf_generator.py`)**: Creates reports with weather, saturation bar, recommendations.
- **Config**: Constants for cities, thresholds.

## How Conclusions Are Drawn and Limitations
- **Decision Flow**: Weather fetch → Feature prep → ML probs → Rule boosts → Thresholding → Risk label.
- **Explanations**: Feature importance, interactive breakdowns, saturation gauges.
- **Accuracy Expectations**: 75-85% in real-world (training optimistic); monitor false positives.
- **Limitations**: Relies on synthetic data (validate with real floods); API dependencies; rules may over-caution.
- **Improvements**: More real data, explicit feature selection, seasonal rule tuning.

The system blends adaptive ML learning with rule-based safety, providing explainable, real-time flood risk insights for Vietnamese subdivisions.