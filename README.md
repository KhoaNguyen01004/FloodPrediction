# 🌧️ Vietnam Flood Prediction App

A machine learning-powered Streamlit application for predicting flood risk in Vietnam using real-time/historical weather data, historical flood patterns from EM-DAT disaster database, and expert rule overrides for safety. The system combines XGBoost gradient boosting with rule-based enhancements for accurate, conservative flood risk assessments across Vietnam's 34 administrative subdivisions.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://khoavietanhnguyenfloodprediction.streamlit.app/)

## 📋 Overview

This application predicts flood risk by processing EM-DAT flood data (2005-2023) into synthetic weather-risk datasets, training an XGBoost model, and integrating live weather APIs with expert rules. It provides interactive predictions, scenario simulations, and PDF reports for Vietnam's subdivisions.

### Key Features

- **Synthetic Data Generation**: Creates training data from disaster records with Gaussian weather distributions
- **Multi-Source Weather Data**: Fetches live/historical data via OpenWeatherMap and Open-Meteo APIs
- **XGBoost + Rules Prediction**: Gradient boosting ML combined with threshold-based boosts for safety
- **Interactive UI**: Streamlit app with predictions, simulators, maps, and breakdowns
- **Comprehensive Coverage**: All 34 Vietnamese administrative subdivisions (2026 structure)

## 🏗️ Project Structure

```
flood-prediction-app/
├── streamlit_app.py                 # Main Streamlit web application
├── src/
│   ├── config.py                   # Configuration constants and API keys
│   ├── coordinates.py              # City coordinate loading and mapping
│   ├── pdf_generator.py             # PDF report generation
│   ├── prediction.py                # Prediction pipeline with ML + expert rules
│   ├── process_disaster_data.py     # Disaster data processing and synthetic feature generation
│   ├── scenario_simulator.py        # Interactive what-if simulator
│   ├── train_model.py               # XGBoost model training and evaluation
│   ├── utils.py                     # Utility functions, cached loaders, UI components
│   └── weather_api.py               # Weather API integration (OpenWeatherMap + Open-Meteo)
├── data/
│   ├── raw/
│   │   ├── disaster_in_vietnam.csv  # EM-DAT disaster database (2005-2023 floods)
│   │   └── vietnam_city_coords.json # City coordinates and names (34 subdivisions)
│   └── processed/
│       └── flood_training.csv       # Synthetic training dataset with weather-risk features
├── models/
│   ├── flood_model.pkl              # Calibrated XGBoost model
│   ├── region_encoder.pkl           # One-hot encoder for regions
│   ├── features.pkl                 # Feature list for alignment
│   ├── feature_importance.pkl       # Feature importance data
│   ├── confusion_matrix.png         # Evaluation plot
│   └── feature_importance.png       # Feature importance plot
├── test.py                          # Utility for generating coords JSON
├── requirements.txt                 # Python dependencies
├── summary.md                       # Comprehensive system summary
├── LICENSE                          # MIT License
└── README.md                        # This documentation
```

## 📁 File Descriptions

### Core Application Files

- **`streamlit_app.py`**: Main Streamlit web app with tabs for predictions and model performance. Includes UI for city/date selection, weather display, risk assessment, maps, feature analysis, PDF export, and scenario simulator.

- **`src/config.py`**: Configuration constants: OpenWeather API key, flood thresholds, and list of 34 Vietnamese subdivisions.

### Data Processing Files

- **`src/coordinates.py`**: Loads city coordinates from JSON, handles DMS-to-decimal conversion, and caches lat/lon for API calls.

- **`src/process_disaster_data.py`**: Generates synthetic training data from EM-DAT floods. Maps locations to cities, assigns risk levels (based on damages/deaths), and creates weather features via Gaussian distributions with noise and oversampling.

### ML and Prediction Files

- **`src/train_model.py`**: Trains/calibrates XGBoost model with class weights, regularization, and subsampling. Evaluates with metrics, confusion matrix, and saves model/encoders.

- **`src/prediction.py`**: Prediction pipeline: selects data source (live/historical/seasonal), fetches weather, engineers features, runs ML + expert rule boosts, and thresholds to risk levels.

- **`src/scenario_simulator.py`**: Interactive simulator with sliders for weather variables, presets, and updated predictions with breakdowns.

### API and Utility Files

- **`src/weather_api.py`**: Integrates OpenWeatherMap (live) and Open-Meteo (historical/archive) for weather data, including cumulatives and fallbacks.

- **`src/utils.py`**: Cached loaders for models/encoders, UI components (calculation breakdowns, charts), and display functions.

- **`src/pdf_generator.py`**: Generates PDF reports with risk, weather, saturation analysis, and recommendations.

### Data Files

- **`data/raw/disaster_in_vietnam.csv`**: EM-DAT database for Vietnam floods (2005-2023).

- **`data/raw/vietnam_city_coords.json`**: Coordinates for 34 subdivisions.

- **`data/processed/flood_training.csv`**: Synthetic dataset with weather-risk features.

### Model Files

- **`models/flood_model.pkl`**: Calibrated XGBoost model.
- **`models/region_encoder.pkl`**: Region one-hot encoder.
- **`models/features.pkl`**: Feature list.
- **`models/feature_importance.pkl`**: Importance data.
- **`models/confusion_matrix.png`**: Evaluation chart.
- **`models/feature_importance.png`**: Importance chart.

- **`test.py`**: Script to extract locations from disaster data into JSON (utility).

## 🔄 Data Pipeline

### 1. Data Preparation Phase
- **Input**: EM-DAT disaster database (`disaster_in_vietnam.csv`)
- **Processing**: Filter floods in Vietnam (2005-2023), map locations to subdivisions, assign risk levels (High: damages >$500k or deaths; Medium: damages/affected/injured thresholds)
- **Synthetic Generation**: For each city/month/year, create weather via risk-based Gaussians (e.g., High: rain~150mm, humidity~95%), add noise/oversampling, balance classes
- **Output**: `flood_training.csv` with millions of samples (weather + region features)

### 2. Model Training Phase
- **Algorithm**: XGBoost (gradient boosting) with calibration
- **Features**: Weather (max, rain, humidity, wind), cyclical months, rain cumulatives, region one-hots, interactions
- **Training**: Class weights, regularization (L1/L2), subsampling, 3-fold CV
- **Evaluation**: Accuracy, F1, Brier score, confusion matrix, feature importance
- **Output**: Calibrated model + encoders in `models/`

### 3. Prediction Phase
- **Input**: City + date
- **Data Source**: Live (≤7 days via OpenWeather), Historical (via Open-Meteo), or Seasonal averages
- **Feature Engineering**: Rolling sums, cyclical encoding, region interactions
- **Prediction**: XGBoost probs → expert boosts (e.g., +0.2 for rain >50mm) → thresholds (High: prob>0.2, etc.)
- **Output**: Risk level, confidence, breakdowns, visualizations

## 🧠 Machine Learning Algorithm

### XGBoost (eXtreme Gradient Boosting)

**Why XGBoost + Calibration?**
- Superior to Random Forest on imbalanced data via gradient boosting and regularization
- Handles complex interactions (e.g., weather-region combos)
- Built-in feature selection via L1/L2 penalties and subsampling
- Calibration adjusts probabilities for realism (reduces overconfidence)

**Algorithm Details:**
- **Boosting**: 100 trees, sequential correction of residuals
- **Depth/Learning**: Max depth 3, learning rate 0.03 for stability
- **Regularization**: L1 (alpha=2.0), L2 (lambda=3.0) to shrink weights
- **Sampling**: 80% data/tree, 80% features/tree, 50% features/node
- **Calibration**: Sigmoid post-training for better probabilities

**Training Process:**
1. Fit initial model on data
2. Build trees to minimize loss on residuals
3. Combine with learning rate
4. Apply class weights for imbalance
5. Calibrate output probabilities

**Performance Metrics (Calibrated Model):**
- **Accuracy**: ~99% (imbalanced: ~98% Low risk)
- **Macro F1**: ~0.97
- **Class Performance** (example from code):
  - Low: Precision 0.99, Recall 1.00, F1 0.99
  - Medium: Precision 0.96, Recall 0.98, F1 ~0.97
  - High: Precision 1.00, Recall 0.50, F1 0.67
- **Brier Score**: Improved calibration (lower for better probs)
- **Overall**: Excellent on Low/Medium, moderate on High due to rarity

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- OpenWeather API key (free at [openweathermap.org](https://openweathermap.org/api))

### Installation

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd flood-prediction-app
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure API key**:
   - Edit `src/config.py`
   - Replace `OPENWEATHER_API_KEY` with your API key

3. **Process data and train model**:
   ```bash
   python src/process_disaster_data.py
   python src/train_model.py
   ```

4. **Run the application**:
   ```bash
   streamlit run streamlit_app.py
   ```

## 📊 Usage

1. **Select Location**: Choose from 34 Vietnamese administrative subdivisions
2. **Fetch Weather**: Click "Predict Flood Risk" to get current conditions
3. **View Results**:
   - Weather metrics dashboard
   - Flood risk prediction (Low/Medium/High) with color coding
   - Feature importance visualization
   - Raw data option for detailed inspection

## 🔧 Configuration

### Flood Risk Thresholds
```python
FLOOD_THRESHOLDS = {
    'low': 20,      # Low: < 20mm rainfall
    'medium': 50    # Medium: 20-50mm, High: >50mm
}
```

### Supported Locations
The app supports all 34 administrative subdivisions of Vietnam:
- Ha Noi, Ho Chi Minh City, Hai Phong, Da Nang, Can Tho, Hue
- Bac-Thai, Ha-Bac, Vinh-Phu, Ha-Tay, Hai-Hung, Ha-Nam-Ninh
- Nghe-Tinh, Quang-Binh, Quang-Tri, Quang-Nam-Da-Nang
- Nghia-Binh, Phu-Khanh, Thuan-Hai, Gia-Lai-Kon-Tum, Dak-Lak, Lam-Dong
- Song-Be, Tay-Ninh, Dong-Nai, Long-An, Dong-Thap, An-Giang
- Tien-Giang, Ben-Tre, Cuu-Long, Hau-Giang, Kien-Giang, Minh-Hai

## 📈 Model Performance

**Training Dataset**: Millions of synthetic samples (balanced via oversampling/noise)
- Low Risk: Majority (natural imbalance)
- Medium/High: Augmented for diversity

**Example Classification Report (Calibrated XGBoost)**:
```
              precision    recall  f1-score   support
Low              0.99       1.00      0.99      ~1526
Medium           0.96       0.98      0.97        ~122
High             1.00       0.50      0.67         ~50
```
- **Accuracy**: ~99%, **Macro F1**: ~0.97
- **Brier Score**: Lower for calibrated probs
- Excels on Low/Medium; High recall limited by rarity

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **EM-DAT**: Disaster database by CRED/OFDA
- **OpenWeatherMap & Open-Meteo**: Weather APIs
- **XGBoost**: Gradient boosting library
- **Streamlit**: Web framework
- **scikit-learn & Plotly**: ML and visualization

## ⚠️ Disclaimer

This application provides educational flood risk estimates based on weather patterns and historical data. Always consult local authorities and official sources for emergency preparedness and flood warnings.

---

**Vietnam Administrative Divisions**: The app uses Vietnam's 34 first-level administrative subdivisions as defined by the Vietnamese government, covering all provinces and centrally-administered cities.
