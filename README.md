# 🌧️ Vietnam Flood Prediction App

A machine learning-powered Streamlit application for predicting flood risk in Vietnam using real-time weather data and historical flood patterns from EM-DAT disaster database.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vietnam-flood-prediction.streamlit.app/)

## 📋 Overview

This application predicts flood risk across Vietnam's 34 administrative subdivisions by combining historical flood data (2005-2023) with real-time weather conditions. The system uses machine learning to analyze weather patterns and provide risk assessments for informed decision-making.

### Key Features

- **Historical Data Integration**: Processes EM-DAT disaster database for flood events
- **Real-time Weather Data**: Fetches current conditions using OpenWeather API
- **Machine Learning Prediction**: RandomForest classifier trained on synthetic weather features
- **Interactive UI**: Streamlit interface for selecting locations and viewing predictions
- **Comprehensive Coverage**: Supports all 34 Vietnamese administrative subdivisions

## 🏗️ Project Structure

```
flood-prediction-app/
├── streamlit_app.py                 # Main Streamlit web application
├── src/
│   ├── config.py                   # Configuration constants and API keys
│   ├── coordinates.py              # City coordinate loading and mapping
│   ├── process_disaster_data.py    # Disaster data processing and feature generation
│   ├── train_model.py              # Machine learning model training
│   └── weather_api.py              # OpenWeather API integration
├── data/
│   ├── raw/
│   │   ├── disaster_in_vietnam.csv # EM-DAT disaster database
│   │   └── vietnam_city_coords.json # City coordinates and names
│   └── processed/
│       └── flood_training.csv      # Processed training dataset
├── models/
│   └── flood_model.pkl             # Trained RandomForest model
├── requirements.txt                # Python dependencies
├── TODO.md                         # Project task tracking
├── LICENSE                         # MIT License
└── README.md                       # This documentation
```

## 📁 File Descriptions

### Core Application Files

- **`streamlit_app.py`**: The main web application built with Streamlit. Provides an interactive interface where users can select a Vietnamese subdivision, fetch current weather data, and receive flood risk predictions with visualizations.

- **`src/config.py`**: Contains configuration constants including the OpenWeather API key, flood risk thresholds (Low: <20mm, Medium: 20-50mm, High: >50mm), and the list of 34 Vietnamese administrative subdivisions.

### Data Processing Files

- **`src/coordinates.py`**: Loads city coordinates from the JSON file and provides functions to get latitude/longitude for any Vietnamese subdivision. Handles coordinate conversion and caching for efficient API calls.

- **`src/process_disaster_data.py`**: Processes the EM-DAT disaster database to create training data. Filters flood events from 2005-2023, maps disaster locations to Vietnamese subdivisions, and generates synthetic weather features based on historical flood risk levels.

- **`src/train_model.py`**: Trains the machine learning model using the processed disaster data. Uses RandomForest classifier with weather features and month to predict flood risk levels.

### API Integration Files

- **`src/weather_api.py`**: Handles communication with the OpenWeather API. Fetches current weather data (temperature, humidity, wind, rainfall, cloud cover, pressure) for any location using coordinates.

### Data Files

- **`data/raw/disaster_in_vietnam.csv`**: EM-DAT International Disaster Database containing historical disaster records for Vietnam, including flood events with location, date, and damage information.

- **`data/raw/vietnam_city_coords.json`**: JSON file containing the 34 administrative subdivisions of Vietnam with their names, latitudes, and longitudes.

- **`data/processed/flood_training.csv`**: Processed training dataset with synthetic weather features and flood risk labels generated from historical disaster data.

- **`models/flood_model.pkl`**: Serialized trained RandomForest model ready for predictions.

## 🔄 Data Pipeline

The flood prediction system follows this data processing and prediction pipeline:

### 1. Data Preparation Phase
- **Input**: EM-DAT disaster database (`disaster_in_vietnam.csv`)
- **Processing**: Filter for flood events in Vietnam (2005-2023)
- **Location Mapping**: Match disaster locations to 34 Vietnamese subdivisions using partial string matching
- **Feature Generation**: Create synthetic weather features based on historical flood risk:
  - Low Risk: Normal weather conditions (low rainfall, moderate temperatures)
  - Medium Risk: Elevated conditions (moderate rainfall, variable temperatures)
  - High Risk: Extreme conditions (high rainfall, low temperatures, high humidity)
- **Output**: Training dataset (`flood_training.csv`) with weather features and risk labels

### 2. Model Training Phase
- **Algorithm**: RandomForest Classifier (ensemble of decision trees)
- **Features**: max/min temperature, wind speed, rainfall, humidity, cloud cover, pressure, month
- **Target**: Flood risk level (0=Low, 1=Medium, 2=High)
- **Validation**: 80/20 train/test split with classification metrics
- **Output**: Trained model (`flood_model.pkl`)

### 3. Prediction Phase
- **Input**: User selects Vietnamese subdivision
- **Weather Fetching**: Get current weather data from OpenWeather API using coordinates
- **Prediction**: Use trained model to predict flood risk based on current conditions
- **Output**: Risk assessment with color-coded indicators and weather visualizations

## 🧠 Machine Learning Algorithm

### Random Forest Classifier

**Why Random Forest?**
- Handles both numerical and categorical features well
- Robust to overfitting through ensemble averaging
- Provides feature importance insights
- Works well with imbalanced datasets (most locations have low flood risk)

**Algorithm Details:**
- **Number of Trees**: 100 decision trees in the forest
- **Splitting Criteria**: Gini impurity for classification
- **Max Depth**: Unlimited (trees grow until pure leaves)
- **Bootstrap Sampling**: Each tree trained on random subset of data
- **Feature Selection**: Random subset of features considered at each split

**Training Process:**
1. Create multiple decision trees using bootstrap sampling
2. Each tree votes on the prediction
3. Final prediction is the majority vote across all trees
4. Model learns patterns between weather conditions and historical flood occurrences

**Performance Metrics:**
- **Accuracy**: 99% on validation set (highly imbalanced dataset with 98.4% low-risk samples)
- **Class Performance**:
  - Low Risk (Class 0): 99% precision, 100% recall, 99% F1-score
  - Medium Risk (Class 1): 67% precision, 20% recall, 31% F1-score
  - High Risk (Class 2): 100% precision, 50% recall, 67% F1-score
- **Overall**: Strong performance on dominant low-risk class, moderate on rare medium/high-risk events

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

**Training Dataset**: 7,752 samples (34 locations × 19 years × 12 months)
- Low Risk: 7,630 samples (98.4%)
- Medium Risk: 83 samples (1.1%)
- High Risk: 39 samples (0.5%)

**Classification Report**:
```
              precision    recall  f1-score   support
Low              0.99       1.00      0.99      1526
Medium           0.67       0.20      0.31        15
High             1.00       0.50      0.67         8
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -m 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **EM-DAT**: Disaster database provided by CRED/OFDA
- **OpenWeatherMap**: Real-time weather API
- **Streamlit**: Web application framework
- **scikit-learn**: Machine learning library

## ⚠️ Disclaimer

This application provides educational flood risk estimates based on weather patterns and historical data. Always consult local authorities and official sources for emergency preparedness and flood warnings.

---

**Vietnam Administrative Divisions**: The app uses Vietnam's 34 first-level administrative subdivisions as defined by the Vietnamese government, covering all provinces and centrally-administered cities.
