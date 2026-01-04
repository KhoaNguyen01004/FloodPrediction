# 🌧️ Vietnam Flood Prediction App

A machine learning-powered Streamlit application for predicting flood risk in Vietnam using real-time weather data and historical flood patterns.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vietnam-flood-prediction.streamlit.app/)

## 📋 Overview

This application leverages machine learning to predict flood risk across 40 Vietnamese provinces and cities. The model is trained on historical weather data from January 2009 to June 2021 and uses live weather data from OpenWeather API to provide real-time flood risk assessments.

### Key Features

- **Real-time Weather Data**: Fetches current weather conditions using OpenWeather API
- **Machine Learning Prediction**: RandomForest classifier trained on historical Vietnam weather data
- **Interactive UI**: User-friendly Streamlit interface with searchable city dropdown
- **Visual Analytics**: Weather feature visualization with flood risk indicators
- **Comprehensive Coverage**: Supports all 40 provinces/cities from the training dataset

## 🏗️ Architecture

```
flood-prediction-app/
├── streamlit_app.py          # Main Streamlit application
├── src/
│   ├── train_model.py        # Model training script
│   ├── weather_api.py        # OpenWeather API integration
│   └── config.py            # Configuration and constants
├── models/
│   └── flood_model.pkl      # Trained RandomForest model
├── data/                    # Training datasets
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenWeather API key (free tier available at [openweathermap.org](https://openweathermap.org/api))

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd flood-prediction-app
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure API Key**
   - Open `src/config.py`
   - Replace `"your_api_key_here"` with your OpenWeather API key

5. **Train the model** (optional - pre-trained model included)
   ```bash
   python src/train_model.py
   ```

6. **Run the application**
   ```bash
   streamlit run streamlit_app.py
   ```

## 📊 Dataset Information

### Training Data
- **Time Period**: January 1, 2009 - June 18, 2021
- **Coverage**: 40 Vietnamese provinces and cities
- **Features**: Temperature, humidity, wind speed, rainfall, cloud cover, pressure
- **Target**: Flood risk classification (Low/Medium/High)

### Supported Cities
Bac Lieu, Ho Chi Minh City, Tam Ky, Ben Tre, Hoa Binh, Tan An, Bien Hoa, Hong Gai, Thai Nguyen, Buon Me Thuot, Hue, Thanh Hoa, Ca Mau, Long Xuyen, Tra Vinh, Cam Pha, My Tho, Tuy Hoa, Cam Ranh, Nam Dinh, Uong Bi, Can Tho, Nha Trang, Viet Tri, Chau Doc, Phan Rang, Vinh, Da Lat, Phan Thiet, Vinh Long, Ha Noi, Play Cu, Vung Tau, Hai Duong, Qui Nhon, Yen Bai, Hai Phong, Rach Gia, Hanoi, Soc Trang

## 🎯 Flood Risk Classification

The model classifies flood risk based on rainfall thresholds:

- **🟢 Low Risk**: Rainfall < 20mm
- **🟡 Medium Risk**: Rainfall 20-50mm
- **🔴 High Risk**: Rainfall > 50mm

## 🧠 Machine Learning Model

### Algorithm
- **RandomForest Classifier** with 100 estimators
- Trained on historical weather-flood correlation data

### Features Used
- Maximum temperature (°C)
- Minimum temperature (°C)
- Wind speed (m/s)
- Rainfall (mm)
- Humidity (%)
- Cloud cover (%)
- Atmospheric pressure (hPa)

### Performance
- Model accuracy: ~85% on validation set
- Trained using scikit-learn with stratified cross-validation

## 🔧 Configuration

### API Configuration (`src/config.py`)
```python
# OpenWeather API Key
OPENWEATHER_API_KEY = "your_api_key_here"

# Flood risk thresholds (mm)
FLOOD_THRESHOLDS = {
    'low': 20,      # Low: < 20mm
    'medium': 50    # Medium: 20-50mm, High: >50mm
}

# Supported cities
VIETNAM_CITIES = [...]
```

## 📱 Usage

1. **Select City**: Use the searchable dropdown to choose a Vietnamese city
2. **Predict Risk**: Click "Predict Flood Risk" to fetch weather data and get prediction
3. **View Results**:
   - Current weather metrics in organized cards
   - Flood risk prediction with color-coded alerts
   - Weather feature visualization
   - Option to view raw API data

## 🔍 API Integration

### OpenWeather API
- **Endpoint**: Current weather data
- **Parameters**: City name, API key, metric units
- **Rate Limit**: 1000 calls/day (free tier)

### Data Processing
- Raw API response transformed to match training features
- Missing data handling with default values
- Feature scaling and preprocessing

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Data Source**: Vietnam weather and flood historical data
- **API Provider**: OpenWeatherMap for real-time weather data
- **Framework**: Streamlit for web application
- **ML Library**: scikit-learn for machine learning

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the Streamlit documentation
- Review OpenWeather API documentation

---

**Note**: This application is for educational and informational purposes. Always consult local authorities for official flood warnings and emergency preparedness.
