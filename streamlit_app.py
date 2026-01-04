import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from src.weather_api import get_weather_data
from src.config import OPENWEATHER_API_KEY, FLOOD_THRESHOLDS, VIETNAM_CITIES

# Load the trained model
model = joblib.load('models/flood_model.pkl')

st.title("🌧️ Flood Prediction App")
st.write("Predict flood risk based on live weather data using machine learning.")

# City selection
city = st.selectbox("Select a city", VIETNAM_CITIES)



if st.button("Predict Flood Risk"):
    try:
        # Fetch live weather data
        weather_data = get_weather_data(city, OPENWEATHER_API_KEY)

        # User-friendly weather display
        st.subheader(f"🌤️ Current Weather in {city}")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Temperature", f"{weather_data['max']:.1f}°C")
            st.metric("Humidity", f"{weather_data['humidi']}%")
        with col2:
            st.metric("Wind Speed", f"{weather_data['wind']:.1f} m/s")
            st.metric("Cloud Cover", f"{weather_data['cloud']}%")
        with col3:
            st.metric("Rainfall", f"{weather_data['rain']:.1f} mm")
            st.metric("Pressure", f"{weather_data['pressure']} hPa")

        # Prepare data for prediction
        features = ['max', 'min', 'wind', 'rain', 'humidi', 'cloud', 'pressure']
        input_data = pd.DataFrame([weather_data])[features]

        # Predict flood risk
        prediction = model.predict(input_data)[0]
        risk_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
        risk = risk_labels[prediction]

        # Display prediction with constant thresholds
        st.subheader("🌊 Flood Risk Prediction")
        if risk == 'Low':
            st.success(f"🟢 Low Flood Risk (Rain < {FLOOD_THRESHOLDS['low']}mm)")
        elif risk == 'Medium':
            st.warning(f"🟡 Medium Flood Risk (Rain {FLOOD_THRESHOLDS['low']}-{FLOOD_THRESHOLDS['medium']}mm)")
        else:
            st.error(f"🔴 High Flood Risk (Rain > {FLOOD_THRESHOLDS['medium']}mm)")

        # Visualization
        st.subheader("📊 Weather Features Visualization")
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(features, input_data.iloc[0], color=['blue', 'blue', 'green', 'cyan', 'purple', 'gray', 'red'])
        ax.set_ylabel('Values')
        ax.set_title('Current Weather Features')
        plt.xticks(rotation=45)

        # Highlight rain bar
        for i, bar in enumerate(bars):
            if features[i] == 'rain':
                bar.set_color('orange')

        st.pyplot(fig)

        # Additional interactivity: Show raw data if button pressed
        if st.button("Show Raw Data"):
            st.json(weather_data)

    except Exception as e:
        st.error(f"Error fetching data or predicting: {str(e)}")
