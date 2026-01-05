import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from datetime import date
import plotly.graph_objects as go
from fpdf import FPDF
import unicodedata
import numpy as np
from src.weather_api import get_weather_data
from src.config import VIETNAM_CITIES, FLOOD_THRESHOLDS

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# Cache the model loading to avoid reloading on every rerun
@st.cache_resource
def load_model():
    return joblib.load('models/flood_model.pkl')

model = load_model()

# Cache coordinates loading
@st.cache_data
def load_coords():
    with open('data/raw/vietnam_city_coords.json', 'r') as f:
        return json.load(f)

coords_data = load_coords()

st.set_page_config(page_title="Vietnam Flood Predictor 2026", layout="wide")

# --- CUSTOM CSS FOR BETTER VISUALS ---
st.markdown("""
    <style>
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #1c83e1; }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-med { color: #ffa500; font-weight: bold; }
    .risk-low { color: #008000; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

st.title("Vietnam Flood Prediction (2026 Admin Structure)")
st.write("Machine Learning analysis based on 34 First-Level Subdivisions.")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Location & Timing")
city = st.sidebar.selectbox("Select Subdivision", VIETNAM_CITIES)
# NEW: Date selection allows the model to "know" the season
target_date = st.sidebar.date_input("Prediction Date", date.today())
st.sidebar.info(f"Analyzing seasonal risk for: {target_date.strftime('%B')}")

# Initialize session state for persisting data across reruns
if 'weather_data' not in st.session_state:
    st.session_state.weather_data = None
if 'proba' not in st.session_state:
    st.session_state.proba = None
if 'risk' not in st.session_state:
    st.session_state.risk = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = None
if 'ml_features' not in st.session_state:
    st.session_state.ml_features = None
if 'pdf_data' not in st.session_state:
    st.session_state.pdf_data = None

# Initialize slider keys with default weather values
if 'rain_s' not in st.session_state:
    st.session_state.rain_s = 10.0
if 'rain72_s' not in st.session_state:
    st.session_state.rain72_s = 50.0
if 'hum_s' not in st.session_state:
    st.session_state.hum_s = 70.0
if 'max_s' not in st.session_state:
    st.session_state.max_s = 25.0
if 'min_s' not in st.session_state:
    st.session_state.min_s = 20.0
if 'wind_s' not in st.session_state:
    st.session_state.wind_s = 5.0
if 'cloud_s' not in st.session_state:
    st.session_state.cloud_s = 50.0
if 'press_s' not in st.session_state:
    st.session_state.press_s = 1010.0
if 'previous_scenario' not in st.session_state:
    st.session_state.previous_scenario = 'Custom'

if st.button("Analyze & Predict Risk"):
    try:
        # 1. Smart Data Source Selection
        date_diff = abs((target_date - date.today()).days)
        if date_diff <= 7:
            # Use live API data
            with st.spinner(f"Fetching live weather forecast for {city}..."):
                weather_data = get_weather_data(city, target_date)
            data_source = "Live Forecast"
        else:
            # Use historical averages
            with st.spinner(f"Calculating seasonal averages for {city} in {target_date.strftime('%B')}..."):
                df = pd.read_csv('data/processed/flood_training.csv')
                month_df = df[df['month'] == target_date.month]
                if month_df.empty:
                    # Fallback defaults if no historical data for the month
                    weather_data = {
                        'max': 25.0,
                        'min': 20.0,
                        'wind': 5.0,
                        'rain': 10.0,
                        'humidi': 70.0,
                        'cloud': 50.0,
                        'pressure': 1010.0,
                        'rain_last_3_days': 50.0,
                        'month': target_date.month
                    }
                else:
                    weather_data = {
                        'max': month_df['max'].mean(),
                        'min': month_df['min'].mean(),
                        'wind': month_df['wind'].mean(),
                        'rain': month_df['rain'].mean(),
                        'humidi': month_df['humidi'].mean(),
                        'cloud': month_df['cloud'].mean(),
                        'pressure': month_df['pressure'].mean(),
                        'rain_last_3_days': month_df['rain_last_3_days'].mean(),
                        'month': target_date.month
                    }
            data_source = "Seasonal Historical Average"

        # 2. Ensure all required keys are present
        required_keys = ['max', 'min', 'wind', 'rain', 'humidi', 'cloud', 'pressure', 'month', 'rain_last_3_days']
        for key in required_keys:
            if key not in weather_data:
                weather_data[key] = 0.0  # Default if missing

        # 3. UI Feedback for Data Source
        if data_source == "Live Forecast":
            st.info(f"📡 **Data Source:** {data_source} (Real-time weather API data)")
        else:
            st.info(f"📊 **Data Source:** {data_source} (Based on historical patterns for {target_date.strftime('%B')})")

        # 4. Display Metrics
        st.subheader(f"🌤️ Current Weather: {city}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Max Temp", f"{weather_data['max']:.1f}°C")
        m2.metric("Humidity", f"{weather_data['humidi']}%")
        m3.metric("Rainfall", f"{weather_data['rain']:.1f} mm")
        m4.metric("Pressure", f"{weather_data['pressure']} hPa")

        # 4. PREPARE FEATURES
        # Add new features: temperature difference and interaction terms
        weather_data['temp_diff'] = abs(weather_data['max'] - weather_data['min'])
        weather_data['rain_humidi_interaction'] = weather_data['rain'] * weather_data['humidi']

        ml_features = ['max', 'min', 'wind', 'rain', 'humidi', 'cloud', 'pressure', 'month', 'rain_last_3_days', 'temp_diff', 'rain_humidi_interaction']
        input_df = pd.DataFrame([weather_data])[ml_features]

        # 5. Predict with adjusted thresholds for better recall
        proba = model.predict_proba(input_df)[0]
        # Rule-based override: If it's peak season and rain is > 50mm, boost high risk probability
        if weather_data['rain'] > 50 and weather_data['month'] in [9, 10, 11]:
            proba[2] += 0.2
            proba[0] -= 0.2
        proba = np.clip(proba, 0, 1)
        proba /= proba.sum()
        thresholds = {0: 0.5, 1: 0.3, 2: 0.2}
        prediction = 0
        if proba[2] > thresholds[2]:
            prediction = 2
        elif proba[1] > thresholds[1]:
            prediction = 1
        risk_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
        risk = risk_labels[prediction]
        confidence = max(proba) * 100  # Confidence score as percentage

        # Store data in session state for persistence
        st.session_state.weather_data = weather_data
        st.session_state.proba = proba
        st.session_state.risk = risk
        st.session_state.confidence = confidence
        st.session_state.ml_features = ml_features
        st.session_state.city = city
        st.session_state.target_date = target_date

        # Generate PDF report and store in session state
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=16, style='B')
        pdf.cell(0, 10, "Vietnam Flood Risk Assessment Report", ln=True, align='C')
        pdf.ln(10)
        pdf.set_font("Arial", size=12)
        pdf.cell(0, 10, f"Location: {strip_accents(city)}", ln=True)
        pdf.cell(0, 10, f"Analysis Date: {target_date.strftime('%Y-%m-%d')}", ln=True)
        pdf.cell(0, 10, f"Risk Level: {risk}", ln=True)
        pdf.cell(0, 10, f"AI Confidence: {confidence:.1f}%", ln=True)
        pdf.ln(5)
        pdf.cell(0, 10, "Weather Conditions:", ln=True)
        pdf.cell(0, 10, f"  - Max Temperature: {weather_data['max']:.1f} deg C", ln=True)
        pdf.cell(0, 10, f"  - Min Temperature: {weather_data['min']:.1f} deg C", ln=True)
        pdf.cell(0, 10, f"  - Humidity: {weather_data['humidi']}%", ln=True)
        pdf.cell(0, 10, f"  - Rainfall (24h): {weather_data['rain']:.1f} mm", ln=True)
        pdf.cell(0, 10, f"  - Rainfall (72h): {weather_data['rain_last_3_days']:.1f} mm", ln=True)
        pdf.cell(0, 10, f"  - Wind Speed: {weather_data['wind']:.1f} m/s", ln=True)
        pdf.cell(0, 10, f"  - Pressure: {weather_data['pressure']} hPa", ln=True)
        pdf.ln(5)
        pdf.cell(0, 10, f"Risk Assessment: {risk} flood risk detected.", ln=True)
        if risk == 'High':
            pdf.cell(0, 10, "Recommendation: Prepare for potential flooding. Monitor weather updates.", ln=True)
        elif risk == 'Medium':
            pdf.cell(0, 10, "Recommendation: Stay alert for localized flooding in low-lying areas.", ln=True)
        else:
            pdf.cell(0, 10, "Recommendation: Conditions appear stable.", ln=True)
        st.session_state.pdf_data = bytes(pdf.output())

        # 6. Display Result & Map
        st.divider()
        col_map, col_res = st.columns([1.5, 1])

        with col_res:
            st.subheader("🌊 Flood Risk Assessment")
            if risk == 'Low':
                st.success("### 🟢 LOW RISK")
                st.write("Conditions appear stable. No immediate flood threat detected for this weather profile.")
            elif risk == 'Medium':
                st.warning("### 🟡 MEDIUM RISK")
                st.write("Elevated risk. Localized flooding possible in low-lying areas. Monitor weather updates.")
            else:
                st.error("### 🔴 HIGH RISK")
                st.write("**URGENT:** High probability of significant flooding. Ensure drainage systems are clear.")

            st.metric("Confidence Score", f"{confidence:.1f}%")
            st.info(f"**Seasonal Context:** Your model is accounting for {target_date.strftime('%B')} patterns in {city}.")

            # 1. Saturation Gauge
            st.subheader("💧 Soil Saturation Level")
            saturation_level = min(weather_data['rain_last_3_days'] / 150.0, 1.0)
            st.progress(saturation_level)
            st.caption(f"Ground has absorbed {weather_data['rain_last_3_days']:.1f}mm in the last 72h.")

            # 2. Risk Probability Breakdown
            st.subheader("🤖 AI Confidence Distribution")
            labels = ['Low Risk', 'Medium Risk', 'High Risk']
            colors = ['#28a745', '#ffc107', '#dc3545']
            fig_proba = go.Figure(data=[go.Pie(labels=labels, values=proba, hole=.4, marker_colors=colors)])
            fig_proba.update_layout(title_text="AI Confidence Distribution")
            st.plotly_chart(fig_proba)

            # 3. Historical "Twin" Comparison
            if risk == 'High':
                st.subheader("📜 Historical Context")
                historical_floods = {
                    'Thanh Hoa': {'month': 10, 'year': 2020, 'description': 'Mã River reached Alarm Level 3'},
                    'Ha Tinh': {'month': 9, 'year': 2020, 'description': 'Severe flooding in coastal areas'},
                    'Quang Binh': {'month': 10, 'year': 2020, 'description': 'Flash floods in mountainous regions'}
                }
                if city in historical_floods and target_date.month == historical_floods[city]['month']:
                    st.info(f"Similar conditions in this region ({historical_floods[city]['month']}/{historical_floods[city]['year']}) resulted in {historical_floods[city]['description']}.")

            # PDF Export Button (moved outside analysis block below)

        with col_map:
            # Filter JSON for current city coordinates
            try:
                city_coord = next(item for item in coords_data if item["location"] == city)
                map_data = pd.DataFrame([{
                    'lat': city_coord['lat'],
                    'lon': city_coord['lon']
                }])
                st.map(map_data, zoom=7, use_container_width=True)
            except StopIteration:
                st.error(f"Coordinates for {city} not found in the database.")
                # Do not attempt to display map if coordinates not found

        # 7. ENHANCED FEATURE ANALYSIS
        st.divider()
        st.subheader("📊 Why did the model predict this?")

        # Unit Mapping
        units = {'max': '°C', 'min': '°C', 'wind': 'm/s', 'rain': 'mm', 'humidi': '%', 'cloud': '%', 'pressure': 'hPa', 'month': 'Mo', 'rain_last_3_days': 'mm', 'temp_diff': '°C', 'rain_humidi_interaction': 'mm*%'}

        # Create feature values table
        feature_table = pd.DataFrame({
            'Feature': [f.replace('_', ' ').title() for f in ml_features],
            'Value': [f"{input_df.iloc[0][f]:.1f}{units[f]}" for f in ml_features]
        })
        st.table(feature_table)

        st.write("**Feature Impact Guide**")
        # Create comprehensive feature impact table
        data_label = "Current" if target_date >= date.today() else "Historical"
        impact_table = pd.DataFrame({
            "Feature": [f.replace('_', ' ').title() for f in ml_features],
            data_label: [f"{weather_data[f]:.1f}{units[f]}" for f in ml_features],
            "Risk Note": [
                "Higher temp = lower risk",
                "Lower temp = higher risk",
                "Strong wind = higher risk",
                "Heavy rain = highest risk",
                "High humidity + rain = very high risk",
                "Cloudy conditions = moderate risk",
                "Low pressure = storm risk",
                "Seasonal patterns affect risk",
                "Accumulated rain = saturation risk",
                "Large temp range = weather instability",
                "Rain × humidity interaction = flood potential"
            ]
        })
        st.table(impact_table)

    except Exception as e:
        st.error(f"Critical Error: {str(e)}")

# PDF Download Button (outside analysis block to avoid rerun issues)
if st.session_state.pdf_data is not None:
    file_name = f"flood_risk_{st.session_state.city}_{st.session_state.target_date.strftime('%Y%m%d')}.pdf"
    st.download_button(
        label="📄 Download Risk Report (PDF)",
        data=st.session_state.pdf_data,
        file_name=file_name,
        mime='application/pdf'
    )

@st.fragment
def scenario_simulator():
    # Results Bar at the top
    results_bar = st.empty()

    st.write("Adjust these values to see how the AI risk changes. Use 'Analyze & Predict Risk' first for real data, or simulate with defaults.")

    # Use actual weather data if available, else defaults
    base_weather = st.session_state.weather_data if st.session_state.weather_data is not None else {
        'rain': 10.0, 'rain_last_3_days': 50.0, 'humidi': 70.0, 'max': 25.0, 'min': 20.0, 'wind': 5.0, 'cloud': 50.0, 'pressure': 1010.0, 'month': 6
    }
    base_proba = st.session_state.proba if st.session_state.proba is not None else [0.6, 0.3, 0.1]  # Default probabilities

    # Define scenario presets with more distinct values
    scenarios = {
        'Clear Sky': {'rain': 0.0, 'rain_last_3_days': 0.0, 'humidi': 60.0, 'max': 30.0, 'min': 25.0, 'wind': 5.0, 'cloud': 20.0, 'pressure': 1020.0},
        'Heavy Monsoon': {'rain': 120.0, 'rain_last_3_days': 250.0, 'humidi': 85.0, 'max': 26.0, 'min': 22.0, 'wind': 8.0, 'cloud': 90.0, 'pressure': 1000.0},
        'Approaching Typhoon': {'rain': 200.0, 'rain_last_3_days': 400.0, 'humidi': 95.0, 'max': 24.0, 'min': 20.0, 'wind': 18.0, 'cloud': 95.0, 'pressure': 980.0}
    }

    def reset_to_actual():
        if st.session_state.weather_data is not None:
            st.session_state.rain_s = st.session_state.weather_data['rain']
            st.session_state.rain72_s = st.session_state.weather_data['rain_last_3_days']
            st.session_state.hum_s = st.session_state.weather_data['humidi']
            st.session_state.max_s = st.session_state.weather_data['max']
            st.session_state.min_s = st.session_state.weather_data['min']
            st.session_state.wind_s = st.session_state.weather_data['wind']
            st.session_state.cloud_s = st.session_state.weather_data['cloud']
            st.session_state.press_s = st.session_state.weather_data['pressure']

    # Quick Scenario Dropdown
    scenario_options = ['Custom', 'Clear Sky', 'Heavy Monsoon', 'Approaching Typhoon']
    selected_scenario = st.selectbox("Quick Scenario", scenario_options, index=0)

    # Update session state when scenario changes
    if selected_scenario != st.session_state.get('previous_scenario', 'Custom'):
        if selected_scenario != 'Custom':
            preset = scenarios[selected_scenario]
            st.session_state.rain_s = preset['rain']
            st.session_state.rain72_s = preset['rain_last_3_days']
            st.session_state.hum_s = preset['humidi']
            st.session_state.max_s = preset['max']
            st.session_state.min_s = preset['min']
            st.session_state.wind_s = preset['wind']
            st.session_state.cloud_s = preset['cloud']
            st.session_state.press_s = preset['pressure']
        st.session_state.previous_scenario = selected_scenario

    # Reset to Actual Weather button (only if data available)
    if st.session_state.weather_data is not None:
        st.button("Reset to Actual Weather", on_click=reset_to_actual)

    # Dashboard Layout: Side-by-side layout
    left_col, right_col = st.columns([1, 1])

    with left_col:
        # Sliders in compact grid (nested columns)
        slider_col1, slider_col2 = st.columns(2)
        with slider_col1:
            s_rain = st.slider("24h Rainfall (mm)", 0.0, 300.0, key="rain_s", step=0.1, help=">50mm is heavy rain, increases flood risk significantly.")
            s_rain_72h = st.slider("72h Total (mm)", 0.0, 500.0, key="rain72_s", step=0.1, help="Accumulated rainfall over 3 days; high values indicate soil saturation.")
            s_humidi = st.slider("Humidity (%)", 0.0, 100.0, key="hum_s", step=0.1, help="High humidity traps water vapor, amplifying rain's impact.")
            s_max = st.slider("Max Temp (°C)", 0.0, 50.0, key="max_s", step=0.1, help="Higher temperatures can evaporate moisture, potentially lowering risk.")
        with slider_col2:
            s_min = st.slider("Min Temp (°C)", 0.0, 40.0, key="min_s", step=0.1, help="Lower temperatures may indicate cooler fronts or storms.")
            s_wind = st.slider("Wind (m/s)", 0.0, 20.0, key="wind_s", step=0.1, help="Strong winds can drive storm surges or rapid weather changes.")
            s_cloud = st.slider("Cloud (%)", 0.0, 100.0, key="cloud_s", step=0.1, help="Cloud cover indicates potential for precipitation.")
            s_pressure = st.slider("Pressure (hPa)", 900.0, 1100.0, key="press_s", step=0.1, help="<1005hPa indicates a storm approaching.")

        # Create test data with overrides using local variables
        test_weather = base_weather.copy()
        test_weather.update({
            'rain': s_rain,
            'rain_last_3_days': s_rain_72h,
            'humidi': s_humidi,
            'max': s_max,
            'min': s_min,
            'wind': s_wind,
            'cloud': s_cloud,
            'pressure': s_pressure,
            'temp_diff': s_max - s_min,
            'rain_humidi_interaction': s_rain * s_humidi
        })
        ml_features = ['max', 'min', 'wind', 'rain', 'humidi', 'cloud', 'pressure', 'month', 'rain_last_3_days', 'temp_diff', 'rain_humidi_interaction']
        test_input_df = pd.DataFrame([test_weather])[ml_features]

        # Predict with test data
        test_proba = model.predict_proba(test_input_df)[0]
        # Enhanced rule-based override for simulator: boost high risk if high rainfall or other extreme conditions
        boost = 0
        if test_weather['rain'] > 50:
            boost += 0.2
        if test_weather['rain_last_3_days'] > 150:
            boost += 0.2
        if test_weather['humidi'] > 80:
            boost += 0.1
        if test_weather['wind'] > 10:
            boost += 0.1
        if boost > 0:
            test_proba[2] = min(1.0, test_proba[2] + boost)
            test_proba[0] = max(0, test_proba[0] - boost * 0.5)
            test_proba[1] = max(0, min(1, test_proba[1] - boost * 0.5))
            # Normalize probabilities
            total = sum(test_proba)
            test_proba = [p / total for p in test_proba]
        thresholds = {0: 0.5, 1: 0.3, 2: 0.2}
        test_prediction = 0
        if test_proba[2] > thresholds[2]:
            test_prediction = 2
        elif test_proba[1] > thresholds[1]:
            test_prediction = 1
        risk_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
        test_risk = risk_labels[test_prediction]
        test_confidence = max(test_proba) * 100

        # Interactive Calculation Explanation
        with st.expander("Interactive Calculation Breakdown", expanded=True):
            st.write("### Prediction Pipeline")

            # Create three visual stages
            step1, step2, step3 = st.columns(3)

            with step1:
                st.markdown("#### 1. ML Engine")
                st.caption("Analyzing 11 weather features")
                st.code(f"Low: {base_proba[0]:.1%}\nMed: {base_proba[1]:.1%}\nHigh: {base_proba[2]:.1%}")

            with step2:
                st.markdown("#### 2. Expert Rules")
                st.caption("Applying local thresholds")
                if boost > 0:
                    st.error(f"Boost: +{boost:.0%}")
                else:
                    st.success("Normal")

                # Status Tiles for Rules
                st.write("**Rule Engine Activity:**")

                def get_status(condition):
                    return "ACTIVE" if condition else "INACTIVE"

                def get_icon(condition):
                    return "🔴" if condition else "🟢"

                r1, r2 = st.columns(2)
                r1.markdown(f"{get_icon(test_weather['rain'] > 50)} **Heavy Rain**\n{get_status(test_weather['rain'] > 50)}")
                r2.markdown(f"{get_icon(test_weather['rain_last_3_days'] > 150)} **Saturated Soil**\n{get_status(test_weather['rain_last_3_days'] > 150)}")

                r3, r4 = st.columns(2)
                r3.markdown(f"{get_icon(test_weather['humidi'] > 80)} **High Humidity**\n{get_status(test_weather['humidi'] > 80)}")
                r4.markdown(f"{get_icon(test_weather['wind'] > 10)} **Storm Wind**\n{get_status(test_weather['wind'] > 10)}")

            with step3:
                st.markdown("#### 3. Final Result")
                st.caption("Normalized Probability")
                color = "red" if test_risk == "High" else "orange" if test_risk == "Medium" else "green"
                st.markdown(f"<h2 style='color:{color};'>{test_risk}</h2>", unsafe_allow_html=True)
                st.metric("Confidence", f"{test_confidence:.1f}%")

            # Visualize the "Boost" Logic with a Horizontal Bar Chart
            st.write("### Risk Contribution Factors")
            impact_data = {
                "Base Model": base_proba[2],
                "Rainfall Boost": 0.2 if test_weather['rain'] > 50 else 0,
                "Saturation Boost": 0.2 if test_weather['rain_last_3_days'] > 150 else 0,
                "Humidity Boost": 0.1 if test_weather['humidi'] > 80 else 0,
                "Wind Boost": 0.1 if test_weather['wind'] > 10 else 0
            }

            fig_impact = go.Figure(go.Bar(
                x=list(impact_data.values()),
                y=list(impact_data.keys()),
                orientation='h',
                marker_color=['#1c83e1', '#ff4b4b', '#ff4b4b', '#ff4b4b', '#ff4b4b']
            ))
            fig_impact.update_layout(
                title="Factors Increasing High-Risk Probability",
                height=300,
                xaxis_title="Probability Boost",
                yaxis_title=""
            )
            st.plotly_chart(fig_impact, use_container_width=True)

            # Mathematical Clarity
            st.write("### Mathematical Formula")
            st.latex(r"P(Risk_{final}) = \frac{Clip(P_{base} + Boost, 0, 1)}{\sum Clip(P + Boost)}")
            st.caption("Probabilities are clipped to [0,1] range and renormalized to sum to 1")

    with right_col:
        # Update Results Bar
        with results_bar.container():
            rb_col1, rb_col2 = st.columns(2)
            with rb_col1:
                st.metric("Predicted Risk", test_risk)
            with rb_col2:
                st.metric("Confidence Score", f"{test_confidence:.1f}%")

        # Display risk level and confidence
        if test_risk == 'Low':
            st.success(f"### 🟢 {test_risk} RISK")
        elif test_risk == 'Medium':
            st.warning(f"### 🟡 {test_risk} RISK")
        else:
            st.error(f"### 🔴 {test_risk} RISK")
        st.metric("Simulated Confidence", f"{test_confidence:.1f}%")

        # Show probability change
        st.write("**Probability Changes:**")
        for i, label in enumerate(['Low', 'Medium', 'High']):
            delta = test_proba[i] - base_proba[i]
            st.metric(f"{label} Risk", f"{test_proba[i]:.1%}", f"{delta:+.1%}")

# 4. Interactive "What-If" Scenario Simulator (always available)
st.divider()
with st.expander("🧪 Scenario Simulator (Manual Override)"):
    scenario_simulator()
