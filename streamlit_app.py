import streamlit as st
import pandas as pd
from datetime import date
import plotly.graph_objects as go
import numpy as np
from src.config import VIETNAM_CITIES, FLOOD_THRESHOLDS
from src.utils import load_model, load_coords, load_region_encoder, load_features, load_feature_importance, CSS_STYLES, display_calculation_breakdown
from src.prediction import perform_prediction
from src.pdf_generator import generate_pdf_report
from src.scenario_simulator import scenario_simulator
from src.weather_api import get_weather_data



st.set_page_config(page_title="Vietnam Flood Predictor 2026", layout="wide")

# --- CUSTOM CSS FOR BETTER VISUALS ---
st.markdown(CSS_STYLES, unsafe_allow_html=True)

st.title("Vietnam Flood Prediction (2026 Admin Structure)")
st.write("Machine Learning analysis based on 34 First-Level Subdivisions.")

tab1, tab2 = st.tabs(["Flood Prediction", "Model Performance"])

with tab2:
    st.header("Model Performance Comparison")
    st.write("Comparison of different ML models on the test set (1671 samples).")

    # Hardcoded comparison data from recent run
    comparison_data = {
        'Model': ['XGBoost (Calibrated)', 'Random Forest', 'SVM', 'Neural Network'],
        'Accuracy': [0.991, 0.996, 0.974, 0.992],
        'Macro F1': [0.966, 0.985, 0.916, 0.975],
        'Medium Precision': [0.955, 0.977, 0.878, 0.977],
        'Medium Recall': [0.977, 1.000, 1.000, 1.000]
    }
    df_comp = pd.DataFrame(comparison_data)
    st.table(df_comp)

    # Bar chart for accuracy
    fig_acc = go.Figure(data=[go.Bar(x=df_comp['Model'], y=df_comp['Accuracy'], marker_color='blue')])
    fig_acc.update_layout(title='Model Accuracy Comparison', yaxis_title='Accuracy')
    st.plotly_chart(fig_acc)

    # Bar chart for Macro F1
    fig_f1 = go.Figure(data=[go.Bar(x=df_comp['Model'], y=df_comp['Macro F1'], marker_color='green')])
    fig_f1.update_layout(title='Macro F1 Score Comparison', yaxis_title='Macro F1')
    st.plotly_chart(fig_f1)

    st.write("**Key Insights:** Random Forest achieves the highest accuracy (0.996) and best handles the imbalanced Medium class. XGBoost with calibration is a strong alternative.")

with tab1:
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
if 'chart_counter' not in st.session_state:
    st.session_state.chart_counter = 0

if st.button("Analyze & Predict Risk"):
    try:
        weather_data, proba, risk, confidence, ml_features, data_source, base_proba = perform_prediction(city, target_date)

        # Recreate input_df for feature analysis
        df_train = pd.read_csv('data/processed/flood_training.csv')

        # Calculate boost for display purposes (using same logic as prediction.py)
        rain_threshold, rain_3d_threshold = 50.0, 100.0

        boost = 0
        if weather_data['rain'] > rain_threshold:
            boost += 0.2
        if weather_data['rain_last_3_days'] > rain_3d_threshold:
            boost += 0.2
        if weather_data['humidi'] > 70:
            boost += 0.1
        if weather_data['wind'] > 8:
            boost += 0.1
        region_map = df_train.set_index('city')['region'].to_dict()
        city_region = region_map.get(city, 'North')
        encoder = load_region_encoder()
        region_input = pd.DataFrame([[city_region]], columns=['region'])
        region_encoded = encoder.transform(region_input)
        region_encoded_df = pd.DataFrame(region_encoded, columns=encoder.get_feature_names_out(['region']))

        # Interaction Features
        weather_data_copy = weather_data.copy()
        weather_data_copy['rain_north'] = weather_data['rain'] * (city_region == 'North')
        weather_data_copy['rain_central'] = weather_data['rain'] * (city_region == 'Central')
        weather_data_copy['rain_south'] = weather_data['rain'] * (city_region == 'South')

        core_weather = ml_features + ['rain_north', 'rain_central', 'rain_south']
        weather_df = pd.DataFrame([weather_data_copy])[core_weather]
        input_df = pd.concat([weather_df, region_encoded_df], axis=1)

        expected_features = load_features()
        input_df = input_df.reindex(columns=expected_features, fill_value=0)

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

        # Store data in session state for persistence
        st.session_state.weather_data = weather_data
        st.session_state.proba = proba
        st.session_state.risk = risk
        st.session_state.confidence = confidence
        st.session_state.ml_features = ml_features
        st.session_state.city = city
        st.session_state.target_date = target_date

        # Sync slider session states for scenario simulator
        st.session_state.rain_s = weather_data['rain']
        st.session_state.rain72_s = weather_data['rain_last_3_days']
        st.session_state.hum_s = weather_data['humidi']
        st.session_state.max_s = weather_data['max']
        st.session_state.min_s = weather_data['min']
        st.session_state.wind_s = weather_data['wind']
        st.session_state.cloud_s = weather_data['cloud']
        st.session_state.press_s = weather_data['pressure']

        # Generate PDF report and store in session state
        st.session_state.pdf_data = generate_pdf_report(city, target_date, risk, confidence, weather_data)

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

            # Risk Gauge for Final Risk Score
            st.subheader("📊 Overall Risk Score")
            if risk == 'High':
                steps = [
                    {'range': [0, 20], 'color': "green"},
                    {'range': [20, 100], 'color': "red"}
                ]
            elif risk == 'Medium':
                steps = [
                    {'range': [0, 30], 'color': "green"},
                    {'range': [30, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "red"}
                ]
            else:
                steps = [
                    {'range': [0, 70], 'color': "green"},
                    {'range': [70, 100], 'color': "red"}
                ]
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence if risk != 'Low' else 100 - confidence,
                title={'text': f"Overall Risk: {risk}"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "black"},
                    'steps': steps
                }
            ))
            st.plotly_chart(fig_gauge, width='stretch')

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
                coords_data = load_coords()
                city_coord = next(item for item in coords_data if item["location"] == city)
                map_data = pd.DataFrame([{
                    'lat': city_coord['lat'],
                    'lon': city_coord['lon']
                }])
                st.map(map_data, zoom=7, width='stretch')
            except StopIteration:
                st.error(f"Coordinates for {city} not found in the database.")
                # Do not attempt to display map if coordinates not found

            # Interactive Calculation Breakdown
            with st.expander("Interactive Calculation Breakdown", expanded=True):
                display_calculation_breakdown(base_proba, boost, weather_data, risk, confidence)

        # 7. ENHANCED FEATURE ANALYSIS
        st.divider()
        st.subheader("📊 Why did the model predict this?")

        # Load feature importance
        feature_importance_df = load_feature_importance()

        # Display Feature Importance Chart
        st.write("**Feature Importance (Global Weights)**")
        st.write("These are the relative weights of each feature in the model's decision-making process, based on the XGBoost algorithm.")
        fig_importance = go.Figure(go.Bar(
            x=feature_importance_df['importance'],
            y=[f.replace('_', ' ').title() for f in feature_importance_df['feature']],
            orientation='h',
            marker_color='skyblue'
        ))
        fig_importance.update_layout(
            title="Feature Importance in Flood Risk Prediction",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=400
        )
        st.plotly_chart(fig_importance, width='stretch')

        # Get all features
        all_features = load_features()

        # Unit Mapping for all features
        units = {'max': '°C', 'rain': 'mm', 'humidi': '%', 'month_sin': '', 'month_cos': '', 'rain_north': 'mm', 'rain_central': 'mm', 'rain_south': 'mm', 'region_North': '', 'region_Central': '', 'region_South': '', 'region_Unknown': ''}

        # Create feature values table with importance
        feature_df = pd.DataFrame({
            'Feature': [f.replace('_', ' ').title() for f in all_features],
            'Value': [f"{input_df.iloc[0][f]:.1f}{units.get(f, '')}" if f in ['max', 'rain', 'humidi', 'month_sin', 'month_cos', 'rain_north', 'rain_central', 'rain_south'] else f"{int(input_df.iloc[0][f])}" for f in all_features],
            'Importance': [feature_importance_df.set_index('feature').loc[f, 'importance'] for f in all_features]
        })
        feature_df = feature_df.sort_values('Importance', ascending=False)
        st.table(feature_df)

        st.write("**Feature Impact Guide**")
        # Create comprehensive feature impact table with importance
        data_label = "Current" if target_date >= date.today() else "Historical"
        risk_notes = [
            "Warmer temperatures help dry the ground, reducing flood risk",
            "Heavy rainfall is the primary cause of flooding",
            "High humidity amplifies the effects of rain, making floods worse",
            "Seasonal patterns affect how weather impacts flood risk (sine component)",
            "Seasonal patterns affect how weather impacts flood risk (cosine component)",
            "Rainfall interaction with North region increases flood risk",
            "Rainfall interaction with Central region increases flood risk",
            "Rainfall interaction with South region increases flood risk",
            "North region has higher baseline flood risk",
            "Central region has moderate baseline flood risk",
            "South region has lower baseline flood risk",
            "Unknown region has no baseline risk data"
        ]
        impact_df = pd.DataFrame({
            "Feature": [f.replace('_', ' ').title() for f in all_features],
            data_label: [f"{weather_data[f]:.1f}{units.get(f, '')}" if f in weather_data and f in ['max', 'rain', 'humidi', 'month_sin', 'month_cos', 'rain_north', 'rain_central', 'rain_south'] else f"{input_df.iloc[0][f]:.1f}" if f in ['max', 'rain', 'humidi', 'month_sin', 'month_cos', 'rain_north', 'rain_central', 'rain_south'] else f"{int(input_df.iloc[0][f])}" for f in all_features],
            "Importance": [feature_importance_df.set_index('feature').loc[f, 'importance'] for f in all_features],
            "Risk Note": risk_notes
        })
        impact_df = impact_df.sort_values('Importance', ascending=False)
        st.table(impact_df)

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

# 4. Interactive "What-If" Scenario Simulator (always available)
st.divider()
with st.expander("🧪 Scenario Simulator (Manual Override)"):
    scenario_simulator(city)


