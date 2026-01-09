import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.utils import load_model, load_encoder, load_region_encoder, load_feature_importance, display_calculation_breakdown

@st.fragment
def scenario_simulator(city):
    """
    Interactive scenario simulator for flood risk prediction.
    """
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
            s_rain = st.slider("24h Rainfall (mm)", 0.0, 300.0, key="rain_s", step=0.1, help=">35mm is heavy rain, increases flood risk significantly.")
            s_rain_72h = st.slider("72h Total (mm)", 0.0, 500.0, key="rain72_s", step=0.1, help="Accumulated rainfall over 3 days; high values indicate soil saturation.")
            s_humidi = st.slider("Humidity (%)", 0.0, 100.0, key="hum_s", step=0.1, help="High humidity traps water vapor, amplifying rain's impact.")
            s_max = st.slider("Max Temp (°C)", 0.0, 50.0, key="max_s", step=0.1, help="Higher temperatures can evaporate moisture, potentially lowering risk.")
        with slider_col2:
            s_min = st.slider("Min Temp (°C)", 0.0, 40.0, key="min_s", step=0.1, help="Lower temperatures may indicate cooler fronts or storms.")
            s_wind = st.slider("Wind (m/s)", 0.0, 20.0, key="wind_s", step=0.1, help="Strong winds can drive storm surges or rapid weather changes.")
            s_cloud = st.slider("Cloud (%)", 0.0, 100.0, key="cloud_s", step=0.1, help="Cloud cover indicates potential for precipitation.")
            s_pressure = st.slider("Pressure (hPa)", 900.0, 1100.0, key="press_s", step=0.1, help="Low pressure indicates a storm approaching.")

        # Create test data with overrides using local variables
        test_weather = base_weather.copy()
        test_weather.update({
            'rain': s_rain,
            'rain_last_3_days': s_rain_72h,
            'humidi': s_humidi,
            'max': s_max,
            'wind': s_wind,
        })

        # Compute month_sin and month_cos for seasonal patterns
        test_weather['month_sin'] = np.sin(2 * np.pi * test_weather['month'] / 12)
        test_weather['month_cos'] = np.cos(2 * np.pi * test_weather['month'] / 12)

        # Load encoders and encode region (use same as main prediction)
        model = load_model()
        region_encoder = load_region_encoder()
        # Get region for the city
        df = pd.read_csv('data/processed/flood_training.csv')
        city_region = df[df['city'] == city]['region'].iloc[0] if not df[df['city'] == city].empty else 'North'
        region_input = pd.DataFrame([[city_region]], columns=['region'])
        region_encoded = region_encoder.transform(region_input)
        region_encoded_df = pd.DataFrame(region_encoded, columns=region_encoder.get_feature_names_out(['region']))

        # Create interaction features
        test_weather['rain_north'] = test_weather['rain'] * (city_region == 'North')
        test_weather['rain_central'] = test_weather['rain'] * (city_region == 'Central')
        test_weather['rain_south'] = test_weather['rain'] * (city_region == 'South')

        ml_features = ['max', 'rain', 'humidi', 'month_sin', 'month_cos', 'rain_north', 'rain_central', 'rain_south']
        test_weather_df = pd.DataFrame([test_weather])[ml_features]
        test_input_df = pd.concat([test_weather_df, region_encoded_df], axis=1)

        # Predict with test data
        test_proba = model.predict_proba(test_input_df)[0]
        # Enhanced rule-based override for simulator: boost high risk if high rainfall or other extreme conditions
        boost = 0
        if test_weather['rain'] > 35:
            boost += 0.2
        if test_weather['rain_last_3_days'] > 75:
            boost += 0.2
        if test_weather['humidi'] > 70:
            boost += 0.1
        if test_weather['wind'] > 8:
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

        # Feature Importance
        st.subheader("Feature Importance")
        feature_importance_df = load_feature_importance()
        fig_importance_sim = go.Figure(go.Bar(
            x=feature_importance_df['importance'],
            y=[f.replace('_', ' ').title() for f in feature_importance_df['feature']],
            orientation='h',
            marker_color='lightgreen'
        ))
        fig_importance_sim.update_layout(
            title="Global Feature Importance",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            height=300
        )
        st.plotly_chart(fig_importance_sim, width='stretch')
        st.caption("Adjusting sliders for features with higher importance will have a greater impact on the predicted risk.")

        # Interactive Calculation Explanation
        with st.expander("Interactive Calculation Breakdown", expanded=True):
            display_calculation_breakdown(base_proba, boost, test_weather, test_risk, test_confidence)

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
