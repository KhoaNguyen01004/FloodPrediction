import streamlit as st
import joblib
import json
import numpy as np
import plotly.graph_objects as go
from .config import VIETNAM_CITIES, FLOOD_THRESHOLDS
import unicodedata

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# --- CACHED LOADERS ---
@st.cache_resource
def load_model():
    return joblib.load('models/flood_model.pkl')

@st.cache_resource
def load_encoder():
    return joblib.load('models/city_encoder.pkl')

@st.cache_resource
def load_region_encoder():
    # Ensuring this returns the loaded object clearly
    model = joblib.load('models/region_encoder.pkl')
    return model

@st.cache_resource
def load_features():
    return joblib.load('models/features.pkl')

@st.cache_resource
def load_feature_importance():
    return joblib.load('models/feature_importance.pkl')

@st.cache_data
def load_coords():
    with open('data/raw/vietnam_city_coords.json', 'r', encoding='utf-8') as f:
        return json.load(f)

# --- UI COMPONENTS ---
def display_calculation_breakdown(base_proba, boost, weather_data, risk, confidence):
    """
    Display the interactive calculation breakdown for flood risk prediction.
    """
    # Note: Removed redundant imports of st and go from inside the function
    
    st.write("### Prediction Pipeline")

    # Create three visual stages
    step1, step2, step3 = st.columns(3)

    with step1:
        st.markdown("#### 1. ML Engine")
        st.caption("Analyzing weather and city features")
        st.code(f"Low: {base_proba[0]:.1%}\nMed: {base_proba[1]:.1%}\nHigh: {base_proba[2]:.1%}")

    with step2:
        st.markdown("#### 2. Expert Rules")
        st.caption("Applying local thresholds")
        if boost > 0:
            st.error(f"Boost: +{boost:.0%}")
        else:
            st.success("Normal")

        st.write("**Rule Engine Activity:**")

        def get_status(condition):
            return "ACTIVE" if condition else "INACTIVE"

        def get_icon(condition):
            return "🔴" if condition else "🟢"

        r1, r2 = st.columns(2)
        r1.markdown(f"{get_icon(weather_data['rain'] > 35)} **Heavy Rain**\n{get_status(weather_data['rain'] > 35)}")
        r2.markdown(f"{get_icon(weather_data['rain_last_3_days'] > 75)} **Saturated Soil**\n{get_status(weather_data['rain_last_3_days'] > 75)}")

        r3, r4 = st.columns(2)
        r3.markdown(f"{get_icon(weather_data['humidi'] > 70)} **High Humidity**\n{get_status(weather_data['humidi'] > 70)}")
        r4.markdown(f"{get_icon(weather_data['wind'] > 8)} **Storm Wind**\n{get_status(weather_data['wind'] > 8)}")

    with step3:
        st.markdown("#### 3. Final Result")
        st.caption("Normalized Probability")
        color = "red" if risk == "High" else "orange" if risk == "Medium" else "green"
        st.markdown(f"<h2 style='color:{color};'>{risk}</h2>", unsafe_allow_html=True)
        st.metric("Confidence", f"{confidence:.1f}%")

    # --- BAR CHART SECTION ---
    st.write("### Risk Contribution Factors")
    impact_data = {
        "Base Model": base_proba[2],
        "Rainfall Boost": 0.2 if weather_data['rain'] > 35 else 0,
        "Saturation Boost": 0.2 if weather_data['rain_last_3_days'] > 75 else 0,
        "Humidity Boost": 0.1 if weather_data['humidi'] > 70 else 0,
        "Wind Boost": 0.1 if weather_data['wind'] > 8 else 0
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
        yaxis_title="",
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    # FIX 1: Provide a unique key to prevent StreamlitDuplicateElementId
    # FIX 2: Replace use_container_width=True with width='stretch' for 2026 compatibility
    # Increment chart counter for unique key
    if 'chart_counter' not in st.session_state:
        st.session_state.chart_counter = 0
    st.session_state.chart_counter += 1
    st.plotly_chart(fig_impact, width='stretch', key=f"impact_chart_{st.session_state.chart_counter}")

    st.write("### Mathematical Formula")
    st.latex(r"P(Risk_{final}) = \frac{Clip(P_{base} + Boost, 0, 1)}{\sum Clip(P + Boost)}")
    st.caption("Probabilities are clipped to [0,1] range and renormalized to sum to 1")

CSS_STYLES = """
    <style>
    .metric-card { background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #1c83e1; }
    .risk-high { color: #ff4b4b; font-weight: bold; }
    .risk-med { color: #ffa500; font-weight: bold; }
    .risk-low { color: #008000; font-weight: bold; }
    </style>
    """