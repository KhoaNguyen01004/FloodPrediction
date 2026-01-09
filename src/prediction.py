import pandas as pd
import numpy as np
from datetime import date
from src.weather_api import get_weather_data
from src.utils import (
    load_model, 
    load_encoder, 
    load_coords, 
    load_features, 
    load_region_encoder
)

def perform_prediction(city, target_date):
    """
    Perform flood risk prediction for given city and date.
    Returns weather_data, proba, risk, confidence, ml_features, data_source, base_proba
    """
    # 1. INITIALIZE & LOAD RESOURCES
    model = load_model()
    city_encoder = load_encoder()
    region_encoder = load_region_encoder()
    expected_features = load_features()
    
    # Load training data once for region mapping and thresholding
    df_train = pd.read_csv('data/processed/flood_training.csv')

    # Pre-compute region mapping for faster lookups
    region_map = df_train.set_index('city')['region'].to_dict()

    # 2. DATA SOURCE SELECTION
    date_diff = abs((target_date - date.today()).days)
    
    if date_diff <= 7:
        # Use live API data
        weather_data = get_weather_data(city, target_date)
        data_source = "Live Forecast"
    else:
        # Determine region for historical lookup using pre-computed dictionary
        city_region = region_map.get(city, 'North')
        
        # Filter by month and city first, then region if no city data
        month_df = df_train[(df_train['month'] == target_date.month) & (df_train['city'] == city)]
        if month_df.empty:
            month_df = df_train[(df_train['month'] == target_date.month) & (df_train['region'] == city_region)]
        
        if month_df.empty:
            month_df = df_train[df_train['month'] == target_date.month]

        if month_df.empty:
            # Fallback to overall monthly averages from training data
            overall_month_df = df_train[df_train['month'] == target_date.month]
            if not overall_month_df.empty:
                numeric_means = overall_month_df.select_dtypes(include=[np.number]).mean()
                weather_data = numeric_means.to_dict()
                weather_data['month'] = target_date.month
            else:
                # Ultimate fallback to hardcoded defaults if no data at all
                weather_data = {
                    'max': 25.0, 'min': 20.0, 'wind': 5.0, 'rain': 10.0,
                    'humidi': 70.0, 'cloud': 50.0, 'pressure': 1010.0,
                    'rain_last_3_days': 50.0, 'month': target_date.month
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

    # 3. FEATURE ENGINEERING
    # Required basic features
    required_keys = ['max', 'rain', 'humidi', 'month', 'rain_last_3_days', 'wind']
    for key in required_keys:
        if key not in weather_data:
            weather_data[key] = 0.0

    # Seasonal cyclical features
    weather_data['month_sin'] = np.sin(2 * np.pi * weather_data['month'] / 12)
    weather_data['month_cos'] = np.cos(2 * np.pi * weather_data['month'] / 12)

    # Approximate 7-day cumulative rain for soil saturation proxy using exponential decay
    k = 0.1  # decay rate per day
    weather_data['rain_last_7_days'] = weather_data['rain_last_3_days'] * (1 - np.exp(-k * 7)) / (1 - np.exp(-k * 3))

    # City Encoding (using DataFrame to avoid Feature Name warnings)
    selected_city = city if city in city_encoder.categories_[0] else 'Hanoi'
    city_input = pd.DataFrame([[selected_city]], columns=['city'])
    city_encoded = city_encoder.transform(city_input)
    city_encoded_df = pd.DataFrame(city_encoded, columns=city_encoder.get_feature_names_out(['city']))

    # Region Encoding
    city_lookup = df_train[df_train['city'] == selected_city]
    city_region = city_lookup['region'].iloc[0] if not city_lookup.empty else 'North'
    region_input = pd.DataFrame([[city_region]], columns=['region'])
    region_encoded = region_encoder.transform(region_input)
    region_encoded_df = pd.DataFrame(region_encoded, columns=region_encoder.get_feature_names_out(['region']))

    # Interaction Features
    weather_data['rain_north'] = weather_data['rain'] * (city_region == 'North')
    weather_data['rain_central'] = weather_data['rain'] * (city_region == 'Central')
    weather_data['rain_south'] = weather_data['rain'] * (city_region == 'South')

    # Construct input dataframe
    ml_features = ['max', 'rain', 'humidi', 'month_sin', 'month_cos']
    core_weather = ml_features + ['rain_north', 'rain_central', 'rain_south']
    weather_df = pd.DataFrame([weather_data])[core_weather]
    
    # CONCATENATION: Reset index to prevent "Arrays must be same length" errors
    input_df = pd.concat([
        weather_df.reset_index(drop=True), 
        region_encoded_df.reset_index(drop=True)
    ], axis=1)

    # Reindex to match the exact order the model expects
    input_df = input_df.reindex(columns=expected_features, fill_value=0)

    # Assert feature alignment to prevent silent accuracy degradation
    assert input_df.columns.tolist() == expected_features, f"Feature alignment mismatch: expected {expected_features}, got {input_df.columns.tolist()}"

    # 4. PREDICTION & EXPERT SYSTEM OVERRIDE
    proba = model.predict_proba(input_df)[0]
    base_proba = proba.copy()

    # Fixed thresholding for consistency across cities
    rain_threshold, rain_3d_threshold = 50.0, 100.0

    # Apply Boosting Rules
    boost = 0.0
    if weather_data['rain'] > rain_threshold: boost += 0.2
    if weather_data['rain_last_3_days'] > rain_3d_threshold: boost += 0.2
    if weather_data['humidi'] > 70: boost += 0.1
    if weather_data['wind'] > 8: boost += 0.1

    if boost > 0:
        # Correctly redistribute probability mass: boost high risk, scale down low and medium proportionally
        new_high = min(1.0, proba[2] + boost)
        remaining = 1.0 - new_high
        scale = remaining / (proba[0] + proba[1]) if (proba[0] + proba[1]) > 0 else 0
        proba[0] *= scale
        proba[1] *= scale
        proba[2] = new_high
        # Ensure no negative probabilities and sum to 1
        proba = np.clip(proba, 0, 1)
        proba = proba / proba.sum() if proba.sum() > 0 else np.array([1/3, 1/3, 1/3])

    # Final Classification based on custom thresholds for better sensitivity
    # (High risk prioritized if it crosses 20%)
    if proba[2] > 0.20:
        prediction = 2
    elif proba[1] > 0.30:
        prediction = 1
    else:
        prediction = 0

    risk_labels = {0: 'Low', 1: 'Medium', 2: 'High'}
    risk = risk_labels[prediction]
    confidence = proba[prediction] * 100

    return weather_data, proba, risk, confidence, ml_features, data_source, base_proba