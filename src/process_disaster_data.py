import pandas as pd
import json
import os
from datetime import datetime
import math
import numpy as np
from utils import strip_accents

def load_city_coords():
    """Load city coordinates from JSON."""
    with open('data/raw/vietnam_city_coords.json', 'r') as f:
        coords = json.load(f)
    return [(item['location'], item['lat'], item['lon']) for item in coords]

def find_closest_city(lat, lon, cities):
    """Find the closest city to given lat/lon."""
    min_dist = float('inf')
    closest = None
    for name, c_lat, c_lon in cities:
        dist = math.sqrt((lat - c_lat)**2 + (lon - c_lon)**2)
        if dist < min_dist:
            min_dist = dist
            closest = name.lower()
    return closest

def create_location_to_cities_mapping():
    """Map disaster location strings to list of affected subdivisions (2026 Administrative Structure)."""
    mapping = {
        'hanoi': ['Hanoi'],
        'ha noi': ['Hanoi'],
        'ho chi minh city': ['Ho Chi Minh City'],
        'saigon': ['Ho Chi Minh City'],
        'hai phong': ['Hai Phong'],
        'da nang': ['Da Nang'],
        'can tho': ['Can Tho'],
        'hue': ['Hue'],
        'mekong delta': ['Dong Thap', 'Can Tho', 'An Giang', 'Vinh Long', 'Ca Mau'],
        'central': ['Da Nang', 'Quang Ngai', 'Khanh Hoa'],
        'north': ['Hanoi', 'Quang Ninh', 'Thai Nguyên', 'Phu Tho', 'Bac Ninh', 'Hung Yen', 'Ninh Binh'],
        'southern coast': ['Khanh Hoa', 'Lam Dong'],
        'china sea coast': ['Hai Phong', 'Quang Ninh'],
        'near cambodia border': ['An Giang', 'Tay Ninh', 'Dong Nai'],
        'quang ngai province': ['Quang Ngai'],
        'binh dinh': ['Quang Ngai'],
        'thanh hoa province': ['Thanh Hoa'],
        'nghe an province': ['Nghe An'],
        'ha tinh province': ['Ha Tinh'],
        'thanh hoa, nghe an, ha tinh provinces': ['Thanh Hoa', 'Nghe An', 'Ha Tinh'],
        'quang tri province': ['Quang Tri'],
        'thua thien hue province': ['Hue'],
        'quang tri, thua thien hue provinces': ['Quang Tri', 'Hue'],
        'gia lai province': ['Gia Lai'],
        'kon tum province': ['Gia Lai'],
        'gia lai kon tum': ['Gia Lai'],
        'dak lak province': ['Dak Lak'],
        'lam dong province': ['Lam Dong'],
        'khanh hoa province': ['Khanh Hoa'],
        'dong nai province': ['Dong Nai'],
        'tay ninh province': ['Tay Ninh'],
        'vinh long province': ['Vinh Long'],
        'dong thap province': ['Dong Thap'],
        'an giang province': ['An Giang'],
        'ca mau province': ['Ca Mau'],
        'cao bang province': ['Cao Bang'],
        'lai chau province': ['Lai Chau'],
        'dien bien province': ['Dien Bien'],
        'son la province': ['Son La'],
        'lang son province': ['Lang Son'],
        'tuyen quang province': ['Tuyen Quang'],
        'lao cai province': ['Lao Cai'],
        'thai nguyen province': ['Thai Nguyên'],
        'phu tho province': ['Phu Tho'],
        'bac ninh province': ['Bac Ninh'],
        'hung yen province': ['Hung Yen'],
        'ninh binh province': ['Ninh Binh'],
        '100 km east from hanoi': ['Bac Ninh', 'Hung Yen'],
        'northern mountainous': ['Cao Bang', 'Lai Chau', 'Dien Bien', 'Son La', 'Lang Son', 'Tuyen Quang', 'Lao Cai'],
        'red river delta': ['Hanoi', 'Hai Phong', 'Quang Ninh', 'Thai Nguyên', 'Phu Tho', 'Bac Ninh', 'Hung Yen', 'Ninh Binh'],
        'central coast': ['Thanh Hoa', 'Nghe An', 'Ha Tinh', 'Quang Tri', 'Hue', 'Da Nang', 'Quang Ngai'],
        'central highlands': ['Gia Lai', 'Dak Lak', 'Lam Dong', 'Khanh Hoa'],
        'southeast': ['Ho Chi Minh City', 'Dong Nai', 'Tay Ninh'],
        'mekong river delta': ['Can Tho', 'An Giang', 'Dong Thap', 'Vinh Long', 'Ca Mau'],
    }
    return mapping

def create_city_to_region_mapping():
    """Map individual cities to broader regions for debiasing."""
    region_mapping = {
        # North
        'Hanoi': 'North', 'Hai Phong': 'North', 'Quang Ninh': 'North', 'Thai Nguyên': 'North',
        'Phu Tho': 'North', 'Bac Ninh': 'North', 'Hung Yen': 'North', 'Ninh Binh': 'North',
        'Cao Bang': 'North', 'Lai Chau': 'North', 'Dien Bien': 'North', 'Son La': 'North',
        'Lang Son': 'North', 'Tuyen Quang': 'North', 'Lao Cai': 'North',

        # Central
        'Thanh Hoa': 'Central', 'Nghe An': 'Central', 'Ha Tinh': 'Central', 'Quang Tri': 'Central',
        'Hue': 'Central', 'Da Nang': 'Central', 'Quang Ngai': 'Central', 'Gia Lai': 'Central',
        'Dak Lak': 'Central', 'Lam Dong': 'Central', 'Khanh Hoa': 'Central',

        # South
        'Ho Chi Minh City': 'South', 'Dong Nai': 'South', 'Tay Ninh': 'South', 'Can Tho': 'South',
        'An Giang': 'South', 'Dong Thap': 'South', 'Vinh Long': 'South', 'Ca Mau': 'South'
    }
    return region_mapping

def process_disaster_data():
    """Process disaster data for flood prediction training."""
    # Load city coordinates
    city_coords = load_city_coords()
    city_dict = {name.lower(): (lat, lon) for name, lat, lon in city_coords}

    # Load region mapping for debiasing
    region_mapping = create_city_to_region_mapping()

    # Load disaster data
    df = pd.read_csv('data/raw/disaster_in_vietnam.csv')

    # Normalize city names in disaster data to match UI
    df['Location'] = df['Location'].apply(lambda x: strip_accents(str(x)) if pd.notna(x) else x)

    # Filter for floods in Vietnam, 2005-2023
    df = df[(df['Disaster Type'] == 'Flood') &
            (df['Start Year'] >= 2005) & (df['Start Year'] <= 2023) &
            (df['Country'] == 'Viet Nam')]

    # Use Start Year, Start Month
    df['year'] = df['Start Year']
    df['month'] = df['Start Month']

    # Location mapping
    location_mapping = create_location_to_cities_mapping()

    def add_gaussian_noise(sample, std_factor=0.02):
        """Add Gaussian noise to weather features for data augmentation."""
        noisy_sample = sample.copy()
        for key in ['max', 'min', 'wind', 'rain', 'humidi', 'cloud', 'pressure', 'rain_last_3_days']:
            if key in noisy_sample:
                value = noisy_sample[key]
                noise = np.random.normal(0, abs(value) * std_factor)
                noisy_sample[key] = max(0, value + noise)  # Ensure non-negative
        return noisy_sample

    # Create training data
    training_data = []

    for city_name, lat, lon in city_coords:
        for year in range(2005, 2024):
            for month in range(1, 13):
                risk = 0  # Default Low
                for _, flood in df[(df['year'] == year) & (df['month'] == month)].iterrows():
                    loc = flood['Location'].lower()
                    affected_cities = []
                    for key, cities in location_mapping.items():
                        if key in loc:
                            affected_cities.extend(cities)
                    if city_name in affected_cities:
                        # Handle NaNs by treating as 0
                        damages = flood.get("Total Damage ('000 US$)", 0) or 0
                        deaths = flood.get('Total Deaths', 0) or 0
                        affected = flood.get('Total Affected', 0) or 0
                        injured = flood.get('Total Injured', 0) or 0

                        # High Risk: Total Damage > 500 OR Total Deaths > 0
                        if damages > 500 or deaths > 0:
                            risk = max(risk, 2)
                        # Medium Risk: Total Damage > 0 OR Total Affected > 5 OR Total Injured > 0
                        elif damages > 0 or affected > 5 or injured > 0:
                            risk = max(risk, 1)

                # Generate synthetic weather features based on risk with overlapping Gaussian distributions
                if risk == 0:  # Low
                    rain = np.random.normal(10, 15)  # Mean 10, std 15, can go negative but we'll clip
                    max_temp = np.random.normal(30, 5)
                    min_temp = np.random.normal(25, 5)
                    wind = np.random.normal(3, 2)
                    humidi = np.random.normal(50, 15)
                    cloud = np.random.normal(40, 20)
                    pressure = np.random.normal(1013, 5)
                    rain_last_3_days = np.random.normal(25, 25)
                elif risk == 1:  # Medium
                    rain = np.random.normal(60, 20)  # Overlaps with Low and High
                    max_temp = np.random.normal(25, 5)
                    min_temp = np.random.normal(20, 5)
                    wind = np.random.normal(8, 2)
                    humidi = np.random.normal(80, 8)
                    cloud = np.random.normal(65, 20)
                    pressure = np.random.normal(1005, 3)
                    rain_last_3_days = np.random.normal(150, 50)
                else:  # High
                    rain = np.random.normal(150, 50)  # Overlaps with Medium
                    max_temp = np.random.normal(20, 5)
                    min_temp = np.random.normal(15, 5)
                    wind = np.random.normal(18, 5)
                    humidi = np.random.normal(95, 5)
                    cloud = np.random.normal(80, 15)
                    pressure = np.random.normal(990, 10)
                    rain_last_3_days = np.random.normal(300, 100)

                # Clip to reasonable ranges to avoid unrealistic values
                rain = np.clip(rain, 0, 500)
                max_temp = np.clip(max_temp, 10, 40)
                min_temp = np.clip(min_temp, 5, 35)
                wind = np.clip(wind, 0, 30)
                humidi = np.clip(humidi, 0, 100)
                cloud = np.clip(cloud, 0, 100)
                pressure = np.clip(pressure, 980, 1020)
                rain_last_3_days = np.clip(rain_last_3_days, 0, 600)

                # Add random noise to introduce chaos (5% of the time, mess up the data)
                if np.random.random() < 0.05:
                    humidi = np.random.uniform(30, 100)
                    cloud = np.random.uniform(0, 100)
                    rain = np.random.uniform(0, 250)
                    rain_last_3_days = np.random.uniform(0, 500)
                    max_temp = np.random.uniform(10, 35)
                    min_temp = np.random.uniform(5, 30)
                    wind = np.random.uniform(1, 25)

                # Get region for debiasing
                region = region_mapping.get(city_name, 'Unknown')

                sample = {
                    'city': city_name,
                    'region': region,
                    'max': max_temp,
                    'min': min_temp,
                    'wind': wind,
                    'rain': rain,
                    'humidi': humidi,
                    'cloud': cloud,
                    'pressure': pressure,
                    'month': month,
                    'rain_last_3_days': rain_last_3_days,
                    'flood_risk': risk
                }

                # Synthetic Booster: oversample High and Medium risk samples with Gaussian noise, reduced to prevent overfitting
                if risk == 2:
                    for _ in range(5):  # Reduced from 15
                        noisy_sample = add_gaussian_noise(sample)
                        training_data.append(noisy_sample)
                elif risk == 1:
                    for _ in range(50):  # Reduced from 1000
                        noisy_sample = add_gaussian_noise(sample)
                        training_data.append(noisy_sample)
                else:
                    training_data.append(sample)

    # Create DataFrame
    training_df = pd.DataFrame(training_data)

    # Save processed data
    os.makedirs('data/processed', exist_ok=True)
    training_df.to_csv('data/processed/flood_training.csv', index=False)

    print(f"Processed {len(training_df)} training samples")
    print(training_df['flood_risk'].value_counts())

    # Generate feature correlation heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns

    os.makedirs('models', exist_ok=True)
    plt.figure(figsize=(10,8))
    corr = training_df[['max', 'min', 'wind', 'rain', 'humidi', 'cloud', 'pressure', 'month', 'rain_last_3_days', 'flood_risk']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Feature Correlation Heatmap')
    plt.tight_layout()
    plt.savefig('models/feature_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Feature correlation heatmap saved to models/feature_correlation.png")

if __name__ == '__main__':
    process_disaster_data()
