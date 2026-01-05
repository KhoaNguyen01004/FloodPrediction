import requests
from datetime import date
from src.coordinates import get_lat_lon
from src.config import OPENWEATHER_API_KEY

def get_weather_data(city, target_date=None):
    """
    Fetch weather data. If target_date is today, use OpenWeather (Live).
    If target_date is in the past, use Open-Meteo (Historical).
    """
    lat, lon = get_lat_lon(city)
    
    # Default to today if no date provided
    if target_date is None:
        target_date = date.today()

    # DECISION: Live vs Historical
    if target_date >= date.today():
        return _get_live_weather(lat, lon)
    else:
        return _get_historical_weather(lat, lon, target_date)

def _get_live_weather(lat, lon):
    """Fetches real-time data from OpenWeatherMap"""
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()

    rain_value = data.get('rain', {}).get('1h', 0)
    return {
        'max': data['main']['temp_max'],
        'min': data['main']['temp_min'],
        'wind': data['wind']['speed'],
        'rain': rain_value,
        'rain_last_3_days': rain_value,  # Set to current rain as minimum
        'humidi': data['main']['humidity'],
        'cloud': data['clouds']['all'],
        'pressure': data['main']['pressure'],
        'month': date.today().month
    }

def _get_historical_weather(lat, lon, target_date):
    from datetime import timedelta
    start_date = target_date - timedelta(days=2)
    
    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
           f"&start_date={start_date.strftime('%Y-%m-%d')}&end_date={target_date.strftime('%Y-%m-%d')}"
           f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,precipitation,cloud_cover,wind_speed_10m"
           f"&timezone=auto")

    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    h = data['hourly']

    # 1. Cumulative rain over all 72 hours (The "Saturation" feature)
    rain_last_3_days = sum(h['precipitation'])

    # 2. Extract the last 24 hours (The "Target Day" features)
    target_day_temp = h['temperature_2m'][-24:]
    target_day_precip = h['precipitation'][-24:]
    
    # 3. Snapshot values (Noon of target day = index -12)
    return {
        'max': max(target_day_temp),
        'min': min(target_day_temp),
        'wind': h['wind_speed_10m'][-12], 
        'rain': sum(target_day_precip),
        'rain_last_3_days': rain_last_3_days,
        'humidi': h['relative_humidity_2m'][-12],
        'cloud': h['cloud_cover'][-12],
        'pressure': h['surface_pressure'][-12]
    }
