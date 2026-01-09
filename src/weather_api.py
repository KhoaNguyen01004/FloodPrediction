import requests
from datetime import date, timedelta, datetime
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
    # Get current weather for instant fields
    url_current = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url_current)
    response.raise_for_status()
    data = response.json()

    # Get forecast for today's cumulative rain
    url_forecast = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response_forecast = requests.get(url_forecast)
    response_forecast.raise_for_status()
    forecast_data = response_forecast.json()

    today_str = date.today().strftime('%Y-%m-%d')
    today_rain = 0
    for item in forecast_data['list']:
        if item['dt_txt'].startswith(today_str):
            today_rain += item.get('rain', {}).get('3h', 0)

    # Get past cumulatives
    past_6_days_rain = get_cumulative_rain_past_days(lat, lon, 6)
    past_2_days_rain = get_cumulative_rain_past_days(lat, lon, 2)

    return {
        'max': data['main']['temp_max'],
        'min': data['main']['temp_min'],
        'wind': data['wind']['speed'],
        'rain': today_rain,
        'rain_last_3_days': past_2_days_rain + today_rain,
        'rain_last_7_days': past_6_days_rain + today_rain,
        'humidi': data['main']['humidity'],
        'cloud': data['clouds']['all'],
        'pressure': data['main']['pressure'],
        'month': date.today().month
    }

def _get_historical_weather(lat, lon, target_date):
    today = date.today()
    days_diff = (today - target_date).days

    if days_diff <= 2:
        # Use Forecast API for recent dates to avoid archive delay
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&past_days={7}&hourly=temperature_2m,relative_humidity_2m,surface_pressure,precipitation,cloud_cover,wind_speed_10m"
               f"&timezone=auto")
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        h = data['hourly']

        # Find indices for the target_date
        target_indices = [i for i, time_str in enumerate(h['time']) if time_str.startswith(target_date.strftime('%Y-%m-%d'))]
        if not target_indices:
            raise ValueError(f"No data available for {target_date}")

        start_idx = target_indices[0]
        end_idx = target_indices[-1] + 1

        # Extract data for the target day
        target_day_temp = h['temperature_2m'][start_idx:end_idx]
        target_day_precip = h['precipitation'][start_idx:end_idx]
        target_day_humidity = h['relative_humidity_2m'][start_idx:end_idx]
        target_day_pressure = h['surface_pressure'][start_idx:end_idx]
        target_day_cloud = h['cloud_cover'][start_idx:end_idx]
        target_day_wind = h['wind_speed_10m'][start_idx:end_idx]

        # Cumulative rain over past 7 days up to target_date
        past_7_days_precip = h['precipitation'][:end_idx]
        rain_last_7_days = sum(past_7_days_precip)

        # Cumulative rain over past 3 days up to target_date
        past_3_days_precip = past_7_days_precip[-72:] if len(past_7_days_precip) >= 72 else past_7_days_precip
        rain_last_3_days = sum(past_3_days_precip)

        # Snapshot at noon (middle of the day)
        noon_idx = start_idx + len(target_day_temp) // 2

        return {
            'max': max(target_day_temp),
            'min': min(target_day_temp),
            'wind': target_day_wind[noon_idx - start_idx],
            'rain': sum(target_day_precip),
            'rain_last_3_days': rain_last_3_days,
            'rain_last_7_days': rain_last_7_days,
            'humidi': target_day_humidity[noon_idx - start_idx],
            'cloud': target_day_cloud[noon_idx - start_idx],
            'pressure': target_day_pressure[noon_idx - start_idx],
            'month': target_date.month
        }
    else:
        # Use Archive API for older dates
        start_date = target_date - timedelta(days=6)

        url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               f"&start_date={start_date.strftime('%Y-%m-%d')}&end_date={target_date.strftime('%Y-%m-%d')}"
               f"&hourly=temperature_2m,relative_humidity_2m,surface_pressure,precipitation,cloud_cover,wind_speed_10m"
               f"&timezone=auto")

        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        h = data['hourly']

        # 1. Cumulative rain over all 168 hours (7 days)
        rain_last_7_days = sum(h['precipitation'])

        # 2. Cumulative rain over last 72 hours (3 days)
        rain_last_3_days = sum(h['precipitation'][-72:]) if len(h['precipitation']) >= 72 else sum(h['precipitation'])

        # 3. Extract the last 24 hours (The "Target Day" features)
        target_day_temp = h['temperature_2m'][-24:]
        target_day_precip = h['precipitation'][-24:]

        # 4. Snapshot values (Noon of target day = index -12)
        return {
            'max': max(target_day_temp),
            'min': min(target_day_temp),
            'wind': h['wind_speed_10m'][-12],
            'rain': sum(target_day_precip),
            'rain_last_3_days': rain_last_3_days,
            'rain_last_7_days': rain_last_7_days,
            'humidi': h['relative_humidity_2m'][-12],
            'cloud': h['cloud_cover'][-12],
            'pressure': h['surface_pressure'][-12],
            'month': target_date.month
        }

def get_cumulative_rain_past_days(lat, lon, days):
    today = date.today()
    url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
           f"&past_days={days}&hourly=precipitation&timezone=auto")
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    # Sum precipitation for dates before today
    total_rain = 0
    for time_str, precip in zip(data['hourly']['time'], data['hourly']['precipitation']):
        entry_date = date.fromisoformat(time_str.split('T')[0])
        if entry_date < today:
            total_rain += precip
    return total_rain
