import requests

def get_weather_data(city, api_key):
    """
    Fetch current weather data from OpenWeather API for a given city.
    Returns a dict with features: max, min, wind, rain, humidi, cloud, pressure
    """
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to fetch data: {response.status_code}")
    
    data = response.json()
    
    # Extract features
    temp = data['main']['temp']
    wind_speed = data['wind']['speed']
    humidity = data['main']['humidity']
    pressure = data['main']['pressure']
    clouds = data['clouds']['all']
    
    # Rain: check if 'rain' key exists (last 1h)
    rain = data.get('rain', {}).get('1h', 0.0)
    
    # For max and min, use current temp (since API doesn't provide daily max/min in free tier)
    max_temp = temp
    min_temp = temp
    
    return {
        'max': max_temp,
        'min': min_temp,
        'wind': wind_speed,
        'rain': rain,
        'humidi': humidity,
        'cloud': clouds,
        'pressure': pressure
    }
