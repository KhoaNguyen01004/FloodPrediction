import json
import re
from typing import Dict, Tuple


def dms_to_decimal(dms_str: str) -> float:
    """
    Convert DMS (Degrees, Minutes, Seconds) coordinate string to decimal degrees.

    Args:
        dms_str: DMS string like "10°46′32″N" or "106°42′07″E"

    Returns:
        float: Decimal degree coordinate

    Raises:
        ValueError: If DMS string format is invalid
    """
    # Handle already decimal format like "21.00°N"
    if '.' in dms_str and '°' in dms_str:
        decimal_part = float(dms_str.split('°')[0])
        direction = dms_str[-1]
    else:
        # Parse DMS format: "10°46′32″N"
        match = re.match(r'(\d+)°(\d+)′(\d+)″?([NSEW])', dms_str)
        if not match:
            # Try simpler format: "10°46′N"
            match = re.match(r'(\d+)°(\d+)′([NSEW])', dms_str)
            if not match:
                raise ValueError(f"Invalid DMS format: {dms_str}")

        if len(match.groups()) == 4:
            degrees, minutes, seconds, direction = match.groups()
            seconds = int(seconds)
        else:
            degrees, minutes, direction = match.groups()
            seconds = 0

        degrees = int(degrees)
        minutes = int(minutes)

        # Convert to decimal
        decimal_part = degrees + minutes / 60 + seconds / 3600

    # Apply sign based on direction
    if direction in ['S', 'W']:
        decimal_part = -decimal_part

    return decimal_part


def load_city_coordinates(json_path: str = 'data/raw/vietnam_city_coords.json') -> Dict[str, Dict[str, float]]:
    """
    Load Vietnamese city coordinates from JSON file.

    Args:
        json_path: Path to the JSON file containing city coordinates

    Returns:
        Dict mapping city names to {"lat": float, "lon": float}

    Raises:
        FileNotFoundError: If JSON file is not found
        json.JSONDecodeError: If JSON is malformed
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Coordinates file not found: {json_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON format in {json_path}: {e.msg}", e.doc, e.pos)

    coordinates = {}

    for item in data:
        try:
            city = item['location']
            lat = item['lat']
            lon = item['lon']

            coordinates[city] = {"lat": lat, "lon": lon}

        except KeyError as e:
            raise ValueError(f"Missing coordinate data for city {item}: {e}")

    return coordinates


# Global cache for coordinates
_CITY_COORDINATES = None


def get_lat_lon(city_name: str) -> Tuple[float, float]:
    """
    Get latitude and longitude coordinates for a Vietnamese city.

    Args:
        city_name: Name of the Vietnamese city

    Returns:
        Tuple of (latitude, longitude) as floats

    Raises:
        ValueError: If city name is not found in coordinates
    """
    global _CITY_COORDINATES

    if _CITY_COORDINATES is None:
        _CITY_COORDINATES = load_city_coordinates()

    if city_name not in _CITY_COORDINATES:
        available_cities = list(_CITY_COORDINATES.keys())
        raise ValueError(f"City '{city_name}' not found. Available cities: {available_cities}")

    coords = _CITY_COORDINATES[city_name]
    return coords["lat"], coords["lon"]


# Example usage and testing
if __name__ == "__main__":
    # Test the functions
    try:
        coords = load_city_coordinates()
        print(f"Loaded coordinates for {len(coords)} cities")

        # Test a few cities
        test_cities = ["Hanoi", "Ho Chi Minh City"]
        for city in test_cities:
            if city in coords:
                lat, lon = get_lat_lon(city)
                print(f"{city}: {lat:.6f}, {lon:.6f}")
            else:
                print(f"{city} not found")

    except Exception as e:
        print(f"Error: {e}")
