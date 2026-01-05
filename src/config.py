# Configuration file for the Flood Prediction App

# OpenWeather API Key (replace with your actual key)
OPENWEATHER_API_KEY = "35cfdd6f5a3d81ab1bde396413da31db"

# Flood risk thresholds (mm rain)
FLOOD_THRESHOLDS = {
    'low': 20,      # Low: < 20mm
    'medium': 50    # Medium: 20-50mm, High: >50mm
}

# List of Vietnamese cities (34 subdivisions from vietnam_city_coords.json)
# src/config.py
VIETNAM_CITIES = [
    # 11 Units that remained unchanged (mostly mountainous/key areas)
    "Hanoi", "Hue", "Quang Ninh", "Thanh Hoa", "Nghe An", "Ha Tinh", 
    "Cao Bang", "Lai Chau", "Dien Bien", "Son La", "Lang Son",
    
    # 23 Newly formed/merged units
    "Ho Chi Minh City", "Hai Phong", "Da Nang", "Can Tho",
    "Tuyen Quang", "Lao Cai", "Thai Nguyen", "Phu Tho", "Bac Ninh", 
    "Hung Yen", "Ninh Binh", "Quang Tri", "Quang Ngai", "Gia Lai", 
    "Khanh Hoa", "Lam Dong", "Dak Lak", "Dong Nai", "Tay Ninh", 
    "Vinh Long", "Dong Thap", "An Giang", "Ca Mau"
]
