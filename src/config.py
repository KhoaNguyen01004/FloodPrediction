# Configuration file for the Flood Prediction App

# OpenWeather API Key (replace with your actual key)
OPENWEATHER_API_KEY = "35cfdd6f5a3d81ab1bde396413da31db"

# Flood risk thresholds (mm rain)
FLOOD_THRESHOLDS = {
    'low': 20,      # Low: < 20mm
    'medium': 50    # Medium: 20-50mm, High: >50mm
}

# List of Vietnamese cities (from dataset: 40 provinces/cities, Jan 2009 - Jun 2021)
VIETNAM_CITIES = [
    "Bac Lieu", "Ho Chi Minh City", "Tam Ky", "Ben Tre", "Hoa Binh", "Tan An", "Bien Hoa", "Hong Gai", "Thai Nguyen",
    "Buon Me Thuot", "Hue", "Thanh Hoa", "Ca Mau", "Long Xuyen", "Tra Vinh", "Cam Pha", "My Tho", "Tuy Hoa", "Cam Ranh",
    "Nam Dinh", "Uong Bi", "Can Tho", "Nha Trang", "Viet Tri", "Chau Doc", "Phan Rang", "Vinh", "Da Lat", "Phan Thiet",
    "Vinh Long", "Ha Noi", "Play Cu", "Vung Tau", "Hai Duong", "Qui Nhon", "Yen Bai", "Hai Phong", "Rach Gia", "Hanoi", "Soc Trang"
]
