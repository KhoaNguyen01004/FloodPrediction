import pandas as pd
import json

file_path = r"D:\SIU\machinlearning\FloodPrediction\data\raw\disaster_in_vietnam.csv"
df = pd.read_csv(file_path)


# 1️⃣ Filter only Flood events
df_flood = df[df["Disaster Type"] == "Flood"]

# 2️⃣ Keep only necessary columns
necessary_cols = [
    "Location", "Start Year", "Start Month", "Start Day",
    "River Basin", "Latitude", "Longitude", "Country"
]
df_flood = df_flood[necessary_cols]

# 3️⃣ Drop NaN values in Location or Country
df_flood = df_flood.dropna(subset=["Location", "Country"])

# 5️⃣ Filter only data from 2005 to 2023
df_flood = df_flood[(df_flood["Start Year"] >= 2005) & (df_flood["Start Year"] <= 2023)]


# 6️⃣ Split multi-location cells into individual rows
all_locations = []

for _, row in df_flood.iterrows():
    locations = str(row["Location"]).split(",")
    for loc in locations:
        loc = loc.strip()
        loc_lower = loc.lower()
        
        # Skip empty or generic entries
        if not loc or loc_lower in ["north", "central", "south"]:
            continue
        
        # Remove 'province'
        loc_clean = loc_lower.replace("province", "").strip()
        
        # Remove 'district' and anything after it
        loc_clean = loc_clean.split("district")[0].strip()
        
        # Remove trailing 'S' if present
        if loc_clean.endswith("s"):
            loc_clean = loc_clean[:-1].strip()
        
        # Capitalize properly
        loc_clean = loc_clean.title()
        
        if not loc_clean:
            continue
        
        all_locations.append({
            "location": loc_clean,
            "start_year": row["Start Year"],
            "start_month": row["Start Month"],
            "start_day": row["Start Day"],
            "river_basin": row["River Basin"],
            "latitude": row["Latitude"],
            "longitude": row["Longitude"]
        })


# 7️⃣ Remove duplicates based on location
unique_locations = {}
for item in all_locations:
    loc_name = item["location"]
    if loc_name not in unique_locations:
        unique_locations[loc_name] = item

# 8️⃣ Save to JSON
output_file = r"D:\SIU\machinlearning\FloodPrediction\data\raw\vietnam_city_coords.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(unique_locations, f, ensure_ascii=False, indent=4)

print(f"✅ JSON file created with {len(unique_locations)} unique Vietnam locations from 2005-2023")
