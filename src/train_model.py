import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# Load data
data = pd.read_csv('data/raw/weather.csv')

# Define flood risk based on rain thresholds
def define_flood_risk(rain):
    if rain < 20:
        return 'Low'
    elif 20 <= rain <= 50:
        return 'Medium'
    else:
        return 'High'

data['flood_risk'] = data['rain'].apply(define_flood_risk)

# Features and target
features = ['max', 'min', 'wind', 'rain', 'humidi', 'cloud', 'pressure']
X = data[features]
y = data['flood_risk']

# Encode target
y = y.map({'Low': 0, 'Medium': 1, 'High': 2})

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/flood_model.pkl')
print("Model saved to models/flood_model.pkl")
