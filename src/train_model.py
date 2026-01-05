import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt

# Load processed data with historical flood risk labels
data = pd.read_csv('data/processed/flood_training.csv')

# Add cumulative rainfall feature: sum of rainfall from the current day and the previous 2 days (total water on the ground)
data['rain_last_3_days'] = data['rain'].rolling(window=3, min_periods=1).sum().fillna(0)

# Add additional features: temperature difference and interaction terms
data['temp_diff'] = data['max'] - data['min']
data['rain_humidi_interaction'] = data['rain'] * data['humidi']

# Features
features = ['max', 'min', 'wind', 'rain', 'humidi', 'cloud', 'pressure', 'month', 'rain_last_3_days', 'temp_diff', 'rain_humidi_interaction']
X = data[features]
y = data['flood_risk']  # Already encoded as 0=Low, 1=Medium, 2=High from historical data

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Define XGBoost model for imbalanced classes
# For multi-class, we'll use sample weights instead of scale_pos_weight

# Hyperparameter tuning with RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7, 10],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 3, 5]
}

# Use StratifiedKFold for cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

model = XGBClassifier(objective='multi:softmax', num_class=3, random_state=42)

random_search = RandomizedSearchCV(
    model, param_distributions=param_dist, n_iter=50, cv=cv,
    scoring='f1_macro', random_state=42, n_jobs=-1, verbose=1
)

random_search.fit(X_train_sm, y_train_sm)
model = random_search.best_estimator_

print(f"Best parameters: {random_search.best_params_}")

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Evaluate with adjusted prediction thresholds
y_proba = model.predict_proba(X_test)
# Adjust thresholds: Lower threshold for Medium and High to increase recall
thresholds = {0: 0.5, 1: 0.3, 2: 0.2}
y_pred_adj = []
for proba in y_proba:
    pred = 0
    if proba[2] > thresholds[2]:
        pred = 2
    elif proba[1] > thresholds[1]:
        pred = 1
    y_pred_adj.append(pred)

print("Adjusted Thresholds Classification Report:")
print(classification_report(y_test, y_pred_adj))

# Feature importance analysis
print("\nFeature Importance Analysis:")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Save model
os.makedirs('models', exist_ok=True)
joblib.dump(model, 'models/flood_model.pkl')
print("Model saved to models/flood_model.pkl")
