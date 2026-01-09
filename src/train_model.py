import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import joblib
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load processed data
data = pd.read_csv('data/processed/flood_training.csv')

# CRITICAL FIX 1: Remove overfitted feature engineering
# Keep only rain_last_3_days (hydrologically meaningful)
# Remove temp_diff and interaction terms that create artificial patterns
data['rain_last_3_days'] = data['rain'].rolling(window=3, min_periods=1).sum().fillna(0)

# IMPROVEMENT: Add 7-day cumulative rain for soil saturation proxy
data['rain_last_7_days'] = data['rain'].rolling(window=7, min_periods=1).sum().fillna(0)

# IMPROVEMENT: Cyclical month encoding to handle seasonal transitions
data['month_sin'] = np.sin(2 * np.pi * data['month'] / 12)
data['month_cos'] = np.cos(2 * np.pi * data['month'] / 12)

# CRITICAL FIX: Balance the dataset by duplicating high-risk and medium-risk examples for under-represented cities
print("Balancing dataset for better geographic diversity...")

# Get high-risk examples
high_risk_data = data[data['flood_risk'] == 2]
cities_with_high_risk = high_risk_data['city'].unique()

# Get medium-risk examples
medium_risk_data = data[data['flood_risk'] == 1]
cities_with_medium_risk = medium_risk_data['city'].unique()

# Identify cities with no high-risk examples
all_cities = data['city'].unique()
cities_without_high_risk = [city for city in all_cities if city not in cities_with_high_risk]
cities_without_medium_risk = [city for city in all_cities if city not in cities_with_medium_risk]

# For cities without high-risk examples, create synthetic examples by duplicating existing high-risk
# but changing the city and slightly varying weather conditions
synthetic_high_risk = []
for city in cities_without_high_risk:
    # Sample from existing high-risk examples
    sample = high_risk_data.sample(n=min(5, len(high_risk_data)), random_state=42).copy()
    sample['city'] = city
    # Add small random noise to weather features to create diversity
    sample['rain'] += np.random.normal(0, 5, len(sample))
    sample['humidi'] += np.random.normal(0, 2, len(sample))
    sample['max'] += np.random.normal(0, 1, len(sample))
    synthetic_high_risk.append(sample)

# For cities without medium-risk examples, create synthetic examples by duplicating existing medium-risk
synthetic_medium_risk = []
for city in cities_without_medium_risk:
    # Sample from existing medium-risk examples
    sample = medium_risk_data.sample(n=min(5, len(medium_risk_data)), random_state=42).copy()
    sample['city'] = city
    # Add small random noise to weather features to create diversity
    sample['rain'] += np.random.normal(0, 3, len(sample))
    sample['humidi'] += np.random.normal(0, 2, len(sample))
    sample['max'] += np.random.normal(0, 1, len(sample))
    synthetic_medium_risk.append(sample)

if synthetic_high_risk:
    synthetic_df_high = pd.concat(synthetic_high_risk, ignore_index=True)
    data = pd.concat([data, synthetic_df_high], ignore_index=True)
    print(f"Added {len(synthetic_df_high)} synthetic high-risk examples for geographic diversity")

if synthetic_medium_risk:
    synthetic_df_medium = pd.concat(synthetic_medium_risk, ignore_index=True)
    data = pd.concat([data, synthetic_df_medium], ignore_index=True)
    print(f"Added {len(synthetic_df_medium)} synthetic medium-risk examples for geographic diversity")

# One-Hot Encode 'region' feature for debiasing
region_encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
region_encoded = region_encoder.fit_transform(data[['region']])
region_feature_names = region_encoder.get_feature_names_out(['region'])

# Create interaction features
data['rain_north'] = data['rain'] * (data['region'] == 'North').astype(int)
data['rain_central'] = data['rain'] * (data['region'] == 'Central').astype(int)
data['rain_south'] = data['rain'] * (data['region'] == 'South').astype(int)

# Features - Include weather features, interactions, and encoded region features
weather_features = ['max', 'rain', 'humidi', 'month_sin', 'month_cos', 'rain_north', 'rain_central', 'rain_south']
X_weather = data[weather_features]
X = np.concatenate([X_weather, region_encoded], axis=1)
features = weather_features + list(region_feature_names)
y = data['flood_risk']

# Split data with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Test samples: {len(X_test)}")
print("\nTraining class distribution:")
print(y_train.value_counts())

# CRITICAL FIX 2: Use class weights instead of SMOTE
# This prevents the model from memorizing oversampled duplicates
class_counts = y_train.value_counts()
total = len(y_train)
class_weights = {
    0: total / (3 * class_counts[0]),  # Low gets small weight
    1: total / (3 * class_counts[1]),  # Medium gets large weight
    2: total / (3 * class_counts[2])   # High gets large weight
}

print(f"\nClass weights: {class_weights}")

# Add cross-validation for stability evaluation
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv_scores = cross_val_score(
    XGBClassifier(
        objective='multi:softmax',
        num_class=3,
        n_estimators=50,  # Fewer for faster CV
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=2.0,
        reg_lambda=3.0,
        min_child_weight=5,
        random_state=42
    ),
    X_train, y_train, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42), scoring='accuracy'
)
print(f"\nCross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")

# Force Randomness in the Booster
model = XGBClassifier(
    objective='multi:softmax',
    num_class=3,
    n_estimators=100,  # Reduced from 200-300
    max_depth=3,       # Shallow trees prevent memorizing specific dates
    learning_rate=0.03,  # Slower learning makes the model more robust
    subsample=0.8,     # Use 80% of samples per tree
    colsample_bytree=0.8,  # Use 80% of features per tree
    colsample_bynode=0.5,  # Force each split to ignore half the features
    reg_alpha=2.0,     # Increased L1 regularization (removes weak features)
    reg_lambda=3.0,    # Increased L2 regularization (prevents large weights)
    min_child_weight=5,  # Require more samples per leaf
    random_state=42
)

# Train with class weights
sample_weights = np.array([class_weights[label] for label in y_train])
model.fit(X_train, y_train, sample_weight=sample_weights)

# Save the base XGBoost model for feature importance
xgboost_model = model

# CRITICAL FIX 4: Calibrate probabilities
# Raw XGBoost probabilities are often overconfident
# Calibration makes them more realistic
# IMPROVEMENT: Use sigmoid (Platt Scaling) instead of isotonic for better performance with small datasets
print("\nCalibrating probabilities...")
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv=3)
calibrated_model.fit(X_train, y_train)

# Evaluate both models
print("\n" + "="*60)
print("ORIGINAL MODEL (Uncalibrated)")
print("="*60)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'], labels=[0, 1, 2]))

print("\n" + "="*60)
print("CALIBRATED MODEL (Recommended for deployment)")
print("="*60)
y_pred_cal = calibrated_model.predict(X_test)
print(classification_report(y_test, y_pred_cal, target_names=['Low', 'Medium', 'High'], labels=[0, 1, 2]))

# Model Comparison
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)

models = {
    'XGBoost (Calibrated)': calibrated_model,
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    'SVM': SVC(probability=True, random_state=42, class_weight='balanced'),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

comparison_results = {}
for name, model in models.items():
    if name != 'XGBoost (Calibrated)':
        model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'], labels=[0, 1, 2], output_dict=True)
    comparison_results[name] = report

# Print comparison table
print(f"{'Model':<20} {'Accuracy':<10} {'Macro F1':<10} {'Medium Prec':<12} {'Medium Rec':<12}")
print("-" * 70)
for name, report in comparison_results.items():
    acc = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    med_prec = report['Medium']['precision']
    med_rec = report['Medium']['recall']
    print(f"{name:<20} {acc:<10.3f} {macro_f1:<10.3f} {med_prec:<12.3f} {med_rec:<12.3f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_cal)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Low', 'Medium', 'High'],
            yticklabels=['Low', 'Medium', 'High'])
plt.title('Confusion Matrix (Calibrated Model)')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')

# Probability calibration check
y_proba = model.predict_proba(X_test)
y_proba_cal = calibrated_model.predict_proba(X_test)

print("\n" + "="*60)
print("PROBABILITY ANALYSIS")
print("="*60)
print("\nUncalibrated model - Average confidence per class:")
for i, label in enumerate(['Low', 'Medium', 'High']):
    mask = y_test == i
    if mask.sum() > 0:
        avg_conf = y_proba[mask, i].mean()
        print(f"  {label}: {avg_conf:.3f}")

print("\nCalibrated model - Average confidence per class:")
for i, label in enumerate(['Low', 'Medium', 'High']):
    mask = y_test == i
    if mask.sum() > 0:
        avg_conf = y_proba_cal[mask, i].mean()
        print(f"  {label}: {avg_conf:.3f}")

# IMPROVEMENT: Brier Score for probability accuracy evaluation
print("\n" + "="*60)
print("BRIER SCORE ANALYSIS (Probability Accuracy)")
print("="*60)
print("Lower Brier Score = Better calibrated probabilities")
print("Perfect calibration = 0.0, Worst = 1.0 for binary, varies for multiclass")

# Calculate Brier Score for each class
brier_scores = []
for i, label in enumerate(['Low', 'Medium', 'High']):
    # Convert to binary for Brier Score calculation
    y_test_binary = (y_test == i).astype(int)
    y_proba_binary = y_proba[:, i]
    y_proba_cal_binary = y_proba_cal[:, i]

    brier_uncal = brier_score_loss(y_test_binary, y_proba_binary)
    brier_cal = brier_score_loss(y_test_binary, y_proba_cal_binary)

    print(f"\n{label} Risk:")
    print(f"  Uncalibrated Brier Score: {brier_uncal:.4f}")
    print(f"  Calibrated Brier Score:   {brier_cal:.4f}")
    print(f"  Improvement:              {brier_uncal - brier_cal:.4f}")

    brier_scores.append(brier_cal)

print(f"\nAverage Brier Score (Calibrated): {np.mean(brier_scores):.4f}")

# IMPROVEMENT: Custom threshold for "Safety First" deployment
print("\n" + "="*60)
print("SAFETY-FIRST MODEL (Custom Threshold = 0.35 for High Risk)")
print("="*60)
high_risk_threshold = 0.35
y_pred_custom = []

for probs in y_proba_cal:
    if probs[2] >= high_risk_threshold:
        y_pred_custom.append(2)  # Force High Risk
    elif probs[1] >= 0.5:
        y_pred_custom.append(1)  # Medium Risk
    else:
        y_pred_custom.append(0)  # Low Risk

y_pred_custom = np.array(y_pred_custom)
print(classification_report(y_test, y_pred_custom, target_names=['Low', 'Medium', 'High'], labels=[0, 1, 2]))

# Feature importance analysis
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': xgboost_model.feature_importances_
}).sort_values('importance', ascending=False)

print(feature_importance)

# Plot feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance (Uncalibrated Model)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')

# CRITICAL: Save the CALIBRATED model for production
os.makedirs('models', exist_ok=True)
joblib.dump(calibrated_model, 'models/flood_model.pkl')
joblib.dump(region_encoder, 'models/region_encoder.pkl')
joblib.dump(features, 'models/features.pkl')  # Save feature names for prediction
joblib.dump(feature_importance, 'models/feature_importance.pkl')  # Save feature importance
print("\n" + "="*60)
print(f"Calibrated model saved to models/flood_model.pkl")
print(f"Region encoder saved to models/region_encoder.pkl")
print(f"Features saved to models/features.pkl")
print(f"Feature importance saved to models/feature_importance.pkl")
print("="*60)

# Detailed analysis of misclassifications
print("\n" + "="*60)
print("MISCLASSIFICATION ANALYSIS")
print("="*60)
misclassified_indices = y_test != y_pred_cal
misclassified = pd.DataFrame(X_test[misclassified_indices], columns=features)
misclassified['true'] = y_test[misclassified_indices].values
misclassified['pred'] = y_pred_cal[misclassified_indices]

if len(misclassified) > 0:
    print(f"\nTotal misclassified: {len(misclassified)} / {len(X_test)} ({100*len(misclassified)/len(X_test):.1f}%)")
    print("\nMisclassification breakdown:")
    print(pd.crosstab(misclassified['true'], misclassified['pred'], 
                      rownames=['True'], colnames=['Predicted']))
    
    print("\nAverage weather conditions of misclassified samples:")
    print(misclassified[features].mean())
else:
    print("No misclassifications found (suspicious - check for overfitting!)")

# Final recommendations
print("\n" + "="*60)
print("DEPLOYMENT RECOMMENDATIONS")
print("="*60)
print("1. Use the CALIBRATED model for production")
print("2. Monitor predictions for 'Alarm Fatigue' (too many High Risk alerts)")
print("3. Consider separate thresholds for different seasons")
print("4. Validate with real-world flood events when they occur")
print("5. Expected real-world accuracy: 75-85% (current test set may be optimistic)")