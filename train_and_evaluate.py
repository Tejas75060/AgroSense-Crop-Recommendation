#!/usr/bin/env python3
"""
AgroSense: Precision Crop Recommendation System
Group 15 - Machine Learning Fundamentals Mini Project
Authors: Group 15
End-to-End Pipeline: Data Loading, Preprocessing, EDA, Model Building,
Cross-Validation, Evaluation, Comparison, and Model Persistence.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "Crop_recommendation.csv")
EDA_DIR = os.path.join(BASE_DIR, "eda_plots")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. DATA LOADING AND UNDERSTANDING
# -------------------------------------------------------------
print("=" * 70)
print(" AgroSense: Crop Recommendation System - ML Training Pipeline")
print("=" * 70)

print(f"\n[1/7] Loading dataset from: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)

print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Info:")
df.info()

print("\nSummary Statistics:")
print(df.describe().round(2))

# -------------------------------------------------------------
# 2. DATA CLEANING & PREPROCESSING
# -------------------------------------------------------------
print("\n[2/7] Preprocessing and Data Quality Verification")
missing = df.isnull().sum()
print("Missing values per column:\n", missing)
duplicates = df.duplicated().sum()
print(f"Duplicate rows detected: {duplicates}")

feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
target_col = 'label'

unique_crops = sorted(df[target_col].unique())
print(f"Total Unique Crops ({len(unique_crops)}): {', '.join(unique_crops)}")
print("\nSamples per crop class (Top 5):\n", df[target_col].value_counts().head(5))

# -------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION
# -------------------------------------------------------------
print("\n[3/7] Generating and saving publication-grade EDA visualizations...")

# Plot 1: Crop Distribution
fig, ax = plt.subplots(figsize=(14, 5))
sns.countplot(data=df, x='label', order=unique_crops, palette='viridis', ax=ax)
ax.set_title("Distribution of Crop Categories (Balanced Dataset: 100 samples/crop)", fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel("Crop Name", fontsize=11, labelpad=8)
ax.set_ylabel("Sample Count", fontsize=11, labelpad=8)
plt.xticks(rotation=60, ha='right', fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(EDA_DIR, "01_crop_distribution.png"), dpi=300)
plt.close()

# Plot 2: Feature Distributions (Histograms with KDE)
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
colors = ['#2e7d32', '#388e3c', '#4caf50', '#d32f2f', '#0288d1', '#7b1fa2', '#f57c00']

for i, col in enumerate(feature_cols):
    sns.histplot(df[col], kde=True, ax=axes[i], color=colors[i], bins=25, edgecolor='black', alpha=0.6)
    axes[i].set_title(f"Distribution of {col.capitalize()}", fontsize=12, fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

axes[7].set_visible(False) # remove empty 8th slot
plt.suptitle("Agronomic & Climate Feature Distributions", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(EDA_DIR, "02_feature_distributions.png"), dpi=300)
plt.close()

# Plot 3: Feature Correlation Heatmap
fig, ax = plt.subplots(figsize=(9, 7))
corr = df[feature_cols].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='Spectral', vmin=-1, vmax=1,
            square=True, linewidths=1, linecolor='white', cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Pearson Correlation Matrix of Soil & Climate Parameters", fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
fig.savefig(os.path.join(EDA_DIR, "03_correlation_heatmap.png"), dpi=300)
plt.close()

# Plot 4: Boxplots for Outlier Analysis
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    sns.boxplot(y=df[col], ax=axes[i], color=colors[i], width=0.4, fliersize=4)
    axes[i].set_title(f"Outlier Assessment: {col.capitalize()}", fontsize=12, fontweight='bold')
    axes[i].set_ylabel(col)

axes[7].set_visible(False)
plt.suptitle("Boxplots of Environmental and Soil Variables", fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
fig.savefig(os.path.join(EDA_DIR, "04_outlier_boxplots.png"), dpi=300)
plt.close()

# Plot 5: Average N, P, K Nutrient Demand per Crop
npk_summary = df.groupby('label')[['N', 'P', 'K']].mean().loc[unique_crops]
fig, ax = plt.subplots(figsize=(16, 6))
npk_summary.plot(kind='bar', stacked=False, ax=ax, colormap='Accent', width=0.8, edgecolor='black', alpha=0.85)
ax.set_title("Mean Primary Macronutrient (N-P-K) Requirements Across Crops", fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel("Crop", fontsize=11)
ax.set_ylabel("Nutrient Level (mg/kg in soil)", fontsize=11)
plt.xticks(rotation=60, ha='right', fontsize=9)
plt.legend(title="Macronutrients", frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(EDA_DIR, "05_npk_ratio_by_crop.png"), dpi=300)
plt.close()

# Plot 6: Rainfall vs Temperature Climatic Clustering
fig, ax = plt.subplots(figsize=(12, 7))
sample_crops = ['rice', 'cotton', 'coffee', 'apple', 'chickpea', 'watermelon', 'banana', 'grapes']
df_sample = df[df['label'].isin(sample_crops)]
sns.scatterplot(data=df_sample, x='rainfall', y='temperature', hue='label',
                style='label', s=70, palette='tab10', alpha=0.85, ax=ax)
ax.set_title("Climatic Niche: Rainfall vs. Temperature for Representative Crops", fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel("Annual Rainfall (mm)", fontsize=11)
ax.set_ylabel("Temperature (°C)", fontsize=11)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Selected Crops")
plt.tight_layout()
fig.savefig(os.path.join(EDA_DIR, "06_rainfall_vs_temp.png"), dpi=300)
plt.close()

print(f"EDA plots successfully generated and saved to '{EDA_DIR}/'")

# -------------------------------------------------------------
# 4. DATA SPLITTING & STRATIFICATION
# -------------------------------------------------------------
print("\n[4/7] Splitting data into Stratified Train and Test sets...")
X = df[feature_cols]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(f"Train features shape: {X_train.shape}, Test features shape: {X_test.shape}")
print(f"Training samples: {len(X_train)} | Testing samples: {len(X_test)}")

# -------------------------------------------------------------
# 5. MODEL TRAINING & COMPARATIVE BENCHMARKING
# -------------------------------------------------------------
print("\n[5/7] Training and evaluating multiple Machine Learning classifiers...")

models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42),
    "Decision Tree": DecisionTreeClassifier(criterion='gini', max_depth=None, random_state=42),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine (RBF)": Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(kernel='rbf', C=10.0, probability=True, random_state=42))
    ]),
    "K-Nearest Neighbors": Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=5))
    ])
}

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

trained_models = {}
test_predictions = {}

for name, model in models.items():
    print(f"\n---> Training: {name}")
    # Stratified 5-Fold Cross Validation on full dataset
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    
    # Train on training split
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    trained_models[name] = model
    test_predictions[name] = y_pred
    
    acc = accuracy_score(y_test, y_pred)
    prec_weighted = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec_weighted = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    prec_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0)
    
    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision (Weighted)": prec_weighted,
        "Recall (Weighted)": rec_weighted,
        "F1-Score (Weighted)": f1_weighted,
        "Precision (Macro)": prec_macro,
        "Recall (Macro)": rec_macro,
        "F1-Score (Macro)": f1_macro,
        "5-Fold CV Mean": cv_scores.mean(),
        "5-Fold CV Std": cv_scores.std()
    })
    
    print(f"Test Accuracy: {acc*100:.2f}% | F1-Score: {f1_weighted*100:.2f}% | 5-Fold CV: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

results_df = pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
comparison_csv_path = os.path.join(BASE_DIR, "model_comparison.csv")
results_df.to_csv(comparison_csv_path, index=False)
print("\n" + "=" * 70)
print(" MODEL BENCHMARKING RESULTS")
print("=" * 70)
print(results_df[["Model", "Accuracy", "Precision (Weighted)", "Recall (Weighted)", "F1-Score (Weighted)", "5-Fold CV Mean"]].to_string(index=False))

# -------------------------------------------------------------
# 6. DETAILED EVALUATION: CONFUSION MATRICES & FEATURE IMPORTANCES
# -------------------------------------------------------------
print("\n[6/7] Detailed Evaluation for Decision Tree and Random Forest...")

# Decision Tree Confusion Matrix
dt_cm = confusion_matrix(y_test, test_predictions["Decision Tree"], labels=unique_crops)
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(dt_cm, annot=True, fmt='d', cmap='Blues', xticklabels=unique_crops, yticklabels=unique_crops, ax=ax, cbar=False)
ax.set_title("Decision Tree Classifier - Confusion Matrix (Test Split)", fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel("Predicted Crop", fontsize=11, labelpad=8)
ax.set_ylabel("True Crop", fontsize=11, labelpad=8)
plt.xticks(rotation=60, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(EDA_DIR, "08_decision_tree_confusion_matrix.png"), dpi=300)
plt.close()

# Random Forest Confusion Matrix
rf_cm = confusion_matrix(y_test, test_predictions["Random Forest"], labels=unique_crops)
fig, ax = plt.subplots(figsize=(14, 11))
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', xticklabels=unique_crops, yticklabels=unique_crops, ax=ax, cbar=False)
ax.set_title("Random Forest Classifier - Confusion Matrix (Test Split)", fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel("Predicted Crop", fontsize=11, labelpad=8)
ax.set_ylabel("True Crop", fontsize=11, labelpad=8)
plt.xticks(rotation=60, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(EDA_DIR, "09_random_forest_confusion_matrix.png"), dpi=300)
plt.close()

# Plot 10: Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(12, 6))
plot_metrics = ['Accuracy', 'Precision (Weighted)', 'Recall (Weighted)', 'F1-Score (Weighted)', '5-Fold CV Mean']
results_plot_df = results_df.set_index('Model')[plot_metrics]
results_plot_df.plot(kind='bar', ax=ax, colormap='viridis', width=0.8, edgecolor='black', alpha=0.9)
ax.set_title("Performance Comparison Across Evaluated ML Models", fontsize=14, fontweight='bold', pad=12)
ax.set_ylabel("Score (0.0 to 1.0)", fontsize=11)
ax.set_ylim(0.85, 1.02)
plt.xticks(rotation=30, ha='right', fontsize=10)
plt.legend(loc='lower right', frameon=True)
plt.tight_layout()
fig.savefig(os.path.join(EDA_DIR, "10_model_comparison_chart.png"), dpi=300)
plt.close()

# Feature Importance Analysis (Random Forest & Decision Tree)
rf_model = trained_models["Random Forest"]
dt_model = trained_models["Decision Tree"]

feat_imp_df = pd.DataFrame({
    'Feature': feature_cols,
    'Random Forest Importance': rf_model.feature_importances_,
    'Decision Tree Importance': dt_model.feature_importances_
}).sort_values(by='Random Forest Importance', ascending=False)

feat_imp_csv_path = os.path.join(BASE_DIR, "feature_importance.csv")
feat_imp_df.to_csv(feat_imp_csv_path, index=False)
print("\nFeature Importance Ranking (Random Forest vs Decision Tree):")
print(feat_imp_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
feat_imp_plot = feat_imp_df.set_index('Feature')
feat_imp_plot.plot(kind='barh', ax=ax, color=['#2e7d32', '#1976d2'], edgecolor='black', alpha=0.85)
ax.invert_yaxis()
ax.set_title("Feature Importance: Gini Impurity Reduction in Random Forest vs Decision Tree", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Relative Importance Score", fontsize=11)
plt.legend(loc='lower right')
plt.tight_layout()
fig.savefig(os.path.join(EDA_DIR, "07_feature_importance.png"), dpi=300)
plt.close()

# Print detailed Classification Report for Best Model (Random Forest)
print("\n" + "=" * 70)
print(" DETAILED CLASSIFICATION REPORT (BEST MODEL: RANDOM FOREST)")
print("=" * 70)
print(classification_report(y_test, test_predictions["Random Forest"], labels=unique_crops, digits=4))

# -------------------------------------------------------------
# 7. MODEL PERSISTENCE & METADATA EXPORT
# -------------------------------------------------------------
print("\n[7/7] Persisting models and metadata for Streamlit deployment...")

rf_model_path = os.path.join(MODELS_DIR, "crop_recommendation_rf.pkl")
dt_model_path = os.path.join(MODELS_DIR, "crop_recommendation_dt.pkl")
# Also save crop_recommendation_model.pkl in root for backwards compatibility
root_model_path = os.path.join(BASE_DIR, "crop_recommendation_model.pkl")

joblib.dump(rf_model, rf_model_path)
joblib.dump(dt_model, dt_model_path)
joblib.dump(rf_model, root_model_path)

print(f"Saved Random Forest model to: {rf_model_path}")
print(f"Saved Decision Tree model to: {dt_model_path}")
print(f"Saved Default deployment model to: {root_model_path}")

# Feature summary statistics for input validation in UI
feature_ranges = {}
for col in feature_cols:
    feature_ranges[col] = {
        "min": float(df[col].min()),
        "max": float(df[col].max()),
        "mean": float(df[col].mean()),
        "std": float(df[col].std()),
        "q25": float(df[col].quantile(0.25)),
        "q50": float(df[col].quantile(0.50)),
        "q75": float(df[col].quantile(0.75))
    }

# Crop Agronomic Database (Season, Water, Soil Type, Benefits)
crop_profiles = {
    "apple": {"season": "Kharif/Rabi (Temperate)", "water": "Moderate", "soil": "Well-drained loam", "ph": "5.5 - 6.5", "desc": "High economic value temperate fruit; requires chilling hours and high potassium."},
    "banana": {"season": "All Year (Tropical)", "water": "High", "soil": "Rich loamy soil", "ph": "6.0 - 7.5", "desc": "Fast-growing fruit requiring warm tropical climate, high humidity, and nitrogen."},
    "blackgram": {"season": "Kharif", "water": "Low to Moderate", "soil": "Loamy/Sandy loam", "ph": "6.5 - 7.8", "desc": "Nitrogen-fixing pulse crop; enriches soil fertility and tolerates modest drought."},
    "chickpea": {"season": "Rabi", "water": "Low", "soil": "Sandy loam/Clay", "ph": "6.0 - 8.0", "desc": "Cool-season legume requiring dry climate, moderate phosphorus, and minimal water."},
    "coconut": {"season": "Perennial (Coastal)", "water": "High", "soil": "Coastal alluvium/Laterite", "ph": "5.2 - 8.0", "desc": "Perennial plantation crop thriving in warm humid coasts with consistent rainfall."},
    "coffee": {"season": "Perennial (Hilly)", "water": "High", "soil": "Deep, fertile volcanic loam", "ph": "6.0 - 6.5", "desc": "Cash crop grown at elevated altitudes with shaded canopy, high rainfall, and cool temps."},
    "cotton": {"season": "Kharif", "water": "Moderate", "soil": "Deep black cotton soil (Regur)", "ph": "6.0 - 8.0", "desc": "Major fiber cash crop requiring warm climate, moderate moisture, and sunny days."},
    "grapes": {"season": "Rabi/Summer", "water": "Moderate", "soil": "Sandy loam to clay loam", "ph": "6.5 - 7.5", "desc": "High-value fruit demanding very high potassium (K) and phosphorus (P) nutrition."},
    "jute": {"season": "Kharif", "water": "Very High", "soil": "Alluvial river basin", "ph": "6.0 - 7.5", "desc": "Natural golden fiber flourishing in warm humid river deltas with heavy monsoon rainfall."},
    "kidneybeans": {"season": "Kharif", "water": "Moderate", "soil": "Rich organic loam", "ph": "5.5 - 6.5", "desc": "High protein pulse needing temperate warmth and moderate moisture levels."},
    "lentil": {"season": "Rabi", "water": "Low", "soil": "Well-drained alluvium", "ph": "6.0 - 7.5", "desc": "Hardy winter pulse with low water requirement; fixes atmospheric nitrogen."},
    "maize": {"season": "Kharif/Rabi", "water": "Moderate", "soil": "Well-drained alluvium", "ph": "5.8 - 7.2", "desc": "Versatile cereal staple demanding substantial nitrogen and balanced sunshine."},
    "mango": {"season": "Perennial (Tropical)", "water": "Moderate", "soil": "Deep alluvial/Lateritic", "ph": "5.5 - 7.5", "desc": "King of fruits thriving in hot frost-free climate with distinct dry spells for flowering."},
    "mothbeans": {"season": "Kharif", "water": "Very Low", "soil": "Sandy/Arid desert soil", "ph": "6.0 - 8.5", "desc": "Extremely drought-resilient legume ideal for arid zones and soil moisture conservation."},
    "mungbean": {"season": "Kharif/Zaid", "water": "Low to Moderate", "soil": "Sandy loam", "ph": "6.2 - 7.2", "desc": "Short-duration pulse suitable for multi-cropping and green manuring."},
    "muskmelon": {"season": "Zaid (Summer)", "water": "Moderate", "soil": "Sandy river beds", "ph": "6.0 - 7.0", "desc": "Warm-season cucurbit flourishing in sandy loam with abundant sunshine and dry air."},
    "orange": {"season": "Subtropical/Tropical", "water": "Moderate", "soil": "Deep, well-drained loam", "ph": "5.5 - 7.0", "desc": "Citrus cash crop preferring moderate temperatures, adequate potassium, and good drainage."},
    "papaya": {"season": "All Year (Tropical)", "water": "Moderate to High", "soil": "Rich volcanic/alluvial", "ph": "6.0 - 7.0", "desc": "Rapidly fruiting tropical tree requiring warm temperatures and frost-free weather."},
    "pigeonpeas": {"season": "Kharif", "water": "Low to Moderate", "soil": "Deep loam to black soil", "ph": "6.5 - 7.5", "desc": "Drought-tolerant pulse with deep taproots that break hard soil pans."},
    "pomegranate": {"season": "Semi-arid", "water": "Low to Moderate", "soil": "Sandy loam to light black", "ph": "6.5 - 7.5", "desc": "Hardy fruit crop flourishing in hot dry summers and cool winters with low humidity."},
    "rice": {"season": "Kharif (Monsoon)", "water": "Very High", "soil": "Heavy clayey or alluvial", "ph": "5.5 - 7.0", "desc": "Premier cereal staple requiring water inundation (200+ mm rainfall) and warm humidity."},
    "watermelon": {"season": "Zaid (Summer)", "water": "Moderate", "soil": "Sandy loam", "ph": "6.0 - 7.0", "desc": "Warm season vine requiring high sunshine, warm soils, and moderate nitrogen."}
}

metadata = {
    "project_title": "AgroSense: An Intelligent Soil & Climate-Driven Precision Crop Recommendation System",
    "group_id": "GROUP 15",
    "features": feature_cols,
    "classes": unique_crops,
    "num_classes": len(unique_crops),
    "feature_ranges": feature_ranges,
    "best_model_name": "Random Forest",
    "best_model_accuracy": float(results_df[results_df['Model'] == 'Random Forest']['Accuracy'].values[0]),
    "best_model_f1": float(results_df[results_df['Model'] == 'Random Forest']['F1-Score (Weighted)'].values[0]),
    "best_model_cv_mean": float(results_df[results_df['Model'] == 'Random Forest']['5-Fold CV Mean'].values[0]),
    "model_comparison": results_df.to_dict(orient='records'),
    "feature_importance": feat_imp_df.to_dict(orient='records'),
    "crop_profiles": crop_profiles
}

metadata_path = os.path.join(MODELS_DIR, "model_metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"Exported model metadata to: {metadata_path}")
print("\n[SUCCESS] Pipeline execution finished successfully!")
