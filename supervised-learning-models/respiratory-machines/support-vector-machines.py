"""
AI-Based Prediction of Respiratory Machine Verification Outcomes
- Loads dataset
- Cleans and engineers domain-specific features
- Trains Support Vector Machines model with GridSearchCV
- Calibrates probability predictions
- Identifies high-risk devices based on predicted failure likelihood
- Identifies next years failure prediction
"""

import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, roc_auc_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# Ensure console prints in UTF-8
sys.stdout.reconfigure(encoding="utf-8")

# Ensure console prints all columns and rows set
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#------------------------------------------------------------------------
# LOAD AND CLEAN DATASET
#------------------------------------------------------------------------

# Load data from CSV file
data = pd.read_csv("files/respiratory_machines_2015_2024.csv", encoding="utf-8")

# Drop unnecessary columns that we won't need for prediction
data.drop(columns=['Broj izvještaja', 'Broj naloga', 'Status verifikacije', 'Zahtjev za verifikaciju', 'Metoda', 'Vrsta','Napomena'], errors='ignore', inplace=True)

# Handle incorrect value - this value was intentionally put in the dataset used for the information that something wasn't measured
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)
data.replace("NIJE MJERENO", np.nan, inplace=True)

# Convert comma decimal separators to dot for all columns that should be numeric 
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].str.replace(',', '.', regex=False)

# Set 'N/A' in Usklađeno column if columns 'Mjerena vrijednost [ml]' || 'Greška mjerenja [%]' are 'NIJE MJERENO' (in our case null)
for i in range(1, 24):
    value_col = next((col for col in data.columns if col.startswith(f"{i}: Mjerena vrijednost")), None)
    error_col = next((col for col in data.columns if col.startswith(f"{i}: Greška mjerenja")), None)
    aligned_col = f"{i}: Usklađeno"

    if all([value_col, error_col, aligned_col in data.columns]):
        # Convert to numeric
        data[value_col] = pd.to_numeric(data[value_col], errors='coerce')
        data[error_col] = pd.to_numeric(data[error_col], errors='coerce')

        # Set N/A if both values are missing or value is zero
        mask = (data[value_col].isna() & data[error_col].isna()) | (data[value_col] == 0)
        data.loc[mask, aligned_col] = 'N/A'

# Convert columns 'Spoljašnji pregled _' to 0/1 for better dataset inspection
external_cols = ['Spoljašnji pregled 1', 'Spoljašnji pregled 2', 'Spoljašnji pregled 3', 'Spoljašnji pregled 4']
for col in external_cols:
    data[col] = data[col].astype(str).str.strip().str.upper()  
    data[col] = data[col].map({'DA': 1, 'NE': 0})

# Encode columns 'Proizvođač' and 'Uređaj' to appropriate unique names
original_values = {}
categorical_cols = ['Proizvođač', 'Uređaj']
for col in categorical_cols:
    data[col], uniques = pd.factorize(data[col])
    original_values[col] = dict(enumerate(uniques))

# Encode target variable 'Verifikacija ispravna' to 0/1
data['Verifikacija ispravna'] = data['Verifikacija ispravna'].map({'DA': 1, 'NE': 0})

# Encode column 'Usklađeno' to 0/1
usk_columns = [col for col in data.columns if 'Usklađeno' in col]
for col in usk_columns:
    data[col] = data[col].map({'DA': 1, 'NE': 0})

#data.to_csv("files/anesthesia_cleaned.csv", index=False, encoding="utf-8")

#------------------------------------------------------------------------
# FEATURE ENGINEERING
#------------------------------------------------------------------------

# External check ('Spoljašnji pregled _') summary - if any external inspection failed, the machine is at risk
data['external_failed'] = data[external_cols].apply(
    lambda row: 1 if (row == 0).any() else 0, axis=1
)

# Count how many measurements were skipped in each row ('Zadana vrijednost', 'Mjerena vrijednost', 'Greška mjerenja')
measure_cols = [col for col in data.columns if 'Zadana vrijednost' in col or 'Mjerena vrijednost' in col or 'Greška mjerenja' in col]
data['skipped_measurements'] = data[measure_cols].isna().sum(axis=1) 

# Percentage of 'Usklađeno' measurement - tells us how consistently the mahine was aligned 
usk_column = [col for col in data.columns if 'Usklađeno' in col]
data['usk_percent'] = data[usk_column].apply(pd.to_numeric, errors='coerce').mean(axis=1) # gives value between 0 and 1 - 0: non were coordinated, 1: all were coordinated

# Error measurement - statistical view of measurement consistency
error_cols = [col for col in data.columns if 'Greška mjerenja' in col]
allowed_cols = [col for col in data.columns if 'Dozvoljeno odstupanje' in col]
data[error_cols] = data[error_cols].apply(pd.to_numeric, errors='coerce') # This will ensure that these error_cols are numeric
data[allowed_cols] = data[allowed_cols].apply(pd.to_numeric, errors='coerce') # This will ensure that these allowed_cols are numeric

new_cols = pd.DataFrame({
    'mean_error': data[error_cols].mean(axis=1), # Average error across all points for a device
    'max_error': data[error_cols].max(axis=1), # Worst error recorded
    'std_error': data[error_cols].std(axis=1), # How consistent/inconsistent the errors are
})

# Function to classify the error into performance categories using scaled thresholds
def classify_scaled_error_score(error, allowed):
    try:
        if pd.isna(error) or pd.isna(allowed) or allowed == 0:
            return np.nan

        abs_error = abs(error)
        threshold = allowed
        
        if abs_error <= 0.2 * threshold:
            return 5  # Excellent
        elif abs_error <= 0.4 * threshold:
            return 4  # Very Good
        elif abs_error <= 0.6 * threshold:
            return 3  # Good
        elif abs_error <= 0.8 * threshold:
            return 2  # Moderate
        elif abs_error <= 1.0 * threshold:
            return 1  # Likely to Fail
        else:
            return 0  # Failed
    except:
        return np.nan
    
# Scaled classification to each error/deviation pair
scaled_score_cols = []
for err_col, dev_col in zip(error_cols, allowed_cols):
    score_col = err_col + '_score'
    data[score_col] = data.apply(lambda row: classify_scaled_error_score(row[err_col], row[dev_col]), axis=1)
    scaled_score_cols.append(score_col)

data['mean_error_score'] = data[scaled_score_cols].mean(axis=1)
data['min_error_score'] = data[scaled_score_cols].min(axis=1)
data['high_risk_score_count'] = data[scaled_score_cols].apply(lambda row: ((row == 0) | (row == 1)).sum(), axis=1)

# Count breaches where error ('Greška mjerenja') is outside ± allowed deviation ('Dozvoljeno odstupanje')- if a devices error is worse than allowed
breaches = []
for err_col, dev_col in zip(error_cols, allowed_cols):
    breach = ((data[err_col] < -data[dev_col]) | (data[err_col] > data[dev_col])).astype(int)
    breaches.append(breach)

# Add everything back to the DataFrame
data = pd.concat([data, new_cols], axis=1)
data = data.copy()
data['total_breaches'] = np.vstack(breaches).sum(axis=0)

# Normalized value of how often a device exceeded the allowed error margins relative to how many tests were performed
data['total_tests'] = data[error_cols].notna().sum(axis=1)
data['breach_ratio'] = data['total_breaches'] / data['total_tests']
data['breach_ratio'] = data['breach_ratio'].fillna(0)

# Time-Based Features: extract year, month, day out of 'Datum izdavanje' column
data['Datum izdavanja'] = pd.to_datetime(data['Datum izdavanja'], dayfirst=True, errors='coerce')
data['year'] = data['Datum izdavanja'].dt.year
data['month'] = data['Datum izdavanja'].dt.month
data['dayofweek'] = data['Datum izdavanja'].dt.dayofweek

#data.to_csv("files/anesthesia_cleaned_with_features.csv", index=False, encoding="utf-8")

#------------------------------------------------------------------------
# MODEL TRAINING AND TUNING
#------------------------------------------------------------------------

y = data['Verifikacija ispravna'] # Target variable

# Feature columns that will be used for prediction
feature_columns = [
    'Proizvođač', 'Uređaj',
    'external_failed', 'usk_percent',
    'mean_error', 'max_error', 'std_error',
    'total_breaches', 'breach_ratio', 'skipped_measurements', 
    'year', 'month', 'dayofweek', 
    'mean_error_score', 'min_error_score', 'high_risk_score_count'
]

for col in feature_columns:
    if data[col].isna().any():
        data[col + "_missing"] = data[col].isna().astype(int)
        data[col] = data[col].fillna(-999)
missing_cols = [col + "_missing" for col in feature_columns if col + "_missing" in data.columns]
feature_columns += missing_cols

X = data[feature_columns]

# Split the data in training and test sets: 80% train & 20% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # stratisfy is used to balance DA/NE classes

# Scale features
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# Tune Gradient Boosting with GridSearchCV - systematically tries all combinations of specified hyperparameters to find the best-performing model
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto'],
    'kernel': ['rbf', 'linear']
}

# Idea: trying out every possible combination of n_estimators, max_depth, min_samples_split and seeing which combination gives the best model performance
grid_search = GridSearchCV( 
    estimator=SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
    param_grid=param_grid, 
    cv=5, 
    scoring='f1', 
    n_jobs=-1, 
    verbose=1
    )

grid_search.fit(X_train_scaled, y_train) # Training the model
print("Best Parameters:", grid_search.best_params_) # See what is the best model performance using GridSearchCV

#------------------------------------------------------------------------
# CALIBRATION AND EVALUATION
#------------------------------------------------------------------------

# CalibratedClassifierCV - used to improve the quality of the probability estimates given by the model
calibrated_svc = CalibratedClassifierCV(grid_search.best_estimator_, cv=3) 
# Make probability outputs more accurate and realistic
calibrated_svc.fit(X_train_scaled, y_train)

# Get trustworthy probabilities for risk assessment
y_pred = calibrated_svc.predict(X_test_scaled) # Classify passed/failed when needed
y_proba = calibrated_svc.predict_proba(X_test_scaled)[:, 1]

# Basic performance report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Plot Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['NE (Failed)', 'DA (Passed)'], yticklabels=['NE (Failed)', 'DA (Passed)'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Support vector Machine')
plt.tight_layout()
plt.show()

# Calculate ROC AUC Score: metric used to evaluate binary classifiers, especially when the probability scores matter
roc_auc = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC Score: {roc_auc:.2f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # baseline
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Support Vector Machine')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()

#------------------------------------------------------------------------
# HIGH-RISK MACHINE ANALYSIS
#------------------------------------------------------------------------

# Get probabilities for both classes Pass and Failure - here we are looking at the whole dataset to get the idea of how risk can be assessed
X_scaled_all = pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)
probas = calibrated_svc.predict_proba(X_scaled_all)
data['Failure_Probability'] = probas[:, 0] 
data['Pass_Probability'] = probas[:, 1]

# Track uncertainty (how close it is to 50%)
data['Uncertainty'] = abs(0.5 - probas[:, 0]) 

# Getting all statistical measurments for analysis
global_failure_stats = data.groupby('Serijski broj').agg({
    'Verifikacija ispravna': lambda x: 1 - x.mean(),  # Global failure rate
    'Failure_Probability': ['mean', 'max'],
    'Datum izdavanja': ['min', 'max', 'count'],
    'Uređaj': 'first',
    'Proizvođač': 'first'
}).reset_index()

global_failure_stats.columns = [
    'Serijski broj', 'Passed/Failed Verifications [%]', 'Avg_Failure_Probability', 'Max_Failure_Probability',
    'First_Verification', 'Last_Verification',
    'Total_Verifications', 'Uređaj', 'Proizvođač'
]

# Map back 'Uređaj' and 'Proizvođač' to original labels for better readability
global_failure_stats['Uređaj'] = global_failure_stats['Uređaj'].map(original_values['Uređaj'])
global_failure_stats['Proizvođač'] = global_failure_stats['Proizvođač'].map(original_values['Proizvođač'])

# Sort values by 'Predicted_Failure_Probability'
global_failure_stats = global_failure_stats.sort_values(by='Avg_Failure_Probability', ascending=False)

print("\n\n\tTop 100 devices most likely to fail (Global Analysis):")
print(global_failure_stats.head(100)) 

# ------------------------------------------------------------------------
# NEXT YEAR FAILURE PREDICTION - DEVICES VERIFIED MORE THAN 5 TIMES
# ------------------------------------------------------------------------

# Group by device and count actual verifications (not just years)
device_years = data.groupby('Serijski broj').agg({
    'year': lambda x: sorted(set(x)),   # Optional, only needed for analysis
    'Datum izdavanja': 'count'          # Count total verifications
}).reset_index()

device_years.rename(columns={
    'year': 'Years',
    'Datum izdavanja': 'Total_Verifications'
}, inplace=True)

# Keep only devices verified more than 3 times
eligible_devices = device_years[device_years['Total_Verifications'] > 3].copy()

# Filter data for only these devices
eligible_data = data[data['Serijski broj'].isin(eligible_devices['Serijski broj'])].copy()

# Get the last record for each eligible device (latest by date)
last_verifications = eligible_data.sort_values(by='Datum izdavanja').groupby('Serijski broj').tail(1).copy()

# Predict next year failure probability
X_next_year = last_verifications[feature_columns]
probas_next_year = calibrated_svc.predict_proba(X_next_year)
last_verifications['Predicted_Failure_Probability'] = probas_next_year[:, 0]
last_verifications['Predicted_Pass_Probability'] = probas_next_year[:, 1]

# Add total verifications for context
last_verifications = last_verifications.merge(
    eligible_devices[['Serijski broj', 'Total_Verifications']],
    on='Serijski broj',
    how='left'
)

# Map back to original labels
last_verifications['Uređaj'] = last_verifications['Uređaj'].map(original_values['Uređaj'])
last_verifications['Proizvođač'] = last_verifications['Proizvođač'].map(original_values['Proizvođač'])

# Sort by predicted risk
next_year_predictions = last_verifications.sort_values(by='Predicted_Failure_Probability', ascending=False)

print("\n\n\tPredicted High-Risk Devices for Next Year:")
print(next_year_predictions[['Serijski broj', 'Uređaj', 'Proizvođač', 'Datum izdavanja',
                             'Predicted_Failure_Probability', 'Total_Verifications']].head(100))
