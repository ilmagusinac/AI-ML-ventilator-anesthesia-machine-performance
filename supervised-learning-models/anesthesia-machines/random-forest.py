"""
AI-Based Prediction of Anesthesia Machine Verification Outcomes
- Loads dataset
- Cleans and engineers domain-specific features
- Trains a Random Forest with GridSearchCV
- Calibrates probability predictions
- Identifies high-risk devices based on predicted failure likelihood
"""

import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# Ensure console prints in UTF-8
sys.stdout.reconfigure(encoding="utf-8")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

#------------------------------------------------------------------------
# LOAD AND CLEAN DATASET
#------------------------------------------------------------------------

# Load data from CSV file
data = pd.read_csv("files/anesthesia_machines_2015_2023.csv", encoding="utf-8")

# Drop unnecessary columns that we won't need for prediction
data.drop(columns=['Broj izvještaja', 'Broj naloga', 'Status verifikacije', 'Zahtjev za verifikaciju', 'Metoda', 'Vrsta','Napomena'], errors='ignore', inplace=True)

# Handle incorrect value - this value was intentionally put in the dataset used for the information that something wasn't measured
data.replace("NIJE MJERENO", np.nan, inplace=True)

# Convert comma decimal separators to dot for all columns that should be numeric
for col in data.columns:
    if data[col].dtype == 'object':  # Only process string columns
        data[col] = data[col].str.replace(',', '.', regex=False)

# Clean external inspection columns
external_cols = ['Spoljašnji pregled 1', 'Spoljašnji pregled 2', 'Spoljašnji pregled 3', 'Spoljašnji pregled 4']
for col in external_cols:
    data[col] = data[col].astype(str).str.strip().str.upper()  
    data[col] = data[col].map({'DA': 1, 'NE': 0})

# Encode categorical variables
original_values = {}
categorical_cols = ['Proizvođač', 'Uređaj']
for col in categorical_cols:
    data[col], uniques = pd.factorize(data[col])
    original_values[col] = dict(enumerate(uniques))

# Encode target variable
data['Verifikacija ispravna'] = data['Verifikacija ispravna'].map({'DA': 1, 'NE': 0})

# Coordinated columns
usk_columns = [col for col in data.columns if 'Usklađeno' in col]
for col in usk_columns:
    data[col] = data[col].map({'DA': 1, 'NE': 0})

#data.to_csv("files/anesthesia_cleaned.csv", index=False, encoding="utf-8")

#------------------------------------------------------------------------
# FEATURE ENGINEERING
#------------------------------------------------------------------------

# External check summary - if any external inspection failed, the machine is at risk
data['external_failed'] = data[external_cols].apply(
    lambda row: 1 if (row == 0).any() else 0, axis=1
)

# Count how many measurements were skipped in each row
measure_cols = [col for col in data.columns if 'Zadana vrijednost' in col or 'Mjerena vrijednost' in col or 'Greška mjerenja' in col]
data['skipped_measurements'] = data[measure_cols].isna().sum(axis=1)

# Percentage of coordinated (Usklađeno) measurement - tells us how consistently the mahine was aligned
usk_column = [col for col in data.columns if 'Usklađeno' in col]
data['usk_percent'] = data[usk_column].mean(axis=1) # gives value between 0 and 1 - 0: non were coordinated, 1: all were coordinated

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

# Count breaches where error is outside ± allowed deviation - if a devices error is worse than allowed
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

# Time-Based Features
data['Datum izdavanja'] = pd.to_datetime(data['Datum izdavanja'], dayfirst=True, errors='coerce')
data['year'] = data['Datum izdavanja'].dt.year
data['month'] = data['Datum izdavanja'].dt.month
data['dayofweek'] = data['Datum izdavanja'].dt.dayofweek

#data.to_csv("files/anesthesia_cleaned_with_features.csv", index=False, encoding="utf-8")

#------------------------------------------------------------------------
# MODEL TRAINING AND TUNING
#------------------------------------------------------------------------

y = data['Verifikacija ispravna'] # target variable

feature_columns = [
    'Proizvođač', 'Uređaj',
    'external_failed', 'usk_percent',
    'mean_error', 'max_error', 'std_error',
    'total_breaches', 'breach_ratio', 'skipped_measurements',
    'year', 'month', 'dayofweek', 
    'mean_error_score', 'min_error_score', 'high_risk_score_count'
]

X = data[feature_columns]

# Split the data in training and test sets: 80% train & 20% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # stratisfy is used to balance DA/NE classes

# Tune Random Forest with GridSearchCV - systematically tries all combinations of specified hyperparameters to find the best-performing model
param_grid = {
    'n_estimators': [100, 200, 300],     # Number of trees in the forest
    'max_depth': [None, 10, 20, 3],      # Max depth of each tree
    'min_samples_split': [2, 5, 10],     # Minimum number of samples required to split an internal node
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print("Best Parameters:", grid_search.best_params_) 

#------------------------------------------------------------------------
# CALIBRATION AND EVALUATION
#------------------------------------------------------------------------

calibrated_rf = CalibratedClassifierCV(grid_search.best_estimator_, cv=3)
calibrated_rf.fit(X_train, y_train)

# Predict calibrated probabilities instead of original
y_proba = calibrated_rf.predict_proba(X_test)[:, 0] # 0: NE- we did not
y_pred = calibrated_rf.predict(X_test)

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
plt.title('Confusion Matrix - Random Forest')
plt.tight_layout()
plt.show()

# Calculate ROC AUC Score
roc_auc = roc_auc_score(y_test == 0, y_proba)
print(f"AUC-ROC Score: {roc_auc:.2f}")

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # baseline
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend(loc="lower right")
plt.grid()
plt.tight_layout()
plt.show()

"""
# Visualise Feature Importance: Why is this machine likely to fail?
importances = grid_search.best_estimator_.feature_importances_
feat_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_df, color='skyblue')
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.show()"""

#------------------------------------------------------------------------
# HIGH-RISK MACHINE ANALYSIS
#------------------------------------------------------------------------

# Copy test features and attach the probabilities
X_test_with_probs = X_test.copy()
X_test_with_probs['Failure_Probability'] = y_proba

# Attach real data info only from test set
test_info = data.loc[X_test.index, ['Proizvođač', 'Uređaj', 'Serijski broj']].copy()
test_info['Proizvođač'] = test_info['Proizvođač'].map(original_values['Proizvođač'])
test_info['Uređaj'] = test_info['Uređaj'].map(original_values['Uređaj'])
test_info['Failure_Probability'] = y_proba

# Group by machine (serial number), get max risk per machine
grouped_test_info = test_info.groupby('Serijski broj').agg({
    'Proizvođač': 'first',
    'Uređaj': 'first',
    'Failure_Probability': 'max'
}).reset_index()

device_test_counts = data['Serijski broj'].value_counts().rename('Total_Tests')
grouped_test_info = grouped_test_info.merge(
    device_test_counts,
    left_on='Serijski broj',
    right_index=True,
    how='left'
)

# Sort and show top 20 riskiest machines in the test set
riskiest_test_machines = grouped_test_info.sort_values(by='Failure_Probability', ascending=False)
print("\n\n\tTop 50 riskiest machines (Test Set Only):")
print(riskiest_test_machines.head(50))

# Most Verified & Risky Overlap
devices_count = data.groupby(['Uređaj', 'Serijski broj']).size().reset_index(name='device_verification_count')
devices_counts_sorted = devices_count.sort_values(by='device_verification_count', ascending=False)

risky_serials = set(riskiest_test_machines['Serijski broj'])
verified_serials = set(devices_counts_sorted['Serijski broj'])

overlap_serials = risky_serials.intersection(verified_serials)

print(f"\n\n\t Devices that are BOTH risky and most verified: {len(overlap_serials)} devices")

overlap_devices = riskiest_test_machines[riskiest_test_machines['Serijski broj'].isin(overlap_serials)]
print("\n\t These devices are both highly risky and highly verified:")
print(overlap_devices.head(50))
