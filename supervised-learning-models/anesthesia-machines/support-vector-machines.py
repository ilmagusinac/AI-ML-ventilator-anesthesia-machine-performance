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

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Load data from CSV file
data = pd.read_csv("files/anesthesia_machines_2015_2023.csv", encoding="utf-8")

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#                                                                    CLEANING THE DATASET

# Drop unnecessary columns
data.drop(columns=['Broj izvještaja', 'Broj naloga', 'Status verifikacije', 'Zahtjev za verifikaciju', 'Metoda', 'Vrsta','Napomena'], errors='ignore', inplace=True)

# Handle incorrect value - this value was intentionally put in the dataset used for the information that something wasn't measured
data.replace("NIJE MJERENO", np.nan, inplace=True)

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

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#                                                                           FEATURES ENGINEERING

# External check summary - if any external inspection failed, the machine is at risk
data['external_failed'] = data[external_cols].apply(
    lambda row: 1 if (row == 0).any() else 0, axis=1
)

# Count how many measurements were skipped in each row
measure_cols = [col for col in data.columns if 'Zadana vrijednost' in col or 'Mjerena vrijednost' in col or 'Greška mjerenja' in col]
data['skipped_measurements'] = data[measure_cols].isna().sum(axis=1)

# Percentage of coordinated (usklađeno) measurement - tells us how consistently the mahine was aligned
usk_column = [col for col in data.columns if 'Usklađeno' in col]
data['usk_percent'] = data[usk_column].mean(axis=1) # gives value between 0 and 1 - 0: non were coordinated, 1: all were coordinated

# Calculate usk_percent per type
"""
measure_mapping = {
    'volumen': 'VERIFIKACIJA GREŠKE MJERILA:',
    'pritisak': 'VERIFIKACIJA GREŠKE MJERILA:.1',
    'protok': 'VERIFIKACIJA GREŠKE MJERILA:.2',
    'koncentracija_anestezioloških_gasova': 'VERIFIKACIJA GREŠKE MJERILA:.3'
}

for label, colname in measure_mapping.items():
    new_col = f'usk_{label}'
    data[new_col] = data.apply(
        lambda row: row[usk_column].mean() if pd.notna(row[colname]) else np.nan,
        axis=1
    )

print(data[['VERIFIKACIJA GREŠKE MJERILA:.3', 'usk_koncentracija_anestezioloških_gasova']].head(40)) 
"""

# Error measurement - statistical view of measurement consistency
error_cols = [col for col in data.columns if 'Greška mjerenja' in col]
allowed_cols = [col for col in data.columns if 'Dozvoljeno odstupanje' in col]

# This will only ensure that these error_cols and allowed_cols are numeric without 'NIJE MJERENO' value
data[error_cols] = data[error_cols].apply(pd.to_numeric, errors='coerce')
data[allowed_cols] = data[allowed_cols].apply(pd.to_numeric, errors='coerce')

new_cols = pd.DataFrame({
    'mean_error': data[error_cols].mean(axis=1), # Average error across all points for a device
    'max_error': data[error_cols].max(axis=1), # Worst error recorded
    'std_error': data[error_cols].std(axis=1), # How consistent/inconsistent the errors are
})

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
error_columns = [col for col in data.columns if 'Greška mjerenja' in col]
data['total_tests'] = data[error_columns].notna().sum(axis=1)
data['breach_ratio'] = data['total_breaches'] / data['total_tests']
data['breach_ratio'] = data['breach_ratio'].fillna(0)

# Time-Based Features
data['Datum izdavanja'] = pd.to_datetime(data['Datum izdavanja'], dayfirst=True, errors='coerce')
data['year'] = data['Datum izdavanja'].dt.year
data['month'] = data['Datum izdavanja'].dt.month
data['dayofweek'] = data['Datum izdavanja'].dt.dayofweek

# ---------------------------------------------------------------------------------------------------------------------------------------------

#                                                                          FEATURE SELECTION 
y = data['Verifikacija ispravna'] # target variable

feature_columns = [
    'Proizvođač', 'Uređaj',
    'external_failed', 'usk_percent',
    'mean_error', 'max_error', 'std_error',
    'total_breaches', 'breach_ratio', 'skipped_measurements',
    'year', 'month', 'dayofweek'
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

print("Train class balance:", y_train.value_counts(normalize=True))
print("Test class balance:", y_test.value_counts(normalize=True))

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# SVM Model
svc = SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)

# Hyperparameter tuning
param_grid = {
    'C': [0.1, 1, 10],
    'gamma': ['scale', 'auto']
}
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=1)
grid_search.fit(X_train_scaled, y_train)

# Best model
best_svc = grid_search.best_estimator_

# Calibrate
calibrated_svc = CalibratedClassifierCV(best_svc, cv=3)
calibrated_svc.fit(X_train_scaled, y_train)

# Predict
y_pred = calibrated_svc.predict(X_test_scaled)
y_proba = calibrated_svc.predict_proba(X_test_scaled)[:, 0]  # probability of NE (not verified)???? should be 1

# Evaluation
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

# Calculate ROC AUC Score
roc_auc = roc_auc_score(1-y_test, y_proba)
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

# ---------------------------------------------------------------------------------------------------------------------------------------------

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

# Sort and show top 10 riskiest machines in the test set
riskiest_test_machines = grouped_test_info.sort_values(by='Failure_Probability', ascending=False)
print("\nTop 10 riskiest machines (Test Set Only):")
print(riskiest_test_machines.head(187))

# ---------------------------------------------------------------------------------------------------------------------------------------------

#                                                               Riskiest and most verified devices


devices_count = data.groupby(['Uređaj', 'Serijski broj']).size().reset_index(name='device_verification_count')
devices_counts_sorted = devices_count.sort_values(by='device_verification_count', ascending=False)
print("Top 60 most verified devices:")
print(devices_counts_sorted.head(187))

risky_serials = set(riskiest_test_machines['Serijski broj'])
verified_serials = set(devices_counts_sorted['Serijski broj'])

overlap_serials = risky_serials.intersection(verified_serials)

print(f"\n Devices that are BOTH risky and most verified: {len(overlap_serials)} out of 187")

overlap_devices = riskiest_test_machines[riskiest_test_machines['Serijski broj'].isin(overlap_serials)]
print("\n These devices are both highly risky and highly verified:")
print(overlap_devices)

