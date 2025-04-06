import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, accuracy_score, roc_auc_score, roc_curve
from sklearn.calibration import CalibratedClassifierCV

# Ensure console prints in UTF-8
sys.stdout.reconfigure(encoding="utf-8")

# Load data from CSV file
data = pd.read_csv("files/anesthesia_machines_2015_2023.csv", encoding="utf-8")

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#                                                                       CLEANING THE DATASET

# Drop unnecessary columns
data.drop(columns=['Broj izvještaja', 'Broj naloga', 'Status verifikacije', 'Zahtjev za verifikaciju', 'Napomena'], errors='ignore', inplace=True)

# Handle incorrect value
data.replace("NIJE MJERENO", np.nan, inplace=True)

# Clean external inspection columns
external_cols = ['Spoljašnji pregled 1', 'Spoljašnji pregled 2', 'Spoljašnji pregled 3', 'Spoljašnji pregled 4']
for col in external_cols:
    data[col] = data[col].astype(str).str.strip().str.upper()  
    data[col] = data[col].map({'DA': 1, 'NE': 0})

# Encode categorical variables
original_values = {}
categorical_cols = ['Metoda', 'Vrsta', 'Proizvođač', 'Uređaj']
for col in categorical_cols:
    data[col], uniques = pd.factorize(data[col])
    original_values[col] = dict(enumerate(uniques))

# Encode target variable
data['Verifikacija ispravna'] = data['Verifikacija ispravna'].map({'DA': 1, 'NE': 0})

# Coordinated columns
usk_columns = [col for col in data.columns if 'Usklađeno' in col]
for col in usk_columns:
    data[col] = data[col].map({'DA': 1, 'NE': 0})

# Fill remaining NaNs with median (for numeric columns)
data.fillna(data.median(numeric_only=True), inplace=True)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#                                                                           FEATURES ENGINEERING

# External check summary - if any external inspection failed, the machine is at risk
data['external_failed'] = data[external_cols].apply(
    lambda row: 1 if (row == 0).any() else 0, axis=1
)

# Percentage of coordinated (usklađeno) measurement 
usk_column = [col for col in data.columns if 'Usklađeno' in col]
data['usk_percent'] = data[usk_column].mean(axis=1) # gives value between 0 and 1 - 0: non were coordinated, 1: all were coordinated

# Error measurement - statistical view of measurement consistency
error_cols = [col for col in data.columns if 'Greška mjerenja' in col]
allowed_cols = [col for col in data.columns if 'Dozvoljeno odstupanje' in col]

data[error_cols] = data[error_cols].apply(pd.to_numeric, errors='coerce')
data[allowed_cols] = data[allowed_cols].apply(pd.to_numeric, errors='coerce')

new_cols = pd.DataFrame({
    'mean_error': data[error_cols].mean(axis=1),
    'max_error': data[error_cols].max(axis=1),
    'std_error': data[error_cols].std(axis=1),
})

# Calculate breaches where error is outside ± allowed deviation
breaches = []
for err_col, dev_col in zip(error_cols, allowed_cols):
    breach = ((data[err_col] < -data[dev_col]) | (data[err_col] > data[dev_col])).astype(int)
    breaches.append(breach)

data = pd.concat([data, new_cols], axis=1)
data = data.copy()

data['total_breaches'] = np.vstack(breaches).sum(axis=0)

# Count of Missing Measurements
measure_cols = [col for col in data.columns if 'Zadana vrijednost' in col or 'Mjerena vrijednost' in col]
data['missing_measurements'] = data[measure_cols].isna().sum(axis=1)

# Time-Based Features - provjeri
data['Datum izdavanja'] = pd.to_datetime(data['Datum izdavanja'], dayfirst=True, errors='coerce')
data['year'] = data['Datum izdavanja'].dt.year
data['month'] = data['Datum izdavanja'].dt.month
data['dayofweek'] = data['Datum izdavanja'].dt.dayofweek

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#                                                                           FEATURE SELECTION 
y = data['Verifikacija ispravna'] # target variable

feature_columns = [
    'Metoda', 'Vrsta', 'Proizvođač', 'Uređaj',
    'external_failed', 'usk_percent',
    'mean_error', 'max_error', 'std_error',
    'total_breaches', 'missing_measurements',
    'year', 'month', 'dayofweek'
]

X = data[feature_columns]

# Split the data in training and test sets: 80% train & 20% test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # stratisfy is used to balance DA/NE classes

# Create the model
rf_model = RandomForestClassifier(
    n_estimators=100,           # number of trees
    max_depth=None,             # allow full growth
    random_state=42,            # reproducibility
    class_weight='balanced'     # handle imbalance between DA/NE
)

# Train the model
rf_model.fit(X_train, y_train)

# Make prediction
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1] # Probability of class 1 (failure)

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
roc_auc = roc_auc_score(y_test, y_proba)
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

# Visualise Feature Importance: Why is this machine likely to fail?
importances = rf_model.feature_importances_
feature_names = X_train.columns

feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_df, color='skyblue')
plt.title('Feature Importance - Random Forest')
plt.tight_layout()
plt.show()

# Tune Random Forest with GridSearchCV
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
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

# Use the best model
rf_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# ------------------------------------------------------------------------------------------------------------------------------------------------------------------------

calibrated_rf = CalibratedClassifierCV(rf_model, cv=3)  # use cv=3 if you don’t have many NE
calibrated_rf.fit(X_train, y_train)

# Predict calibrated probabilities instead of original
y_proba = calibrated_rf.predict_proba(X_test)[:, 1]
y_pred = calibrated_rf.predict(X_test)

# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Copy test features and attach the probabilities
X_test_with_probs = X_test.copy()
X_test_with_probs['Failure_Probability'] = y_proba

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

# Sort and show top 10 riskiest machines in the test set
riskiest_test_machines = grouped_test_info.sort_values(by='Failure_Probability', ascending=False)
print("\nTop 10 riskiest machines (Test Set Only):")
print(riskiest_test_machines.head(10))


# -----------------------------------------------------------------------------------------------------------

# Get real labels from test set
real_labels = y.loc[X_test.index]

# Find risky predictions that were actually OK (False Positives based on high probability)
high_risk_preds = y_proba >= 0.9
false_positives = test_info.loc[high_risk_preds & (real_labels == 1)]

print("\nFalse Positives with high risk prediction (>= 0.9):")
print(false_positives[['Serijski broj', 'Proizvođač', 'Uređaj', 'Failure_Probability']])

# SHAP
# TreeExplainer on rf_model (not calibrated_rf)
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_test)

# Initialize JS for interactive plots (optional, useful in notebooks)
#shap.initjs()

# Waterfall for first machine in test set (class 1 = failure)
shap.plots.waterfall(shap.Explanation(
    values=shap_values[1][0],                     # SHAP values for class 1
    base_values=explainer.expected_value[1],      # expected output for class 1
    data=X_test.iloc[0],                          # input data for this instance
    feature_names=X_test.columns                  # feature names
))
shap.summary_plot(shap_values[1], X_test)
print("X_test shape:", X_test.shape)
print("X_test columns:\n", X_test.columns.tolist())
