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
data = pd.read_csv("files/respiratory_machines_2015_2023.csv", encoding="utf-8")
