import pandas as pd
import sys

# Ensure console prints in UTF-8
sys.stdout.reconfigure(encoding="utf-8")

# Load data from CSV file
data = pd.read_csv("files/respiratory_machines_2015_2023.csv", encoding="utf-8")

devices_count = data.groupby(['Uređaj', 'Serijski broj']).size().reset_index(name='device_verification_count')
devices_counts_sorted = devices_count.sort_values(by='device_verification_count', ascending=False)
print("Top 30 most verified devices:")
print(devices_counts_sorted.head(30))