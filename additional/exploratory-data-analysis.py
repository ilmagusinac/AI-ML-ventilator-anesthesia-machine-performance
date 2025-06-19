import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv("files/respiratory_machines_2015_2024.csv")
print(df.shape)
print(df.head())

print(df.info())  # Data types and non-null values
print(df.describe())  # Summary statistics for numerical columns
print(df.isnull().sum())  # Count missing values per column

sns.countplot(data=df, x='Verifikacija ispravna')
plt.title("Class Distribution: Verification Outcome")
plt.show()

df['Datum izdavanja'] = pd.to_datetime(df['Datum izdavanja'], errors='coerce', dayfirst=True)
df['year'] = df['Datum izdavanja'].dt.year

yearly_counts = df.groupby(['year', 'Verifikacija ispravna']).size().unstack().fillna(0)
yearly_counts.plot(kind='bar', stacked=True)
plt.title("Verifications per Year (Passed vs Failed)")
plt.ylabel("Count")
plt.show()

numeric_cols = df.select_dtypes(include=['float64', 'int64'])
correlation = numeric_cols.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(correlation, annot=False, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

sns.histplot(df['mean_error'], kde=True)
plt.title("Distribution of Mean Error")
plt.show()

sns.boxplot(data=df, x='Verifikacija ispravna', y='mean_error')
plt.title("Mean Error by Verification Outcome")
plt.show()

print(df['Proizvođač'].value_counts())
print(df['Spoljašnji pregled 1'].value_counts())
