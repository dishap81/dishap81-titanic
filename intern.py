# -----------------------------
# Task 1: Data Cleaning & Preprocessing
# Dataset: Titanic Dataset
# -----------------------------

# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Step 1: Import Dataset & Explore
# -----------------------------
# Load Titanic dataset (update path if needed)
df = pd.read_csv(r"C:\Users\Admin\Documents\titanic proj\titanic proj\Titanic-Dataset.csv")

# First 5 rows
print("Head of Dataset:\n")
print(df.head())

# Basic info
print("\nDataset Info:\n")
print(df.info())

# Summary stats
print("\nDataset Description:")
print(df.describe())

# Missing values
print("Missing Values:\n")
print(df.isnull().sum())

# -----------------------------
# Step 2: Handle Missing Values
# -----------------------------
# Fill Age with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked with most frequent value (mode)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column (too many missing values)
df.drop(columns=['Cabin'], inplace=True)

# Verify missing values again
print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# -----------------------------
# Step 3: Convert Categorical to Numerical (Encoding)
# -----------------------------
# Map Sex column (male=0, female=1)
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# One-hot encoding for Embarked
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

print("\nData After Encoding:")
print(df.head())

# -----------------------------
# Step 4: Normalize/Standardize Numerical Features
# -----------------------------
scaler = StandardScaler()
num_cols = ['Age', 'Fare']  # Select numerical features
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nData After Standardization:")
print(df[num_cols].head())

# -----------------------------
# Step 5: Visualize & Handle Outliers
# -----------------------------
# Boxplot before removing outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot of Age & Fare (Before Outlier Removal)")
plt.show()

# Outlier removal using IQR for Fare
Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1

df = df[(df['Fare'] >= Q1 - 1.5 * IQR) & (df['Fare'] <= Q3 + 1.5 * IQR)]

# Boxplot after removing outliers
plt.figure(figsize=(10, 5))
sns.boxplot(data=df[['Age', 'Fare']])
plt.title("Boxplot of Age & Fare (After Outlier Removal)")
plt.show()

# -----------------------------
# Final Dataset Ready
# -----------------------------
print("\nFinal Cleaned Dataset Shape:", df.shape)
print("\nFinal Dataset Sample:")
print(df.head())