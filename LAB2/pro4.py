import pandas as pd
import numpy as np

# Load the data
file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB2\Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

# Show column names
print("Columns:", df.columns.tolist())

# 1. Data Types
print("\n--- 1. Attribute Data Types ---")
print(df.dtypes)

# 2. Categorical Encoding Scheme Suggestion
print("\n--- 2. Encoding Suggestions ---")
for col in df.columns:
    if df[col].dtype == 'object':
        unique_vals = df[col].dropna().unique()
        print(f"{col}: Nominal → {len(unique_vals)} unique values → Use One-Hot Encoding")

# 3. Value Ranges for Numeric Columns
print("\n--- 3. Value Ranges (Min-Max) ---")
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    print(f"{col}: Min = {df[col].min()}, Max = {df[col].max()}")

# 4. Missing Values
print("\n--- 4. Missing Values ---")
print(df.isnull().sum())

# 5. Outlier Detection (using IQR)
print("\n--- 5. Outliers in Numeric Columns ---")
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
    print(f"{col}: {len(outliers)} outliers")

# 6. Mean & Variance for Numeric Columns
print("\n--- 6. Mean and Variance ---")
for col in numeric_cols:
    mean = df[col].mean()
    var = df[col].var()
    std = df[col].std()
    print(f"{col}: Mean = {mean:.2f}, Variance = {var:.2f}, Std Dev = {std:.2f}")
