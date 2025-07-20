import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB2\Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

# Copy to avoid altering original
df_imputed = df.copy()

# === Step 1: Identify column types ===
numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df_imputed.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric Columns: {numeric_cols}")
print(f"Categorical Columns: {categorical_cols}")

# === Step 2: Handle numeric columns ===
for col in numeric_cols:
    if df_imputed[col].isnull().sum() > 0:
        Q1 = df_imputed[col].quantile(0.25)
        Q3 = df_imputed[col].quantile(0.75)
        IQR = Q3 - Q1
        outliers = df_imputed[(df_imputed[col] < Q1 - 1.5 * IQR) | (df_imputed[col] > Q3 + 1.5 * IQR)]

        if len(outliers) > 0:
            # If outliers are present, use median
            median_val = df_imputed[col].median()
            df_imputed[col] = df_imputed[col].fillna(median_val)
            print(f"{col}: Imputed missing values using MEDIAN = {median_val}")
        else:
            # If no outliers, use mean
            mean_val = df_imputed[col].mean()
            df_imputed[col] = df_imputed[col].fillna(mean_val)
            print(f"{col}: Imputed missing values using MEAN = {mean_val:.2f}")

# === Step 3: Handle categorical columns ===
for col in categorical_cols:
    if df_imputed[col].isnull().sum() > 0:
        mode_val = df_imputed[col].mode()[0]
        df_imputed[col] = df_imputed[col].fillna(mode_val)
        print(f"{col}: Imputed missing values using MODE = {mode_val}")

# === Step 4: Summary of Missing Values After Imputation ===
print("\nMissing values after imputation:")
print(df_imputed.isnull().sum())
