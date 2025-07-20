import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Load the dataset
file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB2\Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

# Make a copy for normalization
df_norm = df.copy()

# Select only numeric columns for normalization
numeric_cols = df_norm.select_dtypes(include=[np.number]).columns.tolist()

# Check standard deviation to decide normalization type
print("\n=== Normalization Decisions ===")
columns_minmax = []
columns_standard = []

for col in numeric_cols:
    std = df_norm[col].std()
    unique_vals = df_norm[col].nunique()
    if unique_vals <= 2:
        continue  # skip binary variables
    if std < 1e-5 or std == 0:
        continue  # skip constant columns
    elif df_norm[col].min() >= 0 and df_norm[col].max() <= 1:
        continue  # already normalized
    elif abs(df_norm[col].mean()) < 100:  # simple heuristic
        columns_standard.append(col)
        print(f"{col}: Apply Standardization (Z-score)")
    else:
        columns_minmax.append(col)
        print(f"{col}: Apply Min-Max Scaling")

# Fill missing values for safe transformation
df_norm[numeric_cols] = df_norm[numeric_cols].fillna(df_norm[numeric_cols].mean(numeric_only=True))

# Apply Min-Max Scaling
minmax_scaler = MinMaxScaler()
df_norm[columns_minmax] = minmax_scaler.fit_transform(df_norm[columns_minmax])

# Apply Standardization
standard_scaler = StandardScaler()
df_norm[columns_standard] = standard_scaler.fit_transform(df_norm[columns_standard])

print("\n Normalization completed.")
print("\nFirst few rows of normalized data:")
print(df_norm.head())
