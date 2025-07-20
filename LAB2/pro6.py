import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB2\Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

# Preprocess: handle categorical columns using label encoding
df_cleaned = df.copy()
for col in df_cleaned.columns:
    if df_cleaned[col].dtype == 'object':
        df_cleaned[col] = df_cleaned[col].astype(str)
        le = LabelEncoder()
        df_cleaned[col] = le.fit_transform(df_cleaned[col])

# Fill missing values with column mean (safe default)
df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))

# Take first two observations
v1 = df_cleaned.iloc[0].values.reshape(1, -1)
v2 = df_cleaned.iloc[1].values.reshape(1, -1)

# Calculate cosine similarity
cos_sim = cosine_similarity(v1, v2)[0][0]

# Output
print(f"Cosine Similarity between first two observations: {cos_sim:.4f}")
