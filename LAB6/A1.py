import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB6\DCT_mal.csv")

# Inspect dataset columns
print(df.head())

# Assuming the last column is the continuous outcome (adjust if needed)
outcome_col = df.columns[-1]  # Change if a different column is the target

# Step 1: Equal-width binning into 4 bins
df['bins'] = pd.cut(df[outcome_col], bins=4, labels=False)

# Step 2: Count frequencies of each bin
bin_counts = df['bins'].value_counts().sort_index()

# Step 3: Calculate probabilities
total = len(df)
probabilities = bin_counts / total

# Step 4: Compute entropy using formula
entropy = -np.sum(probabilities * np.log2(probabilities + 1e-9))  # Adding small value to avoid log(0)

print(f"Entropy of the dataset = {entropy:.4f}")
