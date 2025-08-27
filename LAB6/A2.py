import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB6\DCT_mal.csv")

# Assuming last column is continuous outcome (adjust if needed)
outcome_col = df.columns[-1]

#Equal-width binning into 4 bins
df['bins'] = pd.cut(df[outcome_col], bins=4, labels=False)

#Count frequencies of each bin
bin_counts = df['bins'].value_counts().sort_index()

#Calculate probabilities
probabilities = bin_counts / len(df)

#Calculate Gini Index
gini = 1 - np.sum(probabilities ** 2)

print(f"Gini Index of the dataset = {gini:.4f}")
