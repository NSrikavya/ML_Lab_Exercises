import pandas as pd
import numpy as np

# File path
file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB2\Lab Session Data.xlsx"

# Load data from Excel file
df = pd.read_excel(file_path, sheet_name="Purchase data")

# Display column names to verify
print("Column names in sheet:", df.columns.tolist())

# Adjust column names to match what's actually in the sheet
# These names were corrected after inspecting the Excel file
A = df[['Candies (#)', 'Mangoes (Kg)', 'Milk Packets (#)']].values
C = df[['Payment (Rs)']].values

# Find the dimensionality of A matrix
dim = A.shape[1]

# Find the number of vectors in A matrix
vec = A.shape[0]

# Find rank of A matrix
ran_A = np.linalg.matrix_rank(A)

# Calculate pseudo-inverse to estimate product prices
pro_cos = np.linalg.pinv(A) @ C

# Output results
print("Dimensionality: ", dim)
print("Number of vectors: ", vec)
print("Rank of Matrix A: ", ran_A)
print("Estimated Cost per Product:")
print(f"  - Candy: ₹{pro_cos[0][0]:.2f}")
print(f"  - Mango (Kg): ₹{pro_cos[1][0]:.2f}")
print(f"  - Milk Packet: ₹{pro_cos[2][0]:.2f}")
