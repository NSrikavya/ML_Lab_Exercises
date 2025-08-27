import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB-3\telecom_churn.csv")

# Select only numerical features
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Take two customers (rows) as vectors
x = df[num_cols].iloc[0].values   # Customer 1
y = df[num_cols].iloc[1].values   # Customer 2

# Calculate Minkowski distance for r = 1 to 10
distances = []
r_values = range(1, 11)

for r in r_values:
    dist = np.linalg.norm(x - y, ord=r)
    distances.append(dist)

# Plot r vs distance
plt.figure(figsize=(8,5))
plt.plot(r_values, distances, marker='o', color='blue')
plt.title("Minkowski Distance between Two Customers")
plt.xlabel("r (order of Minkowski distance)")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

# Print distances
for r, d in zip(r_values, distances):
    print(f"Minkowski distance (r={r}): {d:.4f}")
