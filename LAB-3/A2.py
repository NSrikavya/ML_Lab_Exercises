import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB-3\telecom_churn.csv")

# Select one feature
feature = "Total day minutes"
data = df[feature].values

# Calculate histogram data
hist_counts, bin_edges = np.histogram(data, bins=20)  # 20 buckets

# Calculate mean and variance
mean_val = np.mean(data)
var_val = np.var(data)

# Print mean and variance
print(f"Mean of {feature}: {mean_val:.2f}")
print(f"Variance of {feature}: {var_val:.2f}")

# Plot histogram
plt.figure(figsize=(8,5))
plt.hist(data, bins=20, color='blue', edgecolor='black')
plt.axvline(mean_val, color='red', linestyle='dashed', linewidth=2, label=f"Mean = {mean_val:.2f}")
plt.title(f"Histogram of {feature}")
plt.xlabel(feature)
plt.ylabel("Number of Customers")
plt.legend()
plt.show()
