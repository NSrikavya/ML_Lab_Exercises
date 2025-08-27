import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB-3\telecom_churn.csv")

# Select only numerical features
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[num_cols].values
y = df["Churn"].astype(int).values  # Convert True/False to 1/0

# Separate into two classes
class0 = X[y == 0]   # Non-churn
class1 = X[y == 1]   # Churn

# Calculate centroids (mean vectors)
centroid0 = np.mean(class0, axis=0)
centroid1 = np.mean(class1, axis=0)

# Calculate spreads (standard deviation vectors)-"How much customers within the same class differ from their centroid."
spread0 = np.std(class0, axis=0)
spread1 = np.std(class1, axis=0)

# Calculate interclass distance (Euclidean distance between centroids)-"How far apart the centroids of two classes are."
centroid_distance = np.linalg.norm(centroid0 - centroid1)

# Print results
print("Centroid (Class 0 - Non Churn):", centroid0)
print("Centroid (Class 1 - Churn):", centroid1)
print("\nSpread (Class 0 - Non Churn):", spread0)
print("Spread (Class 1 - Churn):", spread1)
print("\nInterclass Distance between centroids:", centroid_distance)
