import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-3\LAB5\DCT_mal.csv") 
# Features for clustering (remove LABEL)
X_cluster = df.drop(columns=['LABEL'])
# List to store distortions
distortions = []
k_range = range(2, 20)
# Compute distortion for each k
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_cluster)
    distortions.append(km.inertia_)
# Plot elbow curve
plt.plot(k_range, distortions, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Distortion (Inertia)")
plt.show()
