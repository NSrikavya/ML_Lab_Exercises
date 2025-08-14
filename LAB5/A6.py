# ===== A6: Clustering Scores for Multiple k =====

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-3\LAB5\DCT_mal.csv") 

# Features for clustering (remove LABEL)
X_cluster = df.drop(columns=['LABEL'])

# Lists to store scores
silhouette_scores = []
ch_scores = []
db_scores = []
k_values = range(2, 10)

# Calculate metrics for each k
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X_cluster)
    silhouette_scores.append(silhouette_score(X_cluster, km.labels_))
    ch_scores.append(calinski_harabasz_score(X_cluster, km.labels_))
    db_scores.append(davies_bouldin_score(X_cluster, km.labels_))

# Plot results
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(k_values, silhouette_scores, marker='o')
plt.title("Silhouette Score vs k")
plt.xlabel("k"); plt.ylabel("Score")

plt.subplot(1, 3, 2)
plt.plot(k_values, ch_scores, marker='o')
plt.title("Calinski-Harabasz Score vs k")
plt.xlabel("k"); plt.ylabel("Score")

plt.subplot(1, 3, 3)
plt.plot(k_values, db_scores, marker='o')
plt.title("Davies-Bouldin Index vs k")
plt.xlabel("k"); plt.ylabel("Score")

plt.tight_layout()
plt.show()
