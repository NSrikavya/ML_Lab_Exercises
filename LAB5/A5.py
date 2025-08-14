import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-3\LAB5\DCT_mal.csv") 
# Features for clustering (remove LABEL)
X_cluster = df.drop(columns=['LABEL'])
# Train KMeans
kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X_cluster)
# Calculate metrics
sil = silhouette_score(X_cluster, kmeans.labels_)
ch = calinski_harabasz_score(X_cluster, kmeans.labels_)
db = davies_bouldin_score(X_cluster, kmeans.labels_)
# Print results
print(f"Silhouette Score: {sil}")
print(f"Calinski-Harabasz Score: {ch}")
print(f"Davies-Bouldin Index: {db}")
