import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

# ---------- STEP 1: Load and prepare project data ----------
file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB4\DCT_mal.csv"

df = pd.read_csv(file_path)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Encode labels if categorical
if not pd.api.types.is_numeric_dtype(y):
    le = LabelEncoder()
    y = le.fit_transform(y)

# Split and scale
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- STEP 2: Reduce to 2D with PCA ----------
pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train_scaled)
X_test_2d = pca.transform(X_test_scaled)

print(f"Explained variance ratio of PCA components: {pca.explained_variance_ratio_}")

# ---------- STEP 3: Train and plot decision boundaries ----------
def plot_decision_boundary(X, y, classifier, k_value, title, filename):
    h = 0.05
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu,
                edgecolors='black', s=60)
    plt.title(f"{title} (k={k_value})")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# Try different k values like in A5
k_values = [1, 3, 5, 10]
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_2d, y_train)
    plot_decision_boundary(X_train_2d, y_train, knn, k,
                           "Project Data Decision Boundary",
                           f"project_boundary_k{k}.png")

print("✅ Decision boundary plots saved for k =", k_values)
