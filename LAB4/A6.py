import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 1. Load dataset
data = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB4\DCT_mal.csv")

# 2. Select ONLY two features for visualization (e.g., '0' and '1')
X = data[["0", "1"]].values   # ✅ keep just 2 columns
y = data["LABEL"].values

# 3. Train kNN classifier
k = 3
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X, y)

# 4. Create meshgrid (only in 2D now)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# 5. Predict class for each point in the meshgrid
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 6. Plot decision boundary + training points
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor="k")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title(f"kNN (k={k}) on DCT_mal.csv")
plt.show()
