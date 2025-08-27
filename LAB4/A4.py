import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# 1. Generate 20 random training points 
np.random.seed(42)
X_train = np.random.randint(1, 11, (20, 2))
y_train = np.where(X_train[:, 0] + X_train[:, 1] < 10, 0, 1)

# 2. Generate test data grid 
x_values = np.arange(0, 10.1, 0.1)
y_values = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x_values, y_values)
X_test = np.c_[xx.ravel(), yy.ravel()]  

# 3. Train kNN classifier (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# 4. Predict test labels
y_pred = knn.predict(X_test)

# 5. Scatter plot of test data 
plt.figure(figsize=(7, 7))
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap=plt.cm.bwr, alpha=0.3, s=10)

# Overlay training points (larger, with edge color)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.bwr,
            edgecolor="black", s=100, marker="o", label="Training points")

plt.title("kNN (k=3) Classification with Decision Boundaries")
plt.xlabel("Feature X")
plt.ylabel("Feature Y")
plt.legend()
plt.grid(True)
plt.show()
