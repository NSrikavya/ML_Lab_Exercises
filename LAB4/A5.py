import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# ---------- Training data  ----------
np.random.seed(42)                       # reproducible
X_train = np.random.randint(1, 11, (20, 2))
y_train = np.where(X_train[:, 0] + X_train[:, 1] < 10, 0, 1)  # class 0 (blue) vs class 1 (red)

# ---------- Test grid  ----------
x_values = np.arange(0, 10.1, 0.1)
y_values = np.arange(0, 10.1, 0.1)
xx, yy = np.meshgrid(x_values, y_values)
X_test = np.c_[xx.ravel(), yy.ravel()]   

# ---------- Try multiple k values ----------
k_values = [1, 3, 5, 7, 9, 15]          

# ---------- Plot decision regions for each k ----------
n = len(k_values)
cols = 3
rows = int(np.ceil(n / cols))

fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 5*rows))
axes = np.array(axes).reshape(rows, cols)

for ax, k in zip(axes.ravel(), k_values):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # Predict on the grid
    y_pred = knn.predict(X_test).reshape(xx.shape)

    # Decision regions as an image 
    # 0 -> blue, 1 -> red; we define our own colors to match the spec
    from matplotlib.colors import ListedColormap
    cmap_bg = ListedColormap(["#1f77b4", "#d62728"])  # blue, red

    ax.imshow(
        y_pred,
        origin="lower",
        extent=(x_values.min(), x_values.max(), y_values.min(), y_values.max()),
        interpolation="nearest",
        cmap=cmap_bg,
        alpha=0.35,
        aspect="auto"
    )

    # Overlay training points
    # class 0 (blue)
    ax.scatter(
        X_train[y_train==0, 0], X_train[y_train==0, 1],
        s=100, c="#1f77b4", edgecolor="black", label="Class 0 (Blue)"
    )
    # class 1 (red)
    ax.scatter(
        X_train[y_train==1, 0], X_train[y_train==1, 1],
        s=100, c="#d62728", edgecolor="black", label="Class 1 (Red)"
    )

    ax.set_title(f"kNN Decision Regions (k = {k})")
    ax.set_xlabel("Feature X")
    ax.set_ylabel("Feature Y")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.grid(True, alpha=0.25)

# Hide any empty subplots
for j in range(n, rows*cols):
    fig.delaxes(axes.ravel()[j])

# Only one legend (top-left subplot)
handles, labels = axes.ravel()[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
