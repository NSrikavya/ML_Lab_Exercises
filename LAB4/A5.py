import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# ---------------------------
# Generate 20 random training data points
# ---------------------------
def generate_training_data(n=20, low=1.0, high=10.0, random_state=42):
    rng = np.random.default_rng(random_state)
    X_train = rng.uniform(low, high, size=(n, 2))
    sums = X_train.sum(axis=1)
    threshold = np.median(sums)
    y_train = (sums >= threshold).astype(int)
    return X_train, y_train

# ---------------------------
# Generate test set grid points
# ---------------------------
def generate_test_grid(low=0.0, high=10.0, step=0.1):
    x_vals = np.arange(low, high + step, step)
    y_vals = np.arange(low, high + step, step)
    XX, YY = np.meshgrid(x_vals, y_vals)
    X_test = np.c_[XX.ravel(), YY.ravel()]  # Flatten grid into 2D array
    return X_test, XX, YY

# ---------------------------
# Train kNN Classifier
# ---------------------------
def train_knn(X_train, y_train, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

# ---------------------------
# Scatter plot for test data predictions
# ---------------------------
def plot_test_classification(X_test, y_pred, k):
    plt.figure(figsize=(7, 6))
    colors = np.array(['blue', 'red'])
    plt.scatter(X_test[:, 0], X_test[:, 1], c=colors[y_pred], s=5, alpha=0.6)
    plt.title(f"A5: kNN Classification (k={k})")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    file = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-4\LAB4\DCT_mal.csv" 

    # Step 1: Generate training data
    X_train, y_train = generate_training_data(n=20, low=1.0, high=10.0, random_state=42)

    # Step 2: Generate test set (grid of ~10,000 points)
    X_test, XX, YY = generate_test_grid(low=0.0, high=10.0, step=0.1)

    # Step 3: Try different k values and observe decision boundaries
    k_values = [1, 3, 5, 9, 15]
    for k in k_values:
        # Train kNN for current k
        knn_model = train_knn(X_train, y_train, k=k)

        # Predict test data labels
        y_pred = knn_model.predict(X_test)

        # Plot classified test data
        plot_test_classification(X_test, y_pred, k)

        print(f"=== kNN Classification (k={k}) ===")
        print(f"Predicted Class 0: {(y_pred == 0).sum()} points")
        print(f"Predicted Class 1: {(y_pred == 1).sum()} points\n")
