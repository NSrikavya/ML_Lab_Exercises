import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
def generate_training_data(n=20, low=1.0, high=10.0, random_state=42):
    rng = np.random.default_rng(random_state)
    X_train = rng.uniform(low, high, size=(n, 2))
    sums = X_train.sum(axis=1)
    threshold = np.median(sums)
    y_train = (sums >= threshold).astype(int)
    return X_train, y_train
def generate_test_grid(low=0.0, high=10.0, step=0.1):
    x_vals = np.arange(low, high + step, step)
    y_vals = np.arange(low, high + step, step)
    XX, YY = np.meshgrid(x_vals, y_vals)
    X_test = np.c_[XX.ravel(), YY.ravel()] 
    return X_test, XX, YY
def train_knn(X_train, y_train, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model
def plot_test_classification(X_test, y_pred, XX, YY):
    plt.figure(figsize=(8, 6))
    colors = np.array(['blue', 'red'])
    plt.scatter(X_test[:, 0], X_test[:, 1], c=colors[y_pred], s=5, alpha=0.6)
    plt.title("A4: kNN Classification on Test Set (k=3)")
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

    # Step 3: Train kNN classifier (k=3)
    knn_model = train_knn(X_train, y_train, k=3)

    # Step 4: Predict test data classes
    y_pred = knn_model.predict(X_test)

    # Step 5: Plot classified test data
    plot_test_classification(X_test, y_pred, XX, YY)

    # Display summary
    print("=== Test Data Classification Complete ===")
    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Test Samples: {X_test.shape[0]}")
    print(f"Predicted Class 0: {(y_pred == 0).sum()} points")
    print(f"Predicted Class 1: {(y_pred == 1).sum()} points")
