import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Load dataset
# ---------------------------
def load_dct_data(file_path, feature_cols=('0', '1')):
    """
    Load DCT_mal.csv dataset and extract only the required two features.
    """
    df = pd.read_csv(file_path)
    if feature_cols[0] not in df.columns or feature_cols[1] not in df.columns:
        raise ValueError("Selected feature columns not found in dataset!")
    return df[list(feature_cols)].to_numpy()

# ---------------------------
# Generate random 20 data points between 1 and 10
# ---------------------------
def generate_random_points(n=20, low=1.0, high=10.0, random_state=42):
    rng = np.random.default_rng(random_state)
    return rng.uniform(low, high, size=(n, 2))

# ---------------------------
# Assign classes based on X + Y value threshold
# ---------------------------
def assign_classes(X):
    """
    Use median of (X + Y) sum as threshold for assigning class 0 or 1.
    """
    sums = X.sum(axis=1)
    threshold = np.median(sums)
    return (sums >= threshold).astype(int)

# ---------------------------
# Scatter plot function
# ---------------------------
def plot_scatter(X, y):
    plt.figure(figsize=(7, 5))
    colors = np.array(['blue', 'red'])
    plt.scatter(X[:, 0], X[:, 1], c=colors[y], s=80, edgecolor='black')
    plt.title("A3: Training Data Scatter Plot")
    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.grid(True)
    plt.show()

# ---------------------------
# Main driver
# ---------------------------
if __name__ == "__main__":
    file = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-4\LAB4\DCT_mal.csv"  # Dataset path

    # Load dataset but we're only using 2 features for visualization reference
    _ = load_dct_data(file, feature_cols=('0', '1'))

    # Generate 20 random data points between 1 and 10
    X_train = generate_random_points(n=20, low=1, high=10, random_state=42)

    # Assign classes based on threshold
    y_train = assign_classes(X_train)

    # Show generated data
    print("=== 20 Random Training Data Points ===")
    print(pd.DataFrame(X_train, columns=['X', 'Y']))
    print("\nClass Labels (0=Blue, 1=Red):", y_train)

    # Scatter plot of training data
    plot_scatter(X_train, y_train)
