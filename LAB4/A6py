import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

# ---------------------------
# Function: Plot training data and return training arrays
# ---------------------------
def plot_training_data(data):
    # Identify numerical features
    features = [col for col in data.select_dtypes(include=['float64', 'int64']).columns]

    # Sample 20 points from each class (2 and 99)
    sample_2 = data[data['LABEL'] == 2][features].sample(n=20, random_state=1)
    sample_99 = data[data['LABEL'] == 99][features].sample(n=20, random_state=1)

    # Combine samples and create binary labels
    sample_data = pd.concat([sample_2, sample_99])
    sample_data['Binary'] = sample_data['LABEL'].apply(lambda x: 0 if x == 2 else 1)

    # Use first two features for visualization
    feature_x = features[0]
    feature_y = features[1]

    # Plot training samples
    plt.figure(figsize=(6,5))
    for label, color in zip([0, 1], ['blue', 'red']):
        subset = sample_data[sample_data['Binary'] == label]
        plt.scatter(subset[feature_x], subset[feature_y],
                    c=color, label=f'Class {label}', edgecolor='k', s=100)
    plt.xlabel(f'Feature {feature_x}')
    plt.ylabel(f'Feature {feature_y}')
    plt.title("Scatter Plot of 20 Random Samples (Class 2 → Blue, Class 99 → Red)")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Return training data
    X_train = sample_data[[feature_x, feature_y]].values
    y_train = sample_data['Binary'].values
    return X_train, y_train

# ---------------------------
# Function: Generate KNN decision boundary plots
# ---------------------------
def plot_knn_decision_boundaries(data):
    X_train, y_train = plot_training_data(data)

    # Generate 2D grid for testing
    x_vals = np.arange(0, 10.1, 0.1)
    y_vals = np.arange(0, 10.1, 0.1)
    xx, yy = np.meshgrid(x_vals, y_vals)
    X_test = np.c_[xx.ravel(), yy.ravel()]

    k_values = [1, 3, 5, 7]
    plt.figure(figsize=(16, 12))

    for i, k in enumerate(k_values, 1):
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        plt.subplot(2, 2, i)
        plt.title(f"k = {k}", fontsize=14)

        # Background decision region
        plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='bwr', alpha=0.2, s=10)

        # Overlay training points
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr', edgecolor='black', s=80)

        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.grid(True)

        # Legend
        handles = [
            plt.Line2D([0], [0], marker='o', color='w', label='Class 0', markerfacecolor='blue', markersize=10),
            plt.Line2D([0], [0], marker='o', color='w', label='Class 1', markerfacecolor='red', markersize=10)
        ]
        plt.legend(handles=handles, title='Training Labels')

    plt.tight_layout()
    plt.show()

# ---------------------------
# Main execution
# ---------------------------
data_path = r'C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-4\LAB4\DCT_mal.csv'
data = pd.read_csv(data_path)
plot_knn_decision_boundaries(data)
