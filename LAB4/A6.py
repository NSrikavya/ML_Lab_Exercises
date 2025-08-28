import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Load dataset
file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB4\A7.py"
df = pd.read_csv(file_path, delimiter="\t")

# Features and target
X = df.drop("LABEL", axis=1).values   # all columns except LABEL
y = df["LABEL"].values               # target

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# k-NN model
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predictions
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Evaluation
print("\n--- Training Evaluation ---")
print("Confusion Matrix:\n", confusion_matrix(y_train, y_train_pred))
print(classification_report(y_train, y_train_pred))

print("\n--- Testing Evaluation ---")
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))
print(classification_report(y_test, y_test_pred))

print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy :", accuracy_score(y_test, y_test_pred))

# Visualization (using first 2 features just for plotting)
plt.figure(figsize=(8, 6))
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', label="Train")
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test_pred, marker='x', cmap='coolwarm', label="Test Prediction")
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.title("k-NN Classification (first 2 features)")
plt.legend()
plt.grid(True)
plt.show()
