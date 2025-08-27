import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB-3\telecom_churn.csv")
# Select only numeric features for X
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[num_cols].values
# Target variable (Churn: True/False → 1/0)
y = df["Churn"].astype(int).values
# Train-test split (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
# Train kNN model (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
# Test accuracy
accuracy = knn.score(X_test, y_test)
print(f"Accuracy of kNN (k=3) on test set: {accuracy:.4f}")
# After training
y_pred = knn.predict(X_test)
# Confusion matrix and classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
