import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB-3\telecom_churn.csv")

# Drop non-numeric columns (like 'State' and 'International plan') for now
X = df.drop(columns=['Churn', 'State', 'International plan', 'Voice mail plan'], errors='ignore')
y = df['Churn']

# Convert categorical yes/no to 0/1 if present
X = X.apply(lambda col: pd.to_numeric(col, errors='coerce')).fillna(0)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Store accuracy results
k_values = range(1, 12)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)

# Print results
for k, acc in zip(k_values, accuracies):
    print(f"k={k}: Accuracy = {acc:.4f}")

# Plot accuracy vs k
plt.plot(k_values, accuracies, marker='o')
plt.xlabel("k (Number of Neighbors)")
plt.ylabel("Accuracy")
plt.title("kNN Accuracy for different k values")
plt.grid(True)
plt.show()
