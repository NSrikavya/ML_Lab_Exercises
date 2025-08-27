import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB-3\telecom_churn.csv")

# Select only numerical features
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[num_cols].values
y = df["Churn"].astype(int).values

# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Initialize kNN with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Train (fit) the classifier on training data
knn.fit(X_train, y_train)

print("kNN model trained successfully with k=3")
