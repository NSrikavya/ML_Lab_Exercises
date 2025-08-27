import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB-3\telecom_churn.csv")

# Select numeric features
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
X = df[num_cols].values

# Target variable (Churn: convert True/False -> 0/1)
y = df["Churn"].astype(int).values

# Split dataset into train (70%) and test (30%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Print dataset shapes
print("Training set size:", X_train.shape, "Target size:", y_train.shape)
print("Testing set size:", X_test.shape, "Target size:", y_test.shape)

# Check class balance
print("Class distribution in training set:", np.bincount(y_train))
print("Class distribution in test set:", np.bincount(y_test))
