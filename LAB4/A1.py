import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

#Load dataset
data = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB4\DCT_mal.csv")

print("Dataset columns:", data.columns)

#Separate features (X) and target (y)
# Assuming the last column is the target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Encode labels if categorical
if y.dtype == 'object':
    y = LabelEncoder().fit_transform(y)

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

#Train k-NN model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

#Predictions
y_train_pred = knn.predict(X_train)
y_test_pred = knn.predict(X_test)

# Confusion Matrices
print("\nConfusion Matrix (Train):\n", confusion_matrix(y_train, y_train_pred))
print("\nConfusion Matrix (Test):\n", confusion_matrix(y_test, y_test_pred))

# Classification Reports
print("\nClassification Report (Train):\n", classification_report(y_train, y_train_pred))
print("\nClassification Report (Test):\n", classification_report(y_test, y_test_pred))

# Accuracy Comparison
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Train Accuracy: {train_acc:.3f}")
print(f"Test Accuracy: {test_acc:.3f}")

#Learning Outcome
if train_acc > 0.95 and test_acc < 0.7:
    print("Model is OVERFITTING.")
elif train_acc < 0.7 and test_acc < 0.7:
    print("Model is UNDERFITTING.")
else:
    print("Model is GENERALIZING (regular fit).")
