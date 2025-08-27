import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB-3\telecom_churn.csv")

# Keep only two classes: Churn = 0 or 1
df = df[df['Churn'].isin([0, 1])]

# Convert categorical (string) columns to numeric
df = pd.get_dummies(df, drop_first=True)

# Separate features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train kNN classifier (k=3)
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Predict entire test set
y_pred_all = knn.predict(X_test)
print("Predictions for first 20 test samples:\n", y_pred_all[:20])

# Predict for a single test vector
test_vect = X_test.iloc[0]  # first test sample
prediction = knn.predict([test_vect])  # single prediction
print("\nActual class:", y_test.iloc[0])
print("Predicted class:", prediction[0])

# Check if prediction is correct
if prediction[0] == y_test.iloc[0]:
    print("Correct prediction for this test sample")
else:
    print("Wrong prediction for this test sample")
