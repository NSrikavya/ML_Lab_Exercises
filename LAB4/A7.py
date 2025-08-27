import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load dataset
data = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB4\DCT_mal.csv")

# 2. Select only two features (for consistency with A6)
X = data[["0", "1"]].values
y = data["LABEL"].values

# 3. Split dataset into train & test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 4. Define parameter grid for 'k'
param_grid = {'n_neighbors': list(range(1, 21))}   # k = 1 to 20

# 5. GridSearchCV for hyperparameter tuning
knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid.fit(X_train, y_train)

# 6. Best parameters & accuracy
print("Best k value:", grid.best_params_)
print("Best cross-validation score:", grid.best_score_)

# 7. Evaluate on test set
best_knn = grid.best_estimator_
y_pred = best_knn.predict(X_test)
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
