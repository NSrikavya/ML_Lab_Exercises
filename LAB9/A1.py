from sklearn.ensemble import StackingClassifier, StackingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_diabetes
from sklearn.metrics import accuracy_score, mean_squared_error
# Load dataset
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
base_models = [
    ('knn', KNeighborsClassifier()),
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('rf', RandomForestClassifier(random_state=42))
]

# Define meta-model
meta_model = LogisticRegression()

# Build stacking classifier
stack_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model
)

# Train
stack_clf.fit(X_train, y_train)

# Predict
y_pred = stack_clf.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
# Load dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define base models
base_models = [
    ('knn', KNeighborsRegressor()),
    ('dt', DecisionTreeRegressor(random_state=42)),
    ('rf', RandomForestRegressor(random_state=42))
]

# Define meta-model
meta_model = LinearRegression()

# Build stacking regressor
stack_reg = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model
)

# Train
stack_reg.fit(X_train, y_train)

# Predict
y_pred = stack_reg.predict(X_test)

# Evaluate
print("MSE:", mean_squared_error(y_test, y_pred))
