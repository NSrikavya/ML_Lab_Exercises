import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error
# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-3\LAB5\DCT_mal.csv") 
# Features: all columns except LABEL
X_all = df.drop(columns=['LABEL'])
# Target: column "0" (numeric)
y_all = df['0']
# Train-test split
X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(
    X_all, y_all, test_size=0.3, random_state=42
)
# Train Linear Regression
reg_all = LinearRegression().fit(X_train_all, y_train_all)
# Predictions
y_train_pred_all = reg_all.predict(X_train_all)
y_test_pred_all = reg_all.predict(X_test_all)
# Regression metrics function
def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2
# Print metrics
print("Train Metrics (All Features):", regression_metrics(y_train_all, y_train_pred_all))
print("Test Metrics  (All Features):", regression_metrics(y_test_all, y_test_pred_all))
