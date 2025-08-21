import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
def load_irctc_data(file_path):
    df = pd.read_excel(file_path, sheet_name="IRCTC Stock Price")
    # Keep only required numeric columns
    df = df[['Price', 'Open', 'High', 'Low']].dropna()
    X = df[['Open', 'High', 'Low']].to_numpy()
    y = df['Price'].to_numpy()
    return X, y

def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    eps = 1e-8  # to avoid division by zero in MAPE
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, eps, y_true))) * 100
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

if __name__ == "__main__":
    file = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-4\LAB4\Lab Session Data.xlsx"  

    # Load IRCTC Stock Price data
    X, y = load_irctc_data(file)

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test data
    y_pred = model.predict(X_test)

    # Compute regression metrics
    mse, rmse, mape, r2 = regression_metrics(y_test, y_pred)

    # Display results
    print("=== A2: Price Prediction on IRCTC Stock Data ===")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
    print(f"R² Score: {r2:.4f}")

    # Analysis
    print("\n=== Analysis ===")
    if r2 > 0.9 and mape < 5:
        print("Excellent fit: Model predicts price values accurately.")
    elif r2 > 0.7:
        print("Good fit: Model predicts reasonably well but can be improved.")
    else:
        print("Weak fit: Model may underfit; consider adding more features.")
