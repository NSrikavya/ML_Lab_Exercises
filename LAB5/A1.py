from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
# Load dataset
df = pd.read_csv(r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-3\LAB5\DCT_mal.csv") 
X = df[['1']]  
y = df['0']          
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# Train model
reg = LinearRegression().fit(X_train, y_train)
# Predictions
y_train_pred = reg.predict(X_train)
y_test_pred = reg.predict(X_test)
