import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split  # For splitting data into train/test sets
from sklearn.neighbors import KNeighborsClassifier  # For k-Nearest Neighbors classifier
# Function to load dataset from a CSV file
def load_data(file_path):
    return pd.read_csv(file_path)
# Function to clean column names by removing extra spaces
def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df
# Function to reduce the dataset to two classes (for binary classification)
def filter_binary_classes(df, label_column):
    class_counts = df[label_column].value_counts()
    if len(class_counts) > 2:
        top_two = class_counts.index[:2]  
        df = df[df[label_column].isin(top_two)]  
    return df
# Function to separate features (X) and labels (y)
def split_features_and_labels(df, label_column):
    X = df.drop(columns=[label_column])  
    y = df[label_column]  
    return X, y
# Function to split data into training and testing sets
def split_dataset(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
# Function to train a k-Nearest Neighbors classifier
def train_knn_classifier(X_train, y_train, k=3):
    model = KNeighborsClassifier(n_neighbors=k)  
    model.fit(X_train, y_train)  
    return model
# Function to evaluate model accuracy on test data
def evaluate_accuracy(model, X_test, y_test):
    return model.score(X_test, y_test)  
# Main script execution
if __name__ == "__main__":
    # File path to the dataset and target column
    file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-1\LAB-3\telecom_churn.csv"
    label_column = "Churn"
    # Load and clean the dataset
    df = clean_columns(load_data(file_path))
    # Keep only top 2 classes for binary classification
    df = filter_binary_classes(df, label_column)
    # Use only numeric columns (required for distance-based kNN)
    df_numeric = df.select_dtypes(include=[np.number])
    # Add label column back into numeric DataFrame
    df_numeric[label_column] = df[label_column].values
    # Split into input features and output label
    X, y = split_features_and_labels(df_numeric, label_column)
    # Split into training and test sets
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    # Train the kNN model with k = 3
    knn_model = train_knn_classifier(X_train, y_train, k=3)
    # Evaluate and print the model accuracy on test data
    accuracy = evaluate_accuracy(knn_model, X_test, y_test)
    print(f"Test Accuracy of kNN (k=3): {accuracy:.4f}")
