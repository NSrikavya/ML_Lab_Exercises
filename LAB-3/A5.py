import pandas as pd 
import numpy as np  
from sklearn.model_selection import train_test_split  # To split the dataset into train/test sets
from sklearn.neighbors import KNeighborsClassifier  # For the k-Nearest Neighbors classifier
# Function to load data from a CSV file
def load_data(file_path):
    return pd.read_csv(file_path)
# Function to remove leading/trailing spaces from column names
def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df
# Function to keep only the top two classes in a label column (for binary classification)
def filter_binary_classes(df, label_column):
    class_counts = df[label_column].value_counts()
    if len(class_counts) > 2:
        top_two_classes = class_counts.index[:2] 
        df = df[df[label_column].isin(top_two_classes)] 
    return df
# Function to split the dataset into features (X) and target/label (y)
def split_features_and_labels(df, label_column):
    X = df.drop(columns=[label_column]) 
    y = df[label_column]  
    return X, y
# Function to split data into training and testing sets
def split_dataset(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
# Function to train a k-Nearest Neighbors (kNN) classifier
def train_knn_classifier(X_train, y_train, k=3):
    model = KNeighborsClassifier(n_neighbors=k)  
    model.fit(X_train, y_train) 
    return model
# Main execution block
if __name__ == "__main__":
    # File path and target column
    file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-1\LAB-3\telecom_churn.csv"
    label_column = "Churn"
    # Load and clean the dataset
    df = clean_columns(load_data(file_path))

    # Reduce to binary classification if necessary
    df = filter_binary_classes(df, label_column)

    # Keep only numeric columns (for distance-based models like kNN)
    df_numeric = df.select_dtypes(include=[np.number])

    # Reattach the label column to the numeric DataFrame
    df_numeric[label_column] = df[label_column].values

    # Split the data into features and labels
    X, y = split_features_and_labels(df_numeric, label_column)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    # Train the kNN model using k = 3
    knn_model = train_knn_classifier(X_train, y_train, k=3)

    # Indicate successful training
    print("kNN model with k=3 trained successfully.")
