import pandas as pd  
import numpy as np  
from sklearn.model_selection import train_test_split  # For splitting the dataset into train and test sets
# Function to load CSV data into a DataFrame
def load_data(file_path):
    return pd.read_csv(file_path)
# Function to clean column names by stripping extra whitespace
def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df
# Function to filter the dataset to include only the top two most frequent classes in the label column
def filter_binary_classes(df, label_column):
    class_counts = df[label_column].value_counts()
    if len(class_counts) > 2:
        top_two_classes = class_counts.index[:2]  # Select top 2 classes by frequency
        df = df[df[label_column].isin(top_two_classes)]  # Keep only rows belonging to top 2 classes
    return df

# Function to separate feature columns (X) and label column (y)
def split_features_and_labels(df, label_column):
    X = df.drop(columns=[label_column])  
    y = df[label_column]  
    return X, y
# Function to split the dataset into training and testing sets
def split_dataset(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
# Main block to execute when script is run directly
if __name__ == "__main__":
    # Path to the dataset file
    file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-1\LAB-3\telecom_churn.csv"
    # Load the dataset
    df = load_data(file_path)
    # Clean whitespace from column names
    df = clean_columns(df)
    # Define the name of the label column
    label_column = "Churn"
    # Filter to keep only two classes if more than two exist (binary classification)
    df = filter_binary_classes(df, label_column)
    # Keep only numeric features for model training
    df_numeric = df.select_dtypes(include=[np.number])
    # Add the label column back to the numeric DataFrame
    df_numeric[label_column] = df[label_column].values
    # Split data into features (X) and labels (y)
    X, y = split_features_and_labels(df_numeric, label_column)
    # Split features and labels into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    # Print the number of samples in the training and test sets
    print(f"Train set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
