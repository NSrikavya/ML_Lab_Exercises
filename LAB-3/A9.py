import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.neighbors import KNeighborsClassifier  
# Function to load the dataset from a given file path
def load_data(file_path):
    return pd.read_csv(file_path)
# Function to strip any leading/trailing whitespace from column names
def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df
# Function to reduce multi-class target column to a binary classification task (top 2 classes)
def filter_binary_classes(df, label_column):
    class_counts = df[label_column].value_counts()
    if len(class_counts) > 2:
        top_two = class_counts.index[:2] 
        df = df[df[label_column].isin(top_two)]  
    return df
# Function to separate features (X) and target labels (y)
def split_features_and_labels(df, label_column):
    X = df.drop(columns=[label_column])  
    y = df[label_column] 
    return X, y
# Function to split data into training and testing sets
def split_dataset(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
# Function to train a k-Nearest Neighbors classifier with a specified value of k
def train_knn_classifier(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k) 
    model.fit(X_train, y_train)  
    return model
# Function to evaluate the trained model on test data and return accuracy
def evaluate_accuracy(model, X_test, y_test):
    return model.score(X_test, y_test)
# Function to compute accuracy for a range of k values (from k_min to k_max)
def get_accuracy_for_k_range(X_train, X_test, y_train, y_test, k_min=1, k_max=11):
    accuracies = []
    for k in range(k_min, k_max + 1):
        model = train_knn_classifier(X_train, y_train, k) 
        acc = evaluate_accuracy(model, X_test, y_test) 
        accuracies.append(acc)
    return accuracies

# Function to plot accuracy vs. different values of k
def plot_accuracy_vs_k(k_values, accuracies):
    plt.plot(k_values, accuracies, marker='o')  
    plt.title("Accuracy vs. k in kNN")  
    plt.xlabel("k (Number of Neighbors)") 
    plt.ylabel("Accuracy")  
    plt.grid(True)  
    plt.xticks(k_values)  
    plt.show()  
# Main block to execute the program
if __name__ == "__main__":
    # File path to the dataset and label column name
    file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-1\LAB-3\telecom_churn.csv"
    label_column = "Churn"
    # Load and clean the dataset
    df = clean_columns(load_data(file_path))
    # Filter the data to only include top 2 most frequent classes (binary classification)
    df = filter_binary_classes(df, label_column)
    # Keep only numeric columns (required for kNN)
    df_numeric = df.select_dtypes(include=[np.number])
    # Add the label column back into the numeric DataFrame
    df_numeric[label_column] = df[label_column].values
    # Split into features (X) and labels (y)
    X, y = split_features_and_labels(df_numeric, label_column)
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    # Evaluate accuracy of kNN for k values from 1 to 11
    k_range = list(range(1, 12))
    accuracies = get_accuracy_for_k_range(X_train, X_test, y_train, y_test, 1, 11)
    # Print accuracy for each k value
    for k, acc in zip(k_range, accuracies):
        print(f"k = {k}, Accuracy = {acc:.4f}")
    # Plot accuracy vs. k
    plot_accuracy_vs_k(k_range, accuracies)
