import pandas as pd 
import numpy as np  
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split  
from sklearn.neighbors import KNeighborsClassifier  
# Load dataset from the specified CSV file path
def load_data(file_path):
    return pd.read_csv(file_path)
# Clean column names by removing leading whitespace
def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df
# If the label column contains more than two classes, keep only the top two
def filter_binary_classes(df, label_column):
    class_counts = df[label_column].value_counts()
    if len(class_counts) > 2:
        top_two = class_counts.index[:2]  
        df = df[df[label_column].isin(top_two)]  
    return df
# Separate features and label columns
def split_features_and_labels(df, label_column):
    X = df.drop(columns=[label_column])  
    y = df[label_column]  
    return X, y
# Split the dataset into training and testing sets
def split_dataset(X, y, test_size=0.3, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
# Train a k-Nearest Neighbors classifier for a given k
def train_knn_classifier(X_train, y_train, k):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model
# Evaluate the classifier accuracy on test data
def evaluate_accuracy(model, X_test, y_test):
    return model.score(X_test, y_test)
# Train and evaluate the model for each k in the given range
def get_accuracy_for_k_range(X_train, X_test, y_train, y_test, k_min=1, k_max=11):
    accuracies = []
    for k in range(k_min, k_max + 1):
        model = train_knn_classifier(X_train, y_train, k)
        acc = evaluate_accuracy(model, X_test, y_test)
        accuracies.append(acc)
    return accuracies
# Plot accuracy vs. different values of k
def plot_accuracy_vs_k(k_values, accuracies):
    plt.plot(k_values, accuracies, marker='o')
    plt.title("Accuracy vs. k in kNN")  
    plt.xlabel("k (Number of Neighbors)")  
    plt.ylabel("Accuracy")  
    plt.grid(True)
    plt.xticks(k_values)  
    plt.show()
# Main script logic
if __name__ == "__main__":
    # File path to the dataset and target label
    file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-1\LAB-3\telecom_churn.csv"
    label_column = "Churn"
    # Load and preprocess the dataset
    df = clean_columns(load_data(file_path))
    df = filter_binary_classes(df, label_column)
    # Use only numeric columns (kNN works with numerical data)
    df_numeric = df.select_dtypes(include=[np.number])
    # Add label column back to numeric DataFrame
    df_numeric[label_column] = df[label_column].values
    # Separate into features and labels
    X, y = split_features_and_labels(df_numeric, label_column)
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    # Compute accuracy for k values from 1 to 11
    k_range = list(range(1, 12))
    accuracies = get_accuracy_for_k_range(X_train, X_test, y_train, y_test, 1, 11)
    # Print accuracy for each value of k
    for k, acc in zip(k_range, accuracies):
        print(f"k = {k}, Accuracy = {acc:.4f}")
    # Plot the accuracy vs k graph
    plot_accuracy_vs_k(k_range, accuracies)
