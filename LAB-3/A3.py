# Import required libraries
import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
# Function to load CSV data into a DataFrame
def load_data(file_path):
    return pd.read_csv(file_path)
# Function to clean column names by removing leading/trailing whitespace
def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df
# Function to return only numeric features from the DataFrame
def get_numeric_features(df):
    return df.select_dtypes(include=[np.number])
# Function to compute Minkowski distance of order 'r' between two vectors
def minkowski_distance(vec1, vec2, r):
    return np.power(np.sum(np.abs(vec1 - vec2) ** r), 1 / r)
# Function to compute Minkowski distances for r = 1 to max_r
def compute_distances_over_r(vec1, vec2, max_r=10):
    return [minkowski_distance(vec1, vec2, r) for r in range(1, max_r + 1)]
# Function to plot Minkowski distances vs. different values of r
def plot_minkowski_distances(distances):
    r_values = list(range(1, len(distances) + 1))  # Values of r from 1 to max_r
    plt.plot(r_values, distances, marker='o')  
    plt.title("Minkowski Distance vs r")  
    plt.xlabel("r (Order of Minkowski Distance)")  
    plt.ylabel("Distance")  
    plt.grid(True) 
    plt.show()  
# Main execution block
if __name__ == "__main__":
    # File path to the dataset
    file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-1\LAB-3\telecom_churn.csv"
    # Load and clean the dataset
    df = load_data(file_path)
    df = clean_columns(df)
    # Extract only numeric columns for distance computation
    numeric_df = get_numeric_features(df)
    # Select two numeric rows (vectors) from the DataFrame for distance calculation
    vec1 = numeric_df.iloc[0].values  # First row
    vec2 = numeric_df.iloc[1].values  # Second row
    # Compute Minkowski distances for r = 1 to 10
    distances = compute_distances_over_r(vec1, vec2, max_r=10)
    # Print the distances for each value of r
    for r, dist in enumerate(distances, 1):
        print(f"Minkowski Distance (r={r}): {dist:.4f}")
    # Plot the Minkowski distances
    plot_minkowski_distances(distances)
