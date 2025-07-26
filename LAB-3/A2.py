import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt  # For plotting graphs
# Function to load data from a CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to extract values of a specific feature/column from the DataFrame
def get_feature_data(df, feature_name):
    return df[feature_name].values

# Function to calculate mean and variance of the given data
def cal_mean_variance(data):
    mean_val = np.mean(data)  # Compute the mean of the data
    var_val = np.var(data)    # Compute the variance of the data
    return mean_val, var_val

# Function to plot a histogram for the given data
def plot_histo(data, feature_name, bins=20):
    plt.hist(data, bins=bins, edgecolor='black', alpha=0.7)  # Create histogram
    plt.title(f'Histogram of {feature_name}')  # Set plot title
    plt.xlabel(feature_name)  # Label x-axis
    plt.ylabel('Frequency')  # Label y-axis
    plt.grid(True)  # Enable grid for better readability
    plt.show()  # Display the plot

# Main block to execute when the script is run directly
if __name__ == "__main__":
    file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-1\LAB-3\telecom_churn.csv"
    # Load the dataset
    df = load_data(file_path)
    # Clean column names by removing leading whitespace
    df.columns = df.columns.str.strip()
    # Print available column names for user reference
    print("Available columns:", df.columns.tolist())
    # Define the feature to analyze
    feature_name = 'Total day minutes'
    # Extract data for the selected feature
    data = get_feature_data(df, feature_name)
    # Calculate mean and variance
    mean_val, variance_val = cal_mean_variance(data)
    # Print the statistical results
    print(f"Mean of {feature_name}: {mean_val}")
    print(f"Variance of {feature_name}: {variance_val}")
    # Plot histogram of the feature
    plot_histo(data, feature_name)
