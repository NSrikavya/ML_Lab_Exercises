import numpy as np
import pandas as pd

# Function for equal-width binning
def equal_width_binning(data, num_bins=4):
    min_val = np.min(data)
    max_val = np.max(data)
    bin_width = (max_val - min_val) / num_bins
    bins = [min_val + i * bin_width for i in range(num_bins + 1)]
    binned_data = np.digitize(data, bins, right=False) - 1  # bin indices start at 0
    return binned_data, bins

# Function to calculate entropy
def calculate_entropy(data):
    values, counts = np.unique(data, return_counts=True)
    probabilities = counts / counts.sum()
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy

# Load dataset
file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-3\LAB6\DCT_mal.csv"  # Use uploaded file path
df = pd.read_csv(file_path)

# Target column
target = df['LABEL'].values

# Check if target is continuous
if np.issubdtype(target.dtype, np.number) and len(np.unique(target)) > 10:
    target_binned, bin_edges = equal_width_binning(target, num_bins=4)
    entropy_value = calculate_entropy(target_binned)
    print("Equal-width bin edges:", bin_edges)
else:
    entropy_value = calculate_entropy(target)

print("Entropy of the target variable:", entropy_value)
