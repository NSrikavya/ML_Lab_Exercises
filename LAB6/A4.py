import pandas as pd
import numpy as np

# ---------- Entropy Function ----------
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

# ---------- Binning Function (A4) ----------
def binning(series, bin_type="width", num_bins=5):
    """
    Convert continuous feature into categorical bins.
    Default: Equal-width with 5 bins.
    """
    if not pd.api.types.is_numeric_dtype(series):
        return series  # categorical features are returned as-is

    if bin_type == "width":
        # Equal-width binning
        return pd.cut(series, bins=num_bins, labels=False, duplicates="drop")
    elif bin_type == "frequency":
        # Equal-frequency binning
        return pd.qcut(series, q=num_bins, labels=False, duplicates="drop")
    else:
        raise ValueError("bin_type must be 'width' or 'frequency'")

# ---------- Information Gain Function ----------
def information_gain(data, feature, target):
    total_entropy = entropy(data[target].values)
    values, counts = np.unique(data[feature], return_counts=True)

    weighted_entropy = sum(
        (c/len(data)) * entropy(data.loc[data[feature] == v, target].values)
        for v, c in zip(values, counts)
    )
    return total_entropy - weighted_entropy

# ---------- Root Node Detector ----------
def find_root_node(data, target, bin_type="width", num_bins=5):
    ig_scores = {}
    for col in data.columns:
        if col != target:
            # Apply binning for continuous features
            binned = binning(data[col], bin_type=bin_type, num_bins=num_bins)
            data[col] = binned

            ig_scores[col] = information_gain(data, col, target)

    # Best feature (root node)
    root_feature = max(ig_scores, key=ig_scores.get)
    return root_feature, ig_scores

# ---------- Run ----------
if __name__ == "__main__":
    # Load dataset
    file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB6\DCT_mal.csv"   # adjust path if needed
    df = pd.read_csv(file_path)

    target_col = "LABEL"  # last column is target

    # Try Equal-Width Binning
    root, scores = find_root_node(df.copy(), target_col, bin_type="width", num_bins=5)
    print("Information Gain (Equal-Width Binning):")
    for f, score in scores.items():
        print(f"{f}: {score:.6f}")
    print("\nBest Root Node Feature (Width):", root)

    # Try Equal-Frequency Binning
    root, scores = find_root_node(df.copy(), target_col, bin_type="frequency", num_bins=5)
    print("\nInformation Gain (Equal-Frequency Binning):")
    for f, score in scores.items():
        print(f"{f}: {score:.6f}")
    print("\nBest Root Node Feature (Frequency):", root)
