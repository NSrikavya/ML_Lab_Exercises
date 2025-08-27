import pandas as pd
import numpy as np

# ---------- Entropy Function ----------
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

# ---------- Binning Function ----------
def binning(series, bin_type="width", num_bins=5):
    if not pd.api.types.is_numeric_dtype(series):
        return series
    if bin_type == "width":
        return pd.cut(series, bins=num_bins, labels=False, duplicates="drop")
    elif bin_type == "frequency":
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

# ---------- Find Best Feature ----------
def find_best_feature(data, target, bin_type="width", num_bins=5):
    ig_scores = {}
    for col in data.columns:
        if col != target:
            binned = binning(data[col], bin_type=bin_type, num_bins=num_bins)
            data[col] = binned
            ig_scores[col] = information_gain(data, col, target)

    best_feature = max(ig_scores, key=ig_scores.get)
    return best_feature, ig_scores

# ---------- Decision Tree Builder ----------
def build_tree(data, target, bin_type="width", num_bins=5, depth=0, max_depth=None):
    y = data[target]
    
    # Stopping conditions
    if len(np.unique(y)) == 1:  # Pure node
        return y.iloc[0]
    if len(data.columns) == 1:  # No features left
        return y.mode()[0]
    if max_depth is not None and depth >= max_depth:
        return y.mode()[0]

    # Find best feature
    best_feature, _ = find_best_feature(data.copy(), target, bin_type, num_bins)

    # Create subtree
    tree = {best_feature: {}}
    for value in np.unique(data[best_feature]):
        subset = data[data[best_feature] == value].drop(columns=[best_feature])
        subtree = build_tree(subset, target, bin_type, num_bins, depth+1, max_depth)
        tree[best_feature][value] = subtree
    return tree

# ---------- Prediction ----------
def predict_one(tree, sample):
    if not isinstance(tree, dict):
        return tree
    root = next(iter(tree))
    value = sample[root]
    if value in tree[root]:
        return predict_one(tree[root][value], sample)
    else:
        return None  # unseen value

def predict(tree, data):
    return [predict_one(tree, row) for _, row in data.iterrows()]

# ---------- Run ----------
if __name__ == "__main__":
    # Load dataset
    file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB6\DCT_mal.csv"
    df = pd.read_csv(file_path)

    target_col = "LABEL"

    # Build decision tree 
    decision_tree = build_tree(df.copy(), target_col, bin_type="frequency", num_bins=5, max_depth=3)

    print("Decision Tree (depth 3):")
    print(decision_tree)
