import pandas as pd
import numpy as np

# ---------- Entropy Function ----------
def entropy(y):
    values, counts = np.unique(y, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

# ---------- Information Gain Function ----------
def information_gain(data, feature, target):
    total_entropy = entropy(data[target].values)
    values, counts = np.unique(data[feature], return_counts=True)
    
    # Weighted entropy
    weighted_entropy = sum(
        (c/len(data)) * entropy(data.loc[data[feature] == v, target].values)
        for v, c in zip(values, counts)
    )
    return total_entropy - weighted_entropy

# ---------- Root Node Detector ----------
def find_root_node(data, target):
    ig_scores = {}
    for col in data.columns:
        if col != target:
            # Skip columns with too many unique values (likely continuous)
            if data[col].nunique() > 20:
                continue
            ig_scores[col] = information_gain(data, col, target)
    
    root_feature = max(ig_scores, key=ig_scores.get)
    return root_feature, ig_scores


# ---------- Run on Uploaded File ----------
file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB6\DCT_mal.csv"
df = pd.read_csv(file_path)

target_col = df.columns[-1]   # assume last col is target
root, scores = find_root_node(df, target_col)

print("Information Gain scores:", scores)
print("Best Root Node Feature:", root)
