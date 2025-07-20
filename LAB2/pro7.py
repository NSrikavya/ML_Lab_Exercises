import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# === Load data ===
file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises\LAB2\Lab Session Data.xlsx"
df = pd.read_excel(file_path, sheet_name="thyroid0387_UCI")
df_20 = df.iloc[:20].copy()

# === Step 1: Preprocess for Jaccard & SMC ===
# Convert categorical to numbers
df_binary = df_20.copy()
for col in df_binary.columns:
    if df_binary[col].dtype == 'object':
        df_binary[col] = df_binary[col].astype(str)
        df_binary[col] = LabelEncoder().fit_transform(df_binary[col])

# Fill missing with 0 and threshold to binary (1 if non-zero)
df_binary = df_binary.fillna(0)
df_binary = df_binary.applymap(lambda x: 1 if x != 0 else 0)

# === Step 2: Jaccard & SMC ===
def jaccard_smc_matrix(data):
    n = len(data)
    jc_matrix = np.zeros((n, n))
    smc_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            f11 = f10 = f01 = f00 = 0
            for a, b in zip(data.iloc[i], data.iloc[j]):
                if a == 1 and b == 1:
                    f11 += 1
                elif a == 1 and b == 0:
                    f10 += 1
                elif a == 0 and b == 1:
                    f01 += 1
                else:
                    f00 += 1

            jc = f11 / (f11 + f10 + f01) if (f11 + f10 + f01) else 0
            smc = (f11 + f00) / (f11 + f10 + f01 + f00) if (f11 + f10 + f01 + f00) else 0
            jc_matrix[i, j] = jc
            smc_matrix[i, j] = smc

    return jc_matrix, smc_matrix

jc_matrix, smc_matrix = jaccard_smc_matrix(df_binary)

# === Step 3: Cosine Similarity on full data ===
df_encoded = df_20.copy()
for col in df_encoded.columns:
    if df_encoded[col].dtype == 'object':
        df_encoded[col] = df_encoded[col].astype(str)
        df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col])

df_encoded = df_encoded.fillna(df_encoded.mean(numeric_only=True))
cos_matrix = cosine_similarity(df_encoded)

# === Step 4: Heatmap Function ===
def plot_heatmap(matrix, title):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, annot=False, cmap="coolwarm", xticklabels=range(1, 21), yticklabels=range(1, 21))
    plt.title(title)
    plt.xlabel("Observation")
    plt.ylabel("Observation")
    plt.tight_layout()
    plt.show()

# === Step 5: Plot all 3 heatmaps ===
plot_heatmap(jc_matrix, "Jaccard Coefficient Heatmap (Binary Derived)")
plot_heatmap(smc_matrix, "Simple Matching Coefficient Heatmap (Binary Derived)")
plot_heatmap(cos_matrix, "Cosine Similarity Heatmap (All Attributes)")
