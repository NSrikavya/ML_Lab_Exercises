import pandas as pd
import numpy as np

# Loading data
file_path = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-1\LAB-3\telecom_churn.csv"
df = pd.read_csv(file_path)

# Function to filter two classes
def split_classes(df, target_col='Churn'):
    class1 = df[df[target_col] == True]
    class2 = df[df[target_col] == False]
    return class1, class2

# Function to get numeric features
def get_numeric_features(df):
    return df.select_dtypes(include=[np.float64]).columns.to_list()

# Function to compute centroid (mean vector)
def compute_centroid(class_data, features):
    return class_data[features].mean(axis=0)

# Function to compute spread (standard deviation vector)
def compute_spread(class_data, features):
    return class_data[features].std(axis=0)

# Function to compute interclass distance
def compute_interclass_distance(centroid1, centroid2):
    return np.linalg.norm(centroid1 - centroid2)

# Main execution
if __name__ == "__main__":
    class1, class2 = split_classes(df)
    features = get_numeric_features(df)

    centroid1 = compute_centroid(class1, features)
    centroid2 = compute_centroid(class2, features)

    spread1 = compute_spread(class1, features)
    spread2 = compute_spread(class2, features)

    interclass_distance = compute_interclass_distance(centroid1, centroid2)

    # Printing results
    print("Centroid of Class 1 (Churn=True):\n", centroid1)
    print("\nCentroid of Class 2 (Churn=False):\n", centroid2)
    print("\nSpread of Class 1 (Std Dev):\n", spread1)
    print("\nSpread of Class 2 (Std Dev):\n", spread2)
    print("\nInterclass Distance:", interclass_distance)
