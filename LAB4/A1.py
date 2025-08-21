import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

def load_project_data(file_path, feature_cols=('0', '1')):
    df = pd.read_csv(file_path)
    if 'LABEL' not in df.columns:
        raise ValueError("Expected a 'LABEL' column in the dataset")
    # Subset to required columns
    sub = df[list(feature_cols) + ['LABEL']].dropna()
    # Pick top-2 frequent classes for binary classification
    top_two = sub['LABEL'].value_counts().nlargest(2).index.tolist()
    sub = sub[sub['LABEL'].isin(top_two)].copy()
    # Map top-2 labels to binary values (0,1)
    label_map = {top_two[0]: 0, top_two[1]: 1}
    sub['y'] = sub['LABEL'].map(label_map)
    # Prepare features & labels
    X = sub[list(feature_cols)].to_numpy()
    y = sub['y'].to_numpy()
    return X, y, label_map
def make_knn_pipeline(k=3):
    """
    StandardScaler + kNN Classifier pipeline.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('knn', KNeighborsClassifier(n_neighbors=k))
    ])
def evaluate_classifier(model, X, y):
    """
    Returns confusion matrix + macro metrics.
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_pred, average='macro', zero_division=0
    )
    return cm, precision, recall, f1
def infer_fit_status(train_f1, test_f1, threshold=0.10):
    gap = train_f1 - test_f1
    if gap > threshold:
        return "Overfit"
    elif train_f1 < 0.6 and test_f1 < 0.6 and abs(gap) <= threshold:
        return "Underfit"
    else:
        return "Regularfit"

if __name__ == "__main__":
    # Dataset filename
    file = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-4\LAB4\DCT_mal.csv"
    # Load dataset with two features
    X, y, label_map = load_project_data(file, feature_cols=('0', '1'))
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    # Train kNN model
    model = make_knn_pipeline(k=3)
    model.fit(X_train, y_train)
    # Evaluate on training data
    train_cm, train_prec, train_rec, train_f1 = evaluate_classifier(model, X_train, y_train)
    # Evaluate on testing data
    test_cm, test_prec, test_rec, test_f1 = evaluate_classifier(model, X_test, y_test)
    # Print results
    print("=== A1: Confusion Matrix & Performance Metrics ===")
    print(f"Selected Features: ['0', '1']")
    print(f"Label Mapping (original → binary): {label_map}\n")
    print("Train Confusion Matrix:\n", train_cm)
    print("Train → Precision: {:.3f}, Recall: {:.3f}, F1-Score: {:.3f}".format(train_prec, train_rec, train_f1))
    print("\nTest Confusion Matrix:\n", test_cm)
    print("Test → Precision: {:.3f}, Recall: {:.3f}, F1-Score: {:.3f}".format(test_prec, test_rec, test_f1))
    # Model learning outcome
    fit_status = infer_fit_status(train_f1, test_f1)
    print(f"\nModel Learning Outcome: {fit_status}")
