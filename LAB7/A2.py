# 23CSE301 - Lab 07 (A2): Hyperparameter Tuning with RandomizedSearchCV
# -----------------------------------------------------------
# This script:
# 1) Loads the dataset from variable `file`
# 2) Infers problem type (classification vs regression)
# 3) Builds preprocessing + model pipelines
# 4) Tunes hyperparameters using RandomizedSearchCV
# 5) Prints best scores/params for each model
#
# Note: Ensure scikit-learn is installed.

from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.metrics import make_scorer, accuracy_score, f1_score, r2_score, mean_squared_error

# --- Classifiers ---
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

# --- Regressors ---
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor


# =========================
# Data loading & utilities
# =========================

def load_dataset(file: str, target_hint: str = None) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load CSV and split into X, y.
    Heuristics to find target column:
      1) If 'target_hint' provided and present, use it
      2) Common names: ['label','target','class','y','output']
      3) Fallback: last column
    """
    df = pd.read_csv(file)
    if target_hint and target_hint in df.columns:
        target_col = target_hint
    else:
        common = [c for c in ['label', 'target', 'class', 'y', 'output'] if c in df.columns]
        target_col = common[0] if common else df.columns[-1]

    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Basic cleaning: drop completely empty columns
    X = X.dropna(axis=1, how='all')

    return X, y


def detect_problem_type(y: pd.Series) -> str:
    """
    Decide problem type:
      - classification if dtype is object/category or if number of unique values is small relative to size
      - else regression
    """
    if y.dtype.name in ['object', 'category', 'bool']:
        return 'classification'
    unique = y.nunique()
    if unique <= max(10, int(0.02 * len(y))):
        return 'classification'
    return 'regression'


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Create column transformer:
      - Numeric: StandardScaler
      - Categorical: OneHotEncoder(handle_unknown='ignore')
    """
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(with_mean=False) if len(num_cols) > 500 else StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ],
        remainder='drop'
    )
    return pre


# =======================================
# Model spaces for RandomizedSearchCV
# =======================================

def get_models_and_spaces(problem_type: str, random_state: int = 42) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Return dict: model_name -> (estimator, param_distributions)
    Param grids target the pipeline step name 'model__'
    """
    if problem_type == 'classification':
        models = {
            "LogisticRegression": (
                LogisticRegression(max_iter=500, random_state=random_state),
                {
                    "model__C": np.logspace(-3, 3, 50),
                    "model__penalty": ["l2"],
                    "model__solver": ["lbfgs", "liblinear"],
                }
            ),
            "Perceptron": (
                Perceptron(random_state=random_state),
                {
                    "model__penalty": [None, "l2", "l1", "elasticnet"],
                    "model__alpha": np.logspace(-6, -1, 20),
                    "model__fit_intercept": [True, False],
                    "model__max_iter": [500, 1000, 2000],
                }
            ),
            "SVC": (
                SVC(probability=False, random_state=random_state),
                {
                    "model__C": np.logspace(-3, 3, 50),
                    "model__gamma": ["scale", "auto"] + list(np.logspace(-4, 1, 20)),
                    "model__kernel": ["rbf", "poly", "sigmoid"],
                    "model__degree": [2, 3, 4],
                }
            ),
            "RandomForestClassifier": (
                RandomForestClassifier(random_state=random_state, n_jobs=-1),
                {
                    "model__n_estimators": np.arange(100, 601, 50),
                    "model__max_depth": [None] + list(np.arange(3, 21)),
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__max_features": ["sqrt", "log2", None],
                    "model__bootstrap": [True, False],
                }
            ),
            "KNeighborsClassifier": (
                KNeighborsClassifier(),
                {
                    "model__n_neighbors": np.arange(1, 51),
                    "model__weights": ["uniform", "distance"],
                    "model__p": [1, 2],  # Manhattan vs Euclidean
                }
            ),
            "MLPClassifier": (
                MLPClassifier(max_iter=500, random_state=random_state),
                {
                    "model__hidden_layer_sizes": [(h,) for h in [32, 64, 128, 256]] +
                                                 [(64, 32), (128, 64)],
                    "model__activation": ["relu", "tanh", "logistic"],
                    "model__alpha": np.logspace(-6, -2, 10),
                    "model__learning_rate_init": np.logspace(-4, -2, 10),
                }
            ),
        }
    else:
        models = {
            "Ridge": (
                Ridge(random_state=random_state),
                {"model__alpha": np.logspace(-6, 3, 50)}
            ),
            "Lasso": (
                Lasso(random_state=random_state, max_iter=5000),
                {"model__alpha": np.logspace(-6, 1, 50)}
            ),
            "SVR": (
                SVR(),
                {
                    "model__C": np.logspace(-2, 3, 30),
                    "model__gamma": ["scale", "auto"] + list(np.logspace(-4, 1, 10)),
                    "model__epsilon": np.logspace(-3, 0, 10),
                    "model__kernel": ["rbf", "poly", "sigmoid"],
                    "model__degree": [2, 3, 4],
                }
            ),
            "RandomForestRegressor": (
                RandomForestRegressor(random_state=random_state, n_jobs=-1),
                {
                    "model__n_estimators": np.arange(100, 601, 50),
                    "model__max_depth": [None] + list(np.arange(3, 21)),
                    "model__min_samples_split": [2, 5, 10],
                    "model__min_samples_leaf": [1, 2, 4],
                    "model__max_features": ["sqrt", "log2", 1.0],
                    "model__bootstrap": [True, False],
                }
            ),
            "KNeighborsRegressor": (
                KNeighborsRegressor(),
                {
                    "model__n_neighbors": np.arange(1, 51),
                    "model__weights": ["uniform", "distance"],
                    "model__p": [1, 2],
                }
            ),
            "MLPRegressor": (
                MLPRegressor(max_iter=1000, random_state=random_state),
                {
                    "model__hidden_layer_sizes": [(h,) for h in [32, 64, 128, 256]] +
                                                 [(128, 64), (256, 128)],
                    "model__activation": ["relu", "tanh", "logistic"],
                    "model__alpha": np.logspace(-6, -2, 10),
                    "model__learning_rate_init": np.logspace(-4, -2, 10),
                }
            ),
        }

    return models


def tune_model(X: pd.DataFrame, y: pd.Series,
               model_name: str, estimator, param_distributions: Dict[str, Any],
               problem_type: str, random_state: int = 42,
               n_iter: int = 30, cv_splits: int = 5) -> Dict[str, Any]:
    """
    Build pipeline -> RandomizedSearchCV -> fit -> return summary dict
    """
    pre = build_preprocessor(X)
    pipe = Pipeline([("pre", pre), ("model", estimator)])

    if problem_type == 'classification':
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        scoring = {
            "accuracy": make_scorer(accuracy_score),
            "f1_macro": make_scorer(f1_score, average="macro")
        }
        refit_metric = "accuracy"
    else:
        cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        def rmse(y_true, y_pred):
            return np.sqrt(mean_squared_error(y_true, y_pred))
        scoring = {
            "neg_rmse": make_scorer(lambda yt, yp: -np.sqrt(mean_squared_error(yt, yp))),
            "r2": make_scorer(r2_score)
        }
        refit_metric = "neg_rmse"

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring=scoring,
        refit=refit_metric,
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )

    search.fit(X, y)

    result = {
        "model": model_name,
        "best_score": search.best_score_,
        "refit_metric": refit_metric,
        "best_params": search.best_params_,
        "all_cv_results_keys": list(search.cv_results_.keys())
    }
    return result


# =========
#  main()
# =========

def main():
    # As requested: use variable name `file` for the dataset file.
    # If your file name differs, just change this string.
    file = r"C:\Users\lenovo\OneDrive\Desktop\ML_Lab_Exercises\ML_Lab_Exercises-4\LAB7\DCT_mal.csv"   # <- keep the variable name `file` as per instructions

    print("=== Lab 07 (A2): Hyperparameter Tuning with RandomizedSearchCV ===")
    print(f"Loading dataset from: {file}")

    X, y = load_dataset(file)
    print(f"Data shape: X={X.shape}, y={y.shape}")
    problem_type = detect_problem_type(y)
    print(f"Detected problem type: {problem_type}")

    models = get_models_and_spaces(problem_type, random_state=42)

    print("\nTuning models (this may take a few minutes depending on dataset size)...\n")
    results = []

    for name, (estimator, space) in models.items():
        print(f"-> Tuning {name} ...")
        try:
            res = tune_model(X, y, name, estimator, space, problem_type,
                             random_state=42, n_iter=30, cv_splits=5)
            results.append(res)
            metric_name = res["refit_metric"]
            metric_value = res["best_score"]
            if problem_type == 'classification' and metric_name == 'accuracy':
                print(f"   Best {metric_name}: {metric_value:.4f}")
            elif problem_type == 'regression' and metric_name == 'neg_rmse':
                print(f"   Best RMSE: {(-metric_value):.4f}")  # convert back to positive RMSE
            print(f"   Best Params: {res['best_params']}\n")
        except Exception as e:
            print(f"   Skipped {name} due to error: {e}\n")

    # Summary table
    if results:
        print("=== Summary ===")
        if problem_type == 'classification':
            print("Model\t\tBest_Accuracy\tNotes")
            for r in results:
                print(f"{r['model']}\t{r['best_score']:.4f}\t(refit={r['refit_metric']})")
        else:
            print("Model\t\tBest_RMSE\tNotes")
            for r in results:
                if r['refit_metric'] == 'neg_rmse':
                    print(f"{r['model']}\t{(-r['best_score']):.4f}\t(refit={r['refit_metric']})")
                else:
                    print(f"{r['model']}\tN/A\t(refit={r['refit_metric']})")
    else:
        print("No successful tuning runs. Please verify the dataset and try again.")

if __name__ == "__main__":
    main()
