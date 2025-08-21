from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Features & target
X = data.drop(columns=["LABEL"])
y = data["LABEL"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize features (important for kNN)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define parameter grid for k
param_grid = {'n_neighbors': list(range(1, 31))}

# ----- GridSearchCV -----
grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# ----- RandomizedSearchCV -----
random_search = RandomizedSearchCV(KNeighborsClassifier(), param_grid, n_iter=10, cv=5, 
                                   scoring='accuracy', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)

# Results
print("Best k (GridSearchCV):", grid_search.best_params_, 
      "with score:", grid_search.best_score_)

print("Best k (RandomizedSearchCV):", random_search.best_params_, 
      "with score:", random_search.best_score_)
