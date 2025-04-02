# Import necessary libraries
import pandas as pd
from sklearn.utils import check_random_state
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, RocCurveDisplay

# Load data
file_path = r"C:\Users\josep\Dropbox\AI Centennial\Semester 4\Supervised Learning\Group Assinment\KSI_data_cleaned.csv"
data = pd.read_csv(file_path)

# Create a copy
data_Group2 = data.copy()

# Define features (X) and target variable (y)
X = data_Group2.drop(columns=["ACCLASS"])
y = data_Group2["ACCLASS"]

# Perform Train-Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23, stratify=y)

# Define the parameter grid
param_grid = {
    'criterion': ['gini', 'entropy'],  
    'max_depth': [None, 10, 20, 30],  
    'min_samples_split': [2, 5, 10],  
    'min_samples_leaf': [1, 2, 4]    
}
# GRID SEARCH-------------------------------------------------
# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=DecisionTreeClassifier(random_state=57), param_grid=param_grid, scoring='accuracy', refit=True, verbose=3)

# Fit the model to the training data
grid_search.fit(X_train, y_train)

# Extract the best model
best_model_grid = grid_search.best_estimator_

# Evaluate the model on test data
y_pred_grid = best_model_grid.predict(X_test)

# Print GridSearchCV results
print("Best Parameters (Grid Search):", grid_search.best_params_)
print("Accuracy (Grid Search):", accuracy_score(y_test, y_pred_grid))
print("\nClassification Report (Grid Search):\n", classification_report(y_test, y_pred_grid))
print("\nConfusion Matrix (Grid Search):\n", confusion_matrix(y_test, y_pred_grid))

# RANDOMIZED GRID SEARCH-------------------------------------------------
# Define the randomized parameter distribution
param_distributions = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': np.arange(2, 11),
    'min_samples_leaf': np.arange(1, 5)
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=DecisionTreeClassifier(random_state=57), param_distributions=param_distributions, n_iter=50, scoring='accuracy', refit=True, verbose=3, random_state=42)

# Fit the model to the training data
random_search.fit(X_train, y_train)

# Extract the best model
best_model_random = random_search.best_estimator_

# Evaluate the model on test data
y_pred_random = best_model_random.predict(X_test)

# RandomizedSearchCV results
print("Best Parameters (Randomized Search):", random_search.best_params_)
print("Accuracy (Randomized Search):", accuracy_score(y_test, y_pred_random))
print("\nClassification Report (Randomized Search):\n", classification_report(y_test, y_pred_random))
print("\nConfusion Matrix (Randomized Search):\n", confusion_matrix(y_test, y_pred_random))

#ROC Curve and Model Evaluation--------------------------------------------------------------------
# ROC Curve for the best model from Grid Search
roc_display_grid = RocCurveDisplay.from_estimator(best_model_grid, X_test, y_test)
plt.title('ROC Curve - Best Decision Tree Model (Grid Search)')
plt.savefig('roc_curve_best_decision_tree_grid.png')
plt.show()

# ROC Curve for the best model from Randomized Search
roc_display_random = RocCurveDisplay.from_estimator(best_model_random, X_test, y_test)
plt.title('ROC Curve - Best Decision Tree Model (Randomized Search)')
plt.savefig('roc_curve_best_decision_tree_random.png')
plt.show()