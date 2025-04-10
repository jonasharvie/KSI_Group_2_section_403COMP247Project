# -*- coding: utf-8 -*-
"""
Created on Thu Apr 10 02:05:27 2025

@author: Shaik
"""

#------Rubiya-Logistic Regression Begins-----------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt
import pandas as pd

data_Group2_processed = pd.read_csv('KSI_data_cleaned.csv')
# Load the processed dataset
X = data_Group2_processed.drop(columns=['ACCLASS'])  # Features
y = data_Group2_processed['ACCLASS']  # Target

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Initialize logistic regression
logreg = LogisticRegression(max_iter=1000, random_state=42)

# Train the model
logreg.fit(X_train, y_train)

# Predict on the test set
y_pred = logreg.predict(X_test)

# Evaluation for basic logistic regression
print("\n--- Logistic Regression Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average='weighted'))
print("Recall:", recall_score(y_test, y_pred, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

#-----------------Grid Search--------------------------------

# Hyperparameter grid
param_grid = {
    'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100],
    'logisticregression__penalty': ['l2'],
    'logisticregression__solver': ['liblinear', 'lbfgs', 'saga'],
    'logisticregression__max_iter': [1000, 2000, 5000]
}

# Pipeline with scaling
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(random_state=42)
)

# Grid Search with the pipeline
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters and accuracy
print("\n--- Grid Search Results ---")
print("Best Parameters:", grid_search.best_params_)
print("Best Accuracy:", grid_search.best_score_)

# Predict with best model
y_pred_grid = grid_search.predict(X_test)

# Evaluation for Grid Search
print("\n--- Grid Search Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred_grid))
print("Precision:", precision_score(y_test, y_pred_grid, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_grid, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_grid, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_grid))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_grid))

# ROC Curve for Grid Search
y_prob_grid = grid_search.predict_proba(X_test)[:, 1]
fpr_grid, tpr_grid, _ = roc_curve(y_test, y_prob_grid)
roc_auc_grid = auc(fpr_grid, tpr_grid)

plt.figure(figsize=(8, 6))
plt.plot(fpr_grid, tpr_grid, color='blue', lw=2, label='Grid Search (AUC = %0.2f)' % roc_auc_grid)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Grid Search')
plt.legend(loc='lower right')
plt.show()

#-----------------Randomized Grid Search----------------------------

# Randomized Grid Search with pipeline
random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=10,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    random_state=42
)

# Fit the model
random_search.fit(X_train, y_train)

# Best parameters and accuracy
print("\n--- Randomized Grid Search Results ---")
print("Best Parameters:", random_search.best_params_)
print("Best Accuracy:", random_search.best_score_)

# Predict with best model from random search
y_pred_random = random_search.predict(X_test)

# Evaluation for Randomized Grid Search
print("\n--- Randomized Grid Search Evaluation ---")
print("Accuracy:", accuracy_score(y_test, y_pred_random))
print("Precision:", precision_score(y_test, y_pred_random, average='weighted'))
print("Recall:", recall_score(y_test, y_pred_random, average='weighted'))
print("F1 Score:", f1_score(y_test, y_pred_random, average='weighted'))
print("\nClassification Report:\n", classification_report(y_test, y_pred_random))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_random))

# ROC Curve for Randomized Grid Search
y_prob_random = random_search.predict_proba(X_test)[:, 1]
fpr_random, tpr_random, _ = roc_curve(y_test, y_prob_random)
roc_auc_random = auc(fpr_random, tpr_random)

plt.figure(figsize=(8, 6))
plt.plot(fpr_random, tpr_random, color='green', lw=2, label='Randomized Search (AUC = %0.2f)' % roc_auc_random)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Randomized Grid Search')
plt.legend(loc='lower right')
plt.show()


import pickle

# Save the best model from Grid Search
with open('logistic_model.pkl', 'wb') as f:
    pickle.dump(grid_search.best_estimator_, f)

# -------- Logistic Regression - End --------
