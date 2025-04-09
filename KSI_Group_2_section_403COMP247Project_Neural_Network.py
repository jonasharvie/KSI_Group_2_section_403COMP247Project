# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 15:53:02 2025

@author: shaku
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, RocCurveDisplay
from sklearn.pipeline import Pipeline
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

#---------------- LOADING AND SPLITTING DATA ----------------#
data = pd.read_csv('KSI_features_cleaned_pretransform.csv')

X = data.drop('ACCLASS', axis=1)
y = data['ACCLASS']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=23, stratify=y
)

X_test_classifier, X_test_final, y_test_classifier, y_test_final = train_test_split(
    X_test, y_test, test_size=0.01, random_state=23, stratify=y_test
)

#---------------- TRANSFORMING FUNCTIONS ----------------#
def age_transformer(df):
    df = df.copy()
    label_encoder = LabelEncoder()
    df['INVAGE'] = label_encoder.fit_transform(df['INVAGE'])
    return df

def date_transformer(df):
    df = df.copy()
    df['DATE'] = pd.to_datetime(df['DATE'])
    df['DATE'] = df['DATE'].dt.strftime('%m/%d/%Y')
    df['DATE'] = df['DATE'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y').timetuple().tm_yday)
    return df

def binary_transformer(df):
    df = df.copy()
    binary_cols = ["PEDESTRIAN", "SPEEDING"]
    for col in binary_cols:
        df[col] = np.where(df[col] == "Yes", 1, 0)
    return df

#---------------- LOAD PIPELINE ----------------#
base_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(base_dir, 'Group2_pipeline.joblib')
pipeline = joblib.load(pipeline_path)

#---------------- TRANSFORMING DATA ----------------#
target_encode_cols = ['NEIGHBOURHOOD_158', 'STREET1', 'STREET2', 'ROAD_CLASS', 'INITDIR',
                      'DISTRICT', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND',
                      'INVTYPE', 'MANOEUVER', 'PEDTYPE', 'PEDACT', 'CYCLISTYPE', 'DIVISION',
                      'IMPACTYPE', 'DRIVACT', 'DRIVCOND', 'PEDCOND']
minmax_cols = ['TIME', 'DATE', 'INVAGE']

processed_columns = (
    target_encode_cols +
    minmax_cols +
    [col for col in X_test_classifier.columns if col not in target_encode_cols + minmax_cols]
)

X_train_transformed = pipeline.transform(X_train)
X_train_transformed = pd.DataFrame(X_train_transformed, columns=processed_columns)

X_test_classifier_transformed = pipeline.transform(X_test_classifier)
X_test_classifier_transformed = pd.DataFrame(X_test_classifier_transformed, columns=processed_columns)

#---------------- GRID SEARCH Neural Network----------------#
nn_pipeline_grid = Pipeline(steps=[
    ('mlp_classifier', MLPClassifier(max_iter=1000, random_state=42))
])

param_grid = {
    'mlp_classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'mlp_classifier__activation': ['relu', 'tanh'],
    'mlp_classifier__solver': ['adam'],
    'mlp_classifier__alpha': [0.0001, 0.001],
}

grid_search_nn = GridSearchCV(
    estimator=nn_pipeline_grid,
    param_grid=param_grid,
    scoring='accuracy',
    refit=True,
    verbose=3,
    n_jobs=-1
)

grid_search_nn.fit(X_train_transformed, y_train)
best_nn_grid = grid_search_nn.best_estimator_
y_pred_grid = best_nn_grid.predict(X_test_classifier_transformed)

accuracy_grid = accuracy_score(y_test_classifier, y_pred_grid)
report_grid = classification_report(y_test_classifier, y_pred_grid)
confusion_grid = confusion_matrix(y_test_classifier, y_pred_grid)

print("\n===== GRID SEARCH RESULTS =====")
print("Best Parameters (Grid Search):")
print(grid_search_nn.best_params_)
print("\nAccuracy (Grid Search):", accuracy_grid)
print("\nClassification Report (Grid Search):")
print(report_grid)
print("\nConfusion Matrix (Grid Search):")
print(confusion_grid)

# Save Grid model
joblib.dump(best_nn_grid, 'best_model_nn_grid.pkl')

plt.figure(figsize=(8, 6))
RocCurveDisplay.from_estimator(best_nn_grid, X_test_classifier_transformed, y_test_classifier)
plt.title('ROC Curve - Grid Search (NN)')
plt.savefig('roc_curve_nn_grid.png')
plt.show()

#---------------- RANDOMIZED SEARCH Neural Network ----------------#
nn_pipeline_random = Pipeline(steps=[
    ('mlp_classifier', MLPClassifier(max_iter=1000, random_state=42))
])

param_dist = {
    'mlp_classifier__hidden_layer_sizes': [(50,), (100,), (100, 50), (50, 25)],
    'mlp_classifier__activation': ['relu', 'tanh', 'logistic'],
    'mlp_classifier__solver': ['adam', 'sgd'],
    'mlp_classifier__alpha': [0.0001, 0.001, 0.01],
    'mlp_classifier__learning_rate': ['constant', 'adaptive']
}

random_search_nn = RandomizedSearchCV(
    estimator=nn_pipeline_random,
    param_distributions=param_dist,
    n_iter=10,
    scoring='accuracy',
    refit=True,
    verbose=3,
    random_state=42,
    n_jobs=-1
)

random_search_nn.fit(X_train_transformed, y_train)
best_nn_random = random_search_nn.best_estimator_
y_pred_random = best_nn_random.predict(X_test_classifier_transformed)

accuracy_random = accuracy_score(y_test_classifier, y_pred_random)
report_random = classification_report(y_test_classifier, y_pred_random)
confusion_random = confusion_matrix(y_test_classifier, y_pred_random)

print("\n===== RANDOMIZED SEARCH RESULTS =====")
print("Best Parameters (Randomized Search):")
print(random_search_nn.best_params_)
print("\nAccuracy (Randomized Search):", accuracy_random)
print("\nClassification Report (Randomized Search):")
print(report_random)
print("\nConfusion Matrix (Randomized Search):")
print(confusion_random)

# Save Random model
joblib.dump(best_nn_random, 'best_model_nn_random.pkl')

# Save ROC - Random
plt.figure(figsize=(8, 6))
RocCurveDisplay.from_estimator(best_nn_random, X_test_classifier_transformed, y_test_classifier)
plt.title('ROC Curve - Randomized Search (NN)')
plt.savefig('roc_curve_nn_random.png')
plt.show()
