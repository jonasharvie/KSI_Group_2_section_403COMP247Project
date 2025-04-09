# -*- coding: utf-8 -*-
"""
Created on Sat Apr 6 2025
@author: Jeongho
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, RocCurveDisplay
import joblib
from pipeline_utils import age_transformer, date_transformer, binary_transformer

# -------- Load and Split data - Jeongho - Begin --------
data = pd.read_csv('KSI_features_cleaned_pretransform.csv')

X = data.drop('ACCLASS', axis=1)
y = data['ACCLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=23)

# Create smaller test subset for final testing
X_test_classifier, X_test_final, y_test_classifier, y_test_final = train_test_split(
    X_test, y_test, test_size=0.01, stratify=y_test, random_state=23
)

X_test_final.to_csv('X_test_final.csv', index=False)
y_test_final.to_csv('y_test_final.csv', index=False)
# -------- Load and Split data - Jeongho - End --------

# -------- Load pipeline and transform - Jeongho - Begin --------
base_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_path = os.path.join(base_dir, 'Group2_pipeline.joblib')
pipeline = joblib.load(pipeline_path)

target_encode_cols = ['NEIGHBOURHOOD_158', 'STREET1', 'STREET2', 'ROAD_CLASS', 'INITDIR',
                      'DISTRICT', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND',
                      'INVTYPE', 'MANOEUVER', 'PEDTYPE', 'PEDACT', 'CYCLISTYPE', 'DIVISION',
                      'IMPACTYPE', 'DRIVACT', 'DRIVCOND', 'PEDCOND']

minmax_cols = ['TIME', 'DATE', 'INVAGE']

processed_columns = (
    target_encode_cols +
    minmax_cols +
    [col for col in X.columns if col not in target_encode_cols + minmax_cols]
)

X_train_transformed = pipeline.transform(X_train)
X_train_transformed = pd.DataFrame(X_train_transformed, columns=processed_columns)

X_test_classifier_transformed = pipeline.transform(X_test_classifier)
X_test_classifier_transformed = pd.DataFrame(X_test_classifier_transformed, columns=processed_columns)
# -------- Load pipeline and transform - Jeongho - End --------

# -------- SVM Classifier Grid Search - Jeongho - Begin --------

svm_pipeline = Pipeline(steps=[
    ('svm_classifier', SVC(probability=True, random_state=57))
])

param_grid = {
    'svm_classifier__C': [0.1, 1, 10],
    'svm_classifier__kernel': ['rbf', 'linear', 'poly'],
    'svm_classifier__gamma': ['scale', 'auto']
}

grid_svm = GridSearchCV(estimator=svm_pipeline, param_grid=param_grid, scoring='accuracy', refit=True, verbose=3)
grid_svm.fit(X_train_transformed, y_train)

# Print best results
print("\nBest Parameters:")
print(grid_svm.best_params_)

print("\nBest Estimator:")
print(grid_svm.best_estimator_)

best_model_svm = grid_svm.best_estimator_
best_model_svm.fit(X_train_transformed, y_train)

# Predict and evaluate
y_pred_best_svm = best_model_svm.predict(X_test_classifier_transformed)

accuracy = accuracy_score(y_test_classifier, y_pred_best_svm)
print("Accuracy:", accuracy)

classification_report_result = classification_report(y_test_classifier, y_pred_best_svm)
print("\nClassification Report:\n", classification_report_result)

confusion_matrix_result = confusion_matrix(y_test_classifier, y_pred_best_svm)
print("\nConfusion Matrix:\n", confusion_matrix_result)

# Save results to CSV
results_dict = {
    "Best Parameters": [grid_svm.best_params_],
    "Best Estimator": [str(grid_svm.best_estimator_)],
    "Accuracy": [accuracy],
    "Classification Report": [classification_report_result],
    "Confusion Matrix": [confusion_matrix_result.tolist()]
}

results_df = pd.DataFrame(results_dict)
results_df.to_csv('svm_results.csv', index=False)

# Plot ROC Curve
plt.figure(figsize=(8, 6))
roc_display = RocCurveDisplay.from_estimator(best_model_svm, X_test_classifier_transformed, y_test_classifier)
plt.title('ROC Curve for Best SVM Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('roc_curve_best_svm.png')
plt.show()

# Save model
joblib.dump(best_model_svm, 'best_model_svm.pkl')

# -------- SVM Classifier Grid Search - Jeongho - End --------