# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 17:47:23 2025

@author: Jonas
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import RocCurveDisplay 
from sklearn.pipeline import Pipeline
import joblib

#--------- Random Forest Classifier Grid Search - Jonas Harvie - Begin -------------

########################### change to different file, import the pipeline pickle file

data_Group2_features_pretransform = pd.read_csv('KSI_data_cleaned.csv')

X = data_Group2_features_pretransform.drop('ACCLASS', axis=1)
y = data_Group2_features_pretransform['ACCLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23, stratify=y
)

random_forest_pipeline_Group2 = Pipeline(steps=[
    ('random_forest_classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

param_grid = {
    'random_forest_classifier__n_estimators': [100, 200, 500],  # Number of trees in the forest
    'random_forest_classifier__max_depth': [None, 10, 20, 30],  # Maximum depth of each tree
    'random_forest_classifier__max_features': ['sqrt', 'log2'],  # Number of features to consider for splits
    'random_forest_classifier__min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
    'random_forest_classifier__min_samples_leaf': [1, 2, 4],    # Minimum samples required at a leaf node
    'random_forest_classifier__bootstrap': [True, False]        # Whether bootstrap sampling is used
}
grid_random_forest = GridSearchCV(estimator=random_forest_pipeline_Group2, param_grid=param_grid,scoring='accuracy',refit = True,verbose = 3)

grid_random_forest.fit(X_train, y_train)

# Print out the best parameters 
print("\nbest parameters:\n")
print(grid_random_forest.best_params_)
# Printout the best estimator
print("\nbest estimator:\n")
print(grid_random_forest.best_estimator_)
# Create an object that holds the best model 
best_model_random_forest = grid_random_forest.best_estimator_
# Fit the training data to the best model. Printout the accuracy score
best_model_random_forest.fit(X_train, y_train)

y_pred_best_random_forest = best_model_random_forest.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_best_random_forest)
print("Accuracy:", accuracy)
 
classification_report_result = classification_report(y_test, y_pred_best_random_forest)

print("\nClassification Report:\n", y_pred_best_random_forest)
 
confusion_matrix_result = confusion_matrix(y_test, y_pred_best_random_forest)
print("\nConfusion Matrix:\n", confusion_matrix_result)

results_dict = {
    "Best Parameters": [grid_random_forest.best_params_],
    "Best Estimator": [str(grid_random_forest.best_estimator_)],
    "Accuracy": [accuracy],
    "Classification Report": [classification_report_result],
    "Confusion Matrix": [confusion_matrix_result.tolist()]  # Convert numpy array to list for CSV compatibility
}

results_df = pd.DataFrame(results_dict)
results_df.to_csv('random_forest_results.csv', index=False)

# Plotting the ROC curve for the best model
plt.figure(figsize=(8, 6))
#roc_display = RocCurveDisplay(best_model_random_forest, X_test, y_test)
roc_display = RocCurveDisplay.from_estimator(  # Changed to class method
    best_model_random_forest, 
    X_test, 
    y_test
)
plt.title('ROC Curve for Best Random Forest Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('roc_curve_best_random_forest.png')  # Save ROC curve as PNG
plt.show()

# Save the best model using joblib
joblib.dump(best_model_random_forest, 'best_model_random_forest.joblib')

#--------- Random Forest Classifier Grid Search - Jonas Harvie - End -------------