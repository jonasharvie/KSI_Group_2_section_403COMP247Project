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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, RocCurveDisplay
from sklearn.pipeline import Pipeline
import joblib
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime


#--------- Load and Split data- Jonas Harvie - Begin -------------
data_Group2_features_pretransform = pd.read_csv('KSI_features_cleaned_pretransform.csv')

X = data_Group2_features_pretransform.drop('ACCLASS', axis=1)
y = data_Group2_features_pretransform['ACCLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23, stratify=y
)


X_test_classifier, X_test_final, y_test_classifier, y_test_final = train_test_split(X_test, y_test, test_size=0.01, random_state=23, stratify=y_test
)

#X_test_classifier.to_csv('random_forest_results.csv', index=False)
X_test_final.to_csv('X_test_final.csv', index=False)
y_test_final.to_csv('y_test_final.csv', index=False)

#--------- Load and Split data- Jonas Harvie - End -------------

#--------- Transform data- Jonas Harvie - Begin -------------
# Define transformation functions
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

base_dir = os.path.dirname(os.path.abspath(__file__))

 # Build relative path to files
pipeline_path = os.path.join(base_dir, 'Group2_pipeline.joblib')
pipeline = joblib.load(pipeline_path)

# Columns used in transformations
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
print(X_train_transformed.head())

X_test_classifier_transformed = pipeline.transform(X_test_classifier)

X_test_classifier_transformed = pd.DataFrame(X_test_classifier_transformed, columns=processed_columns)


#--------- Transform data- Jonas Harvie - End -------------


#--------- Random Forest Classifier Grid Search - Jonas Harvie - Begin -------------


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

grid_random_forest.fit(X_train_transformed, y_train)

# Print out the best parameters 
print("\nbest parameters:\n")
print(grid_random_forest.best_params_)
# Printout the best estimator
print("\nbest estimator:\n")
print(grid_random_forest.best_estimator_)
# Create an object that holds the best model 
best_model_random_forest = grid_random_forest.best_estimator_
# Fit the training data to the best model. Printout the accuracy score
best_model_random_forest.fit(X_train_transformed, y_train)

y_pred_best_random_forest = best_model_random_forest.predict(X_test_classifier_transformed)

accuracy = accuracy_score(y_test_classifier, y_pred_best_random_forest)
print("Accuracy:", accuracy)
 
classification_report_result = classification_report(y_test_classifier, y_pred_best_random_forest)

print("\nClassification Report:\n", classification_report_result)
 
confusion_matrix_result = confusion_matrix(y_test_classifier, y_pred_best_random_forest)
print("\nConfusion Matrix:\n", confusion_matrix_result)


results_dict = {
    "Best Parameters": [grid_random_forest.best_params_],
    "Best Estimator": [str(grid_random_forest.best_estimator_)],
    "Accuracy": [accuracy],
    "Classification Report": [classification_report_result],
    "Confusion Matrix": [confusion_matrix_result.tolist()]
}

results_df = pd.DataFrame(results_dict)
results_df.to_csv('random_forest_results.csv', index=False)

# Plotting the ROC curve for the best model
plt.figure(figsize=(8, 6))
roc_display = RocCurveDisplay.from_estimator( 
    best_model_random_forest, 
    X_test_classifier_transformed, 
    y_test_classifier
)
plt.title('ROC Curve for Best Random Forest Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('roc_curve_best_random_forest.png')  # Save ROC curve as PNG
plt.show()

# Save the best model using joblib
joblib.dump(best_model_random_forest, 'best_model_random_forest.pkl')

#--------- Random Forest Classifier Grid Search - Jonas Harvie - End -------------