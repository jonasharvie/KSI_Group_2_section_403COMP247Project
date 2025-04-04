# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 02:59:10 2025

@author: Jonas
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, TargetEncoder, FunctionTransformer,LabelEncoder
from sklearn.utils import resample
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


np.random.seed(42)

#-------- Jonas - Begin --------
#1. Data exploration: a complete review and analysis of the dataset including:

#Load and describe data elements (columns), provide descriptions & types, ranges and values of elements as appropriate. – use pandas, numpy and any other python packages.
data_Group2 = pd.read_csv('TOTAL_KSI_6386614326836635957.csv')

print("\nPrint the shape of data_Group2\n")
print(data_Group2.shape)

print("\nCheck the names and types of columns.\n")
print(data_Group2.dtypes)

#Statistical assessments including means, averages, correlations
print("\nCheck the statistics of the numeric fields\n")
print(data_Group2.describe())

#Missing data evaluations – use pandas, numpy and any other python packages
print("\nCheck the missing values.\n")
print(data_Group2.isnull().sum())

print("\nColumns\n")
data_Group2_original_columns = data_Group2.columns
print(data_Group2_original_columns)

print("\nNumber of unique items\n")
data_Group2_original_columns = data_Group2.columns
print(data_Group2.nunique())

#-------- Jonas - End --------

#-------- Rubiya - Begin --------

# Barplot for road classification
plt.figure(figsize=(10, 6))
sns.countplot(data=data_Group2, x='ROAD_CLASS')
plt.title('Accidents by Road Classification')
plt.xlabel('Road Classification')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()

#Scatterplot for geographic distribution
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data_Group2, x='LONGITUDE', y='LATITUDE', hue='ACCLASS', palette={"Fatal": 'red', "Non-Fatal Injury": 'blue', "Property Damage O": 'blue'})
plt.title('Accidents Location')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

# Create a custom legend
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label='Fatal', markerfacecolor='red'),
                   plt.Line2D([0], [0], marker='o', color='w', label='Non-Fatal', markerfacecolor='blue')]

plt.legend(handles=legend_elements, title='ACCLASS')
plt.show()

#Visualization for Accidents by District
plt.figure(figsize=(10, 6))
sns.countplot(data=data_Group2, x='DISTRICT')
plt.title('Accidents by City District')
plt.xlabel('City District')
plt.ylabel('Number of Accidents')
plt.xticks(rotation=45)
plt.show()

# Heatmap for missing data
plt.figure(figsize=(10, 6))
sns.heatmap(data_Group2.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Visualization')
plt.show()

#-------- Rubiya - End --------

#2. Data modelling:

#Data transformations – includes handling missing data, categorical data management, data normalization and standardizations as needed.

#-------- Jeongho - Begin --------
# Clean up Target variable "ACCLASS"

# Remove rows where ACCLASS is empty 
data_Group2 = data_Group2.dropna(subset=["ACCLASS"])
# Encode target variable as binary (1 = Fatal, 0 = Non-Fatal)
data_Group2["ACCLASS"] = np.where(data_Group2["ACCLASS"] == "Fatal", 1, 0)

#-------- Jeongho - End --------


#-------- Jonas - Begin --------

# Managing imbalanced classes

# The target variable 'ACCLASS' is imbalanced (86% Non-Fatal to 14% Fatal)
acclass_counts = data_Group2['ACCLASS'].value_counts()

# Print the counts for each unique item
print("\nAmount of rows for Fatal vs Non-Fatal BEFORE down-sampling:\n")
print(acclass_counts)

# Separate majority and minority classes
data_Group2_majority = data_Group2[data_Group2['ACCLASS']==0]
data_Group2_minority = data_Group2[data_Group2['ACCLASS']==1]
 
# Downsample majority class
data_Group2_majority_downsampled = resample(data_Group2_majority, replace=False, n_samples=data_Group2['ACCLASS'].value_counts()[1], random_state=123)
 
# Combine minority class with downsampled majority class
data_Group2 = pd.concat([data_Group2_majority_downsampled, data_Group2_minority])
 
acclass_counts = data_Group2['ACCLASS'].value_counts()

# Print the counts for each unique item
print("\nAmount of rows for Fatal vs Non-Fatal After down-sampling:\n")
print(acclass_counts)
#-------- Jonas - End --------

#-------- Jose - Begin --------
# Data Transformation
# List of columns to drop, based on spreadsheet we decided to remove

# Round 1 of dropping columns

# List of columns to drop, because they are arbitrary unique identification generated after the accident
columns_to_drop = ["INDEX", "ACCNUM", "OBJECTID"]

# Drop the columns
data_Group2 = data_Group2.drop(columns=columns_to_drop, axis=1, errors='ignore')

# List of columns to drop, because the description is unclear
columns_to_drop = ["FATAL_NO", "x", "y"]

# Drop the columns
data_Group2 = data_Group2.drop(columns=columns_to_drop, axis=1, errors='ignore')

# List of columns to drop, because of clear duplication of data
columns_to_drop = ["HOOD_158", "HOOD_140", "NEIGHBOURHOOD_140", "INJURY","VEHTYPE"]

# Drop the columns
data_Group2 = data_Group2.drop(columns=columns_to_drop, axis=1, errors='ignore')

# List of columns to drop, because majority of column is empty
columns_to_drop = ["OFFSET"]

# Drop the columns
data_Group2 = data_Group2.drop(columns=columns_to_drop, axis=1, errors='ignore')


# Handle missing values for numerical columns (using mean/median)
numerical_columns = data_Group2.select_dtypes(include=["int64", "float64"]).columns
for col in numerical_columns:
    # Fill missing values with the median of each column
    data_Group2[col].fillna(data_Group2[col].median(), inplace=True)



# List of columns with binary classifications that are "Yes" or blank
binary_columns = ["PEDESTRIAN","CYCLIST","AUTOMOBILE","MOTORCYCLE","TRUCK","TRSN_CITY_VEH","EMERG_VEH","PASSENGER","SPEEDING","AG_DRIV","REDLIGHT","ALCOHOL","DISABILITY"]
# Change binary_columns from Yes/blank to Yes/No
for column in binary_columns:
    data_Group2[column] = np.where(data_Group2[column] == "Yes", "Yes", "No")
    
# Each Neighborhood will always have the same District, use other rows that have complete information in both columns to fill empty elements in the District column 
def fill_district(df):
    return df['DISTRICT'].fillna(df['DISTRICT'].iloc[0])

data_Group2['DISTRICT'] = data_Group2.groupby('NEIGHBOURHOOD_158').apply(fill_district).reset_index(level=0, drop=True)

# Correct typo
data_Group2['ROAD_CLASS'] = data_Group2['ROAD_CLASS'].str.replace("Major Arterial ", "Major Arterial")


# Handle missing values for categorical columns (replacing with "Unknown")
categorical_columns = data_Group2.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    # Fill missing values with "Unknown" to avoid noise
    data_Group2[col].fillna("Unknown", inplace=True)

data_Group2_features_pretransform = data_Group2.copy()

# Convert age ranges to their midpoint
def convert_age(value):
    if 'to' in value:  # Ex. "60 to 64"
        return (int(value.split(' to ')[0]) + int(value.split(' to ')[1])) / 2
    elif value == 'Over 95':  # For "Over 95"
        return 95
    else:
        return np.nan

# Apply the conversion function
data_Group2['INVAGE'] = data_Group2['INVAGE'].apply(convert_age)

# Calculate the average of the column, ignoring NaN values
average_value = data_Group2['INVAGE'].mean(skipna=True)

# Replace NaN values (from "unknown") with the average
data_Group2['INVAGE'] = data_Group2['INVAGE'].fillna(average_value)

# Apply Label Encoding
label_encoder = LabelEncoder()
data_Group2['INVAGE'] = label_encoder.fit_transform(data_Group2['INVAGE'])

print("nunique after filling")
print(data_Group2.nunique())

# Verify changes
print("Columns after dropping and handling missing data:")
print(data_Group2.columns)
print("\nMissing values per column:")
print(data_Group2.isnull().sum())



#Encoding Categorical Values
# List of fields for One-Hot Encoding (fields with relatively fewer categories)
#one_hot_fields = ['ROAD_CLASS', 'INITDIR','DISTRICT', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'INVTYPE', 'MANOEUVER', 'PEDTYPE', 'PEDACT', 'CYCLISTYPE', 'CYCACT', 'DIVISION','IMPACTYPE', 'DRIVACT', 'DRIVCOND', 'PEDCOND', 'CYCCOND']

# Apply One-Hot Encoding
#data_Group2 = pd.get_dummies(data_Group2, columns=one_hot_fields, drop_first=True)

#Encoding Categorical Values
# List of fields for Binary Encoding (fields with relatively fewer categories)
binary_columns = ["PEDESTRIAN","CYCLIST","AUTOMOBILE","MOTORCYCLE","TRUCK", "TRSN_CITY_VEH", "EMERG_VEH", "PASSENGER","SPEEDING","AG_DRIV", "REDLIGHT","ALCOHOL" ,"DISABILITY"]


# Change binary_columns from Yes/blank to 1/0
for column in binary_columns:
    data_Group2[column] = np.where(data_Group2[column] == "Yes", 1, 0)


target_encode_columns = ['NEIGHBOURHOOD_158','STREET1', 'STREET2','ROAD_CLASS', 'INITDIR','DISTRICT', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'INVTYPE', 'MANOEUVER', 'PEDTYPE', 'PEDACT', 'CYCLISTYPE', 'CYCACT', 'DIVISION','IMPACTYPE', 'DRIVACT', 'DRIVCOND', 'PEDCOND', 'CYCCOND']
enc_auto = TargetEncoder(smooth="auto")

data_Group2[target_encode_columns] = enc_auto.fit_transform(data_Group2[target_encode_columns], data_Group2['ACCLASS'])

# Verify the changes
print("\nSample of data after encoding:")
print(data_Group2.head())

# Convert DATE to a day of the year
data_Group2['DATE'] = pd.to_datetime(data_Group2['DATE'])
data_Group2['DATE'] = data_Group2['DATE'].dt.strftime('%m/%d/%Y')
data_Group2['DATE'] = data_Group2['DATE'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y').timetuple().tm_yday)

print("\nSample of data after encoding date time:")
print(data_Group2['DATE'].head())

MinMax_columns = ['TIME', 'DATE']

# Min-Max Normalization (scaling values to range [0, 1])
min_max_scaler = MinMaxScaler()
data_Group2[MinMax_columns] = min_max_scaler.fit_transform(data_Group2[MinMax_columns])
print("\nSample of data after minmax:")
print(data_Group2.head())

# Verify changes
print("\nSample of normalized and standardized TIME values:")
print(data_Group2[['TIME','DATE']].head())
#-------- Jose - End --------


# Tran Test Split

#-------- Jeongho - Begin --------


# Define features (X) and target variable (y)
X = data_Group2.drop(columns=["ACCLASS"])  # Features
y = data_Group2["ACCLASS"]  # Target

# Perform Train-Test Split (80% train, 20% test) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=23, stratify=y
)

# Print class distribution in train and test sets
print("\nTraining set class distribution:\n", y_train.value_counts(normalize=True))
print("\nTest set class distribution:\n", y_test.value_counts(normalize=True))

print("\nTrain-Test Split Completed Successfully!\n")

#-------- Jeongho - End --------

#-------- Jonas - Begin --------
# Random Forest

# Initialize and train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)


# Get feature importance


feature_importances = rf.feature_importances_

# Create a dictionary to store importances for all columns
all_column_importances = {col: [] for col in X.columns}

# Process one-hot encoded columns
for feature, importance in zip(X.columns, feature_importances):
    for col in X.columns:
        if str(feature).startswith(f"{col}_") or feature == col:
            all_column_importances[col].append(importance)
            break

# Calculate statistics for each column
stats = []
for col in X.columns:
    importances = all_column_importances[col]
    if importances:
        stats.append({
            'Column': col,
            'Importance': importances
        })
    

# Create summary DataFrame
summary_df = pd.DataFrame(stats)

summary_df = (summary_df
    .sort_values(by='Importance', ascending=False)
    .reset_index(drop=True))

summary_df['Importance'] = summary_df['Importance'].apply(lambda x: x[0] if isinstance(x, list) else x).round(3)
short_names = data_Group2_original_columns

# Create a dictionary mapping long names to short names
name_mapping = {long_name: short_name 
                for short_name in short_names 
                for long_name in summary_df['Column'] 
                if short_name in long_name}

# Replace the long names with short names
summary_df['Column'] = summary_df['Column'].replace(name_mapping)
summary_df['Column'] = summary_df['Column'].replace(name_mapping)

summary_df = summary_df.sort_values('Importance', ascending=False).drop_duplicates('Column', keep='first').sort_index()
print(summary_df)


# List of columns to drop, because they are of low importance
columns_to_drop = ["TRUCK", "AUTOMOBILE", "PASSENGER", "AG_DRIV", "CYCLIST", "CYCCOND", "CYCACT", "TRSN_CITY_VEH", "REDLIGHT", "MOTORCYCLE", "ALCOHOL", "DISABILITY", "EMERG_VEH"]

data_Group2_features_pretransform = data_Group2_features_pretransform.drop(columns=columns_to_drop, axis=1, errors='ignore')

data_Group2_features_pretransform.to_csv('KSI_features_cleaned_pretransform.csv', index=False)

print("Done")
#-------- Jonas - End --------


#Pipelines
#-------- Shakuntala - Begin --------

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
    binary_cols = ["PEDESTRIAN","SPEEDING"]
    for col in binary_cols:
        df[col] = np.where(df[col] == "Yes", 1, 0)
    return df

target_encode_cols = ['NEIGHBOURHOOD_158','STREET1', 'STREET2','ROAD_CLASS', 'INITDIR','DISTRICT', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND', 'INVTYPE', 'MANOEUVER', 'PEDTYPE', 'PEDACT', 'CYCLISTYPE', 'DIVISION','IMPACTYPE', 'DRIVACT', 'DRIVCOND', 'PEDCOND']

minmax_cols = ['TIME', 'DATE','INVAGE']

Group2_pipeline = Pipeline([
    ('preprocessing', Pipeline([
        ('age_conversion', FunctionTransformer(age_transformer)),
        ('date_conversion', FunctionTransformer(date_transformer)),
        ('binary_encoding', FunctionTransformer(binary_transformer))
    ])),
    
    ('feature_encoding', ColumnTransformer([
        ('target_encoder', TargetEncoder(smooth="auto"), target_encode_cols),
        ('minmax_scaler', MinMaxScaler(), minmax_cols)
    ], remainder='passthrough'))
])


# Set features and target
X = data_Group2_features_pretransform.drop('ACCLASS', axis=1)
y = data_Group2_features_pretransform['ACCLASS']


# Fit and transform
data_Group2_processed = Group2_pipeline.fit_transform(X, y)

# Convert back to DataFrame 
processed_columns = (
    target_encode_cols + 
    minmax_cols + 
    [col for col in X.columns if col not in target_encode_cols + minmax_cols]
)
data_Group2_processed = pd.DataFrame(data_Group2_processed, columns=processed_columns)

print(data_Group2_processed['INVAGE'].head())

joblib.dump(Group2_pipeline, 'Group2_pipeline.joblib')




print("\nList of columns to drop, because they are of low importance (<0.01)\n")
print(columns_to_drop)

# Drop the columns
data_Group2_processed = data_Group2_processed.drop(columns=columns_to_drop, axis=1, errors='ignore')
print(data_Group2_processed.shape)
print(y.shape)
data_Group2_processed = pd.concat([data_Group2_processed, y.reset_index(drop=True)], axis=1)
print(data_Group2_processed)

data_Group2_processed.to_csv('KSI_data_cleaned_and_transformed.csv', index=False)

data_Group2_processed = data_Group2_processed.drop(columns=columns_to_drop, axis=1, errors='ignore')



print("\ndone")

#------Rubiya-Logistic Regression Begins-----------------------------

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

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

# -------- Logistic Regression - End --------
