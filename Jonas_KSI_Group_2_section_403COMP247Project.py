# -*- coding: utf-8 -*-
"""
Created on Fri Mar 21 11:31:12 2025

@author: Jonas
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 02:59:10 2025

@author: Jonas
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier



#1. Data exploration: a complete review and analysis of the dataset including:

#Load and describe data elements (columns), provide descriptions & types, ranges and values of elements as appropriate. – use pandas, numpy and any other python packages.
data_Group2 = pd.read_csv('TOTAL_KSI_6386614326836635957.csv')
"""
print("\nPrint the shape of data_Group2:\n")
print(data_Group2.shape)

print("\nCheck the names and types of columns:\n")
print(data_Group2.dtypes)

#Statistical assessments including means, averages, correlations
print("\nCheck the statistics of the numeric fields:\n")
print(data_Group2.describe())

#Missing data evaluations – use pandas, numpy and any other python packages
print("\nCheck the missing values:\n")
print(data_Group2.isnull().sum())

print("\nNumber of unique values in each column:\n")
print(data_Group2.nunique())
"""



#Graphs and visualizations – use pandas, matplotlib, seaborn, numpy and any other python packages, you also can use power BI desktop.


"""
# Heatmap for missing data
plt.figure(figsize=(10, 6))
sns.heatmap(data_Group2.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Visualization')
plt.show()

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

# Barplot for Accidents by Neighbourhood 
plt.figure(figsize=(10, 25))  # Adjust figure size to accommodate vertical bars

# Count the accidents for each neighborhood and sort in descending order
neighborhood_counts = data_Group2['NEIGHBOURHOOD_158'].value_counts()

# Create the horizontal barplot with the sorted data
sns.barplot(y=neighborhood_counts.index, x=neighborhood_counts.values, order=neighborhood_counts.index)

plt.title('Accidents by Neighbourhood')
plt.ylabel('Neighbourhood')  
plt.xlabel('Number of Accidents')
plt.tight_layout()  # Ensure all labels are visible
plt.show()
"""



#2. Data modelling:

#Data transformations – includes handling missing data, categorical data management, data normalization and standardizations as needed.


# Clean up Target variable "ACCLASS"

# Remove rows where ACCLASS is empty
data_Group2 = data_Group2.dropna(subset=["ACCLASS"])
# Encode target variable as binary (1 = Fatal, 0 = Non-Fatal)
data_Group2["ACCLASS"] = np.where(data_Group2["ACCLASS"] == "Fatal", "Fatal", "Non-Fatal")


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

data_Group2['DATE'] = pd.to_datetime(data_Group2['DATE'])
data_Group2['DATE'] = data_Group2['DATE'].dt.strftime('%m/%d/%Y')
data_Group2['DATE'] = data_Group2['DATE'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y').timetuple().tm_yday)

# List of columns with binary classifications that are "Yes" or blank
binary_columns = ["PEDESTRIAN","CYCLIST","AUTOMOBILE","MOTORCYCLE","TRUCK","TRSN_CITY_VEH","EMERG_VEH","PASSENGER","SPEEDING","AG_DRIV","REDLIGHT","ALCOHOL","DISABILITY"]
# Change binary_columns from Yes/blank to Yes/No
for column in binary_columns:
    data_Group2[column] = np.where(data_Group2[column] == "Yes", "Yes", "No")


# change Other / Undefined in "PEDTYPE" to Unknown
# change Insufficient information (to determine cyclist crash type). in CYCLISTYPE to Unknown
categorical_columns = ["PEDTYPE","PEDACT","PEDCOND"]
# Check PEDESTRIAN column and fill categorical columns with "NA" when PEDESTRIAN = 0
for column in categorical_columns:
    data_Group2[column] = data_Group2.apply(lambda row: "NA" if row["PEDESTRIAN"] == 0 else row[column], axis=1)

categorical_columns = ["CYCLISTYPE","CYCACT","CYCCOND"]
# Check PEDESTRIAN column and fill categorical columns with "NA" when PEDESTRIAN = 0
for column in categorical_columns:
    data_Group2[column] = data_Group2.apply(lambda row: "NA" if row["CYCLIST"] == 0 else row[column], axis=1)
   

# Replace the specified text with "Unknown" in the CYCLISTYPE column
data_Group2['CYCLISTYPE'] = data_Group2['CYCLISTYPE'].replace("Insufficient information (to determine cyclist crash type).", "Unknown")

# Remove rows where TRAFFCTL is empty
data_Group2 = data_Group2.dropna(subset=["TRAFFCTL"])

# Remove rows where INVTYPE is empty
data_Group2 = data_Group2.dropna(subset=["INVTYPE"])

# Each Neighborhood will always have the same District, use other rows that have complete information in both columns to fill empty elements in the District column 
def fill_district(df):
    return df['DISTRICT'].fillna(df['DISTRICT'].iloc[0])

data_Group2['DISTRICT'] = data_Group2.groupby('NEIGHBOURHOOD_158').apply(fill_district).reset_index(level=0, drop=True)

data_Group2['ROAD_CLASS'] = data_Group2['ROAD_CLASS'].str.replace("Major Arterial ", "Major Arterial")

categorical_columns = ["PEDTYPE","CYCCOND","PEDCOND", "DRIVCOND","PEDACT", "CYCACT","DRIVACT", "MANOEUVER", "INITDIR","IMPACTYPE","RDSFCOND","LIGHT", "CYCLISTYPE","VISIBILITY", "ACCLOC", "ROAD_CLASS","STREET2"]
# Fill remaining blanks with "Unknown" for each column
for column in categorical_columns:
    data_Group2[column] = data_Group2[column].fillna("Unknown")

data_Group2 = data_Group2.replace("Other", "Unknown")
"""
# Heatmap for missing data
plt.figure(figsize=(10, 6))
sns.heatmap(data_Group2.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Data Visualization All Filled')
plt.show()
"""





# Managing imbalanced classes


# The target variable 'ACCLASS' is imbalanced (86% Non-Fatal to 14% Fatal)
acclass_counts = data_Group2['ACCLASS'].value_counts()

# Print the counts for each unique item
print("\nAmount of rows for Fatal vs Non-Fatal BEFORE down-sampling:\n")
print(acclass_counts)

# Separate majority and minority classes
df_majority = data_Group2[data_Group2['ACCLASS']=="Non-Fatal"]
df_minority = data_Group2[data_Group2['ACCLASS']=="Fatal"]
 
# Downsample majority class
df_majority_downsampled = resample(df_majority, replace=False, n_samples=data_Group2['ACCLASS'].value_counts()["Fatal"], random_state=123)
 
# Combine minority class with downsampled majority class
data_Group2 = pd.concat([df_majority_downsampled, df_minority])
 
acclass_counts = data_Group2['ACCLASS'].value_counts()

# Print the counts for each unique item
print("\nAmount of rows for Fatal vs Non-Fatal After down-sampling:\n")
print(acclass_counts)




# Pipeline
#data_Group2 = pd.read_csv('TOTAL_KSI_6cleaned.csv')

def transform_binary_category(X):
    #return X.replace({'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, '': 0}).fillna(0) 
    # Create a mapping dictionary
    mapping = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, '': 0}
    
    # Use numpy.vectorize to apply the mapping to the array
    vectorized_map = np.vectorize(lambda x: mapping.get(x, x))
    
    # Apply the mapping and convert to float
    result = vectorized_map(X).astype(float)
    
    # Replace any remaining non-numeric values with 0
    result[np.isnan(result)] = 0
    
    return result

def transform_date(X):
    days = [datetime.strptime(date[0].split()[0], '%m/%d/%Y').timetuple().tm_yday for date in X]
    return np.array(days).reshape(-1, 1)
    
    


drop_columns = ["INDEX", "ACCNUM", "OBJECTID", "FATAL_NO", "x", "y", "HOOD_158", "HOOD_140", "NEIGHBOURHOOD_140", "INJURY","VEHTYPE","OFFSET"]
MinMax_columns = ['TIME']
onehot_columns = ["PEDTYPE","CYCCOND","PEDCOND", "DRIVCOND","PEDACT", "CYCACT","DRIVACT", "MANOEUVER", "INITDIR","IMPACTYPE","RDSFCOND","LIGHT", "CYCLISTYPE","VISIBILITY", "ACCLOC", "ROAD_CLASS","STREET1","STREET2","NEIGHBOURHOOD_158","DISTRICT","LATITUDE","LONGITUDE","TRAFFCTL","INVTYPE","INVAGE","DIVISION"]
binary_convert_columns = ["PEDESTRIAN","CYCLIST","AUTOMOBILE","MOTORCYCLE","TRUCK", "TRSN_CITY_VEH", "EMERG_VEH", "PASSENGER","SPEEDING","AG_DRIV", "REDLIGHT","ALCOHOL" ,"DISABILITY"]
date_column = ['DATE']



MinMax_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

onehot_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

binary_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='No')),
    ('binary', FunctionTransformer(transform_binary_category))
])

date_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
    ('date_transform', FunctionTransformer(transform_date)),
    ('scaler', MinMaxScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('normalize', MinMax_transformer, MinMax_columns),
        ('onehot', onehot_transformer, onehot_columns),
        ('binary_convert', binary_transformer, binary_convert_columns)#,
        #('date', date_pipeline, date_column)
    ],
    remainder='passthrough'
)

pipeline_Group2 = Pipeline(steps=[
    ('preprocessor', preprocessor)
])


data_Group2_preprocessed = preprocessor.fit_transform(data_Group2.drop(columns=['ACCLASS'], axis=1, errors='ignore'))

feature_names = (
    MinMax_columns + preprocessor.named_transformers_['onehot'].named_steps['onehot'].get_feature_names_out(onehot_columns).tolist()
    + binary_convert_columns
    + date_column
)

#print(len(feature_names))

#print(data_Group2_preprocessed.shape)


data_Group2_transformed = pd.DataFrame(data_Group2_preprocessed, columns=feature_names)



# train test split


# Split dataset into features (X) and target variable (y)
X = data_Group2_transformed
y = np.where(data_Group2["ACCLASS"] == "Fatal", 1, 0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y, random_state=42)

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
short_names = data_Group2.columns

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


print("Done")

