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
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold


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

#Graphs and visualizations – use pandas, matplotlib, seaborn, numpy and any other python packages, you also can use power BI desktop.
# created a pairplot of all variables in the dataset
#data_plot_Group2 = sns.pairplot(data_Group2)
#plt.show()



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
sns.scatterplot(data=data_Group2, x='LONGITUDE', y='LATITUDE')
plt.title('Accidents Location')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
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



#2. Data modelling:

#Data transformations – includes handling missing data, categorical data management, data normalization and standardizations as needed.

# fisrt round of deciding which variables to drop
def missing_data_percentage(df, threshold=20):
    # amount of rows
    total_rows = len(df)
    
    # missing values per column
    missing_values = df.isnull().sum()
    
    # percentage of missing values
    missing_percentage = (missing_values / total_rows) * 100
    
    # create dataframe listing the missing values, perentage, and Drop? data
    missing_df = pd.DataFrame({
        'Column Name': missing_values.index,
        'Missing Values': missing_values.values,
        'Percentage': missing_percentage.round(2),
        'Drop?': ['Yes' if pct > threshold else 'No' for pct in missing_percentage]
    })
    
    # sort missing_df in descending order
    missing_df = missing_df.sort_values('Percentage', ascending=False).reset_index(drop=True)
    
    return missing_df

missing_data_results = missing_data_percentage(data_Group2)

print("\nMissing Data Results (Drop? = Yes when Percentage > 20):\n")
print(missing_data_results)

columns_to_drop = missing_data_results[missing_data_results['Drop?'] == 'Yes']['Column Name'].tolist()

#data_Group2 = data_Group2.drop(columns_to_drop, axis=1)



unique_counts = data_Group2.nunique()


columns_to_drop = unique_counts[unique_counts > 5000].index

data_Group2 = data_Group2.drop(columns=columns_to_drop)

print("\nUnique counts:\n")
print(data_Group2.nunique())

data_Group2['DATE'] = pd.to_datetime(data_Group2['DATE'])
data_Group2['time2'] = data_Group2['DATE'].dt.time
print("\nAll unique times:\n")
unique_times = data_Group2['time2'].unique()
for time in sorted(unique_times):
    print(time)



# Remove rows where ACCLASS is empty
data_Group2 = data_Group2.dropna(subset=["ACCLASS"])


#Jose's Data Transformation ----------------------Begin----------------------
# List of columns to drop, based on spreadsheet we decided to remove
columns_to_drop = [
    "INDEX_", "ACCNUM", "DATE", "OFFSET", "LATITUDE", "LONGITUDE", "INJURY", "FATAL_NO", 
    "VEHTYPE", "PEDTYPE", "PEDACT", "CYCLISTYPE", "CYCACT", "AUTOMOBILE", "EMERG_VEH", 
    "ALCOHOL", "DISABILITY", "NEIGHBOURHOOD_158", "HOOD_140", "NEIGHBOURHOOD_140", 
    "DIVISION", "ObjectID", "x", "y"
]

# Drop the columns
data_Group2 = data_Group2.drop(columns=columns_to_drop, axis=1, errors='ignore')

# Handle missing values for numerical columns (using mean/median)
numerical_columns = data_Group2.select_dtypes(include=["int64", "float64"]).columns
for col in numerical_columns:
    # Fill missing values with the median of each column
    data_Group2[col].fillna(data_Group2[col].median(), inplace=True)

# Handle missing values for categorical columns (replacing with "Unknown")
categorical_columns = data_Group2.select_dtypes(include=["object"]).columns
for col in categorical_columns:
    # Fill missing values with "Unknown" to avoid noise
    data_Group2[col].fillna("Unknown", inplace=True)

# Verify changes
print("Columns after dropping and handling missing data:")
print(data_Group2.columns)
print("\nMissing values per column:")
print(data_Group2.isnull().sum())

#Encoding Categorical Values
# List of fields for One-Hot Encoding (fields with relatively fewer categories)
one_hot_fields = ['ROAD_CLASS', 'TRAFFCTL', 'LIGHT', 'VISIBILITY', 'RDSFCOND']

# Apply One-Hot Encoding
data_Group2 = pd.get_dummies(data_Group2, columns=one_hot_fields, drop_first=True)

# List of fields for Label Encoding (fields where a numeric label suffices)
label_encode_fields = ['IMPACTYPE', 'DRIVACT', 'DRIVCOND', 'PEDCOND', 'CYCCOND', 'PEDESTRIAN', 
                       'CYCLIST', 'MOTORCYCLE', 'TRUCK', 'PASSENGER', 'SPEEDING', 
                       'AG_DRIV', 'REDLIGHT', 'HOOD_158']

# Apply Label Encoding
label_encoder = LabelEncoder()
for field in label_encode_fields:
    data_Group2[field] = label_encoder.fit_transform(data_Group2[field])

# Verify the changes
print("\nSample of data after encoding:")
print(data_Group2.head())

#Date and Time Handling
# Extract hour from the TIME column
data_Group2['Hour'] = data_Group2['TIME'] // 100  # Assuming TIME is in HHMM format (e.g., 2359)

# Create categorical time periods (Morning, Afternoon, Evening, Night)
def categorize_time(hour):
    if 6 <= hour < 12:
        return "Morning"
    elif 12 <= hour < 18:
        return "Afternoon"
    elif 18 <= hour < 24:
        return "Evening"
    else:
        return "Night"

# Apply the function to create a new column
data_Group2['Time_Period'] = data_Group2['Hour'].apply(categorize_time)

# Check unique categories to verify
print("\nUnique values in Time_Period:")
print(data_Group2['Time_Period'].unique())

# Apply One-Hot Encoding robustly to include all possible categories
time_period_dummies = pd.get_dummies(data_Group2['Time_Period'], prefix="Time_Period", drop_first=False)

# Concatenate the encoded columns back to the DataFrame
data_Group2 = pd.concat([data_Group2, time_period_dummies], axis=1)

# Drop the original 'Time_Period' column as it's now encoded
data_Group2.drop(columns=['Time_Period'], inplace=True)

# Verify changes
print("\nSample of data with extracted time features:")
print(data_Group2[['TIME', 'Hour', 'Time_Period_Morning', 'Time_Period_Afternoon', 'Time_Period_Evening', 'Time_Period_Night']].head())

#Normalizing or Standardizing Features
# List of numeric fields to normalize/standardize
numeric_fields = ['TIME']

# Min-Max Normalization (scaling values to range [0, 1])+
min_max_scaler = MinMaxScaler()
data_Group2['TIME_Normalized'] = min_max_scaler.fit_transform(data_Group2[['TIME']])

# Standardization (scaling values to have mean 0 and standard deviation 1)
standard_scaler = StandardScaler()
data_Group2['TIME_Standardized'] = standard_scaler.fit_transform(data_Group2[['TIME']])

# Verify changes
print("\nSample of normalized and standardized TIME values:")
print(data_Group2[['TIME', 'TIME_Normalized', 'TIME_Standardized']].head())
#Jose's Data Transformation ----------------------END----------------------

#Feature Selection
#Based on spreadsheet filled out by group members


# Convert target variable into binary classification
# Combining 'Property Damage O' with 'Non-Fatal Injury' into 'Non-Fatal'
data_Group2["ACCLASS"] = data_Group2["ACCLASS"].replace("Property Damage O", "Non-Fatal")

# Encode target variable as binary (1 = Fatal, 0 = Non-Fatal)
data_Group2["ACCLASS"] = np.where(data_Group2["ACCLASS"] == "Fatal", 1, 0)

# Define features (X) and target variable (y)
X = data_Group2.drop(columns=["ACCLASS"])  # Features
y = data_Group2["ACCLASS"]  # Target

# Perform Train-Test Split (80% train, 20% test) with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=23, stratify=y
)

# Print class distribution in train and test sets
print("Training set class distribution:\n", y_train.value_counts(normalize=True))
print("Test set class distribution:\n", y_test.value_counts(normalize=True))

print("Train-Test Split Completed Successfully!")

#######################
"""
Both the police department and the “general public” would make use of a software product that can give them an idea about the likelihood of fatal collisions that involve loss of life. For the police department it would assist them in taking better measures of security and better planning for road conditions around certain neighborhoods. For the public individuals, it would help them assess the need for additional precautions at certain times and weather conditions and neighborhoods

Police want:
    better measures of security
    better planning for road conditions around certain neighborhoods
Public Individuals want:
    assess the need for additional precautions at certain
        times
        weather conditions
        neighborhoods    

"""
#######################
"""

Although it could be assumed that the columns with high amounts of missing data are empty because 
that the answer is no to that condition meaning that condition was present in the accident, it impossible to know if the empty data is an answer of "No" or if it is actually missing data. 
the safest decision is to drop those features because there is a high chance of incorrect data, some of the ones with very high percent of missing data could also represent outliers or factors that are not representative of common details that affect crashes.
Any column with more than 20% missing data is to be dropped because there is not enough data to accurately represent the population of the data.

"""
#######################
"""
NOTE
Still need to manually review the remaining variables and decide which to drop or keep
AND provide justification
"""
#######################
"""
Keep ACCLASS
ACCLASS is the target variable
it contains the following options
Fatal
Non-Fatal Injury
Property Damage O (combine with "Non-Fatal Injury" to create "Non-Fatal" option)
empty (drop this row)
"""
#######################
"""
Other Columns to Drop:
    
 drop these are just ID numbers for tracking that were generated after the crash
    OBJECTID
    INDEX
    ACCNUM
    
######################
NOTE: check for columns with too many unique variables 
SHOULD we drop all columns with too many unique variables?
decide on cutoff point for too many unique variables 
######################

 there are too many unique values, also the road names are easier representations of location
    LATITUDE
    LONGITUDE

 drop because these variables are not listed or explained in the KSI_Glossary
    x
    y 

 drop because is old version of new variables (HOOD_158 and NEIGHBOURHOOD_158)
    HOOD_140
    NEIGHBOURHOOD_140

 drop because duplicate info of HOOD_158 (HOOD_158 is unique id for NEIGHBOURHOOD_158 which is name of neighborhood)
    NEIGHBOURHOOD_158

"""
#######################


#Feature selection – use pandas and sci-kit learn. (The group needs to justify each feature used and any data columns discarded)
#Train, Test data splitting – use numpy, sci-kit learn.
#Managing imbalanced classes if needed. Check here for info: https://elitedatascience.com/imbalanced-classes
#Use pipelines class to streamline all the pre-processing transformations

print("done")








