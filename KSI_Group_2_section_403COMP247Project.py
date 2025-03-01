# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 02:59:10 2025

@author: Jonas
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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






