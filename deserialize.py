# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 02:01:23 2025

@author: Jonas
"""
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import numpy as np

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

# Load the pipeline from the pickle file
try:
    
    pipeline = joblib.load('Group2_pipeline.joblib')
    print(type(pipeline))

    print("Pipeline loaded successfully.")
except FileNotFoundError:
    print("Error: 'pipeline.pkl' not found. Please ensure the file exists in the working directory.")
    exit()

# Load the dataset to be used for fitting
try:
    data = pd.read_csv('KSI_features_cleaned_pretransform.csv')
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: 'data.csv' not found. Please ensure the file exists in the working directory.")
    exit()

# Fit the pipeline with the data
# Assuming 'data' has features and target columns, adjust as needed
try:
    X = data.drop(columns=['ACCLASS'])  
    y = data['ACCLASS'] 

    data = pipeline.fit_transform(X, y)
    print("Pipeline fitted successfully.")
    
    # Convert back to DataFrame 
    processed_columns = (
        target_encode_cols + 
        minmax_cols + 
        [col for col in X.columns if col not in target_encode_cols + minmax_cols]
    )
    data = pd.DataFrame(data, columns=processed_columns)

    # Print the head of the fitted data 
    print("Head of the input data:")
    print(data.head())
    
    print(data['INVAGE'].head())
except Exception as e:
    print(f"An error occurred during fitting: {e}")

