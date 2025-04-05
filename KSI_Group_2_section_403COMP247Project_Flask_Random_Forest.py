# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 21:21:07 2025

@author: Jonas
"""

from flask import Flask, request
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import numpy as np
import os

app = Flask(__name__)
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

@app.route("/predict", methods=['GET','POST'])


def predict():
    
   
    
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Build relative path to files
    csv_path = os.path.join(base_dir, 'test_feature_no_target.csv')
    
    new_data = pd.read_csv(csv_path)
    

    # Columns used in transformations
    target_encode_cols = ['NEIGHBOURHOOD_158', 'STREET1', 'STREET2', 'ROAD_CLASS', 'INITDIR',
                          'DISTRICT', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND',
                          'INVTYPE', 'MANOEUVER', 'PEDTYPE', 'PEDACT', 'CYCLISTYPE', 'DIVISION',
                          'IMPACTYPE', 'DRIVACT', 'DRIVCOND', 'PEDCOND']

    minmax_cols = ['TIME', 'DATE', 'INVAGE']


    # Build relative path to files
    pipeline_path = os.path.join(base_dir, 'Group2_pipeline.joblib')
    pipeline = joblib.load(pipeline_path)

    processed_columns = (
        target_encode_cols +
        minmax_cols +
        [col for col in new_data.columns if col not in target_encode_cols + minmax_cols]
    )

    
    # Process JSON input
    json_ = request.json
    print("Input JSON:", json_)
    
    query_df = pd.DataFrame(json_)  # Convert single JSON object to DataFrame
    
    # Ensure all expected columns are present
    print("##############################")
    print("Processed Query DataFrame:")
    print(query_df)
    print("##############################")
    # Apply transform (not fit_transform) to the dataset
    transformed_new_data = pipeline.transform(query_df)

    transformed_new_data = pd.DataFrame(transformed_new_data, columns=processed_columns)

    print(transformed_new_data.head())

    # Build relative path to files
    model_path = os.path.join(base_dir, 'best_model_random_forest.joblib')
    # Load the best random forest model
    model = joblib.load(model_path)

    # Make predictions using the transformed data
    predictions = model.predict(transformed_new_data)

    prediction_result = f"Prediction: {'Fatal' if predictions[0] == 1 else 'Non-Fatal'}\n\n{json_}"
    
    return prediction_result
if __name__ == '__main__':
    app.run(debug=True,port=12345)
