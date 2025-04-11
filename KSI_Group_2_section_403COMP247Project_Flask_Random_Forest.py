from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import numpy as np
import os

app = Flask(__name__)
CORS(app)

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
        if col not in df.columns:
            df[col] = "No"  
        df[col] = np.where(df[col] == "Yes", 1, 0)
    return df

@app.route("/predict", methods=['POST'])
def predict():
    print("=== Starting /predict endpoint ===")
    if not request.is_json:
        print("Error: Request content-type is not JSON")
        return jsonify({"error": "Invalid Content-Type. Expected application/json"}), 415

    json_ = request.get_json()
    print("Input JSON:\n", json_)

    try:
        query_df = pd.DataFrame(json_)
        print("Processed Query DataFrame:\n", query_df)
    except Exception as e:
        print(f"Error converting JSON to DataFrame: {e}")
        return jsonify({"error": str(e)}), 400

    # Load pipeline and model
    pipeline_path = r"C:\Users\josep\Downloads\Group2_pipeline.joblib"
    model_path = r"C:\Users\josep\Downloads\best_model_random_forest.joblib"

    try:
        print("Loading pipeline and model...")
        pipeline = joblib.load(pipeline_path)
        model = joblib.load(model_path)
        print("Pipeline and model loaded successfully.")
    except Exception as e:
        print(f"Error loading pipeline or model: {e}")
        return jsonify({"error": f"Failed to load model or pipeline: {str(e)}"}), 500

    # Define required columns for processing
    target_encode_cols = [
        'NEIGHBOURHOOD_158', 'STREET1', 'STREET2', 'ROAD_CLASS', 'INITDIR',
        'DISTRICT', 'ACCLOC', 'TRAFFCTL', 'VISIBILITY', 'LIGHT', 'RDSFCOND',
        'INVTYPE', 'MANOEUVER', 'PEDTYPE', 'PEDACT', 'CYCLISTYPE', 'DIVISION',
        'IMPACTYPE', 'DRIVACT', 'DRIVCOND', 'PEDCOND'
    ]
    minmax_cols = ['TIME', 'DATE', 'INVAGE']
    processed_columns = target_encode_cols + minmax_cols

    # Apply transformations
    try:
        print("Adding default values for missing columns...")
        # Add missing binary columns
        binary_cols = ["PEDESTRIAN", "SPEEDING"]
        for col in binary_cols:
            if col not in query_df.columns:
                query_df[col] = "Yes"

        # Add missing categorical columns with 'Unknown'
        categorical_cols = [
            'CYCLISTYPE', 'DRIVACT', 'PEDACT', 'INVTYPE', 'ACCLOC',
            'DISTRICT', 'PEDCOND', 'PEDTYPE', 'INITDIR', 'LIGHT', 'MANOEUVER'
        ]
        for col in categorical_cols:
            if col not in query_df.columns:
                query_df[col] = "Unknown"

        # Add missing geographical columns with default values
        if 'LATITUDE' not in query_df.columns:
            query_df['LATITUDE'] = 0.0
        if 'LONGITUDE' not in query_df.columns:
            query_df['LONGITUDE'] = 0.0

        print("Applying transformations...")
        query_df = age_transformer(query_df)
        print("Columns after age_transformer:\n", query_df.columns)

        query_df = date_transformer(query_df)
        print("Columns after date_transformer:\n", query_df.columns)

        query_df = binary_transformer(query_df)
        print("Columns after binary_transformer:\n", query_df.columns)

        # Map unseen categorical values to "Other"
        valid_values = ['No Control', 'Stop Sign', 'Traffic Signal'] 
        query_df['TRAFFCTL'] = query_df['TRAFFCTL'].apply(lambda x: x if x in valid_values else 'Other')

        print("Unique values in categorical columns:")
        for col in ['TRAFFCTL', 'VISIBILITY', 'IMPACTYPE']:
            print(f"Unique values in {col}:", query_df[col].unique())

        print("DataFrame dtypes before pipeline transformation:\n", query_df.dtypes)
    except Exception as e:
        print(f"Transformation failed: {e}")
        return jsonify({"error": f"Transformation failed: {str(e)}"}), 500

    # Pipeline transformation
    try:
        print("Passing data through the pipeline...")
        transformed_new_data = pipeline.transform(query_df)
        print("Transformed DataFrame:\n", transformed_new_data)
    except Exception as e:
        print(f"Pipeline transformation failed: {e}")
        return jsonify({"error": f"Pipeline transformation failed: {str(e)}"}), 500

    # Model prediction
    try:
        print("Making predictions...")
        predictions = model.predict(transformed_new_data)
        print("Predictions:\n", predictions)
        prediction_result = {
            "prediction": "Fatal" if predictions[0] == 1 else "Non-Fatal",
            "input": json_
        }
        print("Returning prediction result.")
        return jsonify(prediction_result)
    except Exception as e:
        print(f"Prediction failed: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500

@app.route('/get-options', methods=['GET'])
def get_options():
    print("=== Starting /get-options endpoint ===")
    # Path to the CSV file
    csv_path = r"C:\Users\josep\Downloads\KSI_features_cleaned_pretransform.csv"

    try:
        data = pd.read_csv(csv_path)
        print("CSV file loaded successfully.")

        # dropdown fields
        options = {
            "NEIGHBOURHOOD_158": data['NEIGHBOURHOOD_158'].dropna().unique().tolist(),
            "STREET1": data['STREET1'].dropna().unique().tolist(),
            "STREET2": data['STREET2'].dropna().unique().tolist(),
            "ROAD_CLASS": data['ROAD_CLASS'].dropna().unique().tolist(),
            "TRAFFCTL": data['TRAFFCTL'].dropna().unique().tolist(),
            "VISIBILITY": data['VISIBILITY'].dropna().unique().tolist(),
            "RDSFCOND": data['RDSFCOND'].dropna().unique().tolist(),
            "LIGHT": data['LIGHT'].dropna().unique().tolist(),
            "MANOEUVER": data['MANOEUVER'].dropna().unique().tolist(),
            "IMPACTYPE": data['IMPACTYPE'].dropna().unique().tolist(),
            "DRIVCOND": data['DRIVCOND'].dropna().unique().tolist(),
            "DIVISION": data['DIVISION'].dropna().unique().tolist()

        }
        print("Returning dropdown options.")
        return jsonify(options)
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return {"error": f"Failed to read CSV file: {str(e)}"}, 500

if __name__ == '__main__':
    app.run(debug=True, port=12345)