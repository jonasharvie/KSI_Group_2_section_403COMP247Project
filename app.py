# -*- coding: utf-8 -*-
"""
Flask API for SVM Fatal Collision Prediction
Created on Apr 6, 2025
@author: Jeongho
"""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import traceback
import os

# Initialize Flask app
app = Flask(__name__)

# Load models
base_dir = os.path.dirname(os.path.abspath(__file__))
svm_model = joblib.load(os.path.join(base_dir, 'best_model_svm.pkl'))
pipeline = joblib.load(os.path.join(base_dir, 'Group2_pipeline.joblib'))

# Define prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input
        input_data = request.get_json()

        # Convert to DataFrame
        input_df = pd.DataFrame([input_data])

        # Apply pipeline transformation
        processed_input = pipeline.transform(input_df)

        # Predict using the SVM model
        prediction = svm_model.predict(processed_input)

        # Return result
        return jsonify({
            'prediction': int(prediction[0])  # 0 = Non-fatal, 1 = Fatal
        })

    except Exception as e:
        return jsonify({
            'error': str(e),
            'trace': traceback.format_exc()
        })

# Run the app
if __name__ == '__main__':
    app.run(debug=True)