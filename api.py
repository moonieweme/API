from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import keras
import json
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load your model (make sure to load the correct model file)
model = keras.models.load_model("trainedStudentPerformanceModel.keras")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data sent from the frontend
        data = request.get_json()
        print(f"Received data: {data}")  # Log received data to the terminal for debugging

        # Extract features from the data
        features_array = np.array([[
            data.get('hours_studied'),
            data.get('attendance'),
            data.get('parental_involvement'),
            data.get('access_to_resources'),
            data.get('extracurricular_activities'),
            data.get('sleep_hours'),
            data.get('previous_scores'),
            data.get('motivation_level'),
            data.get('internet_access'),
            data.get('tutoring_sessions'),
            data.get('family_income'),
            data.get('teacher_quality'),
            data.get('school_type'),
            data.get('peer_influence'),
            data.get('physical_activity'),
            data.get('learning_disabilities'),
            data.get('parental_education_level'),
            data.get('distance_from_home'),
            data.get('gender')
        ]], dtype=np.float32)

        # Model prediction
        prediction = model.predict(features_array)
        print(f"Prediction: {prediction}")

        # Return prediction
        return jsonify({"prediction": int(prediction[0][0])})
    except Exception as e:
        print(f"Error: {e}")  # Print the error to the logs for debugging
        return jsonify({"error": f"An error occurred while predicting: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Default to 5000 if PORT is not set
    app.run(host="0.0.0.0", port=port)
