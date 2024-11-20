from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import keras
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load your model (make sure to load the correct model file)
model = keras.models.load_model("trainedStudentPreformanceModel.keras")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data sent from the frontend
        data = request.get_json()
        print(f"Received data: {data}")  # Log received data to the terminal for debugging

        # Extract features from the data
        hours_studied = data.get('hours_studied')
        attendance = data.get('attendance')
        parental_involvement = data.get('parental_involvement')
        access_to_resources = data.get('access_to_resources')
        extracurricular_activities = data.get('extracurricular_activities')
        sleep_hours = data.get('sleep_hours')
        previous_scores = data.get('previous_scores')
        motivation_level = data.get('motivation_level')
        internet_access = data.get('internet_access')
        tutoring_sessions = data.get('tutoring_sessions')
        family_income = data.get('family_income')
        teacher_quality = data.get('teacher_quality')
        school_type = data.get('school_type')
        peer_influence = data.get('peer_influence')
        physical_activity = data.get('physical_activity')
        learning_disabilities = data.get('learning_disabilities')
        parental_education_level = data.get('parental_education_level')
        distance_from_home = data.get('distance_from_home')
        gender = data.get('gender')

        # Log the extracted data
        print(f"Extracted features: {hours_studied}, {attendance}, {parental_involvement}, {access_to_resources}, {extracurricular_activities}, {sleep_hours}, {previous_scores}, {motivation_level}, {internet_access}, {tutoring_sessions}, {family_income}, {teacher_quality}, {school_type}, {peer_influence}, {physical_activity}, {learning_disabilities}, {parental_education_level}, {distance_from_home}, {gender}")

        # Make sure all data is valid
        features_array = np.array([[
            hours_studied, attendance, parental_involvement, access_to_resources, 
            extracurricular_activities, sleep_hours, previous_scores, motivation_level, 
            internet_access, tutoring_sessions, family_income, teacher_quality, 
            school_type, peer_influence, physical_activity, learning_disabilities, 
            parental_education_level, distance_from_home, gender
        ]], dtype=np.float32)

        # Model prediction (ensure your model is loaded and ready for prediction)
        prediction = model.predict(features_array)
        print(f"Prediction: {prediction}")

        return jsonify({"prediction": int(prediction[0][0])})  # Return prediction
    except Exception as e:
        print(f"Error: {e}")  # Print the error to the logs for debugging
        return jsonify({"error": f"An error occurred while predicting: {e}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
