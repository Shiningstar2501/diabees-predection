from flask import Flask, send_from_directory, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('diabetes_model.pkl', 'rb'))

# Serve the homepage (index.html) from the public folder
@app.route('/')
def index():
    return send_from_directory('public', 'index.html')

# Serve the result page (result.html) from the public folder
@app.route('/result')
def result():
    return send_from_directory('public', 'result.html')

# Endpoint to handle predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from form
        input_features = [
            float(request.form['pregnancies']),
            float(request.form['glucose']),
            float(request.form['blood_pressure']),
            float(request.form['skin_thickness']),
            float(request.form['insulin']),
            float(request.form['bmi']),
            float(request.form['diabetes_pedigree']),
            float(request.form['age']),
            float(request.form['feature_9']),  # Replace with actual feature name
            float(request.form['feature_10'])  # Replace with actual feature name
        ]

        # Convert input data to numpy array and reshape for prediction
        input_array = np.array(input_features).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(input_array)

        # Interpret prediction result
        result = "You have diabetes." if prediction[0] == 1 else "You do not have diabetes."

        return send_from_directory('public', 'result.html')

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
