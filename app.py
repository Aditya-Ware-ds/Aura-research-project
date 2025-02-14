from flask import Flask, render_template, request, jsonify
import pandas as pd
from catboost import CatBoostClassifier

app = Flask(__name__)

# Load the pre-trained CatBoost model
clf = CatBoostClassifier()
clf.load_model(r"C:\Users\adity\OneDrive\Desktop\RBL\Catboost\catboost_model.cbm")

# Preprocessing function (if any)
def preprocess_data(data):
    # Example preprocessing (map gender, for instance)
    if 'Gender' in data:
        data['Gender'] = 0 if data['Gender'].lower() == 'male' else 1
    
    # Convert to DataFrame
    df = pd.DataFrame([data])
    
    return df

@app.route('/')
def index():
    return render_template('alpha.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    age = request.form.get('age')
    gender = request.form.get('gender')
    weight = request.form.get('weight')
    height = request.form.get('height')
    bmi = request.form.get('bmi')
    diseases = request.form.get('diseases')
    emotional_state = request.form.get('emotional_state')
    frequency = request.form.get('frequency')
    amplitude = request.form.get('amplitude')
    biofield_energy = request.form.get('biofield_energy')

    # Construct data for prediction
    input_data = {
        'Age': age,
        'Gender': gender,
        'Weight (kg)': weight,
        'Height (cm)': height,
        'BMI': bmi,
        'Current Diseases': diseases,
        'Emotional State': emotional_state,
        'Frequency (Hz)': frequency,
        'Amplitude (µV)': amplitude,
        'Biofield Energy (J)': biofield_energy
    }

    # Preprocess data
    processed_data = preprocess_data(input_data)

    # Predict disease
    predictions = clf.predict(processed_data)
    predicted_probabilities = clf.predict_proba(processed_data)

    # Return results as a JSON response (you can also render it directly)
    result = {
        'prediction': str(predictions[0]),
        'probabilities': predicted_probabilities[0].tolist()
    }
    
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
