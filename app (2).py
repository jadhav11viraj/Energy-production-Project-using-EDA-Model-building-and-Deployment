from flask import Flask, request, render_template, jsonify
import xgboost as xgb
import pandas as pd

app = Flask(__name__)

# Load the re-trained XGBoost model without 'temperature'
model = xgb.XGBRegressor()
model.load_model('xgboost_model_without_temperature.json')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the POST request
        data = request.get_json()
        
        # Define required features
        required_features = ['exhaust_vacuum', 'amb_pressure', 'r_humidity']
        
        # Check for missing features
        missing_features = [feature for feature in required_features if feature not in data]
        if missing_features:
            return jsonify({'error': f'Missing features: {", ".join(missing_features)}'})
        
        # Convert data to DataFrame with correct feature names
        features = pd.DataFrame([data], columns=required_features)

        # Make prediction
        prediction = model.predict(features)
        
        # Convert numpy float32 to native Python float
        predicted_value = float(prediction[0])
        
        return jsonify({'predicted_energy_production': predicted_value})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
 