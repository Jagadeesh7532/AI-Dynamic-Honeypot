from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

# Load model and scaler
model_path = 'models/rf_model.pkl'
scaler_path = 'models/scaler.pkl'

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    print('Model or scaler files not found.')
    exit()

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

@app.route('/api/logs', methods=['GET'])
def get_logs():
    try:
        df_logs = pd.read_csv('attack_logs.csv')
        return df_logs.to_json(orient='records')
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        features = pd.DataFrame([data['features']])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return jsonify({'prediction': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
