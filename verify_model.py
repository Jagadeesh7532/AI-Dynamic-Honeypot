import joblib

# Load model and scaler
model = joblib.load('models/rf_model.pkl')
scaler = joblib.load('models/scaler.pkl')

print('Model and scaler loaded successfully.')
