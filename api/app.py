from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import joblib
import json
import os

app = Flask(__name__)
CORS(app)

# Load model and scaler on startup
model = tf.keras.models.load_model('lstm_flight_delay.keras')
scaler = joblib.load('scaler_flight_delay.pkl')

@app.route('/')
def home():
    return jsonify({'status': 'FlightSense API is running'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Feature order: DAY_OF_WEEK, DEP_HOUR, ARR_HOUR, DISTANCE, CARRIER_ENC, ORIGIN_ENC, DEST_ENC
    features = np.array([[
        data['day_of_week'],
        data['dep_hour'],
        data['arr_hour'],
        data['distance'],
        data['carrier'],
        data['origin'],
        data['dest']
    ]])
    
    # Scale and reshape for LSTM (1, 7, 1)
    features_scaled = scaler.transform(features)
    features_seq = features_scaled.reshape(1, 7, 1)
    
    # Predict
    prob = float(model.predict(features_seq, verbose=0)[0][0])
    
    return jsonify({
        'probability': round(prob, 4),
        'prediction': 'delayed' if prob > 0.5 else 'on_time'
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)