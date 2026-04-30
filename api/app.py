from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

print("Loading model...")
model = tf.keras.models.load_model('lstm_flight_delay.keras')
scaler = joblib.load('scaler_flight_delay.pkl')
print("Model loaded successfully!")

@app.route('/')
def home():
    return jsonify({'status': 'FlightSense API is running'})

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response

    data = request.get_json()

    features = np.array([[
        data['day_of_week'],
        data['dep_hour'],
        data['arr_hour'],
        data['distance'],
        data['carrier'],
        data['origin'],
        data['dest']
    ]])

    features_scaled = scaler.transform(features)
    features_seq    = features_scaled.reshape(1, 7, 1)
    prob = float(model.predict(features_seq, verbose=0)[0][0])

    response = jsonify({
        'probability': round(prob, 4),
        'prediction':  'delayed' if prob > 0.5 else 'on_time'
    })
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)