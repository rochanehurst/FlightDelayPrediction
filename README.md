# ✈️ FlightSense — Flight Delay Prediction with Deep Learning

> Predicting U.S. domestic flight delays using a Bidirectional LSTM trained on 4.2 million flights from the Bureau of Transportation Statistics.

🔗 **[Live Demo](https://rochanehurst.github.io/FlightDelayPrediction)**

---

## Overview

FlightSense is a deep learning project that predicts whether a flight will arrive 15 or more minutes late, using only schedule-based features available at booking time. The project compares three models — Logistic Regression, a Multilayer Perceptron (MLP), and a Bidirectional LSTM — and includes a fully interactive web demo that runs the trained LSTM model directly in the browser via TensorFlow.js.

This was developed as the final project for CS 478 — Deep Learning at California State University, under Dr. Muhammad Lutfor Rahman.

---

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.6099 | 0.3305 | 0.6452 | 0.4371 | 0.6552 |
| MLP | 0.6189 | 0.3414 | 0.6711 | 0.4526 | 0.6847 |
| **Bidirectional LSTM** | **0.6230** | **0.3453** | **0.6761** | **0.4571** | **0.6905** |

The Bidirectional LSTM outperforms both baselines across all primary metrics. Each model is strictly better than the one before it, demonstrating that treating flight features as a sequence captures meaningful dependencies that a feedforward approach misses.

> **Note on accuracy:** The dataset is ~80% not-delayed. A model that always predicts "not delayed" scores 80% accuracy without learning anything. We prioritize Recall and F1-score as primary metrics.

---

## Dataset

- **Source:** [U.S. Bureau of Transportation Statistics — Airline On-Time Performance](https://www.transtats.bts.gov/)
- **Period:** May – October 2025
- **Size:** ~4.2 million domestic flight records
- **Target:** Binary — arrival delay of 15 minutes or more (1 = delayed, 0 = on time)
- **Split:** 80% train / 10% validation / 10% test (stratified)

### Features Used

| Feature | Description |
|---|---|
| `DAY_OF_WEEK` | Day the flight operates (1=Mon, 7=Sun) |
| `DEP_HOUR` | Scheduled departure hour (0–23) |
| `ARR_HOUR` | Scheduled arrival hour (0–23) |
| `DISTANCE` | Flight distance in miles |
| `CARRIER_ENC` | Airline carrier (label encoded) |
| `ORIGIN_ENC` | Origin airport (label encoded) |
| `DEST_ENC` | Destination airport (label encoded) |

---

## Model Architecture

### Bidirectional LSTM (main model)

Each flight record is treated as a 7-step sequence — one timestep per feature — allowing the LSTM to learn dependencies across features such as the relationship between departure hour and carrier, or between origin and destination.

```
Input: (batch, 7 timesteps, 1 feature)
    ↓
Bidirectional LSTM (64 units) + BatchNormalization + Dropout(0.3)
    ↓
Bidirectional LSTM (32 units) + BatchNormalization + Dropout(0.3)
    ↓
Dense(64, relu) + Dropout(0.2)
    ↓
Dense(32, relu)
    ↓
Dense(1, sigmoid) → delay probability
```

### MLP

3 hidden layers (256 → 128 → 64) with BatchNormalization, ReLU activation, Dropout, and L2 regularization. Treats all features as a flat input vector.

### Training Details

- Optimizer: Adam (lr=1e-3)
- Loss: Binary cross-entropy
- Class imbalance: handled via `class_weight='balanced'`
- Early stopping: patience=5, monitoring val AUC
- Learning rate scheduler: ReduceLROnPlateau (factor=0.5, patience=3)
- Batch size: 512
- Max epochs: 50

---

## Web Demo

The live demo at [rochanehurst.github.io/FlightDelayPrediction](https://rochanehurst.github.io/FlightDelayPrediction) includes:

- **Predictor** — enter flight details and get a real-time delay probability from the trained LSTM model running in the browser via TensorFlow.js
- **EDA Charts** — exploratory data analysis visualizations including delay patterns by hour, day, distance, and carrier
- **Model Results** — full comparison table, confusion matrices, ROC curves, and Precision-Recall curves
- **About** — project details, team, dataset info, and architecture summary

The model is converted to TensorFlow.js format and loaded client-side — no backend required.

---

## Repository Structure

```
FlightDelayPrediction/
├── index.html                          # Web demo (single file, no framework)
├── images/                             # All EDA and results charts
│   ├── fig_class_distribution.png
│   ├── fig_delay_by_hour.png
│   ├── fig_delay_by_day.png
│   ├── fig_delay_by_distance.png
│   ├── fig_correlation_heatmap.png
│   ├── fig_training_curves.png
│   ├── fig_model_comparison.png
│   ├── fig_confusion_matrices.png
│   ├── fig_roc_curves.png
│   └── fig_pr_curves.png
├── tfjs_lstm_model/                    # TensorFlow.js model weights
│   ├── model.json
│   └── group1-shard1of1.bin
└── scaler_params.json                  # StandardScaler parameters for browser inference
```

### Notebooks (run in Google Colab)

| Notebook | Description |
|---|---|
| `CS478-Final_Project__Load___Clean_Step_.ipynb` | Data loading, merging, cleaning, and feature engineering |
| `CS478_Flight_Delay_LSTM.ipynb` | Full modeling pipeline — EDA, Logistic Regression, MLP, Bidirectional LSTM, evaluation |

---

## How to Run

### In Google Colab

1. Open `CS478_Flight_Delay_LSTM.ipynb` in Colab
2. Run all cells — the dataset downloads automatically from Google Drive
3. Training takes approximately 10–15 minutes on a standard Colab GPU

### Local inference (Python)

```python
import joblib
import numpy as np
import tensorflow as tf

# Load models
lstm_model = tf.keras.models.load_model("lstm_flight_delay.keras")
scaler     = joblib.load("scaler_flight_delay.pkl")

# Example flight: Friday, depart 17:00, arrive 19:00, 850 miles, Southwest, Southeast -> Southwest
# Features: [DAY_OF_WEEK, DEP_HOUR, ARR_HOUR, DISTANCE, CARRIER_ENC, ORIGIN_ENC, DEST_ENC]
flight = np.array([[5, 17, 19, 850, 5, 1, 3]])
flight_scaled = scaler.transform(flight)
flight_seq    = flight_scaled.reshape(1, 7, 1)

prob = lstm_model.predict(flight_seq)[0][0]
print(f"Delay probability: {prob:.1%}")
print(f"Prediction: {'Delayed' if prob > 0.5 else 'On time'}")
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Data processing | Python, Pandas, NumPy |
| Machine learning | Scikit-learn |
| Deep learning | TensorFlow / Keras |
| Visualization | Matplotlib, Seaborn |
| Web demo | HTML, CSS, JavaScript, TensorFlow.js |
| Hosting | GitHub Pages |
| Development | Google Colab, VSCode |

---

## Team

| Name | GitHub |
|---|---|
| Rochane Hurst | [@rochanehurst](https://github.com/rochanehurst) |
| Elias Estacion | |
| Meliton Rojas | |
| Bricio Blancas Salgado | |
| Wendy Santiago | |
| Michael Vu | |

---

## References

1. U.S. Bureau of Transportation Statistics. (2025). Airline On-Time Performance Data. https://www.transtats.bts.gov/
2. Cao et al. (2024). Flight Delay Prediction Using Deep Learning Techniques. Transportation Research Part C.
3. Stanford University CS230. (2019). Flight Delay Prediction using LSTNet Architecture.
4. Yazdi et al. (2020). Flight Delay Prediction Using Machine Learning. Journal of Big Data.

---

## Future Work

- Incorporate NOAA weather station data matched by departure airport and time
- Add historical delay propagation features — delays cascade through a carrier's daily route network
- Explore attention mechanisms over the feature sequence
- Expand to a true time-series formulation where each flight's context includes prior flights on the same route

---

*CS 478 — Deep Learning | California State University San Marcos | Spring 2026*