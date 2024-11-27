# verification/signature_verification.py

import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load models
bi_rnn_model = load_model(os.path.join(BASE_DIR, 'verification/static/models/bi_rnn_signature_verification_model.h5'))
rnn_model = load_model(os.path.join(BASE_DIR, 'verification/static/models/rnn_signature_verification_model.h5'))

def verify_signature(signature_file):
    # Preprocess the uploaded signature image
    signature = cv2.imdecode(np.frombuffer(signature_file.read(), np.uint8), cv2.IMREAD_GRAYSCALE)
    signature_resized = cv2.resize(signature, (128, 128)).astype('float32') / 255.0
    signature_reshaped = np.expand_dims(signature_resized, axis=(0, -1))  # Shape: (1, 128, 128, 1)

    # Prediction using the bi-RNN model
    bi_rnn_prediction = bi_rnn_model.predict(signature_reshaped)[0][0]

    # Prediction using the RNN model
    rnn_prediction = rnn_model.predict(signature_reshaped)[0][0]

    # Combine predictions and derive result
    average_confidence = (bi_rnn_prediction + rnn_prediction) / 2
    result = "Genuine Signature" if average_confidence > 0.5 else "Forged Signature"

    return {
        'result': result,
        'bi_rnn_confidence': round(bi_rnn_prediction * 100, 2),
        'rnn_confidence': round(rnn_prediction * 100, 2),
        'average_confidence': round(average_confidence * 100, 2),
    }
