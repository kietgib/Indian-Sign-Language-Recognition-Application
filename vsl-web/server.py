from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import json
import os

app = Flask(__name__)
CORS(app)

# --- Configuration ---
CHECKPOINT_DIR = r"d:\signlanguagemodel\recognition-models\checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model")
MAPPING_PATH = os.path.join(CHECKPOINT_DIR, "action_mapping.json")

load_errors = []

# 1. Load Action Mappings
print("[INFO] Loading Action Mappings...")
try:
    with open(MAPPING_PATH, "r") as f:
        action_map = json.load(f)
except Exception as e:
    err_str = f"Mapping Load Error: {e}"
    print(f"[ERROR] {err_str}")
    load_errors.append(err_str)
    action_map = {}

# 2. Load Model
print("[INFO] Loading TensorFlow Model (this might take a moment)...")
model = None
try:
    # Using tf.saved_model.load because Keras 3 dropped direct SavedModel support
    _saved_model = tf.saved_model.load(MODEL_PATH)
    model = _saved_model.signatures["serving_default"]
    print("[INFO] Model loaded successfully via tf.saved_model.")
except Exception as e:
    err_str = f"Model Load Error: {e}"
    print(f"[ERROR] {err_str}")
    load_errors.append(err_str)

import cv2
import sys
import tempfile

# MediaPipe is no longer needed on the backend! The browser handles it natively.

@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "ok"})

@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify(action_map)

@app.route('/predict_npy', methods=['POST'])
def predict_npy():
    if not model:
        return jsonify({'error': f"Initialization failed: {' | '.join(load_errors)}"}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
        
    try:
        raw_bytes = file.read()
        
        # Determine if it's a real .npy file vs raw Float32 browser buffer
        if raw_bytes.startswith(b'\x93NUMPY'):
            import io
            sequence = np.load(io.BytesIO(raw_bytes), allow_pickle=True)
        else:
            # It's a raw float32 array sent from the JS web tracking
            sequence = np.frombuffer(raw_bytes, dtype=np.float32)
            sequence = sequence.reshape((-1, 1662))
            
        # Auto-detect the required sequence length (frames) from the model architecture!
        # This prevents crash when switching between 'mlp' (201 frames), 'seed_0' (93 frames), etc.
        try:
            expected_shape = model.inputs[0].shape
            TARGET_FRAMES = expected_shape[1] if expected_shape[1] is not None else 201
        except:
            TARGET_FRAMES = 201
            
        current_frames = sequence.shape[0]
        
        if current_frames < TARGET_FRAMES:
            # Pad with zeros at the end
            padding = np.zeros((TARGET_FRAMES - current_frames, 1662), dtype=np.float32)
            sequence = np.vstack([sequence, padding])
        elif current_frames > TARGET_FRAMES:
            # Truncate to max frames
            sequence = sequence[:TARGET_FRAMES]
        
        # Predict using raw signature
        input_tensor = tf.constant(np.expand_dims(sequence, axis=0), dtype=tf.float32)
        outputs = model(input_tensor)
        prediction = list(outputs.values())[0].numpy()[0]
        
        top_3_indices = np.argsort(prediction)[::-1][:3]
        
        results = []
        for idx in top_3_indices:
            results.append({
                'label': action_map.get(str(idx), "Unknown"),
                'confidence': float(prediction[idx]) * 100
            })
            
        return jsonify({
            'success': True,
            'predictions': results,
            'sequence': sequence.tolist()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Video prediction is now handled in the front-end directly via CDN!

if __name__ == '__main__':
    print("\n" + "="*50)
    print("VSL WEB BACKEND SERVER RUNNING ON PORT 5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
