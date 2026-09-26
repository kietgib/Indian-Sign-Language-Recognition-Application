"""
Configuration file for Sign Language Action Detection
"""
import os
from pathlib import Path

# ==================== PATHS ====================
DATA_DIR = Path("/mnt/ngan/vsl_data")          # Raw videos (read-only)
RECOGNITION_DIR = Path("/mnt/ngan/recognition")  # All outputs go here

SEQUENCE_PATH  = RECOGNITION_DIR / "sequences"   # Extracted .npy sequences

# Create sequences directory
SEQUENCE_PATH.mkdir(parents=True, exist_ok=True)

# ==================== MEDIAPIPE SETTINGS ====================
MP_MIN_DETECTION_CONFIDENCE = 0.5
MP_MIN_TRACKING_CONFIDENCE = 0.5

# ==================== DATA SETTINGS ====================
SEQUENCE_LENGTH = None  # No fixed length - use all frames, pad at load time

# ==================== KEYPOINT DIMENSIONS ====================
POSE_LANDMARKS = 33 * 4   # 33 landmarks × (x, y, z, visibility) = 132
FACE_LANDMARKS = 468 * 3  # 468 landmarks × (x, y, z)             = 1404
HAND_LANDMARKS = 21 * 3   # 21 landmarks  × (x, y, z) per hand    = 63

TOTAL_KEYPOINTS = POSE_LANDMARKS + FACE_LANDMARKS + (HAND_LANDMARKS * 2)  # 1662

# ==================== COLORS FOR VISUALIZATION ====================
COLORS = {
    'face':       ((0, 255, 0),   (0, 200, 0)),
    'pose':       ((0, 0, 255),   (0, 0, 200)),
    'left_hand':  ((255, 0, 0),   (200, 0, 0)),
    'right_hand': ((255, 0, 255), (200, 0, 200))
}

# ==================== MODEL SELECTION ====================
# 'mlp'         → MLP branches + Cross-Part Gating + BiLSTM + Attention (hybrid.py)
# 'transformer' → Transformer Encoder branches + same downstream (transformer/model.py)
# Override via environment variable: MODEL_TYPE=transformer python main.py train
MODEL_TYPE = os.environ.get('MODEL_TYPE', 'mlp')

# Separate checkpoint/log dirs per model type to avoid overwriting
CHECKPOINT_DIR = RECOGNITION_DIR / 'checkpoints' / MODEL_TYPE
LOGS_DIR       = RECOGNITION_DIR / 'logs'        / MODEL_TYPE
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ==================== TRAINING CONFIG ====================
TRAINING_CONFIG = {
    'batch_size': 32,
    'epochs': 500,
    'learning_rate': 0.001,
    'early_stopping_patience': 40,
    'reduce_lr_patience': 15,
    'train_split': 0.7,
    'val_split': 0.15,
    'test_split': 0.15
}

