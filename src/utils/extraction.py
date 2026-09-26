"""
Utility functions for keypoint extraction using MediaPipe
"""
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import holistic as mp_holistic_module
from mediapipe.python.solutions import drawing_utils as mp_drawing
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config import MP_MIN_DETECTION_CONFIDENCE, MP_MIN_TRACKING_CONFIDENCE

# MediaPipe Holistic module
mp_holistic = mp_holistic_module


def mediapipe_detection(image, model):
    """
    Perform MediaPipe detection on image
    
    Args:
        image: BGR image from OpenCV
        model: MediaPipe Holistic model instance
        
    Returns:
        image: RGB image
        results: MediaPipe detection results
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                   # Image is no longer writeable
    results = model.process(image)                  # Make prediction
    image.flags.writeable = True                    # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR CONVERSION RGB 2 BGR
    return image, results


def normalize_keypoints(pose, face, lh, rh):
    """
    Normalize keypoints relative to shoulder midpoint and shoulder width.
    
    This makes the model invariant to:
    - Where the person stands in the frame (translation)
    - How far the person is from the camera (scale)
    
    Reference point: midpoint between left shoulder (idx 11) and right shoulder (idx 12)
    Scale: Euclidean distance between the two shoulders
    
    Pose format:  [x0,y0,z0,vis0, x1,y1,z1,vis1, ...] → stride 4
    Face/Hand format: [x0,y0,z0, x1,y1,z1, ...]       → stride 3
    
    Args:
        pose: np.array (132,)  - Pose keypoints
        face: np.array (1404,) - Face keypoints
        lh:   np.array (63,)   - Left hand keypoints
        rh:   np.array (63,)   - Right hand keypoints
    
    Returns:
        Tuple of normalized (pose, face, lh, rh)
    """
    # --- Extract shoulder coordinates from pose ---
    # Left shoulder  = landmark 11 → pose[44:48] = [x, y, z, vis]
    # Right shoulder = landmark 12 → pose[48:52] = [x, y, z, vis]
    left_shoulder_x,  left_shoulder_y  = pose[44], pose[45]
    right_shoulder_x, right_shoulder_y = pose[48], pose[49]

    # If both shoulders are zero (pose not detected), skip normalization
    if left_shoulder_x == 0.0 and right_shoulder_x == 0.0:
        return pose, face, lh, rh

    # --- Compute center (translation reference) ---
    center_x = (left_shoulder_x + right_shoulder_x) / 2.0
    center_y = (left_shoulder_y + right_shoulder_y) / 2.0

    # --- Compute scale (shoulder width, to normalize distance from camera) ---
    shoulder_width = np.sqrt(
        (right_shoulder_x - left_shoulder_x) ** 2 +
        (right_shoulder_y - left_shoulder_y) ** 2
    )
    # Avoid division by zero
    scale = shoulder_width if shoulder_width > 1e-6 else 1.0

    # --- Helper: subtract center and divide by scale for x, y only ---
    def _normalize_xy(arr, stride):
        """Normalize x (offset 0) and y (offset 1) within each stride-sized block."""
        arr = arr.copy()
        arr[0::stride] = (arr[0::stride] - center_x) / scale   # x coords
        arr[1::stride] = (arr[1::stride] - center_y) / scale   # y coords
        return arr

    pose_norm = _normalize_xy(pose, stride=4)   # (x,y,z,vis) per landmark
    face_norm = _normalize_xy(face, stride=3)   # (x,y,z) per landmark
    lh_norm   = _normalize_xy(lh,   stride=3)
    rh_norm   = _normalize_xy(rh,   stride=3)

    return pose_norm, face_norm, lh_norm, rh_norm


def extract_keypoints(results):
    """
    Extract keypoint values from MediaPipe results.
    Keypoints are normalized relative to shoulder midpoint and shoulder width
    to be invariant to position and distance from camera.
    
    Args:
        results: MediaPipe Holistic results
        
    Returns:
        Concatenated numpy array of all keypoints (1662 values), normalized
    """
    # Pose: 33 landmarks × 4 values (x, y, z, visibility) = 132
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33*4)
    
    # Face: 468 landmarks × 3 values (x, y, z) = 1404
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468*3)
    
    # Left hand: 21 landmarks × 3 values = 63
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21*3)
    
    # Right hand: 21 landmarks × 3 values = 63
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21*3)

    # Normalize relative to shoulder midpoint and shoulder width
    pose, face, lh, rh = normalize_keypoints(pose, face, lh, rh)

    return np.concatenate([pose, face, lh, rh])


def get_holistic_model(min_detection_confidence=None, min_tracking_confidence=None):
    """
    Create and return MediaPipe Holistic model
    
    Args:
        min_detection_confidence: Minimum detection confidence (default from config)
        min_tracking_confidence: Minimum tracking confidence (default from config)
        
    Returns:
        MediaPipe Holistic model instance
    """
    if min_detection_confidence is None:
        min_detection_confidence = MP_MIN_DETECTION_CONFIDENCE
    if min_tracking_confidence is None:
        min_tracking_confidence = MP_MIN_TRACKING_CONFIDENCE
    
    return mp_holistic.Holistic(
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence
    )
