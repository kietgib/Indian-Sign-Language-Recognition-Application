"""
MLP Branches for Feature Extraction
Specialized Dense (MLP) branches for different body parts (pose, face, hands)
Uses fully connected layers to extract features from keypoints
"""
import tensorflow as tf
from tensorflow.keras import layers, Model


def create_hand_branch(input_dim=126, name_prefix='hand'):
    """
    DEEP MLP branch for hands (most important for sign language)
    126 dims → 64 features (3 Dense layers)
    
    Args:
        input_dim: Input dimension (21 landmarks × 2 hands × 3 coords = 126)
        name_prefix: Prefix for layer names
    
    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(input_dim,), name=f'{name_prefix}_input')
    
    x = layers.Dense(256, activation='relu', name=f'{name_prefix}_dense1')(inputs)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128, activation='relu', name=f'{name_prefix}_dense2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64, activation='relu', name=f'{name_prefix}_dense3')(x)
    
    return Model(inputs=inputs, outputs=x, name=f'{name_prefix}_branch')


def create_face_branch(input_dim=1404, name_prefix='face'):
    """
    DEEP MLP branch for face (important for expressions)
    1404 dims → 128 features (4 Dense layers - face has many points)
    
    Args:
        input_dim: Input dimension (468 landmarks × 3 coords = 1404)
        name_prefix: Prefix for layer names
    
    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(input_dim,), name=f'{name_prefix}_input')
    
    x = layers.Dense(512, activation='relu', name=f'{name_prefix}_dense1')(inputs)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(256, activation='relu', name=f'{name_prefix}_dense2')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn2')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128, activation='relu', name=f'{name_prefix}_dense3')(x)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn3')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128, activation='relu', name=f'{name_prefix}_dense4')(x)
    
    return Model(inputs=inputs, outputs=x, name=f'{name_prefix}_branch')


def create_pose_branch(input_dim=132, name_prefix='pose'):
    """
    SHALLOW MLP branch for pose (less important, fewer points)
    132 dims → 64 features (2 Dense layers)
    
    Args:
        input_dim: Input dimension (33 landmarks × 4 coords = 132)
        name_prefix: Prefix for layer names
    
    Returns:
        Keras Model
    """
    inputs = layers.Input(shape=(input_dim,), name=f'{name_prefix}_input')
    
    x = layers.Dense(128, activation='relu', name=f'{name_prefix}_dense1')(inputs)
    x = layers.BatchNormalization(name=f'{name_prefix}_bn1')(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(64, activation='relu', name=f'{name_prefix}_dense2')(x)
    
    return Model(inputs=inputs, outputs=x, name=f'{name_prefix}_branch')
