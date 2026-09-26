"""
Hybrid MLP + Bidirectional LSTM Model with:
  - Adaptive Body-Part Gating: learns dynamic per-frame importance weights
    for each body part (pose/face/hand) instead of hardcoded fusion
  - Temporal Attention: learns which frames matter most for classification
    (extends the Masking layer idea to non-padding frames)
  - Multi-stream MLP branches with varying depth per body part
"""
import tensorflow as tf
from tensorflow.keras import layers, Model

# Handle both relative and absolute imports
try:
    from .components import create_hand_branch, create_face_branch, create_pose_branch
except ImportError:
    from components import create_hand_branch, create_face_branch, create_pose_branch


def create_hybrid_multistream_model(num_classes, sequence_length):
    """
    Hybrid MLP + BiLSTM with Adaptive Body-Part Gating + Temporal Attention.

    Architecture:
      1. MLP branches  : per-body-part feature extraction (pose/face/hand)
      2. Body-Part Gate: softmax gate that weights each part's contribution
                         dynamically per frame (learned, not hardcoded)
      3. Shared layers : cross-part interaction
      4. BiLSTM ×2     : temporal modeling (forward + backward)
      5. Temporal Attn : weighted sum over frames (learns which frames matter)
      6. Head          : Dense → softmax

    Args:
        num_classes:     Number of action classes
        sequence_length: Frames per sequence (after padding)

    Returns:
        Keras Model
    """
    # === INPUT ===
    inputs = layers.Input(shape=(sequence_length, 1662), name='sequence_input')
    x = layers.Masking(mask_value=0.0)(inputs)

    # === SPLIT KEYPOINTS ===
    # Pose:  indices   0:132  (33 landmarks × 4)
    # Face:  indices 132:1536 (468 landmarks × 3)
    # Hands: indices 1536:    (21×2 landmarks × 3)
    pose_keypoints = layers.Lambda(lambda t: t[:, :, :132],    name='pose_split')(x)
    face_keypoints = layers.Lambda(lambda t: t[:, :, 132:1536], name='face_split')(x)
    hand_keypoints = layers.Lambda(lambda t: t[:, :, 1536:],   name='hand_split')(x)

    # === MLP BRANCHES ===
    pose_branch = create_pose_branch(132,  'pose')   # shallow → 64 dims
    face_branch = create_face_branch(1404, 'face')   # deep    → 128 dims
    hand_branch = create_hand_branch(126,  'hand')   # deep    → 64 dims

    pose_features = layers.TimeDistributed(pose_branch, name='pose_features')(pose_keypoints)  # (B,T,64)
    face_features = layers.TimeDistributed(face_branch, name='face_features')(face_keypoints)  # (B,T,128)
    hand_features = layers.TimeDistributed(hand_branch, name='hand_features')(hand_keypoints)  # (B,T,64)

    # === CROSS-PART CONTEXTUAL GATING ===
    # Each part first "asks" the other two parts for context before computing
    # its gate weight. This allows cross-part interaction before gating.
    # pose_ctx = f(face, hand): pose learns what the other parts are doing
    pose_ctx = layers.TimeDistributed(
        layers.Dense(64, activation='relu', name='pose_ctx_dense'),
        name='pose_ctx'
    )(layers.Concatenate(name='pose_ctx_input')([face_features, hand_features]))   # (B,T,64)

    face_ctx = layers.TimeDistributed(
        layers.Dense(64, activation='relu', name='face_ctx_dense'),
        name='face_ctx'
    )(layers.Concatenate(name='face_ctx_input')([pose_features, hand_features]))   # (B,T,64)

    hand_ctx = layers.TimeDistributed(
        layers.Dense(64, activation='relu', name='hand_ctx_dense'),
        name='hand_ctx'
    )(layers.Concatenate(name='hand_ctx_input')([pose_features, face_features]))   # (B,T,64)

    # Enrich each part with cross-part context before computing gate weights
    pose_enriched = layers.Concatenate(name='pose_enriched')([pose_features, pose_ctx])  # (B,T,128)
    face_enriched = layers.Concatenate(name='face_enriched')([face_features, face_ctx])  # (B,T,192)
    hand_enriched = layers.Concatenate(name='hand_enriched')([hand_features, hand_ctx])  # (B,T,128)

    gate_input = layers.Concatenate(name='gate_input')([pose_enriched, face_enriched, hand_enriched])  # (B,T,448)
    gate = layers.TimeDistributed(
        layers.Dense(3, activation='softmax', name='gate_dense'),
        name='body_part_gate'
    )(gate_input)  # (B,T,3)

    # Extract per-part gate scalars and broadcast-multiply
    pose_scale = layers.Lambda(lambda g: g[:, :, 0:1], name='pose_gate')(gate)   # (B,T,1)
    face_scale = layers.Lambda(lambda g: g[:, :, 1:2], name='face_gate')(gate)   # (B,T,1)
    hand_scale = layers.Lambda(lambda g: g[:, :, 2:3], name='hand_gate')(gate)   # (B,T,1)

    pose_gated = layers.Multiply(name='pose_gated')([pose_features, pose_scale])  # (B,T,64)
    face_gated = layers.Multiply(name='face_gated')([face_features, face_scale])  # (B,T,128)
    hand_gated = layers.Multiply(name='hand_gated')([hand_features, hand_scale])  # (B,T,64)

    merged = layers.Concatenate(name='feature_fusion')([pose_gated, face_gated, hand_gated])  # (B,T,256)

    # === SHARED LAYERS (cross-part interaction) ===
    x = layers.TimeDistributed(
        layers.Dense(256, activation='relu', name='shared1'), name='shared_td1'
    )(merged)
    x = layers.Dropout(0.3)(x)

    x = layers.TimeDistributed(
        layers.Dense(128, activation='relu', name='shared2'), name='shared_td2'
    )(x)
    x = layers.Dropout(0.3)(x)

    # === BIDIRECTIONAL LSTM ===
    # bilstm1: 64×2 = 128 dims,  bilstm2: 32×2 = 64 dims
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True),  name='bilstm1')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True),  name='bilstm2')(x)  # keep sequences for attention
    x = layers.Dropout(0.3)(x)

    # === TEMPORAL ATTENTION ===
    # Extends Masking: not just ignoring zeros, but weighting real frames by importance.
    # attn_scores: tanh score per frame → softmax over time → weighted sum
    attn_scores  = layers.TimeDistributed(
        layers.Dense(1, activation='tanh', name='attn_score'), name='attn_td'
    )(x)                                                                          # (B,T,1)
    attn_weights = layers.Softmax(axis=1, name='temporal_attention')(attn_scores) # (B,T,1)
    context      = layers.Multiply(name='attn_apply')([x, attn_weights])          # (B,T,64)
    context      = layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1), name='context_vector'
    )(context)                                                                     # (B,64)

    # === CLASSIFICATION HEAD ===
    x = layers.Dense(128, activation='relu', name='dense1')(context)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs,
                  name='MLP_BiLSTM_CrossPartGating_Attention_Model')
    return model


if __name__ == "__main__":
    import numpy as np
    
    print("Creating MLP+LSTM Model...")
    print("\nArchitecture:")
    print("  MLP Branches (Feature Extraction):")
    print("    - Hand:  3 Dense layers (deep) → 64 features")
    print("    - Face:  4 Dense layers (deep) → 128 features")
    print("    - Pose:  2 Dense layers (shallow) → 64 features")
    print("  Shared Dense Layers:")
    print("    - 2 layers for cross-part interaction learning")
    print("  Temporal Modeling:")
    print("    - 2 Bidirectional LSTM layers (64×2=128 → 32×2=64)")
    print()
    
    model = create_hybrid_multistream_model(num_classes=76, sequence_length=33)
    model.summary()
    
    # Test
    print("\nTesting prediction...")
    test_input = np.random.rand(2, 33, 1662)
    test_output = model.predict(test_input, verbose=0)
    
    print(f"\nTest passed ")
    print(f"Input:  {test_input.shape}")
    print(f"Output: {test_output.shape}")
    print(f"Probabilities sum: {test_output[0].sum():.4f}")
