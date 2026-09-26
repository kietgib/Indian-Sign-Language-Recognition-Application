"""
Hybrid Transformer Encoder + BiLSTM model with:
  - Transformer Encoder branches : per-body-part self-attention feature extraction
                                   (replaces TimeDistributed MLP from hybrid.py)
  - Cross-Part Contextual Gating : each part enriched with context from other
                                   parts before computing gate weights
  - Shared layers                : cross-part interaction
  - BiLSTM ×2                    : sequential temporal modeling
  - Temporal Attention           : weighted frame aggregation
  - Head                         : Dense → softmax

Key difference vs hybrid.py:
  MLP (per-frame, independent) → Transformer Encoder (self-attention over T)
  Downstream pipeline (Gating, BiLSTM, Attention) stays identical.
"""
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, Model


# ──────────────────────────────────────────────────────────────────────────────
# BUILDING BLOCKS
# ──────────────────────────────────────────────────────────────────────────────

def _transformer_encoder_block(x, d_model, num_heads, dff, dropout_rate, name):
    """
    Single Transformer Encoder block (Post-LN):
      x → MHA(x,x) + Add & Norm → FFN + Add & Norm
    """
    # Multi-Head Self-Attention
    attn = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=d_model // num_heads,
        dropout=dropout_rate,
        name=f'{name}_mha'
    )(x, x)
    attn = layers.Dropout(dropout_rate)(attn)
    x = layers.LayerNormalization(epsilon=1e-6, name=f'{name}_ln1')(x + attn)

    # Position-wise Feed-Forward Network
    ffn = layers.Dense(dff, activation='relu', name=f'{name}_ffn1')(x)
    ffn = layers.Dropout(dropout_rate)(ffn)
    ffn = layers.Dense(d_model, name=f'{name}_ffn2')(ffn)
    ffn = layers.Dropout(dropout_rate)(ffn)
    x = layers.LayerNormalization(epsilon=1e-6, name=f'{name}_ln2')(x + ffn)

    return x


def _create_part_transformer(input_dim, d_model, num_heads, num_blocks, dff,
                              name, dropout_rate=0.1):
    """
    Transformer Encoder branch for one body part.
    Replaces TimeDistributed(MLP) in hybrid.py.

    Self-attention across the temporal axis captures within-part temporal
    dependencies (e.g. hand trajectory shape) before cross-part gating.

    Args:
        input_dim  : raw keypoint dim for this part (132 / 1404 / 126)
        d_model    : output embedding dim (64 / 128 / 64)
        num_heads  : attention heads
        num_blocks : number of stacked encoder blocks
        dff        : feed-forward hidden dim (typically 2×d_model)
        name       : prefix for all layer/model names
        dropout_rate: dropout applied inside MHA and FFN

    Returns:
        Keras Model: (B, T, input_dim) → (B, T, d_model)
    """
    inputs = layers.Input(shape=(None, input_dim), name=f'{name}_input')

    # Linear projection: raw keypoints → d_model embedding space
    x = layers.Dense(d_model, name=f'{name}_proj')(inputs)          # (B,T,d_model)
    x = layers.LayerNormalization(epsilon=1e-6, name=f'{name}_proj_ln')(x)

    # Stack Transformer Encoder blocks
    for i in range(num_blocks):
        x = _transformer_encoder_block(
            x, d_model, num_heads, dff, dropout_rate,
            name=f'{name}_enc{i + 1}'
        )

    return Model(inputs=inputs, outputs=x, name=f'{name}_transformer')


# ──────────────────────────────────────────────────────────────────────────────
# FULL MODEL
# ──────────────────────────────────────────────────────────────────────────────

def create_hybrid_transformer_model(num_classes, sequence_length):
    """
    Hybrid Transformer Encoder + BiLSTM with Cross-Part Contextual Gating
    + Temporal Attention.

    Architecture:
      1. Transformer branches  : self-attention over T per body part
      2. Cross-Part Gating     : inter-part context → softmax gate weights
      3. Shared layers         : cross-part interaction
      4. BiLSTM ×2             : sequential temporal modeling
      5. Temporal Attention    : weighted frame aggregation → context vector
      6. Head                  : Dense → softmax

    Args:
        num_classes:     Number of action classes
        sequence_length: Frames per sequence (after padding)

    Returns:
        Keras Model
    """
    # ── INPUT ────────────────────────────────────────────────────────────────
    inputs = layers.Input(shape=(sequence_length, 1662), name='sequence_input')
    x = layers.Masking(mask_value=0.0)(inputs)

    # ── SPLIT KEYPOINTS ──────────────────────────────────────────────────────
    # Pose:  [  0:132 ]  33 landmarks × 4 = 132
    # Face:  [132:1536]  468 landmarks × 3 = 1404
    # Hands: [1536:   ]  21×2 landmarks × 3 = 126
    pose_kp = layers.Lambda(lambda t: t[:, :, :132],    name='pose_split')(x)
    face_kp = layers.Lambda(lambda t: t[:, :, 132:1536], name='face_split')(x)
    hand_kp = layers.Lambda(lambda t: t[:, :, 1536:],   name='hand_split')(x)

    # ── TRANSFORMER ENCODER BRANCHES ─────────────────────────────────────────
    # Dimensions chosen to match MLP branch output dims in hybrid.py:
    #   pose → d_model=64,  4 heads (key_dim=16), 2 blocks, dff=128
    #   face → d_model=128, 4 heads (key_dim=32), 2 blocks, dff=256
    #   hand → d_model=64,  4 heads (key_dim=16), 2 blocks, dff=128
    pose_enc = _create_part_transformer(132,  d_model=64,  num_heads=4,
                                        num_blocks=2, dff=128,  name='pose')
    face_enc = _create_part_transformer(1404, d_model=128, num_heads=4,
                                        num_blocks=2, dff=256,  name='face')
    hand_enc = _create_part_transformer(126,  d_model=64,  num_heads=4,
                                        num_blocks=2, dff=128,  name='hand')

    pose_features = pose_enc(pose_kp)   # (B, T, 64)
    face_features = face_enc(face_kp)   # (B, T, 128)
    hand_features = hand_enc(hand_kp)   # (B, T, 64)

    # ── CROSS-PART CONTEXTUAL GATING ─────────────────────────────────────────
    # Each part queries the other two to build a context vector, then all
    # enriched features are concatenated to compute per-frame gate weights.
    pose_ctx = layers.TimeDistributed(
        layers.Dense(64, activation='relu', name='pose_ctx_dense'), name='pose_ctx'
    )(layers.Concatenate(name='pose_ctx_input')([face_features, hand_features]))   # (B,T,64)

    face_ctx = layers.TimeDistributed(
        layers.Dense(64, activation='relu', name='face_ctx_dense'), name='face_ctx'
    )(layers.Concatenate(name='face_ctx_input')([pose_features, hand_features]))   # (B,T,64)

    hand_ctx = layers.TimeDistributed(
        layers.Dense(64, activation='relu', name='hand_ctx_dense'), name='hand_ctx'
    )(layers.Concatenate(name='hand_ctx_input')([pose_features, face_features]))   # (B,T,64)

    pose_enriched = layers.Concatenate(name='pose_enriched')([pose_features, pose_ctx])  # (B,T,128)
    face_enriched = layers.Concatenate(name='face_enriched')([face_features, face_ctx])  # (B,T,192)
    hand_enriched = layers.Concatenate(name='hand_enriched')([hand_features, hand_ctx])  # (B,T,128)

    gate_input = layers.Concatenate(name='gate_input')(
        [pose_enriched, face_enriched, hand_enriched]
    )  # (B,T,448)
    gate = layers.TimeDistributed(
        layers.Dense(3, activation='softmax', name='gate_dense'),
        name='body_part_gate'
    )(gate_input)  # (B,T,3)

    pose_scale = layers.Lambda(lambda g: g[:, :, 0:1], name='pose_gate')(gate)
    face_scale = layers.Lambda(lambda g: g[:, :, 1:2], name='face_gate')(gate)
    hand_scale = layers.Lambda(lambda g: g[:, :, 2:3], name='hand_gate')(gate)

    pose_gated = layers.Multiply(name='pose_gated')([pose_features, pose_scale])  # (B,T,64)
    face_gated = layers.Multiply(name='face_gated')([face_features, face_scale])  # (B,T,128)
    hand_gated = layers.Multiply(name='hand_gated')([hand_features, hand_scale])  # (B,T,64)

    merged = layers.Concatenate(name='feature_fusion')(
        [pose_gated, face_gated, hand_gated]
    )  # (B,T,256)

    # ── SHARED LAYERS ────────────────────────────────────────────────────────
    x = layers.TimeDistributed(
        layers.Dense(256, activation='relu', name='shared1'), name='shared_td1'
    )(merged)
    x = layers.Dropout(0.3)(x)

    x = layers.TimeDistributed(
        layers.Dense(128, activation='relu', name='shared2'), name='shared_td2'
    )(x)
    x = layers.Dropout(0.3)(x)

    # ── BIDIRECTIONAL LSTM ───────────────────────────────────────────────────
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True),  name='bilstm1')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True),  name='bilstm2')(x)
    x = layers.Dropout(0.3)(x)

    # ── TEMPORAL ATTENTION ───────────────────────────────────────────────────
    attn_scores  = layers.TimeDistributed(
        layers.Dense(1, activation='tanh', name='attn_score'), name='attn_td'
    )(x)                                                                           # (B,T,1)
    attn_weights = layers.Softmax(axis=1, name='temporal_attention')(attn_scores)  # (B,T,1)
    context      = layers.Multiply(name='attn_apply')([x, attn_weights])           # (B,T,64)
    context      = layers.Lambda(
        lambda t: tf.reduce_sum(t, axis=1), name='context_vector'
    )(context)                                                                     # (B,64)

    # ── CLASSIFICATION HEAD ──────────────────────────────────────────────────
    x = layers.Dense(128, activation='relu', name='dense1')(context)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs,
                  name='Transformer_BiLSTM_CrossPartGating_Attention_Model')
    return model
