"""
Main training script
"""
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
from pathlib import Path
import sys
import json

import numpy as np
sys.path.append(str(Path(__file__).parent.parent))

from training.data_loader import load_sequences, split_data, create_tf_dataset
from config import TRAINING_CONFIG, CHECKPOINT_DIR, LOGS_DIR


def configure_gpu():
    """Configure GPU settings for TensorFlow"""
    print("\n" + "="*70)
    print("GPU CONFIGURATION")
    print("="*70)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            # Enable memory growth to prevent TF from allocating all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            print(f"Found {len(gpus)} GPU(s):")
            for i, gpu in enumerate(gpus):
                print(f"  GPU {i}: {gpu.name}")
            
            # Set visible devices
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"Using GPU: {gpus[0].name}")
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU detected - training will use CPU (slower)")
        print("  Check CUDA installation and TensorFlow-GPU compatibility")
    
    print("="*70 + "\n")


def train_model():
    """Main training function"""
    
    # Configure GPU first
    configure_gpu()
    
    print("="*70)
    print("SIGN LANGUAGE RECOGNITION - TRAINING")
    print("="*70)
    
    # 1. Load data
    print("\n[1/5] Loading data...")
    X, y, action_names, is_original = load_sequences()  # Load ALL sequences
    num_classes = len(action_names)
    
    # 2. Split data (pass is_original to prevent augmented data leaking into val/test)
    print("\n[2/5] Splitting data...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        X, y,
        train_size=TRAINING_CONFIG['train_split'],
        val_size=TRAINING_CONFIG['val_split'],
        is_original=is_original
    )
    
    # 3. Create datasets
    print("\n[3/5] Creating TensorFlow datasets...")
    print("Creating training dataset...")
    train_ds = create_tf_dataset(X_train, y_train, batch_size=TRAINING_CONFIG['batch_size'], shuffle=True)
    print("Training dataset created")
    print("Creating validation dataset...")
    val_ds = create_tf_dataset(X_val, y_val, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)
    print("Validation dataset created")
    print("Creating test dataset...")
    test_ds = create_tf_dataset(X_test, y_test, batch_size=TRAINING_CONFIG['batch_size'], shuffle=False)
    print(" Test dataset created")
    
    # 4. Build model
    print("\n[4/5] Building model...")
    print("  → Creating model architecture...")
    
    # Get sequence length from data shape
    sequence_length = X_train.shape[1]
    keypoint_dim = X_train.shape[2]      # Should be 1662
    
    print(f"     Input shape: ({sequence_length}, {keypoint_dim})")
    
    from config import MODEL_TYPE
    if MODEL_TYPE == 'transformer':
        from models.transformer.model import create_hybrid_transformer_model
        print("  → Using Transformer Encoder + BiLSTM architecture")
        model = create_hybrid_transformer_model(
            num_classes=num_classes,
            sequence_length=sequence_length
        )
    else:
        from models.hybrid import create_hybrid_multistream_model
        print("  → Using MLP + BiLSTM Hybrid architecture")
        model = create_hybrid_multistream_model(
            num_classes=num_classes,
            sequence_length=sequence_length
        )
    print("  Model architecture created")
    
    print(" Compiling model...")
    model.compile(
        optimizer=Adam(learning_rate=TRAINING_CONFIG['learning_rate']),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )
    print("Model compiled")
    
    print("Model summary:")
    model.summary()
    print("Model ready")
    
    # 5. Setup callbacks
    print("\n[5/5] Setting up training...")
    CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
    LOGS_DIR.mkdir(exist_ok=True, parents=True)
    
    callbacks = [
        ModelCheckpoint(
            filepath=str(CHECKPOINT_DIR / 'best_model'),
            monitor='val_accuracy',
            save_best_only=True,
            save_format='tf',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_accuracy',
            patience=TRAINING_CONFIG['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=TRAINING_CONFIG['reduce_lr_patience'],
            verbose=1
        ),
        TensorBoard(
            log_dir=str(LOGS_DIR / 'fit'),
            histogram_freq=1
        )
    ]
    
    # 6. Train
    print("\n" + "="*70)
    print("STARTING TRAINING")
    print("="*70)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Batch size: {TRAINING_CONFIG['batch_size']}")
    print(f"Epochs: {TRAINING_CONFIG['epochs']}")
    print("="*70 + "\n")
    
    import sys
    sys.stdout.flush()  # Force flush before model.fit
    
    # Compute class weights to handle imbalance
    from sklearn.utils.class_weight import compute_class_weight
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights_array))
    print(f"Class weights computed: min={min(class_weights_array):.3f}, max={max(class_weights_array):.3f}")

    print("[DEBUG] About to call model.fit()...")
    sys.stdout.flush()
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TRAINING_CONFIG['epochs'],
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    print("[DEBUG] model.fit() completed successfully!")
    sys.stdout.flush()
    
    # 7. Evaluate on test set
    print("\n" + "="*70)
    print("EVALUATING ON TEST SET")
    print("="*70 + "\n")
    
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    print(f"\nTest Accuracy: {test_acc*100:.2f}%")
    
    # 8. Save final model
    model.save(str(CHECKPOINT_DIR / 'final_model'), save_format='tf')
    
    # 9. Save action mapping
    mapping = {i: name for i, name in enumerate(action_names)}
    with open(CHECKPOINT_DIR / 'action_mapping.json', 'w') as f:
        json.dump(mapping, f, indent=2)
    
    print(f"\nTraining complete")
    print(f"   Best model: {CHECKPOINT_DIR / 'best_model'}")
    print(f"   TensorBoard: tensorboard --logdir={LOGS_DIR}")
    
    return history, test_acc


if __name__ == "__main__":
    train_model()
