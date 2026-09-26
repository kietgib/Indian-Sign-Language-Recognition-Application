"""
Run full training and evaluation pipeline
"""
import sys
from pathlib import Path
import os

# ========================================
# ENVIRONMENT CONFIGURATION
# ========================================
print("="*70)
print("CONFIGURING ENVIRONMENT")
print("="*70)

# Disable oneDNN optimizations (prevent crashes/instability)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print("✓ Disabled oneDNN optimizations")

# Set reasonable thread limits
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
print("✓ Set thread limits")

print("="*70)
print()

import tensorflow as tf
import numpy as np
import random

sys.path.append(str(Path(__file__).parent.parent))

# ========================================
# GPU CONFIGURATION
# ========================================
print("="*70)
print("GPU CONFIGURATION")
print("="*70)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth (prevent OOM)
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Found {len(gpus)} GPU(s)")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        print("GPU memory growth enabled")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")
else:
    print("No GPU detected - training will use CPU (slower)")

print("="*70)
print()

from training.trainer import train_model
from training.evaluator  import evaluate_model


def set_seed(seed=42):
    """
    Set random seeds for reproducible results
    
    Args:
        seed: Random seed value (default: 42)
    """
    print("="*70)
    print("SETTING RANDOM SEEDS FOR REPRODUCIBILITY")
    print("="*70)
    print(f"Random seed: {seed}")
    
    # Python random
    random.seed(seed)
    
    # Numpy
    np.random.seed(seed)
    
    # TensorFlow
    tf.random.set_seed(seed)
    
    # For hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # TensorFlow deterministic operations (slower but reproducible)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    
    print("All random seeds set")
    print("="*70 + "\n")


def setup_gpu():
    """
    Configure GPU settings
    """
    print("="*70)
    print("GPU CONFIGURATION")
    print("="*70)
    
    # Check GPU availability
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        print(f"GPUs detected: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        
        try:
            # Enable memory growth (prevent TensorFlow from allocating all GPU memory)
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
            
            # Set visible devices (optional, use first GPU)
            tf.config.set_visible_devices(gpus[0], 'GPU')
            print(f"Using GPU: {gpus[0].name}")
            
        except RuntimeError as e:
            print(f"GPU configuration error: {e}")
    else:
        print("No GPU detected - training will use CPU (slower)")
    
    print("="*70 + "\n")
    
    return gpus


def run_full_pipeline():
    """
    Run complete pipeline: train and evaluate
    """
    # Step 0: Set seed for reproducibility
    set_seed()
    
    # Step 0.5: Setup GPU
    setup_gpu()
    
    print("="*70)
    print("FULL TRAINING & EVALUATION PIPELINE")
    print("="*70)
    
    # Step 1: Train model
    print("\n" + "="*70)
    print("STEP 1: TRAINING")
    print("="*70)
    history, train_test_acc = train_model()
    
    # Step 2: Evaluate on test set
    print("\n" + "="*70)
    print("STEP 2: DETAILED EVALUATION")
    print("="*70)
    
    try:
        eval_acc, confusion_matrix = evaluate_model()
        print("Evaluation completed successfully")
    except Exception as e:
        print(f" Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        eval_acc = 0.0
        confusion_matrix = None
    
    # Summary
    print("\n" + "="*70)
    print("PIPELINE COMPLETED")
    print("="*70)
    print(f"Training completed successfully")
    if eval_acc > 0:
        print(f"Final test accuracy: {eval_acc*100:.2f}%")
        print(f"Confusion matrix saved to checkpoints/confusion_matrix.png")
    else:
        print("Evaluation step encountered errors (see above).")
        
    print("="*70)
    
    return history, eval_acc, confusion_matrix


if __name__ == "__main__":
    run_full_pipeline()
