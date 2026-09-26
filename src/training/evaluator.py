"""
Evaluate model and generate metrics
"""
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from training.data_loader import load_sequences, split_data
from config import CHECKPOINT_DIR


def evaluate_model(model_path=None):
    """
    Evaluate model on test set
    """
    # Load best model
    if model_path is None:
        model_path = CHECKPOINT_DIR / 'best_model'
    
    print(f"Loading model: {model_path}")
    print(f"Model path exists: {Path(model_path).exists()}")
    
    try:
        model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print(f"\nTroubleshooting:")
        print(f"  1. Check if model file exists at: {model_path}")
        print(f"  2. Verify the model was saved correctly")
        print(f"  3. Try retraining the model")
        raise
    
    # Get model's expected sequence length from input shape
    model_input_shape = model.input_shape
    expected_seq_length = model_input_shape[1]  # (None, seq_length, features)
    print(f"\nModel input shape: {model_input_shape}")
    print(f"Expected sequence length: {expected_seq_length} frames")
    
    # Load data with target sequence length matching the model
    print("\nLoading test data...")
    X, y, action_names, is_original = load_sequences(target_length=expected_seq_length)
    _, _, X_test, _, _, y_test = split_data(X, y, is_original=is_original)
    
    # Verify shapes match
    print(f"\nShape verification:")
    print(f"  Model expects: {model_input_shape}")
    print(f"  Data shape: {X_test.shape}")
    if X_test.shape[1:] != model_input_shape[1:]:
        raise ValueError(f"Shape mismatch! Model expects {model_input_shape[1:]}, but data has {X_test.shape[1:]}")
    
    # Predict
    print("\nPredicting...")
    y_pred = model.predict(X_test, verbose=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Get unique classes in test set (some may be missing due to random split)
    unique_classes_in_test = np.unique(y_test)
    present_action_names = [action_names[i] for i in unique_classes_in_test]
    
    # Check if all classes are present
    if len(unique_classes_in_test) < len(action_names):
        missing_classes = set(range(len(action_names))) - set(unique_classes_in_test)
        print(f"\nWARNING: {len(missing_classes)} class(es) not in test set (due to random split):")
        for cls_idx in sorted(list(missing_classes))[:5]:
            print(f"     Class {cls_idx}: {action_names[cls_idx]}")
        if len(missing_classes) > 5:
            print(f"     ... and {len(missing_classes) - 5} more")
    
    # Metrics
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    # Use labels parameter to specify which classes to include
    print(classification_report(y_test, y_pred_classes, 
                                labels=unique_classes_in_test,
                                target_names=present_action_names,
                                zero_division=0))
    
    # Confusion matrix (only for classes in test set)
    cm = confusion_matrix(y_test, y_pred_classes, labels=unique_classes_in_test)
    
    plt.figure(figsize=(20, 20))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=present_action_names, yticklabels=present_action_names)
    plt.title(f'Confusion Matrix ({len(present_action_names)} classes in test set)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    output_path = CHECKPOINT_DIR / 'confusion_matrix.png'
    plt.savefig(output_path, dpi=150)
    print(f"\nConfusion matrix saved: {output_path}")
    
    # Accuracy
    accuracy = (y_test == y_pred_classes).mean()
    print(f"\nTest Accuracy: {accuracy*100:.2f}%")
    
    return accuracy, cm


if __name__ == "__main__":
    evaluate_model()
