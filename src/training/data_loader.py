"""
Load sequences and create train/val/test datasets
"""
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys

sys.path.append(str(Path(__file__).parent.parent))
from config import SEQUENCE_PATH


def load_sequences(sequence_path=SEQUENCE_PATH, max_sequences_per_action=None, target_length=None):
    """
    Load all sequences and labels, padding to uniform length.
    Tracks which sequences are original vs augmented to prevent data leakage.
    
    Args:
        sequence_path: Path to sequences directory
        max_sequences_per_action: Limit sequences per action (None = all). Use for testing with limited RAM.
        target_length: Target sequence length for padding (None = auto-detect from data)
    
    Returns:
        X: np.array (num_samples, max_length, 1662) - padded sequences
        y: np.array (num_samples,) - label indices
        action_names: list - Action names
        is_original: np.array (num_samples,) - True if original file, False if augmented
    """
    print("Loading sequences...")
    if max_sequences_per_action:
        print(f"  [LIMIT MODE] Max {max_sequences_per_action} sequences per action")
    
    # Get all action folders
    action_folders = sorted([d for d in sequence_path.iterdir() if d.is_dir()])
    action_names = [d.name for d in action_folders]
    
    X = []
    y = []
    sequence_lengths = []
    is_original = []  # Track original vs augmented
    
    for label_idx, action_folder in enumerate(action_folders):
        npy_files = sorted(action_folder.glob('*.npy'))
        
        # Separate original and augmented files
        original_files = [f for f in npy_files if '_aug' not in f.stem]
        augmented_files = [f for f in npy_files if '_aug' in f.stem]
        
        # Limit sequences if specified (only limit originals)
        if max_sequences_per_action:
            original_files = original_files[:max_sequences_per_action]
        
        all_files = original_files + augmented_files
        
        for npy_file in all_files:
            seq = np.load(npy_file)
            X.append(seq)
            y.append(label_idx)
            sequence_lengths.append(len(seq))
            is_original.append('_aug' not in npy_file.stem)
        
        print(f"  [{label_idx+1}/{len(action_folders)}] {action_folder.name}: "
              f"{len(original_files)} original + {len(augmented_files)} augmented")
    
    # Determine padding length
    if target_length is not None:
        max_length = target_length
        print(f"\n  Using target sequence length: {max_length} frames")
    else:
        # Use 95th percentile instead of max to reduce noise from extreme outliers
        max_length = int(np.percentile(sequence_lengths, 95))
        print(f"\n  Auto-detected 95th percentile sequence length: {max_length} frames")
        print(f"  (max was {max(sequence_lengths)}, using 95th pct to reduce zero-padding)")
    
    print(f"  Sequence length stats:")
    print(f"    Min: {min(sequence_lengths)} frames")
    print(f"    Max: {max(sequence_lengths)} frames")
    print(f"    Mean: {np.mean(sequence_lengths):.1f} frames")
    
    # Pad or truncate all sequences to max_length
    print(f"\n  Padding/truncating all sequences to {max_length} frames...")
    X_padded = []
    for seq in X:
        if len(seq) < max_length:
            # Pad with zeros
            padding = np.zeros((max_length - len(seq), seq.shape[1]))
            seq_padded = np.vstack([seq, padding])
        elif len(seq) > max_length:
            # Truncate to max_length
            seq_padded = seq[:max_length]
        else:
            seq_padded = seq
        X_padded.append(seq_padded)
    
    X = np.array(X_padded)
    y = np.array(y)
    is_original = np.array(is_original)
    
    n_orig = is_original.sum()
    n_aug = (~is_original).sum()
    print(f"\nLoaded {len(X)} sequences ({n_orig} original + {n_aug} augmented)")
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Actions: {len(action_names)}")
    
    return X, y, action_names, is_original


def split_data(X, y, train_size=0.7, val_size=0.15, random_state=42, is_original=None):
    """
    Split into train/val/test with smart stratification.
    If is_original is provided, splits ONLY on original sequences to prevent
    augmented data leaking into val/test sets.
    
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    # If original mask provided: split originals only, add augmented to train
    if is_original is not None and (~is_original).sum() > 0:
        print(f"\n[DATA LEAKAGE PREVENTION] Splitting on original sequences only")
        X_orig = X[is_original]
        y_orig = y[is_original]
        X_aug  = X[~is_original]
        y_aug  = y[~is_original]
    else:
        X_orig, y_orig = X, y
        X_aug, y_aug = np.empty((0, X.shape[1], X.shape[2])), np.empty((0,), dtype=y.dtype)

    # Check class distribution on originals
    unique_classes, class_counts = np.unique(y_orig, return_counts=True)
    min_samples = class_counts.min()
    
    print(f"\nClass distribution (originals only):")
    print(f"  Total classes: {len(unique_classes)}")
    print(f"  Minimum samples per class: {min_samples}")
    print(f"  Maximum samples per class: {class_counts.max()}")
    print(f"  Average samples per class: {class_counts.mean():.1f}")
    
    temp_size = 1 - train_size
    min_samples_in_temp = int(np.floor(min_samples * temp_size))
    can_stratify = min_samples_in_temp >= 2
    
    if not can_stratify:
        problematic = []
        for i in range(len(unique_classes)):
            samples_in_temp = int(np.floor(class_counts[i] * temp_size))
            if samples_in_temp < 2:
                problematic.append((unique_classes[i], class_counts[i], samples_in_temp))
        print(f"\n⚠️  WARNING: {len(problematic)} class(es) will have < 2 samples in val+test set")
        for cls_idx, total_count, temp_count in problematic[:5]:
            print(f"     Class {cls_idx}: {total_count} total → ~{temp_count} in val+test")
        if len(problematic) > 5:
            print(f"     ... and {len(problematic) - 5} more")
    
    if can_stratify:
        print(f"  ✓ Using STRATIFIED split (balanced distribution)")
        X_orig_train, X_temp, y_orig_train, y_temp = train_test_split(
            X_orig, y_orig, train_size=train_size, random_state=random_state, stratify=y_orig
        )
        val_ratio = val_size / (1 - train_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=val_ratio, random_state=random_state, stratify=y_temp
        )
    else:
        print(f"  ⚠️  Using RANDOM split (stratification not possible)")
        X_orig_train, X_temp, y_orig_train, y_temp = train_test_split(
            X_orig, y_orig, train_size=train_size, random_state=random_state
        )
        val_ratio = val_size / (1 - train_size)
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, train_size=val_ratio, random_state=random_state
        )
    
    # Add ALL augmented data to training only (no leakage)
    if len(X_aug) > 0:
        X_train = np.concatenate([X_orig_train, X_aug], axis=0)
        y_train = np.concatenate([y_orig_train, y_aug], axis=0)
        print(f"  + Added {len(X_aug)} augmented sequences to train only")
    else:
        X_train, y_train = X_orig_train, y_orig_train
    
    total = len(X_orig)  # Report % based on originals
    print(f"\nDataset split:")
    print(f"  Train: {len(X_train)} ({len(X_orig_train)}/{len(X_orig)} orig + {len(X_aug)} aug)")
    print(f"  Val:   {len(X_val)} ({len(X_val)/total*100:.1f}% of originals)")
    print(f"  Test:  {len(X_test)} ({len(X_test)/total*100:.1f}% of originals) ← NO augmented data")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def create_tf_dataset(X, y, batch_size=32, shuffle=True):
    """
    Create tf.data.Dataset
    """
    # One-hot encode labels
    num_classes = y.max() + 1
    y_onehot = tf.keras.utils.to_categorical(y, num_classes)
    
    dataset = tf.data.Dataset.from_tensor_slices((X, y_onehot))
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(X))
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    return dataset


if __name__ == "__main__":
    # Test
    X, y, actions, is_original = load_sequences()
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, is_original=is_original)
    
    train_ds = create_tf_dataset(X_train, y_train, batch_size=32)
    
    print(f"\nDataset created")
    for batch_X, batch_y in train_ds.take(1):
        print(f"   Batch X shape: {batch_X.shape}")  # (32, 130, 1662)
        print(f"   Batch y shape: {batch_y.shape}")  # (32, num_classes)
