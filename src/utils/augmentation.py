"""
Data augmentation utilities for sign language sequences
"""
import numpy as np
from typing import Tuple


def temporal_reverse(sequence: np.ndarray) -> np.ndarray:
    """
    Reverse sequence in time (backwards video)
    
    Args:
        sequence: (num_frames, num_keypoints) array
    
    Returns:
        Reversed sequence
    """
    return np.flip(sequence, axis=0).copy()


def temporal_subsample(sequence: np.ndarray, ratio: float = 0.8) -> np.ndarray:
    """
    Randomly subsample frames and interpolate back to original length
    
    Args:
        sequence: (num_frames, num_keypoints) array
        ratio: Fraction of frames to keep (0.5 - 1.0)
    
    Returns:
        Subsampled and interpolated sequence
    """
    num_frames = len(sequence)
    num_keep = max(int(num_frames * ratio), 2)
    
    # Randomly select frames to keep
    indices = sorted(np.random.choice(num_frames, num_keep, replace=False))
    
    # Interpolate back to original length
    from scipy.interpolate import interp1d
    interpolator = interp1d(
        indices, 
        sequence[indices], 
        axis=0, 
        kind='linear',
        fill_value='extrapolate'
    )
    
    new_indices = np.arange(num_frames)
    augmented = interpolator(new_indices)
    
    return augmented.astype(np.float32)


def add_noise(sequence: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """
    Add small Gaussian noise to keypoints
    
    Args:
        sequence: (num_frames, num_keypoints) array
        noise_std: Standard deviation of Gaussian noise
    
    Returns:
        Noisy sequence
    """
    noise = np.random.normal(0, noise_std, sequence.shape)
    return (sequence + noise).astype(np.float32)


def temporal_scale(sequence: np.ndarray, scale: float) -> np.ndarray:
    """
    Speed up or slow down sequence by temporal scaling
    
    Args:
        sequence: (num_frames, num_keypoints) array
        scale: Speed multiplier (0.5 = slower, 2.0 = faster)
    
    Returns:
        Scaled sequence (same length as input)
    """
    from scipy.interpolate import interp1d
    
    num_frames = len(sequence)
    original_indices = np.arange(num_frames)
    
    # Create new time indices
    new_length = int(num_frames / scale)
    new_indices = np.linspace(0, num_frames - 1, new_length)
    
    # Interpolate
    interpolator = interp1d(
        original_indices,
        sequence,
        axis=0,
        kind='linear',
        fill_value='extrapolate'
    )
    
    scaled = interpolator(new_indices)
    
    # Pad or crop to original length
    if len(scaled) < num_frames:
        # Pad by repeating last frame
        padding = np.repeat(scaled[-1:], num_frames - len(scaled), axis=0)
        scaled = np.vstack([scaled, padding])
    elif len(scaled) > num_frames:
        # Crop
        scaled = scaled[:num_frames]
    
    return scaled.astype(np.float32)


def random_temporal_crop(sequence: np.ndarray, crop_ratio: float = 0.9) -> np.ndarray:
    """
    Randomly crop sequence in time and pad back
    
    Args:
        sequence: (num_frames, num_keypoints) array
        crop_ratio: Fraction to keep (e.g., 0.9 = keep 90%)
    
    Returns:
        Cropped and padded sequence
    """
    num_frames = len(sequence)
    crop_length = int(num_frames * crop_ratio)
    
    # Random start position
    max_start = num_frames - crop_length
    start = np.random.randint(0, max_start + 1) if max_start > 0 else 0
    
    # Crop
    cropped = sequence[start:start + crop_length]
    
    # Pad back to original length
    pad_length = num_frames - crop_length
    if pad_length > 0:
        # Pad with zeros at the end
        padding = np.zeros((pad_length, sequence.shape[1]), dtype=np.float32)
        cropped = np.vstack([cropped, padding])
    
    return cropped


def spatial_jitter(sequence: np.ndarray, jitter_std: float = 0.02) -> np.ndarray:
    """
    Add small random spatial offset to all x, y coordinates each frame.
    Simulates the signer standing slightly off-center or camera angle variation.
    
    Pose format:  stride 4 (x, y, z, vis) → offset indices 0::4 and 1::4
    Face/Hand format: stride 3 (x, y, z)  → offset indices 0::3 and 1::3
    
    Args:
        sequence: (num_frames, 1662) array - normalized keypoints
        jitter_std: Std of per-frame Gaussian jitter (default 0.02 ≈ 2% shoulder width)
    
    Returns:
        Jittered sequence (same shape)
    """
    seq = sequence.copy()
    num_frames = len(seq)
 
    # Pose: indices 0:132, stride 4
    # Face: indices 132:1536, stride 3
    # Hands: indices 1536:1662, stride 3
    for frame_idx in range(num_frames):
        # Skip zero-padded frames
        if np.all(seq[frame_idx] == 0):
            continue

        dx = np.random.normal(0, jitter_std)
        dy = np.random.normal(0, jitter_std)

        # Pose (stride 4)
        seq[frame_idx, 0:132:4] += dx
        seq[frame_idx, 1:132:4] += dy
        # Face (stride 3)
        seq[frame_idx, 132:1536:3] += dx
        seq[frame_idx, 133:1536:3] += dy
        # Hands (stride 3)
        seq[frame_idx, 1536::3] += dx
        seq[frame_idx, 1537::3] += dy

    return seq.astype(np.float32)


def augment_sequence(sequence: np.ndarray, 
                     method: str = 'reverse',
                     **kwargs) -> np.ndarray:
    """
    Apply augmentation to a sequence
    
    Args:
        sequence: (num_frames, num_keypoints) array
        method: Augmentation method name
        **kwargs: Additional parameters for specific methods
    
    Returns:
        Augmented sequence
    """
    if method == 'reverse':
        return temporal_reverse(sequence)
    elif method == 'subsample':
        ratio = kwargs.get('ratio', 0.8)
        return temporal_subsample(sequence, ratio)
    elif method == 'noise':
        noise_std = kwargs.get('noise_std', 0.01)
        return add_noise(sequence, noise_std)
    elif method == 'scale':
        scale = kwargs.get('scale', 0.8)
        return temporal_scale(sequence, scale)
    elif method == 'crop':
        crop_ratio = kwargs.get('crop_ratio', 0.9)
        return random_temporal_crop(sequence, crop_ratio)
    elif method == 'jitter':
        jitter_std = kwargs.get('jitter_std', 0.02)
        return spatial_jitter(sequence, jitter_std)
    else:
        raise ValueError(f"Unknown augmentation method: {method}")


def get_augmentation_methods() -> list:
    """Get list of available augmentation methods"""
    return ['reverse', 'subsample', 'noise', 'scale', 'crop', 'jitter']
