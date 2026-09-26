"""
Utilities for stateful LSTM inference with variable-length sequences.
"""
import numpy as np


def predict_sequence_stateful(model, frames, strategy='last'):
    """
    Predict action class for a variable-length sequence using stateful model.
    
    Args:
        model: Stateful LSTM model
        frames: Numpy array of shape (num_frames, 1662) - keypoint sequences
        strategy: Prediction strategy - 'last', 'average', or 'weighted'
    
    Returns:
        Final prediction (1, num_classes)
    """
    model.reset_states()
    predictions = []
    
    # Process each frame
    for frame in frames:
        frame_input = frame.reshape(1, 1, 1662)
        pred = model.predict(frame_input, verbose=0)
        predictions.append(pred)
    
    # Apply prediction strategy
    if strategy == 'last':
        # Use only last frame prediction
        final_pred = predictions[-1]
    elif strategy == 'average':
        # Average all predictions
        final_pred = np.mean(predictions, axis=0)
    elif strategy == 'weighted':
        # Weighted average (recent frames matter more)
        weights = np.linspace(0.5, 1.0, len(predictions))
        final_pred = np.average(predictions, weights=weights, axis=0)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    return final_pred


def predict_batch_videos(model, video_list, strategy='last'):
    """
    Predict action classes for multiple videos.
    
    Args:
        model: Stateful LSTM model
        video_list: List of numpy arrays, each shape (num_frames, 1662)
        strategy: Prediction strategy
    
    Returns:
        List of predictions
    """
    results = []
    
    for i, video in enumerate(video_list):
        pred = predict_sequence_stateful(model, video, strategy=strategy)
        predicted_class = np.argmax(pred)
        confidence = pred[0, predicted_class]
        
        results.append({
            'video_idx': i,
            'prediction': pred,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'num_frames': len(video)
        })
    
    return results


def predict_streaming(model, frame_generator, strategy='last', window_size=None):
    """
    Real-time streaming prediction from frame generator (e.g., webcam).
    
    Args:
        model: Stateful LSTM model
        frame_generator: Generator yielding frames (1662,)
        strategy: Prediction strategy
        window_size: If set, reset states every N frames
    
    Yields:
        Predictions for each frame
    """
    model.reset_states()
    predictions = []
    frame_count = 0
    
    for frame in frame_generator:
        frame_input = frame.reshape(1, 1, 1662)
        pred = model.predict(frame_input, verbose=0)
        predictions.append(pred)
        frame_count += 1
        
        # Apply strategy
        if strategy == 'last':
            final_pred = pred
        elif strategy == 'average':
            final_pred = np.mean(predictions, axis=0)
        elif strategy == 'weighted':
            weights = np.linspace(0.5, 1.0, len(predictions))
            final_pred = np.average(predictions, weights=weights, axis=0)
        
        yield final_pred
        
        # Reset states if window_size reached
        if window_size and frame_count >= window_size:
            model.reset_states()
            predictions = []
            frame_count = 0


if __name__ == "__main__":
    print("Stateful Inference Utilities")
    print("=" * 50)
    print("\nAvailable functions:")
    print("  - predict_sequence_stateful(model, frames, strategy)")
    print("  - predict_batch_videos(model, video_list, strategy)")
    print("  - predict_streaming(model, frame_generator, strategy)")
    print("\nPrediction strategies:")
    print("  - 'last': Use last frame prediction only")
    print("  - 'average': Average all frame predictions")
    print("  - 'weighted': Weighted average (recent frames matter more)")
