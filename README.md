# VSL Recognition - Sign Language Recognition

CNN + LSTM model for Sign Language recognition using MediaPipe keypoints.

## Quick Start

```bash
# 1. Prepare dataset (extract + augment)
python -m src.data.prepare_pipeline

# 2. Train model
python -m src.training.pipeline

# 3. Run inference
python -m src.inference.realtime --mode webcam
```

## Project Structure

```
│   .gitignore
│   environment.yml
│   main.py
│   README.md
│   requirements.txt
│
├───src
│   │   config.py
│   │
│   │
│   ├───data
│   │       augment.py
│   │       check_distribution.py
│   │       extract.py
│   │       prepare_pipeline.py
│   │
│   ├───models
│   │   │   components.py
│   │   │   hybrid.py
│   │   │
│   │   └───transformer
│   │           model.py
│   │           __init__.py
│   │
│   ├───training
│   │       data_loader.py
│   │       evaluator.py
│   │       pipeline.py
│   │       trainer.py
│   │       __init__.py
│   │
│   └───utils
│       │   augmentation.py
│       │   extraction.py
│       │   inference_utils.py
│       └───__init__.py
│
└───vsl-web
        favicon.svg
        index.html
        script.js
        server.py
        style.css
```

## Model Architecture

```
Input (33 frames, 1662 keypoints)
    ↓
MLP Branches → LSTM → Softmax (76 classes)
```

## Usage

### Data Preparation
```bash
python -m src.data.prepare_pipeline
```

### Training
```bash
python -m src.training.pipeline
```

### Inference
```bash
python -m src.inference.realtime --mode webcam
```

## Requirements

```bash
pip install -r requirements.txt
```

---

## Server Deployment

### Setup on Server

```bash
# 1. Clone/pull code
cd /home/islabworker2/mya/vsl-recognition
git pull

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Verify GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Training on Server

```bash
# Run training (foreground)
python main.py train

# Or with nohup (background)
nohup python main.py train > training.log 2>&1 &
tail -f training.log
```

### Monitor Training

```bash
# Check processes
ps aux | grep "main.py"

# Monitor GPU
nvidia-smi

# View logs
tail -f training.log
tail -f logs/training_*.log
```

---

## Web Dashboard UI

The project now includes a beautiful, real-time Web Dashboard for visualizing sign language predictions!

### 1. Start the Backend Server
The backend requires `Flask` and `Flask-Cors` (added in requirements).
```bash
cd vsl-web
python server.py
```
*Note: Make sure your `config.py` MODEL_TYPE and the paths inside `server.py` point to your correctly trained `saved_model`.*

### 2. Open the UI
Simply open `vsl-web/index.html` in any modern web browser (Edge, Chrome, Firefox).

### 3. Usage
- Click **Import Media (.mp4 / .npy)** to upload a video.
- The browser will **automatically execute MediaPipe Holistic** processing to extract exactly 1662 keypoints natively in the client.
- The web app dynamically handles frame padding based on the active loaded TensorFlow model's required sequence length.
- View the skeleton overlaid live alongside the raw video and live top-3 predictions.

