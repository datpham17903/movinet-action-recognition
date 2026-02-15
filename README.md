# Movinet Action Recognition

Human action recognition using MoViNet-style 3D CNN (R3D) with PyTorch.

## Features

- **GPU/CPU Support**: Automatic fallback to CPU when GPU is not compatible
- **Streaming Mode**: Real-time frame-by-frame inference
- **Batch Mode**: Video file classification
- **GUI Application**: Easy-to-use Tkinter interface
- **Multiple Models**: Support for A0, A1, A2, A3 variants
- **600 Action Classes**: Based on Kinetics-400 dataset

## Requirements

- Python 3.12+
- NVIDIA GPU with CUDA support (recommended)
- 8GB+ RAM

## Installation

```bash
# Clone repository
git clone https://github.com/datpham17903/movinet-action-recognition.git
cd movinet-action-recognition

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Python API

```python
from movinet_classifier import MovinetClassifier

# Initialize classifier (with GPU)
classifier = MovinetClassifier(model_id="a0", use_streaming=False)

# Predict from video file
results = classifier.predict("path/to/video.mp4", top_k=5)
print(results)
# [('dancing', 0.85), ('running', 0.10), ...]

# Or use streaming mode for webcam
classifier_stream = MovinetClassifier(model_id="a0", use_streaming=True)
```

### GUI Application

```bash
python gui_app.py
```

The GUI allows you to:
- Select model variant (A0-A3)
- Enable streaming mode for real-time inference
- Load video files for batch prediction
- Use webcam for live action recognition

## Model Variants

| Model | Parameters | Use Case |
|-------|------------|----------|
| A0 | ~3.3M | Mobile/Edge |
| A1 | ~5.3M | Balanced |
| A2 | Larger | Server |
| A3 | Largest | High accuracy |

## Project Structure

```
movinet_action_recognition/
├── movinet_classifier.py   # Core classifier
├── gui_app.py              # GUI application
├── test_movinet.py         # Unit tests
├── requirements.txt         # Dependencies
└── README.md               # This file
```

## Testing

```bash
python test_movinet.py
```

## GPU Compatibility Note

**RTX 5060 Ti (Ada Lovelace / sm_120)**: Current PyTorch versions (including nightly builds as of Feb 2026) do not fully support the sm_120 compute capability. The classifier automatically falls back to CPU mode when GPU inference fails.

To check your PyTorch CUDA support:
```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

## License

MIT License
