# Installation Guide

## Quick Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Test the setup:**
   ```bash
   python3 test_advanced_features.py
   ```

3. **Download models (if needed):**
   ```bash
   # For training
   wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolo11l.pt
   
   # For inference (if not already present)
   wget https://raw.githubusercontent.com/droneforge/yolov11n-UAV-finetune/main/yolov11n-UAV-finetune.pt
   ```

## Dependencies

The new advanced features require these additional packages:

- **supervision**: Advanced computer vision utilities
- **trackers**: Object tracking implementation (SORTTracker)
- **ultralytics**: YOLO implementation
- **roboflow**: Dataset management (optional)

## Troubleshooting

### Import Errors
If you get import errors, ensure all dependencies are installed:
```bash
pip install ultralytics supervision trackers==2.0.2rc0 roboflow
```

### GPU Support
For GPU acceleration:
```bash
# Install PyTorch with CUDA support first
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Then install other dependencies
pip install -r requirements.txt
```

### Python Version
Ensure you're using Python 3.8 or higher:
```bash
python3 --version
```

## Verification

Run the test script to verify everything is working:
```bash
python3 test_advanced_features.py
```

You should see all tests pass with ✓ marks.
