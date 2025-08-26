# YOLOV11N-UAV FINETUNE

**OPEN SOURCE. ADVANCED UAV DETECTION WITH SUPERVISION.**

```
┌─────────────────────────────────────┐
│       TRAINING PERFORMANCE          │
├─────────────────────────────────────┤
│  dj-air3     │ 96.8% mAP50          │
│  uav         │ 80.1% mAP50          │
└─────────────────────────────────────┘
```

█████████████████████████████████████████

## █ CORE

**Advanced YOLO11n and YOLO11L finetuned for unmanned aerial vehicles.**  
**Built for edge deployment with supervision library integration.**
**Features InferenceSlicer for small object detection and object tracking.**
**Additional synthetic data generation tool for robust environment handling.**

## █ NEW FEATURES

```
┌─────────────────────────────────────┐
│  FEATURE         │ DESCRIPTION      │
├─────────────────────────────────────┤
│  InferenceSlicer │ Better small UAV │
│                  │ detection        │
│  Object Tracking │ SORT tracker     │
│  Advanced Train  │ Optimized params │
│  Data Prep       │ Roboflow + synth │
└─────────────────────────────────────┘
```

## █ SPECS

```
┌─────────────────────────────────────┐
│  MODEL       │ YOLO11n backbone    │
│  INPUT       │ 640x640 RGB         │
│  OUTPUT      │ UAV bounding boxes  │
│  TARGET      │ Mobile/Edge deploy  │
└─────────────────────────────────────┘
```

█████████████████████████████████████████

## █ DATASET

```
████ TRAIN: 6877 images (70%)
████ VALID: 1966 images (20%) 
████ TEST:   985 images (10%)
```

**PREPROCESSING**: Auto-orient → Resize to 640x640  
**AUGMENTATIONS**: None applied.

█████████████████████████████████████████

## █ WEIGHTS

```
├── yolov11n-UAV-finetune.pt   [PyTorch]
├── yolov11n-UAV-finetune.onnx [ONNX]
└── yolo12l-UAV-finetune.pt    [Larger model]
```

█████████████████████████████████████████

## █ USAGE

### Basic Video Inference
```bash
python video_inference.py -i video.mp4
```

### Advanced Video Inference (NEW!)
```bash
# With InferenceSlicer and tracking
python advanced_video_inference.py -i video.mp4

# Custom slice size and compression
python advanced_video_inference.py -i video.mp4 --slice-size 512 --compress

# Single frame testing
python advanced_video_inference.py -i video.mp4 --single-frame --frame-number 100 --use-slicer
```

### Advanced Training (NEW!)
```bash
# Train with optimized parameters
python advanced_train.py --data ./uav-detection-3/data.yaml

# Validate model
python advanced_train.py --mode val --weights ./runs/detect/train/weights/best.pt

# Export to ONNX
python advanced_train.py --mode export --weights ./runs/detect/train/weights/best.pt
```

### Data Preparation (NEW!)
```bash
# Prepare synthetic data
python prepare_synthetic_data.py --skip-download

# With Roboflow integration
python prepare_synthetic_data.py --api-key YOUR_API_KEY
```

### Test Setup
```bash
python test_advanced_features.py
```

█████████████████████████████████████████

## █ CITATION

```bibtex
@misc{
uav-detection-blxxz_dataset,
title = { uav-detection Dataset },
type = { Open Source Dataset },
author = { UAVdetection },
howpublished = { \url{ https://universe.roboflow.com/uavdetection-msr99/uav-detection-blxxz } },
url = { https://universe.roboflow.com/uavdetection-msr99/uav-detection-blxxz },
journal = { Roboflow Universe },
publisher = { Roboflow },
year = { 2024 },
month = { apr },
note = { visited on 2025-08-16 },
}
```

█████████████████████████████████████████

```
███████████████████████████████████████████████████
█ OPEN SOURCE UAV DETECTION FOR ALL              █
███████████████████████████████████████████████████
```
