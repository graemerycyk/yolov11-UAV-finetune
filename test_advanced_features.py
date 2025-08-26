#!/usr/bin/env python3
"""
Test script for advanced YOLOv11 features
Quick testing of supervision library integration
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test if all required packages can be imported"""
    print("Testing imports...")
    
    try:
        import ultralytics
        print(f"✓ ultralytics: {ultralytics.__version__}")
    except ImportError as e:
        print(f"✗ ultralytics: {e}")
        return False
    
    try:
        import supervision as sv
        print(f"✓ supervision: {sv.__version__}")
    except ImportError as e:
        print(f"✗ supervision: {e}")
        return False
    
    try:
        from trackers import SORTTracker
        print("✓ trackers: SORTTracker imported successfully")
    except ImportError as e:
        print(f"✗ trackers: {e}")
        return False
    
    try:
        import numpy as np
        print(f"✓ numpy: {np.__version__}")
    except ImportError as e:
        print(f"✗ numpy: {e}")
        return False
    
    try:
        import cv2
        print(f"✓ opencv: {cv2.__version__}")
    except ImportError as e:
        print(f"✗ opencv: {e}")
        return False
    
    return True

def test_model_loading():
    """Test model loading"""
    print("\nTesting model loading...")
    
    # Check for available model files
    model_files = [
        "yolov11n-UAV-finetune.pt",
        "yolov11n.pt", 
        "yolo11n.pt",
        "yolo11l.pt"
    ]
    
    found_model = None
    for model_file in model_files:
        if os.path.exists(model_file):
            found_model = model_file
            break
    
    if not found_model:
        print("✗ No model files found. Please ensure you have a YOLO model file.")
        return False
    
    try:
        from ultralytics import YOLO
        model = YOLO(found_model)
        print(f"✓ Model loaded successfully: {found_model}")
        return True
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        return False

def test_supervision_features():
    """Test supervision library features"""
    print("\nTesting supervision features...")
    
    try:
        import supervision as sv
        import numpy as np
        
        # Test color palette
        COLOR = sv.ColorPalette.from_hex([
            "#ffff00", "#ff9b00", "#ff66ff", "#3399ff"
        ])
        print("✓ Color palette created")
        
        # Test annotators
        box_annotator = sv.BoxAnnotator(color=COLOR, thickness=2)
        label_annotator = sv.LabelAnnotator(color=COLOR, text_color=sv.Color.BLACK)
        print("✓ Annotators created")
        
        # Test InferenceSlicer callback (dummy)
        def dummy_callback(image_slice: np.ndarray):
            # Return empty detections for testing
            return sv.Detections.empty()
        
        slicer = sv.InferenceSlicer(
            callback=dummy_callback, 
            slice_wh=(640, 640)
        )
        print("✓ InferenceSlicer created")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing supervision features: {e}")
        return False

def test_tracker():
    """Test tracker functionality"""
    print("\nTesting tracker...")
    
    try:
        from trackers import SORTTracker
        import supervision as sv
        
        tracker = SORTTracker()
        print("✓ SORTTracker initialized")
        
        # Test with empty detections
        empty_detections = sv.Detections.empty()
        tracked = tracker.update(empty_detections)
        print("✓ Tracker update test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Error testing tracker: {e}")
        return False

def test_video_files():
    """Check for available video files"""
    print("\nChecking for video files...")
    
    video_files = [
        "anduril_swarm.mp4",
        "anduril_swarm_detections.mp4",
        "synthetic_drone_swarm_sim/infer_vids/anduril_swarm.mp4"
    ]
    
    found_videos = []
    for video_file in video_files:
        if os.path.exists(video_file):
            found_videos.append(video_file)
            print(f"✓ Found: {video_file}")
    
    if not found_videos:
        print("! No video files found for testing")
        return False
    
    return True

def print_usage_examples():
    """Print usage examples for the new scripts"""
    print("\n" + "="*60)
    print("USAGE EXAMPLES")
    print("="*60)
    
    print("\n1. Advanced Video Inference:")
    print("   python advanced_video_inference.py -i anduril_swarm.mp4")
    print("   python advanced_video_inference.py -i video.mp4 --slice-size 512 --compress")
    
    print("\n2. Advanced Training:")
    print("   python advanced_train.py --data ./uav-detection-3/data.yaml")
    print("   python advanced_train.py --mode val --weights ./runs/detect/train/weights/best.pt")
    
    print("\n3. Data Preparation:")
    print("   python prepare_synthetic_data.py --skip-download")
    print("   python prepare_synthetic_data.py --api-key YOUR_API_KEY")
    
    print("\n4. Single Frame Testing:")
    print("   python advanced_video_inference.py -i video.mp4 --single-frame --frame-number 100 --use-slicer")

def main():
    print("YOLOv11 Advanced Features Test")
    print("="*50)
    
    all_tests_passed = True
    
    # Run tests
    tests = [
        ("Import Test", test_imports),
        ("Model Loading Test", test_model_loading),
        ("Supervision Features Test", test_supervision_features),
        ("Tracker Test", test_tracker),
        ("Video Files Check", test_video_files)
    ]
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        try:
            result = test_func()
            if not result:
                all_tests_passed = False
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            all_tests_passed = False
    
    # Summary
    print("\n" + "="*50)
    if all_tests_passed:
        print("🎉 All tests passed! Advanced features are ready to use.")
        print_usage_examples()
        return 0
    else:
        print("❌ Some tests failed. Please check the requirements and setup.")
        print("\nTo install missing dependencies:")
        print("pip install -r requirements.txt")
        return 1

if __name__ == "__main__":
    sys.exit(main())
