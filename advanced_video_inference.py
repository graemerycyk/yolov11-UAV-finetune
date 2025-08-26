#!/usr/bin/env python3
"""
Advanced YOLOv11 UAV Detection with Supervision Library
Features:
- InferenceSlicer for better detection of small objects
- Object tracking using SORTTracker
- Enhanced video processing pipeline
- Color-coded detections with tracking IDs
"""

import argparse
import sys
import os
from pathlib import Path
import numpy as np
import supervision as sv
from ultralytics import YOLO
from trackers import SORTTracker

def load_model(model_path):
    """Load YOLOv11 model"""
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        sys.exit(1)
    
    model = YOLO(model_path)
    print(f"Model loaded: {model_path}")
    return model

def setup_annotators():
    """Setup supervision annotators with custom color palette"""
    COLOR = sv.ColorPalette.from_hex([
        "#ffff00", "#ff9b00", "#ff66ff", "#3399ff", "#ff66b2", "#ff8080",
        "#b266ff", "#9999ff", "#66ffff", "#33ff99", "#66ff66", "#99ff00"
    ])
    
    box_annotator = sv.BoxAnnotator(
        color=COLOR, 
        thickness=2, 
        color_lookup=sv.ColorLookup.TRACK
    )
    
    label_annotator = sv.LabelAnnotator(
        color=COLOR, 
        text_color=sv.Color.BLACK, 
        color_lookup=sv.ColorLookup.TRACK
    )
    
    return box_annotator, label_annotator

def create_slicer_callback(model, conf_threshold=0.10):
    """Create callback function for InferenceSlicer"""
    def slicer_callback(image_slice: np.ndarray) -> sv.Detections:
        result = model(image_slice, conf=conf_threshold, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)
    
    return slicer_callback

def process_video_advanced(args):
    """Process video with advanced features: slicing and tracking"""
    
    # Validate input
    if not os.path.exists(args.input):
        print(f"Error: Input video '{args.input}' not found!")
        return 1
    
    # Load model
    model = load_model(args.model)
    
    # Setup annotators
    box_annotator, label_annotator = setup_annotators()
    
    # Setup tracker
    tracker = SORTTracker()
    
    # Setup InferenceSlicer
    slicer_callback = create_slicer_callback(model, args.confidence)
    slicer = sv.InferenceSlicer(
        callback=slicer_callback, 
        slice_wh=(args.slice_size, args.slice_size),
        iou_threshold=args.iou
    )
    
    # Setup paths
    src_path = Path(args.input)
    if args.output:
        target_path = Path(args.output)
    else:
        target_path = src_path.parent / f"{src_path.stem}-advanced-result{src_path.suffix}"
    
    target_path_compressed = target_path.parent / f"{target_path.stem}-compressed{target_path.suffix}"
    
    print(f"Input: {src_path}")
    print(f"Output: {target_path}")
    print(f"Compressed: {target_path_compressed}")
    print(f"Slice size: {args.slice_size}x{args.slice_size}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"IoU threshold: {args.iou}")
    print(f"Max detection area: {args.max_area}")
    
    def callback(frame: np.ndarray, frame_index: int) -> np.ndarray:
        """Frame processing callback with slicing and tracking"""
        
        # Run inference with slicing
        detections = slicer(frame)
        
        # Filter detections by area (remove very large detections)
        if args.max_area > 0:
            detections = detections[detections.area < args.max_area]
        
        # Update tracker
        detections = tracker.update(detections)
        
        # Skip annotation if no detections
        if len(detections) == 0:
            return frame
        
        # Create labels with class name, confidence, and track ID
        labels = []
        for i in range(len(detections)):
            class_name = detections['class_name'][i] if 'class_name' in detections.data else f"class_{detections.class_id[i]}"
            confidence = detections.confidence[i]
            track_id = detections.tracker_id[i] if detections.tracker_id is not None else "N/A"
            labels.append(f"{class_name} {confidence:.2f} ID:{track_id}")
        
        # Annotate frame
        annotated_frame = frame.copy()
        annotated_frame = box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels
        )
        
        return annotated_frame
    
    # Process video
    print("\nProcessing video with advanced pipeline...")
    print("Features: InferenceSlicer + Object Tracking")
    
    try:
        sv.process_video(
            source_path=str(src_path),
            target_path=str(target_path),
            callback=callback
        )
        
        print(f"\nVideo processing completed!")
        print(f"Output saved to: {target_path}")
        
        # Compress video if requested
        if args.compress:
            print(f"\nCompressing video...")
            os.system(f'ffmpeg -y -i "{target_path}" -vcodec libx264 -crf 28 "{target_path_compressed}"')
            print(f"Compressed video saved to: {target_path_compressed}")
        
        return 0
        
    except Exception as e:
        print(f"Error during video processing: {e}")
        return 1

def process_single_frame(args):
    """Process a single frame for testing"""
    
    # Load model
    model = load_model(args.model)
    
    # Setup annotators
    box_annotator, label_annotator = setup_annotators()
    
    # Load video and get frame
    frame_generator = sv.get_video_frames_generator(
        args.input, 
        start=args.frame_number, 
        iterative_seek=True
    )
    frame = next(frame_generator)
    
    print(f"Processing frame {args.frame_number} from {args.input}")
    
    if args.use_slicer:
        # Use InferenceSlicer
        slicer_callback = create_slicer_callback(model, args.confidence)
        slicer = sv.InferenceSlicer(
            callback=slicer_callback, 
            slice_wh=(args.slice_size, args.slice_size),
            iou_threshold=args.iou
        )
        detections = slicer(frame)
        print(f"Detections with slicer: {len(detections)}")
    else:
        # Direct inference
        result = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        print(f"Detections without slicer: {len(detections)}")
    
    # Create labels
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections['class_name'], detections.confidence)
    ]
    
    # Annotate frame
    annotated_frame = frame.copy()
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame,
        detections=detections
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels
    )
    
    # Save annotated frame
    output_path = f"frame_{args.frame_number}_annotated.jpg"
    sv.cv2.imwrite(output_path, annotated_frame)
    print(f"Annotated frame saved to: {output_path}")
    
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Advanced YOLOv11 UAV Detection with Supervision Library",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process full video with advanced pipeline
  python advanced_video_inference.py -i video.mp4 -m yolo11l-UAV-finetune.pt
  
  # Process with custom slice size and compression
  python advanced_video_inference.py -i video.mp4 --slice-size 512 --compress
  
  # Test single frame with slicer
  python advanced_video_inference.py -i video.mp4 --single-frame --frame-number 100 --use-slicer
        """
    )
    
    # Input/Output arguments
    parser.add_argument('--input', '-i', required=True,
                       help='Input video file path')
    parser.add_argument('--output', '-o',
                       help='Output video file path (auto-generated if not provided)')
    parser.add_argument('--model', '-m', default='yolo11l-UAV-finetune.pt',
                       help='Model file path (default: yolo11l-UAV-finetune.pt)')
    
    # Processing options
    parser.add_argument('--single-frame', action='store_true',
                       help='Process single frame instead of full video')
    parser.add_argument('--frame-number', type=int, default=0,
                       help='Frame number to process (for single-frame mode)')
    parser.add_argument('--use-slicer', action='store_true',
                       help='Use InferenceSlicer for single-frame processing')
    
    # Model parameters
    parser.add_argument('--confidence', '-c', type=float, default=0.10,
                       help='Confidence threshold (default: 0.10)')
    parser.add_argument('--iou', type=float, default=0.3,
                       help='IoU threshold for NMS (default: 0.3)')
    parser.add_argument('--slice-size', type=int, default=640,
                       help='Slice size for InferenceSlicer (default: 640)')
    parser.add_argument('--max-area', type=int, default=2000,
                       help='Maximum detection area (0 to disable, default: 2000)')
    
    # Output options
    parser.add_argument('--compress', action='store_true',
                       help='Create compressed version of output video')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.confidence < 0.0 or args.confidence > 1.0:
        print("Error: Confidence threshold must be between 0.0 and 1.0")
        return 1
    
    if args.iou < 0.0 or args.iou > 1.0:
        print("Error: IoU threshold must be between 0.0 and 1.0")
        return 1
    
    if args.slice_size < 64:
        print("Error: Slice size must be at least 64")
        return 1
    
    # Process based on mode
    if args.single_frame:
        return process_single_frame(args)
    else:
        return process_video_advanced(args)

if __name__ == "__main__":
    sys.exit(main())
