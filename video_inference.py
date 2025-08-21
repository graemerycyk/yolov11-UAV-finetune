import onnxruntime
import numpy as np
import cv2
import json
import argparse
import sys
import os
from pathlib import Path

def load_config():
    with open('config.json', 'r') as f:
        config = json.load(f)
    with open('preprocessor.json', 'r') as f:
        preprocess_config = json.load(f)
    return config, preprocess_config

def preprocess_frame(frame, preprocess_config):
    """Preprocess video frame with proper handling for Instagram story aspect ratio"""
    # Convert BGR to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get preprocessing parameters
    target_size = preprocess_config['pad_size']  # Should be 640
    rescale_factor = preprocess_config['rescale_factor']
    
    # Resize maintaining aspect ratio
    height, width = img.shape[:2]
    scale = min(target_size / width, target_size / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    resized = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    # Create padded image (center padding)
    padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    dx = (target_size - new_width) // 2
    dy = (target_size - new_height) // 2
    padded[dy:dy+new_height, dx:dx+new_width] = resized
    
    # Convert to float32 and normalize
    input_tensor = padded.astype(np.float32) * rescale_factor
    
    # Transpose from HWC to CHW format and add batch dimension
    input_tensor = np.transpose(input_tensor, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, 0)
    
    return input_tensor, (scale, (dx, dy))

def postprocess_output(output, frame_shape, preprocessing_info, conf_threshold=0.3, iou_threshold=0.45):
    """Postprocess model output for 3-class fine-tuned model"""
    scale, (dx, dy) = preprocessing_info
    
    # Custom class mapping for fine-tuned model
    class_names = {
        0: "dj-air3",
        1: "uav", 
        2: "UAV"
    }
    
    # Output format: [batch, num_classes + 4, num_boxes]
    output = output[0]  # Remove batch dimension: [7, 8400]
    output = output.transpose()  # Convert to [8400, 7]
    
    # Extract boxes and scores
    boxes = output[:, :4]  # x_center, y_center, width, height
    class_scores = output[:, 4:]  # 3 class scores
    
    # Convert from center format to corner format
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    
    # Get best class for each detection
    best_class_ids = np.argmax(class_scores, axis=1)
    best_scores = np.max(class_scores, axis=1)
    
    # Filter based on confidence threshold
    conf_mask = best_scores > conf_threshold
    boxes = boxes[conf_mask]
    best_scores = best_scores[conf_mask]
    best_class_ids = best_class_ids[conf_mask]
    
    if len(boxes) == 0:
        return []
    
    # Scale back to original frame
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dx) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dy) / scale
    
    # Clamp boxes to frame boundaries
    boxes[:, 0] = np.clip(boxes[:, 0], 0, frame_shape[1])  # x1
    boxes[:, 1] = np.clip(boxes[:, 1], 0, frame_shape[0])  # y1
    boxes[:, 2] = np.clip(boxes[:, 2], 0, frame_shape[1])  # x2
    boxes[:, 3] = np.clip(boxes[:, 3], 0, frame_shape[0])  # y2
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), best_scores.tolist(), conf_threshold, iou_threshold)
    
    results = []
    
    if len(indices) > 0:
        for idx in indices:
            box = boxes[idx]
            score = best_scores[idx]
            class_id = best_class_ids[idx]
            label = class_names[class_id]
            
            results.append({
                'box': box.tolist(),
                'score': float(score),
                'label': label
            })
    
    return results

def draw_results(frame, results):
    """Draw bounding boxes and labels on frame"""
    # Define colors for each class
    colors = {
        "dj-air3": (0, 255, 0),      # Green
        "uav": (255, 0, 0),          # Blue  
        "UAV": (0, 0, 255)           # Red
    }
    
    for result in results:
        box = result['box']
        label = f"{result['label']} {result['score']:.2f}"
        color = colors.get(result['label'], (255, 255, 255))
        
        x1, y1, x2, y2 = map(int, box)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw background rectangle for text
        cv2.rectangle(frame, (x1, y1 - text_height - baseline), (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1, y1 - baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    return frame

def process_video(args):
    """Main video processing function with CLI arguments"""
    # Validate input file
    if not os.path.exists(args.input):
        print(f"Error: Input video '{args.input}' not found!")
        return 1
    
    # Validate model file
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        return 1
    
    # Load configurations
    config, preprocess_config = load_config()
    
    # Set up execution providers
    providers = ['CPUExecutionProvider']
    if args.device == 'gpu':
        providers.insert(0, 'CUDAExecutionProvider')
    
    # Initialize ONNX Runtime session with fine-tuned model
    session = onnxruntime.InferenceSession(args.model, providers=providers)
    input_name = session.get_inputs()[0].name
    
    print(f"Using model: {args.model}")
    print(f"Execution providers: {session.get_providers()}")
    print("Model classes: dj-air3, uav, UAV")
    
    # Open input video
    cap = cv2.VideoCapture(args.input)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file '{args.input}'")
        return 1
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video: {args.input}")
    print(f"Video properties: {width}x{height} @ {fps}fps, {total_frames} frames")
    print(f"Aspect ratio: {width/height:.2f}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"IoU threshold: {args.iou}")
    
    # Generate output path if not provided
    if not args.output:
        input_path = Path(args.input)
        output_path = str(input_path.parent / f"{input_path.stem}_detections{input_path.suffix}")
    else:
        output_path = args.output
    
    # Setup output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    detections_count = 0
    
    print(f"\nProcessing video to: {output_path}")
    print("Press Ctrl+C to stop processing")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Preprocess frame
            input_tensor, preprocessing_info = preprocess_frame(frame, preprocess_config)
            
            # Run inference
            outputs = session.run(None, {input_name: input_tensor})
            output = outputs[0]
            
            # Post-process results
            results = postprocess_output(output, frame.shape, preprocessing_info, 
                                       conf_threshold=args.confidence, iou_threshold=args.iou)
            
            if results:
                detections_count += len(results)
            
            # Draw results on frame
            frame_with_detections = draw_results(frame, results)
            
            # Write frame to output video
            out.write(frame_with_detections)
            
            # Progress indicator
            if frame_count % 30 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
    
    # Clean up
    cap.release()
    out.release()
    
    print(f"\nProcessing complete!")
    print(f"Processed {frame_count} frames")
    print(f"Total detections: {detections_count}")
    print(f"Output saved to: {output_path}")
    
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="YOLOv11 UAV Detection - Process videos to detect UAVs/drones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python video_inference.py -i my_video.mp4
  
  # Custom output path and confidence threshold
  python video_inference.py -i input.mp4 -o output_detected.mp4 -c 0.5
  
  # Use GPU acceleration
  python video_inference.py -i video.mp4 --device gpu
        """
    )
    
    # Arguments (input defaults to original hardcoded video for backward compatibility)
    parser.add_argument('--input', '-i', default='anduril_swarm.mp4',
                       help='Input video file path (default: anduril_swarm.mp4)')
    
    # Other optional arguments
    parser.add_argument('--output', '-o', 
                       help='Output video file path (auto-generated if not provided)')
    
    parser.add_argument('--model', '-m', default='yolov11n-UAV-finetune.onnx',
                       help='Model file path (default: yolov11n-UAV-finetune.onnx)')
    
    parser.add_argument('--confidence', '-c', type=float, default=0.3,
                       help='Confidence threshold for detections (default: 0.3)')
    
    parser.add_argument('--iou', type=float, default=0.45,
                       help='IoU threshold for Non-Maximum Suppression (default: 0.45)')
    
    parser.add_argument('--device', choices=['cpu', 'gpu'], default='cpu',
                       help='Execution device: cpu or gpu (default: cpu)')
    

    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if args.confidence < 0.0 or args.confidence > 1.0:
        print("Error: Confidence threshold must be between 0.0 and 1.0")
        return 1
    
    if args.iou < 0.0 or args.iou > 1.0:
        print("Error: IoU threshold must be between 0.0 and 1.0")
        return 1
    
    # Process video
    return process_video(args)

if __name__ == "__main__":
    sys.exit(main())
