#!/usr/bin/env python3
"""
Advanced YOLOv11 Training Script
Based on the optimized parameters from the Colab notebook
"""

import argparse
import os
import sys
from pathlib import Path
from ultralytics import YOLO

def train_model(args):
    """Train YOLOv11 model with advanced parameters"""
    
    # Validate data file
    if not os.path.exists(args.data):
        print(f"Error: Data configuration file not found: {args.data}")
        return 1
    
    # Load the pre-trained YOLO model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    
    print("Training Configuration:")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Data: {args.data}")
    print(f"Epochs: {args.epochs}")
    print(f"Image size: {args.imgsz}")
    print(f"Batch size: {args.batch}")
    print(f"Workers: {args.workers}")
    print(f"Optimizer: {args.optimizer}")
    print(f"Learning rate: {args.lr0}")
    print(f"Device: {args.device}")
    print(f"Project: {args.project}")
    print(f"Name: {args.name}")
    
    # Training parameters optimized from Colab notebook
    train_params = {
        'data': args.data,
        'epochs': args.epochs,
        'imgsz': args.imgsz,
        'batch': args.batch,
        'workers': args.workers,
        'optimizer': args.optimizer,
        'lr0': args.lr0,
        'device': args.device,
        'project': args.project,
        'name': args.name,
        'save': True,
        'save_period': args.save_period,
        'plots': True,
        'verbose': True
    }
    
    # Add additional parameters if specified
    if args.patience:
        train_params['patience'] = args.patience
    
    if args.resume:
        train_params['resume'] = args.resume
    
    try:
        print("\nStarting training...")
        train_results = model.train(**train_params)
        
        print("\nTraining completed successfully!")
        print(f"Results saved to: {train_results.save_dir}")
        
        # Print training summary
        if hasattr(train_results, 'results_dict'):
            results = train_results.results_dict
            print("\nTraining Summary:")
            print("-" * 30)
            for key, value in results.items():
                print(f"{key}: {value}")
        
        return 0
        
    except Exception as e:
        print(f"Error during training: {e}")
        return 1

def validate_model(args):
    """Validate trained model"""
    
    # Find the best weights
    if args.weights == "auto":
        # Look for best.pt in the training results
        project_path = Path(args.project) / args.name
        weights_path = project_path / "weights" / "best.pt"
        
        if not weights_path.exists():
            print(f"Error: Cannot find best.pt at {weights_path}")
            print("Please specify weights path manually with --weights")
            return 1
        
        args.weights = str(weights_path)
    
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        return 1
    
    print(f"Validating model: {args.weights}")
    
    model = YOLO(args.weights)
    
    # Validation parameters
    val_params = {
        'data': args.data,
        'split': 'val',
        'imgsz': args.imgsz,
        'batch': args.batch,
        'device': args.device,
        'plots': True
    }
    
    try:
        metrics = model.val(**val_params)
        
        print("\nValidation Results:")
        print("=" * 50)
        print(f"mAP50-95: {metrics.box.map:.4f}")
        print(f"mAP50: {metrics.box.map50:.4f}")
        print(f"mAP75: {metrics.box.map75:.4f}")
        
        if hasattr(metrics.box, 'maps') and metrics.box.maps is not None:
            print("\nPer-class mAPs:")
            for i, map_val in enumerate(metrics.box.maps):
                print(f"  Class {i}: {map_val:.4f}")
        
        return 0
        
    except Exception as e:
        print(f"Error during validation: {e}")
        return 1

def export_model(args):
    """Export model to different formats"""
    
    if not os.path.exists(args.weights):
        print(f"Error: Weights file not found: {args.weights}")
        return 1
    
    print(f"Exporting model: {args.weights}")
    print(f"Format: {args.format}")
    
    model = YOLO(args.weights)
    
    try:
        export_path = model.export(format=args.format, imgsz=args.imgsz)
        print(f"Model exported successfully to: {export_path}")
        return 0
        
    except Exception as e:
        print(f"Error during export: {e}")
        return 1

def main():
    parser = argparse.ArgumentParser(
        description="Advanced YOLOv11 Training Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with default optimized parameters
  python advanced_train.py --data ./uav-detection-3/data.yaml
  
  # Train with custom parameters
  python advanced_train.py --data ./data.yaml --epochs 200 --batch 32
  
  # Validate trained model
  python advanced_train.py --mode val --weights ./runs/detect/train/weights/best.pt
  
  # Export model to ONNX
  python advanced_train.py --mode export --weights ./runs/detect/train/weights/best.pt --format onnx
        """
    )
    
    # Mode selection
    parser.add_argument('--mode', choices=['train', 'val', 'export'], default='train',
                       help='Mode: train, validate, or export (default: train)')
    
    # Data and model
    parser.add_argument('--data', default='./uav-detection-3/data.yaml',
                       help='Path to data.yaml file')
    parser.add_argument('--model', default='yolo11l.pt',
                       help='Model to use for training (default: yolo11l.pt)')
    parser.add_argument('--weights', default='auto',
                       help='Weights file for validation/export (auto finds best.pt)')
    
    # Training parameters (optimized from Colab notebook)
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs (default: 100)')
    parser.add_argument('--imgsz', type=int, default=640,
                       help='Image size (default: 640)')
    parser.add_argument('--batch', type=int, default=64,
                       help='Batch size (default: 64)')
    parser.add_argument('--workers', type=int, default=6,
                       help='Number of workers (default: 6)')
    parser.add_argument('--optimizer', default='Adam',
                       help='Optimizer (default: Adam)')
    parser.add_argument('--lr0', type=float, default=3e-4,
                       help='Initial learning rate (default: 3e-4)')
    
    # Training options
    parser.add_argument('--device', default='0',
                       help='Device to use (default: 0)')
    parser.add_argument('--project', default='runs/detect',
                       help='Project directory (default: runs/detect)')
    parser.add_argument('--name', default='train',
                       help='Experiment name (default: train)')
    parser.add_argument('--save-period', type=int, default=10,
                       help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--patience', type=int,
                       help='Early stopping patience (epochs)')
    parser.add_argument('--resume', 
                       help='Resume training from checkpoint')
    
    # Export options
    parser.add_argument('--format', default='onnx',
                       help='Export format (default: onnx)')
    
    args = parser.parse_args()
    
    # Execute based on mode
    if args.mode == 'train':
        return train_model(args)
    elif args.mode == 'val':
        return validate_model(args)
    elif args.mode == 'export':
        return export_model(args)
    else:
        print(f"Unknown mode: {args.mode}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
