#!/usr/bin/env python3
"""
Prepare synthetic drone swarm data for training
Based on the workflow from the Colab notebook
"""

import os
import shutil
import argparse
from pathlib import Path
from roboflow import Roboflow

def download_roboflow_dataset(api_key, workspace="uavdetection-msr99", project="uav-detection-blxxz", version=3):
    """Download dataset from Roboflow"""
    print(f"Downloading dataset from Roboflow...")
    print(f"Workspace: {workspace}")
    print(f"Project: {project}")
    print(f"Version: {version}")
    
    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    version_obj = project_obj.version(version)
    dataset = version_obj.download("yolov11")
    
    print(f"Dataset downloaded successfully!")
    return dataset

def copy_synthetic_data(synthetic_images_dir, synthetic_labels_dir, target_images_dir, target_labels_dir):
    """Copy synthetic simulation data to the training dataset"""
    
    # Create target directories if they don't exist
    os.makedirs(target_images_dir, exist_ok=True)
    os.makedirs(target_labels_dir, exist_ok=True)
    
    # Check if source directories exist
    if not os.path.exists(synthetic_images_dir):
        print(f"Warning: Synthetic images directory not found: {synthetic_images_dir}")
        return False
        
    if not os.path.exists(synthetic_labels_dir):
        print(f"Warning: Synthetic labels directory not found: {synthetic_labels_dir}")
        return False
    
    # Copy images
    print(f"Copying images from {synthetic_images_dir} to {target_images_dir}")
    image_files = list(Path(synthetic_images_dir).glob("*.jpg"))
    for img_file in image_files:
        shutil.copy2(img_file, target_images_dir)
    print(f"Copied {len(image_files)} images")
    
    # Copy labels
    print(f"Copying labels from {synthetic_labels_dir} to {target_labels_dir}")
    label_files = list(Path(synthetic_labels_dir).glob("*.txt"))
    for label_file in label_files:
        shutil.copy2(label_file, target_labels_dir)
    print(f"Copied {len(label_files)} labels")
    
    return True

def prepare_dataset(args):
    """Main function to prepare the dataset"""
    
    # Set up paths
    synthetic_base = Path(args.synthetic_dir)
    synthetic_images_dir = synthetic_base / "images"
    synthetic_labels_dir = synthetic_base / "labels"
    
    dataset_base = Path(args.dataset_dir)
    target_images_dir = dataset_base / "train" / "images"
    target_labels_dir = dataset_base / "train" / "labels"
    
    print("Dataset Preparation")
    print("=" * 50)
    print(f"Synthetic data source: {synthetic_base}")
    print(f"Target dataset: {dataset_base}")
    
    # Download Roboflow dataset if API key provided
    if args.api_key:
        try:
            dataset = download_roboflow_dataset(
                args.api_key, 
                args.workspace, 
                args.project, 
                args.version
            )
        except Exception as e:
            print(f"Error downloading Roboflow dataset: {e}")
            if not args.skip_download:
                return 1
    
    # Copy synthetic data
    success = copy_synthetic_data(
        str(synthetic_images_dir),
        str(synthetic_labels_dir),
        str(target_images_dir),
        str(target_labels_dir)
    )
    
    if not success:
        print("Failed to copy synthetic data")
        return 1
    
    # Print summary
    print("\nDataset preparation completed!")
    print(f"Images directory: {target_images_dir}")
    print(f"Labels directory: {target_labels_dir}")
    
    # Count files
    image_count = len(list(target_images_dir.glob("*.jpg")))
    label_count = len(list(target_labels_dir.glob("*.txt")))
    
    print(f"Total images: {image_count}")
    print(f"Total labels: {label_count}")
    
    if image_count != label_count:
        print("Warning: Number of images and labels don't match!")
    
    return 0

def main():
    parser = argparse.ArgumentParser(
        description="Prepare synthetic drone swarm data for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Prepare dataset with Roboflow download
  python prepare_synthetic_data.py --api-key YOUR_API_KEY
  
  # Prepare dataset without Roboflow (skip download)
  python prepare_synthetic_data.py --skip-download
  
  # Custom paths
  python prepare_synthetic_data.py --synthetic-dir ./outputs/my_swarm --dataset-dir ./uav-detection-3
        """
    )
    
    # Required arguments
    parser.add_argument('--synthetic-dir', default='./synthetic_drone_swarm_sim/outputs/my_swarm',
                       help='Directory containing synthetic images and labels (default: ./synthetic_drone_swarm_sim/outputs/my_swarm)')
    parser.add_argument('--dataset-dir', default='./uav-detection-3',
                       help='Target dataset directory (default: ./uav-detection-3)')
    
    # Roboflow arguments
    parser.add_argument('--api-key', 
                       help='Roboflow API key (required for dataset download)')
    parser.add_argument('--workspace', default='uavdetection-msr99',
                       help='Roboflow workspace (default: uavdetection-msr99)')
    parser.add_argument('--project', default='uav-detection-blxxz',
                       help='Roboflow project (default: uav-detection-blxxz)')
    parser.add_argument('--version', type=int, default=3,
                       help='Roboflow dataset version (default: 3)')
    parser.add_argument('--skip-download', action='store_true',
                       help='Skip Roboflow dataset download')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.api_key and not args.skip_download:
        print("Error: Either provide --api-key or use --skip-download")
        return 1
    
    return prepare_dataset(args)

if __name__ == "__main__":
    import sys
    sys.exit(main())
