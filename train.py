from ultralytics import YOLO

# Load the pre-trained YOLO model (using larger model for better performance)
model = YOLO("yolo11l.pt")  # Using yolo11l instead of yolo11n for better accuracy

# Train the model with improved parameters from Colab notebook
train_results = model.train(
    data="/workspace/UAV-3/data.yaml",  # Path to dataset configuration file
    epochs=100,  # Number of training epochs
    imgsz=640,  # Image size for training
    batch=64,  # Increased batch size for better training stability
    workers=6,  # Number of workers for data loading
    optimizer="Adam",  # Adam optimizer for better convergence
    lr0=3e-4,  # Learning rate optimized for UAV detection
    device="0",  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
)