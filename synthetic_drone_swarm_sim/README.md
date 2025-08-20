# Synthetic Swarm Drone Simulation

## Preview (Simulated Synthetic Data)

<div align="center">
  <img src="assets/sim_preview1.gif" alt="Drone Swarm Simulation Preview 1" width="400"/>
  <img src="assets/sim_preview2.gif" alt="Drone Swarm Simulation Preview 2" width="400"/>
  <img src="assets/sim_preview3.gif" alt="Drone Swarm Simulation Preview 3" width="400"/>
  <img src="assets/sim_preview4.gif" alt="Drone Swarm Simulation Preview 3" width="400"/>
</div>

A Python simulation that creates realistic drone swarm videos by overlaying multiple drones with smooth trajectories onto a background video using alpha blending. Now includes ground truth generation for machine learning training datasets.

## Output from Object Detection Model (trained on Simulated Synthetic Data)
<div align="center">
  <img src="assets/anduril_infer.gif" alt="Drone Swarm Simulation Preview 1" width="400"/>
  <img src="assets/bg_terrain1_infer.gif" alt="Drone Swarm Simulation Preview 2" width="400"/>
  <!-- add video -->
</div>

## Features

- **Smooth Trajectories**: Uses linear motion with fast movement for realistic drone flight paths
- **Alpha Blending**: Properly blends drone images with transparency onto background video
- **Dynamic Scaling**: Varies drone sizes to simulate depth and distance
- **Color Variations**: Applies random color tinting and transformations to drones
- **Ground Truth Generation**: Creates YOLO-compatible bounding box annotations and annotated videos
- **Configurable**: Adjustable number of drones, input files, and output settings

## Requirements

- Python 3.7+
- OpenCV
- NumPy
- tqdm (for progress bars)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your media files:
   - Place your background video in `assets/background.mp4`
   - Place your drone PNG image (with transparency) in `assets/drone.png`

## Project Structure

```
sawrm_drones_sim/
├── assets/
│   ├── background.mp4    # Background video file
│   └── drone.png         # Drone image with transparency
├── outputs/              # Generated videos and annotations
│   ├── output_swarm.mp4  # Main simulation video
│   ├── bbox_*.mp4        # Annotated videos (if --gen_groundtruth used)
│   └── labels/           # YOLO format annotations (if --gen_groundtruth used)
│       ├── 000000.txt    # Frame 0 annotations
│       ├── 000001.txt    # Frame 1 annotations
│       └── ...
├── sim.py                # Main simulation script
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Usage

### Basic Usage
```bash
python sim.py
```
This creates `outputs/output_swarm.mp4` with 10 drones using default assets.

### Generate Ground Truth Data (for ML training)
```bash
python sim.py --gen_groundtruth
```
This creates:
- `outputs/output_swarm.mp4` - Main simulation video
- `outputs/bbox_output_swarm.mp4` - Video with green bounding boxes around drones
- `outputs/labels/` - Directory with YOLO format annotation files

### Advanced Usage
```bash
python sim.py --background assets/my_sky.mp4 --drone assets/my_drone.png --num-drones 15 --output outputs/my_swarm.mp4 --gen_groundtruth
```

### Parameters
- `--background`: Background video file (default: assets/background.mp4)
- `--drone`: Drone PNG image file with transparency (default: assets/drone.png)
- `--num-drones`: Number of drones in swarm (default: 10)
- `--output`: Output video filename (default: outputs/output_swarm.mp4)
- `--gen_groundtruth`: Generate YOLO-compatible ground truth annotations and annotated video

### Ground Truth Output Format

When `--gen_groundtruth` is used, the following files are generated:

1. **YOLO Annotations** (`outputs/labels/*.txt`): Each frame gets a corresponding text file with bounding box annotations in YOLO format:
   ```
   class_id x_center y_center width height
   ```
   - `class_id`: Always 0 (representing UAV/drone)
   - All coordinates are normalized (0.0 to 1.0)
   - Example: `0 0.523456 0.341234 0.084567 0.123789`

2. **Annotated Video** (`outputs/bbox_<video_name>.mp4`): A copy of the simulation with green bounding boxes and "UAV" labels drawn around each drone.

## Input Requirements

### Background Video (assets/background.mp4)
- Format: MP4
- Content: Empty scene with sky or landscape
- Duration: 5-30 seconds recommended
- Resolution: Any (720p or 1080p recommended)

### Drone Image (assets/drone.png)
- Format: PNG with alpha channel (transparency)
- Content: Side or angled view of a drone
- Resolution: 200x200 to 500x500 pixels recommended
- Background: Must be transparent

## How It Works

1. **Trajectory Generation**: Each drone gets a unique linear trajectory from edge to edge
2. **Visual Effects**: Applies random scaling, rotation, color tinting, and alpha blending
3. **Continuous Spawning**: New drones spawn randomly to maintain swarm density
4. **Video Processing**: Overlays drones frame-by-frame onto the background video
5. **Ground Truth Generation**: Calculates bounding boxes and exports YOLO annotations (optional)

## Example Output

The simulation creates:
- **Main Video**: Multiple drones with fast, linear movement across the scene
- **Annotated Video** (with `--gen_groundtruth`): Same video with green bounding boxes
- **YOLO Annotations** (with `--gen_groundtruth`): Machine learning training labels

## Machine Learning Integration

The ground truth generation feature makes this simulator perfect for:
- Training YOLO object detection models
- Creating synthetic datasets for drone detection
- Testing computer vision algorithms
- Benchmarking detection performance

## Tips for Best Results

- Use a high-contrast drone image with clear silhouette
- Ensure background video has minimal motion for better detection training
- Higher drone counts (15-25) create more training examples per frame
- Use `--gen_groundtruth` to create ML training datasets
- The YOLO format is compatible with most modern object detection frameworks
