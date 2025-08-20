#!/usr/bin/env python3
"""
Drone Swarm Simulation
======================
Overlays N drones with smooth trajectories onto a background video using alpha blending.
"""

import cv2
import numpy as np
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import os


class DroneTrajectory:
    """Generates fast, smooth drone trajectories with linear motion."""
    
    def __init__(self, start_pos, end_pos, speed_pixels_per_frame):
        self.start_pos = np.array(start_pos, dtype=float)
        self.end_pos = np.array(end_pos, dtype=float)
        
        # Calculate duration based on distance and speed
        distance = np.linalg.norm(self.end_pos - self.start_pos)
        self.duration_frames = max(1, int(distance / speed_pixels_per_frame))
        
        # Direction vector for smooth linear motion
        self.direction = (self.end_pos - self.start_pos) / self.duration_frames
    
    def get_position(self, frame):
        """Get drone position at given frame using linear motion."""
        if frame >= self.duration_frames:
            return self.end_pos
        
        # Simple linear interpolation for fast, smooth motion
        return self.start_pos + (self.direction * frame)
    
    def is_finished(self, frame):
        """Check if drone has reached its destination."""
        return frame >= self.duration_frames


class DroneSwarmSimulator:
    """Main simulator class that handles video processing and drone overlay."""
    
    def __init__(self, background_video, drone_image, num_drones=10, gen_groundtruth=False):
        self.background_video = background_video
        self.drone_image_path = drone_image
        self.num_drones = num_drones
        self.gen_groundtruth = gen_groundtruth
        
        # Load and prepare multiple drone images
        self.drone_images = []
        drone_paths = []
        
        # Find all drone images in the assets directory
        assets_dir = Path(drone_image).parent
        for drone_file in assets_dir.glob("drone*.png"):
            drone_paths.append(str(drone_file))
        
        # Sort to ensure consistent ordering
        drone_paths.sort()
        
        # If no drone*.png files found, fall back to the specified drone_image
        if not drone_paths:
            drone_paths = [drone_image]
        
        # Load all drone images
        for drone_path in drone_paths:
            drone_img = cv2.imread(drone_path, cv2.IMREAD_UNCHANGED)
            if drone_img is None:
                print(f"Warning: Could not load drone image: {drone_path}")
                continue
            
            # Ensure drone image has alpha channel
            if drone_img.shape[2] == 3:
                # Add alpha channel if missing
                alpha = np.ones((drone_img.shape[0], drone_img.shape[1], 1), dtype=drone_img.dtype) * 255
                drone_img = np.concatenate([drone_img, alpha], axis=2)
            
            self.drone_images.append(drone_img)
        
        if not self.drone_images:
            raise ValueError(f"Could not load any drone images from: {drone_paths}")
        
        print(f"Loaded {len(self.drone_images)} drone models: {[Path(p).name for p in drone_paths]}")
        
        # Keep reference to first drone image for backward compatibility
        self.drone_img = self.drone_images[0]
        
        # Load video
        self.cap = cv2.VideoCapture(background_video)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {background_video}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames")
        
        # Initialize drone list for continuous stream
        self.drones = []
        
        # Create initial drones
        for _ in range(self.num_drones):
            self.drones.append(self._create_new_drone(0))
    
    def _generate_random_edge_position(self, side):
        """Generate a random position on the specified edge."""
        if side == 'left':
            return [0, random.randint(0, self.height)]
        elif side == 'right':
            return [self.width, random.randint(0, self.height)]
        elif side == 'top':
            return [random.randint(0, self.width), 0]
        else:  # bottom
            return [random.randint(0, self.width), self.height]
    
    def _create_new_drone(self, current_frame):
        """Create a new drone with random trajectory."""
        # Choose random start and end edges (ensure they're different)
        edges = ['left', 'right', 'top', 'bottom']
        start_side = random.choice(edges)
        end_side = random.choice([e for e in edges if e != start_side])
        
        start_pos = self._generate_random_edge_position(start_side)
        end_pos = self._generate_random_edge_position(end_side)
        
        # Fast speed for quick movement across screen
        speed = random.uniform(3, 8)  # pixels per frame
        
        trajectory = DroneTrajectory(start_pos, end_pos, speed)
        
        # Color options for drone tinting
        colors = [
            (255, 255, 255),  # White
            (0, 0, 0),        # Black
            (0, 0, 255),      # Red
            (0, 255, 0),      # Green
            (255, 0, 0),      # Blue
            (0, 255, 255),    # Yellow
            (255, 0, 255),    # Magenta
            (255, 128, 0),    # Orange
            (128, 0, 255),    # Purple
            (0, 128, 255)     # Light Blue
        ]
        
        return {
            'trajectory': trajectory,
            'start_frame': current_frame,
            'scale': random.uniform(0.015, 0.075),
            'rotation': random.uniform(0, 30),
            'flip_horizontal': random.choice([True, False]),
            'color': random.choice(colors),
            'alpha_multiplier': random.uniform(0.85, 1.0),
            'drone_image_index': random.randint(0, len(self.drone_images) - 1)
        }
    
    def _transform_drone(self, drone_img, scale, rotation, flip_horizontal, color):
        """Resize, rotate, flip, and tint drone image."""
        # Resize
        h, w = drone_img.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        resized = cv2.resize(drone_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Apply color tinting
        if resized.shape[2] == 4:  # RGBA image
            # Separate RGB and alpha channels
            rgb = resized[:, :, :3].astype(float)
            alpha = resized[:, :, 3:4]
            
            # Apply color tint to RGB channels only
            # Normalize existing colors and apply tint
            tinted_rgb = rgb * np.array(color) / 255.0
            tinted_rgb = np.clip(tinted_rgb, 0, 255).astype(np.uint8)
            
            # Recombine with alpha channel
            resized = np.concatenate([tinted_rgb, alpha], axis=2)
        
        # Flip horizontally if requested
        if flip_horizontal:
            resized = cv2.flip(resized, 1)  # 1 = horizontal flip
        
        # Rotate
        center = (new_w // 2, new_h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation, 1.0)
        transformed = cv2.warpAffine(resized, rotation_matrix, (new_w, new_h), 
                                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(0, 0, 0, 0))
        
        return transformed
    
    def _blend_drone_onto_frame(self, frame, drone_img, position, alpha_multiplier=1.0):
        """Alpha blend drone onto frame at given position."""
        drone_h, drone_w = drone_img.shape[:2]
        x, y = int(position[0] - drone_w // 2), int(position[1] - drone_h // 2)
        
        # Check bounds
        if x + drone_w <= 0 or y + drone_h <= 0 or x >= self.width or y >= self.height:
            return frame
        
        # Calculate overlapping region
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(self.width, x + drone_w)
        y2 = min(self.height, y + drone_h)
        
        # Calculate corresponding regions in drone image
        dx1 = max(0, -x)
        dy1 = max(0, -y)
        dx2 = dx1 + (x2 - x1)
        dy2 = dy1 + (y2 - y1)
        
        if x2 <= x1 or y2 <= y1:
            return frame
        
        # Extract regions
        frame_region = frame[y1:y2, x1:x2]
        drone_region = drone_img[dy1:dy2, dx1:dx2]
        
        # Alpha blending
        if drone_region.shape[2] == 4:
            alpha = drone_region[:, :, 3:4].astype(float) / 255.0 * alpha_multiplier
            drone_rgb = drone_region[:, :, :3]
            
            # Blend
            blended = frame_region * (1 - alpha) + drone_rgb * alpha
            frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        
        return frame
    
    def _get_drone_bounding_box(self, position, drone_img):
        """Calculate bounding box for a drone at given position."""
        drone_h, drone_w = drone_img.shape[:2]
        x_center = position[0]
        y_center = position[1]
        
        # Calculate bounding box corners
        x1 = max(0, x_center - drone_w // 2)
        y1 = max(0, y_center - drone_h // 2)
        x2 = min(self.width, x_center + drone_w // 2)
        y2 = min(self.height, y_center + drone_h // 2)
        
        # Only return bbox if it's visible in frame
        if x2 > x1 and y2 > y1:
            return (x1, y1, x2, y2)
        return None
    
    def _bbox_to_yolo_format(self, bbox):
        """Convert bounding box to YOLO format (class_id, x_center, y_center, width, height) normalized."""
        if bbox is None:
            return None
        
        x1, y1, x2, y2 = bbox
        x_center = (x1 + x2) / 2.0 / self.width
        y_center = (y1 + y2) / 2.0 / self.height
        width = (x2 - x1) / self.width
        height = (y2 - y1) / self.height
        
        # Class ID 0 for UAV
        return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    def _draw_bounding_box(self, frame, bbox, label="UAV"):
        """Draw green bounding box with label on frame."""
        if bbox is None:
            return frame
        
        x1, y1, x2, y2 = bbox
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        
        # Draw green bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(frame, (x1, y1 - label_size[1] - 5), (x1 + label_size[0], y1), (0, 255, 0), -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return frame
    
    def simulate(self, output_path="output_swarm.mp4"):
        """Run the simulation and save output video."""
        # Create directory structure: output_path/<vid_name>/<images/ | vid_name.mp4 | bbox_vid_name.mp4 | labels/>
        output_path_obj = Path(output_path)
        vid_name = output_path_obj.stem  # Get filename without extension
        
        # Create main video directory
        vid_dir = output_path_obj.parent / vid_name
        vid_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        images_dir = vid_dir / "images"
        images_dir.mkdir(exist_ok=True)
        
        # Update output paths
        video_output_path = vid_dir / f"{vid_name}.mp4"
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_output_path), fourcc, self.fps, (self.width, self.height))
        
        # Setup ground truth generation if enabled
        bbox_out = None
        labels_dir = None
        if self.gen_groundtruth:
            # Create annotated video in the same directory
            bbox_video_path = vid_dir / f"bbox_{vid_name}.mp4"
            bbox_out = cv2.VideoWriter(str(bbox_video_path), fourcc, self.fps, (self.width, self.height))
            
            # Create labels directory
            labels_dir = vid_dir / "labels"
            labels_dir.mkdir(exist_ok=True)
            
            print(f"Ground truth generation enabled. Annotated video: {bbox_video_path}")
            print(f"YOLO labels will be saved in: {labels_dir}")
        
        # Process frames with tqdm progress bar
        with tqdm(total=self.total_frames, desc="Processing frames", unit="frame") as pbar:
            frame_num = 0
            
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                # Update drone positions and remove finished drones
                active_drones = []
                frame_bboxes = []
                
                for drone in self.drones:
                    relative_frame = frame_num - drone['start_frame']
                    
                    if not drone['trajectory'].is_finished(relative_frame):
                        # Drone is still active
                        position = drone['trajectory'].get_position(relative_frame)
                        
                        # Transform drone image (use the specific drone model for this drone)
                        selected_drone_img = self.drone_images[drone['drone_image_index']]
                        transformed_drone = self._transform_drone(
                            selected_drone_img, drone['scale'], drone['rotation'], 
                            drone['flip_horizontal'], drone['color']
                        )
                        
                        # Generate bounding box for ground truth if enabled
                        if self.gen_groundtruth:
                            bbox = self._get_drone_bounding_box(position, transformed_drone)
                            if bbox:
                                frame_bboxes.append(bbox)
                        
                        # Blend onto frame
                        frame = self._blend_drone_onto_frame(
                            frame, transformed_drone, position, drone['alpha_multiplier']
                        )
                        
                        active_drones.append(drone)
                
                # Replace finished drones with new ones
                while len(active_drones) < self.num_drones:
                    # Add some randomness to spawn timing
                    if random.random() < 0.3:  # 30% chance per frame
                        active_drones.append(self._create_new_drone(frame_num))
                
                self.drones = active_drones
                
                # Save individual frame to images directory
                frame_filename = images_dir / f"{frame_num:06d}.jpg"
                cv2.imwrite(str(frame_filename), frame)
                
                # Write ground truth data if enabled
                if self.gen_groundtruth:
                    # Write YOLO format labels
                    label_file = labels_dir / f"{frame_num:06d}.txt"
                    with open(label_file, 'w') as f:
                        for bbox in frame_bboxes:
                            yolo_annotation = self._bbox_to_yolo_format(bbox)
                            if yolo_annotation:
                                f.write(yolo_annotation + '\n')
                    
                    # Create annotated frame with bounding boxes on the frame that has drones
                    if bbox_out:
                        annotated_frame = frame.copy()
                        for bbox in frame_bboxes:
                            annotated_frame = self._draw_bounding_box(annotated_frame, bbox, "UAV")
                        bbox_out.write(annotated_frame)
                
                out.write(frame)
                frame_num += 1
                pbar.update(1)
        
        # Cleanup
        self.cap.release()
        out.release()
        if bbox_out:
            bbox_out.release()
        
        print(f"Simulation complete! Output saved in: {vid_dir}")
        print(f"  - Video: {video_output_path}")
        print(f"  - Images: {images_dir}")
        if self.gen_groundtruth:
            print(f"  - Ground truth labels: {labels_dir}")
            print(f"  - Annotated video: {bbox_video_path}")


def main():
    parser = argparse.ArgumentParser(description="Drone Swarm Video Simulator")
    parser.add_argument("--background", default="assets/background.mp4", 
                       help="Background video file (default: assets/background.mp4)")
    parser.add_argument("--drone", default="assets/drone.png",
                       help="Drone PNG image file (default: assets/drone.png)")
    parser.add_argument("--num-drones", type=int, default=10,
                       help="Number of drones in swarm (default: 10)")
    parser.add_argument("--output", default="outputs/output_swarm.mp4",
                       help="Output video file (default: outputs/output_swarm.mp4)")
    parser.add_argument("--gen_groundtruth", action="store_true",
                       help="Generate ground truth bounding boxes in YOLO format and annotated video")
    
    args = parser.parse_args()
    
    # Check if required files exist
    if not Path(args.background).exists():
        print(f"Error: Background video '{args.background}' not found!")
        print("Please add your background video in 'assets/background.mp4' or specify with --background")
        return 1
    
    if not Path(args.drone).exists():
        print(f"Error: Drone image '{args.drone}' not found!")
        print("Please add your drone PNG in 'assets/drone.png' or specify with --drone")
        return 1
    
    try:
        # Ensure output directory exists
        output_dir = Path(args.output).parent
        output_dir.mkdir(exist_ok=True)
        
        # Create simulator and run
        simulator = DroneSwarmSimulator(args.background, args.drone, args.num_drones, args.gen_groundtruth)
        simulator.simulate(args.output)
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())