import cv2
import os
import numpy as np
import open3d as o3d
import mediapipe as mp
from datetime import datetime

# Configuration
INPUT_DIR = os.path.join(os.path.dirname(__file__), "input_videos")  # digital-twin-runpod/input_videos/
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output_videos")  # digital-twin-runpod/input_videos/
os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class VideoProcessor:
    def __init__(self):
        # Initialize MediaPipe models
        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=True,
            min_detection_confidence=0.7
        )
        self.face_detector = mp.solutions.face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        )
        self.blur_kernel = (99, 99)  # Kernel size for face blurring

    def _anonymize_frame(self, frame):
        """Blur faces and remove identifiable features"""
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect and blur faces
        results = self.face_detector.process(rgb_frame)
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure coordinates are within frame bounds
                x, y = max(0, x), max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                # Apply Gaussian blur to face region
                frame[y:y+height, x:x+width] = cv2.GaussianBlur(
                    frame[y:y+height, x:x+width],
                    self.blur_kernel,
                    30
                )
        return frame

    def _get_skeletal_points(self, frame):
        """Extract anonymized 3D skeletal points"""
        frame = self._anonymize_frame(frame)
        results = self.pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None
            
        return np.array([
            [lm.x, lm.y, lm.z] 
            for lm in results.pose_landmarks.landmark
        ])

    def process_videos(self, video1_path, video2_path):
        """Process video pair and generate 3D model"""
        # Open video files
        cap1 = cv2.VideoCapture(video1_path)
        cap2 = cv2.VideoCapture(video2_path)
        
        # Verify videos opened successfully
        if not cap1.isOpened() or not cap2.isOpened():
            raise ValueError("Failed to open video files")
        
        # Process first frame from each video
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            raise ValueError("Failed to read frames from videos")
        
        # Get 3D skeletal points
        points1 = self._get_skeletal_points(frame1)
        points2 = self._get_skeletal_points(frame2)
        
        if points1 is None or points2 is None:
            raise ValueError("No poses detected in videos")
        
        # Triangulate 3D points (simplified projection)
        points_3d = cv2.triangulatePoints(
            projMatr1=np.eye(3, 4),  # Identity matrix for camera 1
            projMatr2=np.array([[1, 0, 0, 0.5], [0, 1, 0, 0], [0, 0, 1, 0]]),  # Camera 2 is 0.5 units to the right
            projPoints1=points1[:, :2].T,
            projPoints2=points2[:, :2].T
        )
        points_3d /= points_3d[3]  # Convert from homogeneous to 3D coordinates
        
        # Create and save point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d[:3].T)
        
        # Generate timestamped output filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"twin_{timestamp}.ply"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Save the 3D model
        o3d.io.write_point_cloud(output_path, pcd)
        
        # Release video resources
        cap1.release()
        cap2.release()
        
        return output_path

if __name__ == "__main__":
    # Initialize processor
    processor = VideoProcessor()
    
    # Define input video paths
    video1_path = os.path.join(INPUT_DIR, "camera1.mp4")
    video2_path = os.path.join(INPUT_DIR, "camera2.mp4")
    
    # Verify input files exist
    if not os.path.exists(video1_path) or not os.path.exists(video2_path):
        print(f"Error: Input videos not found in {INPUT_DIR}")
        print("Please ensure both camera1.mp4 and camera2.mp4 exist in the input_videos folder")
        exit(1)
    
    # Process videos
    try:
        output_path = processor.process_videos(video1_path, video2_path)
        print(f"Successfully generated 3D model: {output_path}")
        print(f"Output directory contents: {os.listdir(OUTPUT_DIR)}")
    except Exception as e:
        print(f"Error processing videos: {str(e)}")