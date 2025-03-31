import runpod
from reconstructor import TwinReconstructor
from pose_estimator import PrivacyPoseDetector
import cv2
import os
import json

detector = PrivacyPoseDetector()
reconstructor = TwinReconstructor()

def process(job):
    input = json.loads(job["input"])
    
    # Load videos
    cap1 = cv2.VideoCapture(input["video1_path"])
    cap2 = cv2.VideoCapture(input["video2_path"])
    
    # Process first frame
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()
    
    if not ret1 or not ret2:
        return {"error": "Failed to read videos"}
    
    # Generate 3D twin
    landmarks1 = detector.get_landmarks(frame1)
    landmarks2 = detector.get_landmarks(frame2)
    pcd = reconstructor.create_twin(landmarks1, landmarks2)
    
    if pcd is None:
        return {"error": "No valid poses detected"}
    
    # Save output
    output_path = os.path.join(os.environ.get("OUTPUT_DIR", "/output"), "twin.ply")
    o3d.io.write_point_cloud(output_path, pcd)
    
    return {"output_path": output_path}

runpod.serverless.start({"handler": process})