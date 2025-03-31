import cv2
import mediapipe as mp
import numpy as np

class PrivacyPoseDetector:
    def __init__(self):
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

    def _anonymize_faces(self, frame):
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                x, y = int(bbox.xmin * w), int(bbox.ymin * h)
                width, height = int(bbox.width * w), int(bbox.height * h)
                
                # Apply Gaussian blur to face region
                frame[y:y+height, x:x+width] = cv2.GaussianBlur(
                    frame[y:y+height, x:x+width], 
                    (99, 99), 
                    30
                )
        return frame

    def get_landmarks(self, frame):
        anonymized_frame = self._anonymize_faces(frame)
        results = self.pose.process(cv2.cvtColor(anonymized_frame, cv2.COLOR_BGR2RGB))
        
        if not results.pose_landmarks:
            return None
            
        return np.array([
            [lm.x, lm.y, lm.z] 
            for lm in results.pose_landmarks.landmark
        ])