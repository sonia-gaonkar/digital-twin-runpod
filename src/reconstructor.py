import open3d as o3d
import cv2
import numpy as np
from typing import Optional

class TwinReconstructor:
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.cam_matrix = self._load_calibration(config_path)
        
    def _load_calibration(self, config_path):
        # Load camera intrinsics/extrinsics
        # (Mocked - replace with actual calibration)
        return {
            'K1': np.eye(3),
            'K2': np.eye(3),
            'baseline': 0.5
        }

    def triangulate_points(self, pts1, pts2):
        proj_mat1 = np.hstack([self.cam_matrix['K1'], np.zeros((3, 1))])
        proj_mat2 = np.hstack([self.cam_matrix['K2'], 
                              np.array([[self.cam_matrix['baseline'], 0, 0]]).T])
        
        pts_4d = cv2.triangulatePoints(
            proj_mat1, proj_mat2,
            pts1.T, pts2.T
        )
        return (pts_4d[:3] / pts_4d[3]).T  # Convert to 3D

    def create_twin(self, landmarks1, landmarks2) -> Optional[o3d.geometry.PointCloud]:
        if landmarks1 is None or landmarks2 is None:
            return None
            
        points_3d = self.triangulate_points(
            landmarks1[:, :2], 
            landmarks2[:, :2]
        )
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        return pcd