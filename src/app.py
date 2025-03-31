from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from reconstructor import TwinReconstructor
from pose_estimator import PrivacyPoseDetector
import cv2
import tempfile
import os

app = FastAPI()
detector = PrivacyPoseDetector()
reconstructor = TwinReconstructor()

@app.post("/process")
async def process_videos(
    video1: UploadFile = File(...),
    video2: UploadFile = File(...)
):
    with tempfile.NamedTemporaryFile(delete=False) as tmp1, \
         tempfile.NamedTemporaryFile(delete=False) as tmp2:
        
        # Save uploaded videos
        tmp1.write(video1.file.read())
        tmp2.write(video2.file.read())
        
        # Process frames
        cap1 = cv2.VideoCapture(tmp1.name)
        cap2 = cv2.VideoCapture(tmp2.name)
        
        _, frame1 = cap1.read()
        _, frame2 = cap2.read()
        
        landmarks1 = detector.get_landmarks(frame1)
        landmarks2 = detector.get_landmarks(frame2)
        pcd = reconstructor.create_twin(landmarks1, landmarks2)
        
        if pcd is None:
            return {"status": "error", "message": "No poses detected"}
        
        # Save output
        output_path = "/workspace/output/twin.ply"
        o3d.io.write_point_cloud(output_path, pcd)
        
    return FileResponse(
        output_path,
        media_type='application/octet-stream',
        filename="digital_twin.ply"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)