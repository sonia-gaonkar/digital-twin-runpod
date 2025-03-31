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
    cap1 = cv2.VideoCapture(input["https://sonia28awsbucket.s3.us-east-1.amazonaws.com/input_videos/7089608-uhd_3840_2160_25fps-_111.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=ASIAXLL3NLU4OFBPCBBG%2F20250331%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250331T072558Z&X-Amz-Expires=300&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDgaCXVzLWVhc3QtMSJHMEUCIQCEMBIsblm0p0SKjay8fynzHsMU0YhKg9UM5i5vho4J0gIgSws9gBGigPcx61Q9jcLYDq1f%2Fj4%2B4LFdviebi1INqPIq9QIIof%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARACGgw1MDU0NTQzNTM3MjAiDGS3NpFgJlbjMiCHDCrJAjRJudg9VuM9mBhtjp4Kj5l2MMX0OVXnQHp%2BZL6mFW97XJXU2V6RSBGb59MH6CEzkKgIpAQJf0Gm1lCyCRseiveqcxuruA90XghRodu9UeIzoROrm8wjV1PuSpFkguV8%2FOUl2B27%2F0zsikKS95VaIcRC0VwkRr%2BBNtCn42sdiSt4j132AfK1e%2Fzov89neGer3s59ero0gmZxMlZTvrSf8QWhiSjP81U4V5AhYZrELyqc1z4zx%2Bgzp%2Bra%2FEmFokUlxOS2kicW188aFPWz%2Fy4WRqJVcNdGldoV0R1VZOphcl0bJyUVoWeLdkakRAmdFiHZg5DPXBOOIVhB77i3UzT9z2PHL8LllbeVLuHPn795tkjN7O0A5QWCTlSOeRFFHced8Z8x4Tfb%2FPRPzEEqL1XwN2WlsLRZIC81%2BcqYVm3AARHx7CGwk4Ypt7dFMIqDqb8GOrMC5%2BcJH4tsnCDdILWAEvluW%2BRpu%2FOQmGNd9RbxApjwMwRqN3ZnN4ymbjaZk%2BXIL4RTJao7VAm9pc6R%2BRfrZmi2j8ScPa1pV0qXBVnScTqluIJDQeXlsF9JN4%2FyQOtSf4bUF5NBCvZ%2Fmy2wYBW7OAS77Pp8a0Vtg52Lo4%2BMavw4AQfm6St625bqpa0Jf%2FT0pNTbX7uilhRjVsG%2F9390XegJh6i9DjcxZK%2FJak00fqeUde5x%2FQIjjVCHhPYS2DCRNh5jqcanF1J0T%2F34NH0XQdhx76Ppn7fncRTGNVImh7j%2FmQUgAoqrrT51Q4q0JoB6l4%2Bc3pfu2W8nUAj77%2F9cfIuW38ADYYgM2mG%2BeCArhTKLWe1UhCjvfzpbZCydDYo9CpopuSZtRI63m5phbaAQfqs4xHBD2w%3D%3D&X-Amz-Signature=431170356e341f83b512e7cf2af7ee73f0daad1dee62ff5c50ec5fc14bc7c88e&X-Amz-SignedHeaders=host&response-content-disposition=inline"])
    
    cap2 = cv2.VideoCapture(input["https://sonia28awsbucket.s3.us-east-1.amazonaws.com/input_videos/7089608-uhd_3840_2160_25fps-_111.mp4?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Content-Sha256=UNSIGNED-PAYLOAD&X-Amz-Credential=ASIAXLL3NLU4OFBPCBBG%2F20250331%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20250331T072558Z&X-Amz-Expires=300&X-Amz-Security-Token=IQoJb3JpZ2luX2VjEDgaCXVzLWVhc3QtMSJHMEUCIQCEMBIsblm0p0SKjay8fynzHsMU0YhKg9UM5i5vho4J0gIgSws9gBGigPcx61Q9jcLYDq1f%2Fj4%2B4LFdviebi1INqPIq9QIIof%2F%2F%2F%2F%2F%2F%2F%2F%2F%2FARACGgw1MDU0NTQzNTM3MjAiDGS3NpFgJlbjMiCHDCrJAjRJudg9VuM9mBhtjp4Kj5l2MMX0OVXnQHp%2BZL6mFW97XJXU2V6RSBGb59MH6CEzkKgIpAQJf0Gm1lCyCRseiveqcxuruA90XghRodu9UeIzoROrm8wjV1PuSpFkguV8%2FOUl2B27%2F0zsikKS95VaIcRC0VwkRr%2BBNtCn42sdiSt4j132AfK1e%2Fzov89neGer3s59ero0gmZxMlZTvrSf8QWhiSjP81U4V5AhYZrELyqc1z4zx%2Bgzp%2Bra%2FEmFokUlxOS2kicW188aFPWz%2Fy4WRqJVcNdGldoV0R1VZOphcl0bJyUVoWeLdkakRAmdFiHZg5DPXBOOIVhB77i3UzT9z2PHL8LllbeVLuHPn795tkjN7O0A5QWCTlSOeRFFHced8Z8x4Tfb%2FPRPzEEqL1XwN2WlsLRZIC81%2BcqYVm3AARHx7CGwk4Ypt7dFMIqDqb8GOrMC5%2BcJH4tsnCDdILWAEvluW%2BRpu%2FOQmGNd9RbxApjwMwRqN3ZnN4ymbjaZk%2BXIL4RTJao7VAm9pc6R%2BRfrZmi2j8ScPa1pV0qXBVnScTqluIJDQeXlsF9JN4%2FyQOtSf4bUF5NBCvZ%2Fmy2wYBW7OAS77Pp8a0Vtg52Lo4%2BMavw4AQfm6St625bqpa0Jf%2FT0pNTbX7uilhRjVsG%2F9390XegJh6i9DjcxZK%2FJak00fqeUde5x%2FQIjjVCHhPYS2DCRNh5jqcanF1J0T%2F34NH0XQdhx76Ppn7fncRTGNVImh7j%2FmQUgAoqrrT51Q4q0JoB6l4%2Bc3pfu2W8nUAj77%2F9cfIuW38ADYYgM2mG%2BeCArhTKLWe1UhCjvfzpbZCydDYo9CpopuSZtRI63m5phbaAQfqs4xHBD2w%3D%3D&X-Amz-Signature=431170356e341f83b512e7cf2af7ee73f0daad1dee62ff5c50ec5fc14bc7c88e&X-Amz-SignedHeaders=host&response-content-disposition=inline"])
    
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