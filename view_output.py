import open3d as o3d
import os
import glob

# 1. Locate the latest PLY file
output_dir = "output_videos"
ply_files = sorted(glob.glob(os.path.join(output_dir, "twin_*.ply")))

if not ply_files:
    print(f"No PLY files found in {output_dir}")
    print("Available files:", os.listdir(output_dir))
else:
    # 2. Load the most recent file
    latest_ply = ply_files[-1]  # Gets last file in sorted list
    print(f"Loading: {latest_ply}")
    
    # 3. Read and visualize
    pcd = o3d.io.read_point_cloud(latest_ply)
    if pcd.is_empty():
        print("Error: Point cloud is empty")
    else:
        print(f"Point cloud loaded with {len(pcd.points)} points")
        o3d.visualization.draw_plotly([pcd])