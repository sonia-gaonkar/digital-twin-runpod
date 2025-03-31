import open3d as o3d
pcd = o3d.io.read_point_cloud("output_videos/twin_*.ply")
o3d.visualization.draw_plotly([pcd])
