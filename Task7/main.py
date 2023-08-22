import open3d as o3d
import os
import numpy as np


def custom_draw_geometry(pcd, name):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(pcd)
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    title = str(name) + ".jpg"
    vis.capture_screen_image(title)
    vis.destroy_window()


for i , item in enumerate(np.sort(os.listdir())):
    if item.endswith("pcd"):
        pcd = o3d.io.read_point_cloud(item)
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        custom_draw_geometry(pcd , i)

