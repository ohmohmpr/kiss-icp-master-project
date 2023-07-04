import numpy as np
import open3d as o3d

# color_codes = np.random.rand(1000, 3)
# np.save('color_codes.npy', color_codes)
# p = np.load('./canonical_points.npy', allow_pickle=True)
p = np.load('canonical_points.npy',allow_pickle='TRUE').item()

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(p[30])

o3d.visualization.draw_geometries([pcd])