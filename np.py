from bbox import  BBox3D
from bbox.metrics import iou_3d
import numpy as np
from pyquaternion import Quaternion

import open3d as o3d

bounding_boxes = np.load('thisdict.npy', allow_pickle='TRUE').item()
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

max = 0 


number_frames = len(bounding_boxes)
for t in range(number_frames - 1):
    t_1 = t + 1
        
    num_boxes_frame_i = len(bounding_boxes[t])
    num_boxes_frame_j = len(bounding_boxes[t_1])
    prev_boxes = bounding_boxes[t]
    current_boxes = bounding_boxes[t_1]
    
    for i in range(num_boxes_frame_i):
        for j in range(num_boxes_frame_j):
            prev_b = prev_boxes[i]
            current_b = current_boxes[j]
            
            prev_b_axis_angles = np.array([0, 0, prev_b[6] + 1e-10])
            prev_b_rot = o3d.geometry.get_rotation_matrix_from_axis_angle(prev_b_axis_angles)
            prev_b_q8d = Quaternion(matrix=prev_b_rot)
            
            current_b_axis_angles = np.array([0, 0, current_b[6] + 1e-10])
            current_b_rot = o3d.geometry.get_rotation_matrix_from_axis_angle(current_b_axis_angles)
            current_b_q8d = Quaternion(matrix=current_b_rot)
    
            box1 = BBox3D(*prev_b[:-1], prev_b_q8d)
            box2 = BBox3D(*current_b[:-1], current_b_q8d)
        
            if (iou_3d(box1, box2) > 0):
                print("number_frames", t, i, j)
        
    

