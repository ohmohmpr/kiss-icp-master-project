from dataclasses import dataclass
from typing import List
import pprint

from bbox import  BBox3D
from bbox.metrics import iou_3d
from pyquaternion import Quaternion

import open3d as o3d
import numpy as np
import copy

@dataclass
class BoundingBox3D:
    x: float
    y: float
    z: float
    length: float
    width: float
    height: float
    yaw: float

@dataclass
class Frame:
    frames: List[BoundingBox3D]


class InstanceAssociation:
    def __init__(self):
        self.idx_instance = 0
        self.idx_frame = -1
        self.data_asso = {
            "instances": [
                # {
                #     "idx": 1,
                #     "frames": [
                #     {
                #         "frame_id": 1,
                #         "g_pose": {
                #             "x": 1,
                #             "y": 1,
                #             "z": 1,
                #             "length": 1,
                #             "width": 1,
                #             "height": 1,
                #             "yaw": 1,
                #         },
                #         "s_pose": {
                #             "x": 1,
                #             "y": 1,
                #             "z": 1,
                #             "length": 1,
                #             "width": 1,
                #             "height": 1,
                #             "yaw": 1,
                #         }
                #     },
                #     {
                #         "frame_id": 2,
                #         "g_pose": {
                #             "x": 2,
                #             "y": 2,
                #             "z": 2,
                #             "length": 2,
                #             "width": 2,
                #             "height": 2,
                #             "yaw": 2,
                #         },
                #         "s_pose": {
                #             "x": 2,
                #             "y": 2,
                #             "z": 2,
                #             "length": 2,
                #             "width": 2,
                #             "height": 2,
                #             "yaw": 2,
                #         }
                #     },
                #     ]
                # },
            ],
        }
        
    def add(self, frame, pose):
        # print("POSE", pose)
        if self.idx_frame != -1:
            for i in range(len(frame)):
                self.check_existance(frame[i], pose)
        else:
            for i in range(len(frame)):
                instance = self.create_new_instance(frame[i], pose)
                self.add_new_instance(instance)

        self.idx_frame = self.idx_frame + 1
        # self.show_current_frame()

        print("##########################################################################################")

    def check_existance(self, frame, pose):
        is_found = False
        instance = self.create_new_instance(frame, pose)
        
        length_tmp = len(self.data_asso["instances"])
        for i in range(length_tmp):
            box1 = BBox3D(
                instance['frames'][0]['g_pose']['x'],
                instance['frames'][0]['g_pose']['y'],
                instance['frames'][0]['g_pose']['z'],
                instance['frames'][0]['g_pose']['length'],
                instance['frames'][0]['g_pose']['width'],
                instance['frames'][0]['g_pose']['height'],
                self.convert_to_quaternion(instance['frames'][0]['g_pose']['yaw']),
            )
            
            frame_id = self.data_asso["instances"][i]['frames'][-1]['frame_id']
            if (frame_id < self.idx_frame - 2):
                continue
            
            box2 = BBox3D(
                self.data_asso["instances"][i]['frames'][-1]['g_pose']['x'],
                self.data_asso["instances"][i]['frames'][-1]['g_pose']['y'],
                self.data_asso["instances"][i]['frames'][-1]['g_pose']['z'],
                self.data_asso["instances"][i]['frames'][-1]['g_pose']['length'],
                self.data_asso["instances"][i]['frames'][-1]['g_pose']['width'],
                self.data_asso["instances"][i]['frames'][-1]['g_pose']['height'],
                self.convert_to_quaternion(self.data_asso["instances"][i]['frames'][-1]['g_pose']['yaw']),
            )
            if (iou_3d(box1, box2) > 0):
                is_found = True
                self.append_instance(instance, self.data_asso["instances"][i]['idx'])
                break
            
        if not is_found:
            self.add_new_instance(instance)
        
    def append_instance(self, instance, idx):
        
        frame = {
                "frame_id": instance['frames'][0]['frame_id'],
                "g_pose": {
                    "x": instance['frames'][0]['g_pose']['x'],
                    "y": instance['frames'][0]['g_pose']['y'],
                    "z": instance['frames'][0]['g_pose']['z'],
                    "length": instance['frames'][0]['g_pose']['length'],
                    "width": instance['frames'][0]['g_pose']['width'],
                    "height": instance['frames'][0]['g_pose']['height'],
                    "yaw": instance['frames'][0]['g_pose']['yaw'],
                },
                "s_pose": {
                    "x": instance['frames'][0]['s_pose']['x'],
                    "y": instance['frames'][0]['s_pose']['y'],
                    "z": instance['frames'][0]['s_pose']['z'],
                    "length": instance['frames'][0]['s_pose']['length'],
                    "width": instance['frames'][0]['s_pose']['width'],
                    "height": instance['frames'][0]['s_pose']['height'],
                    "yaw": instance['frames'][0]['s_pose']['yaw'],
                }
        }
        
        self.data_asso["instances"][idx]["frames"].append(frame)

    def show_current_frame(self):
        length_tmp = len(self.data_asso["instances"])
        for i in range(length_tmp):
            frame_id = self.data_asso["instances"][i]['frames'][-1]['frame_id']
            idx_obj = self.data_asso["instances"][i]['idx']
            if (frame_id < self.idx_frame - 2):
                continue
            # if idx_obj != 1:
            #     continue
            pprint.pprint(self.data_asso["instances"][i], sort_dicts=False)
        
    def convert_to_quaternion(self, yaw):
        axis_angles = np.array([0, 0, yaw + 1e-10])
        rot = o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        yaw_y = Quaternion(matrix=rot)
        return yaw_y
    
    def create_new_instance(self, frame, pose):
        box_t = np.hstack((  np.array([frame.x, frame.y, frame.z, 1]) ))
        hom_pose = pose @ box_t
        g = hom_pose[0:3]
        
        instance = {
            "frames": [
                {
                    "frame_id": self.idx_frame,
                    "g_pose": {
                        "x": g[0],
                        "y": g[1],
                        "z": g[2],
                        "length": frame.length,
                        "width": frame.width,
                        "height": frame.height,
                        "yaw": frame.yaw,
                    },
                    "s_pose": {
                        "x": frame.x,
                        "y": frame.y,
                        "z": frame.z,
                        "length": frame.length,
                        "width": frame.width,
                        "height": frame.height,
                        "yaw": frame.yaw,
                    }
                }
            ]
        }
        return instance

    def add_new_instance(self, instance):
        instance["idx"] = self.idx_instance
        self.data_asso["instances"].append(instance)
        self.idx_instance = self.idx_instance + 1
        
    def return_current_obj(self):
        length_tmp = len(self.data_asso["instances"])
        bboxes = []
        for i in range(length_tmp):
            frame_id = self.data_asso["instances"][i]['frames'][-1]['frame_id']
            idx_obj = self.data_asso["instances"][i]['idx']

            if (frame_id == self.idx_frame - 1):
            # if idx_obj != 1:
            #     continue
            
                instance = {
                    "frames": [
                        {
                            "idx": self.data_asso["instances"][i]['idx'],
                            "frame_id": self.data_asso["instances"][i]['frames'][-1]['frame_id'],
                            "g_pose": {
                                "x": self.data_asso["instances"][i]['frames'][-1]['g_pose']['x'],
                                "y": self.data_asso["instances"][i]['frames'][-1]['g_pose']['y'],
                                "z": self.data_asso["instances"][i]['frames'][-1]['g_pose']['z'],
                                "length": self.data_asso["instances"][i]['frames'][-1]['g_pose']['length'],
                                "width": self.data_asso["instances"][i]['frames'][-1]['g_pose']['width'],
                                "height": self.data_asso["instances"][i]['frames'][-1]['g_pose']['height'],
                                "yaw": self.data_asso["instances"][i]['frames'][-1]['g_pose']['yaw'],
                            },
                            "s_pose": {
                                "x": self.data_asso["instances"][i]['frames'][-1]['s_pose']['x'],
                                "y": self.data_asso["instances"][i]['frames'][-1]['s_pose']['y'],
                                "z": self.data_asso["instances"][i]['frames'][-1]['s_pose']['z'],
                                "length": self.data_asso["instances"][i]['frames'][-1]['s_pose']['length'],
                                "width": self.data_asso["instances"][i]['frames'][-1]['s_pose']['width'],
                                "height": self.data_asso["instances"][i]['frames'][-1]['s_pose']['height'],
                                "yaw": self.data_asso["instances"][i]['frames'][-1]['s_pose']['yaw'],
                            }
                        }
                    ]
                }
                        
                bboxes.append(instance)
            
        return bboxes
        
        
        
        
        
        
        
        
