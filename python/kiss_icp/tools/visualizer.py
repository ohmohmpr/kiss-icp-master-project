# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import copy
import importlib
import os
from abc import ABC
from functools import partial
from typing import Callable, List

import numpy as np

YELLOW = np.array([1, 0.706, 0])
RED = np.array([128, 0, 0]) / 255.0
BLACK = np.array([0, 0, 0]) / 255.0
BLUE = np.array([0.4, 0.5, 0.9])
SPHERE_SIZE = 0.20


class StubVisualizer(ABC):
    def __init__(self):
        pass

    def update(self, source, keypoints, target_map, pose):
        pass


class RegistrationVisualizer(StubVisualizer):
    # Public Interaface ----------------------------------------------------------------------------
    def __init__(self):
        try:
            self.o3d = importlib.import_module("open3d")
        except ModuleNotFoundError as err:
            print(f'open3d is not installed on your system, run "pip install open3d"')
            exit(1)

        # Initialize GUI controls
        self.block_vis = True
        self.play_crun = False
        self.reset_bounding_box = True

        # Create data
        self.source = self.o3d.geometry.PointCloud()
        self.keypoints = self.o3d.geometry.PointCloud()
        self.target = self.o3d.geometry.PointCloud()
        self.line_set = self.o3d.geometry.LineSet()
        self.frames = []

        # Initialize visualizer
        self.vis = self.o3d.visualization.VisualizerWithKeyCallback()
        self._register_key_callbacks()
        self._initialize_visualizer()

        # Visualization options
        self.render_map = True
        self.render_source = True
        self.render_keypoints = False
        self.global_view = False
        self.render_trajectory = True
        # Cache the state of the visualizer
        self.state = (
            self.render_map,
            self.render_keypoints,
            self.render_source,
        )

    def update(self, source, keypoints, target_map, pose):
        target = target_map.point_cloud()
        self._update_geometries(source, keypoints, target, pose)
        while self.block_vis:
            self.vis.poll_events()
            self.vis.update_renderer()
            if self.play_crun:
                break
        self.block_vis = not self.block_vis

    # Private Interaface ---------------------------------------------------------------------------
    def _initialize_visualizer(self):
        w_name = self.__class__.__name__
        self.vis.create_window(window_name=w_name, width=1920, height=1080)
        self.vis.add_geometry(self.source)
        self.vis.add_geometry(self.keypoints)
        

            
        # box = np.array([24.0245,   9.1368,  -1.0325,   3.9734,   1.6260,   1.6061,  -3.1267])
        # box = np.array([20.7825, -21.2953,  -0.0972,   0.6851,   0.7327,   1.7784,   1.8844])
        
        # self.line_set, box3d = self.translate_boxes_to_open3d_instance(box)
        # self.line_set.paint_uniform_color((0, 1, 0))
        # self.vis.add_geometry(self.line_set)

# 0
# [[ 24.0245,   9.1368,  -1.0325,   3.9734,   1.6260,   1.6061,  -3.1267],
#         [ 20.7825, -21.2953,  -0.0972,   0.6851,   0.7327,   1.7784,   1.8844]]

# 1
# [[ 2.0819e+01,  9.2048e+00, -1.0291e+00,  3.4018e+00,  1.5101e+00,
#           1.4456e+00,  3.1432e+00],
#         [ 6.9949e+00, -2.1386e+01, -4.5070e-01,  4.4062e+00,  1.7177e+00,
#           1.5711e+00, -2.4824e+00],
#         [ 5.1868e+00,  3.2407e+01, -1.8581e+00,  8.0439e-01,  6.3408e-01,
#           1.9083e+00,  8.7143e-01],
#         [ 2.4092e+01,  2.0004e-01, -5.9887e-01,  3.7187e+00,  1.6607e+00,
#           1.6982e+00,  1.0615e-02],
#         [ 1.1965e+01,  1.8768e+01, -1.1935e+00,  4.9087e+00,  1.9989e+00,
#           2.1694e+00,  1.0965e-01]]


        self._set_black_background(self.vis)
        self.vis.get_render_option().point_size = 1
        print(
            f"{w_name} initialized. Press:\n"
            "\t[SPACE] to pause/sdfdsstart\n"
            "\t  [ESC] to exit\n"
            "\t    [N] to step\n"
            "\t    [F] to toggle on/off the input cloud to the pipeline\n"
            "\t    [K] to toggle on/off the subsbampled frame\n"
            "\t    [M] to toggle on/off the local map\n"
            "\t    [V] to toggle ego/global viewpoint\n"
            "\t    [T] to toggle the trajectory view(only available in global view)\n"
            "\t    [C] to center the viewpoint\n"
            "\t    [W] to toggle a white background\n"
            "\t    [B] to toggle a black background\n"
        )

    def _register_key_callback(self, keys: List, callback: Callable):
        for key in keys:
            self.vis.register_key_callback(ord(str(key)), partial(callback))

    def _register_key_callbacks(self):
        self._register_key_callback(["Ā", "Q", "\x1b"], self._quit)
        self._register_key_callback([" "], self._start_stop)
        self._register_key_callback(["N"], self._next_frame)
        self._register_key_callback(["V"], self._toggle_view)
        self._register_key_callback(["C"], self._center_viewpoint)
        self._register_key_callback(["F"], self._toggle_source)
        self._register_key_callback(["K"], self._toggle_keypoints)
        self._register_key_callback(["M"], self._toggle_map)
        self._register_key_callback(["T"], self._toggle_trajectory)
        self._register_key_callback(["B"], self._set_black_background)
        self._register_key_callback(["W"], self._set_white_background)

    def _set_black_background(self, vis):
        vis.get_render_option().background_color = [0.0, 0.0, 0.0]

    def _set_white_background(self, vis):
        vis.get_render_option().background_color = [1.0, 1.0, 1.0]

    def _quit(self, vis):
        print("Destroying Visualizer")
        vis.destroy_window()
        os._exit(0)

    def _next_frame(self, vis):
        self.block_vis = not self.block_vis

    def _start_stop(self, vis):
        self.play_crun = not self.play_crun

    def _toggle_source(self, vis):
        if self.render_keypoints:
            self.render_keypoints = False
            self.render_source = True
        else:
            self.render_source = not self.render_source
        return False

    def _toggle_keypoints(self, vis):
        if self.render_source:
            self.render_source = False
            self.render_keypoints = True
        else:
            self.render_keypoints = not self.render_keypoints

        return False

    def _toggle_map(self, vis):
        self.render_map = not self.render_map
        return False

    def _toggle_view(self, vis):
        self.global_view = not self.global_view
        self._trajectory_handle()

    def _center_viewpoint(self, vis):
        vis.reset_view_point(True)

    def _toggle_trajectory(self, vis):
        if not self.global_view:
            return False
        self.render_trajectory = not self.render_trajectory
        self._trajectory_handle()
        return False

    def _trajectory_handle(self):
        if self.render_trajectory and self.global_view:
            for frame in self.frames:
                self.vis.add_geometry(frame, reset_bounding_box=False)
        else:
            for frame in self.frames:
                self.vis.remove_geometry(frame, reset_bounding_box=False)

    def _update_geometries(self, source, keypoints, target, pose):
        # Source hot frame
        if self.render_source:
            self.source.points = self.o3d.utility.Vector3dVector(source)
            self.source.paint_uniform_color(YELLOW)
            if self.global_view:
                self.source.transform(pose)
        else:
            self.source.points = self.o3d.utility.Vector3dVector()

        # Keypoints
        if self.render_keypoints:
            self.keypoints.points = self.o3d.utility.Vector3dVector(keypoints)
            self.keypoints.paint_uniform_color(YELLOW)
            if self.global_view:
                self.keypoints.transform(pose)
        else:
            self.keypoints.points = self.o3d.utility.Vector3dVector()

        # Target Map
        if self.render_map:
            target = copy.deepcopy(target)
            self.target.points = self.o3d.utility.Vector3dVector(target)
            if not self.global_view:
                self.target.transform(np.linalg.inv(pose))
        else:
            self.target.points = self.o3d.utility.Vector3dVector()

        # Update always a list with all the trajectories
        new_frame = self.o3d.geometry.TriangleMesh.create_sphere(SPHERE_SIZE)
        new_frame.paint_uniform_color(BLUE)
        new_frame.compute_vertex_normals()
        new_frame.transform(pose)
        self.frames.append(new_frame)
        # Render trajectory, only if it make sense (global view)
        if self.render_trajectory and self.global_view:
            self.vis.add_geometry(self.frames[-1], reset_bounding_box=False)

        self.vis.update_geometry(self.keypoints)
        self.vis.update_geometry(self.source)
        self.vis.update_geometry(self.target)

        # self.vis.remove_geometry(self.line_set)
        # box = np.array([24.0245 , 9.1368, -1.0325, 3.9734, 1.6260, 1.6061, -3.1267])
        # self.line_set, box3d = self.translate_boxes_to_open3d_instance(box)
        # self.line_set.paint_uniform_color((0, 1, 0))
        # self.vis.add_geometry(self.line_set)
        
        box = np.array([[24.0245, 9.1368, -1.0325, 3.9734, 1.6260, 1.6061, -3.1267],
            [ 34.389, 33.686, -0.895, 4.037, 1.665, 1.550, -2.751],
            [ 30.760, -17.894, -0.418, 0.778, 0.661, 1.782, 0.042],
            [ 13.682, 19.854, -0.428, 0.337, 0.520, 1.822, -1.453]])
        
        # for i in range(box.shape[0]):
        #     self.line_set, box3d = self.translate_boxes_to_open3d_instance(box[i])
        #     self.line_set.paint_uniform_color((0, 1, 0))
        #     self.vis.add_geometry(self.line_set)
            
        self.vis = self.draw_box(box)
        # vis = draw_box(self, vis, gt_boxes, color=(0, 1, 0), ref_labels=None, score=None):
        
        if self.reset_bounding_box:
            self.vis.reset_view_point(True)
            self.reset_bounding_box = False



    def translate_boxes_to_open3d_instance(self, gt_boxes):
        """
                4-------- 6
            /|         /|
            5 -------- 3 .
            | |        | |
            . 7 -------- 1
            |/         |/
            2 -------- 0
        """
        center = gt_boxes[0:3]
        lwh = gt_boxes[3:6]
        axis_angles = np.array([0, 0, gt_boxes[6] + 1e-10])
        rot = self.o3d.geometry.get_rotation_matrix_from_axis_angle(axis_angles)
        box3d = self.o3d.geometry.OrientedBoundingBox(center, rot, lwh)

        line_set = self.o3d.geometry.LineSet.create_from_oriented_bounding_box(box3d)

        # import ipdb; ipdb.set_trace(context=20)
        lines = np.asarray(line_set.lines)
        lines = np.concatenate([lines, np.array([[1, 4], [7, 6]])], axis=0)

        line_set.lines = self.o3d.utility.Vector2iVector(lines)
        return line_set, box3d


    def draw_box(self, gt_boxes):
        for i in range(gt_boxes.shape[0]):
            line_set, box3d = self.translate_boxes_to_open3d_instance(gt_boxes[i])
            # if ref_labels is None:
            line_set.paint_uniform_color((0, 1, 0))
            # else:
            #     line_set.paint_uniform_color(box_colormap[ref_labels[i]])
            
            self.vis.add_geometry(line_set)
            # if score is not None:
            #     corners = box3d.get_box_points()
            #     vis.add_3d_label(corners[5], '%.2f' % score[i])
        return self.vis
