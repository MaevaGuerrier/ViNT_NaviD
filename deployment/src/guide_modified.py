from platform import node
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, distance_transform_edt

import cv2
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2


import importlib.util
import os
import open3d as o3d
from tsdf_cost_map import TsdfCostMap
from costmap_cfg import CostMapConfig

from PIL import Image as PILImage

import time

# ROS 
import rospy


#CONSTANT 
LOG_DIR_VIZ = "../logs_viz"


# DEBUG
from scipy import ndimage
import math



def from_numpy(array: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(array).float()

def check_tensor(tensor, name="tensor"):
    if tensor.grad is not None:
        print(f"{name} grad: {tensor.grad}")
    else:
        print(f"{name} grad is None")

def vis_depth(image, depth):

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
        # convert image from pil to opencv2
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        split_region = np.ones((image.shape[0], 50, 3), dtype=np.uint8) * 255
        combined_result = cv2.hconcat([image, split_region, depth])
        
        cv2.imwrite(os.path.join(LOG_DIR_VIZ, f'depth_image.png'), combined_result)

class PathGuide:

    def __init__(self, device, ACTION_STATS, guide_cfgs=None):
        """
        Parameters:
        """
        self.device = device
        self.guide_cfgs = guide_cfgs
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.robot_width = 0.7 # TODO robot width 
        self.spatial_resolution = 0.1
        self.max_distance = 10
        self.delta_min = from_numpy(ACTION_STATS['min']).to(self.device)
        self.delta_max = from_numpy(ACTION_STATS['max']).to(self.device)
        
        # camera_intrinsics 
        # TODO: MAKE IT A YAML FILE will do once the codebase works fully
        # self.camera_intrinsics = np.array([[607.99658203125, 0, 642.2532958984375],
        #                 [0, 607.862060546875, 366.3480224609375],
        #                 [0, 0, 1]])
        # FISHEYE 
        # self.camera_intrinsics = np.array([ [262.4592858300215, 1.9161601094375007, 327.6999606754441],
        #                                     [0.0, 263.41990773924925, 224.45937199538153],
        #                                     [0.0, 0.0, 1.0]])
        # D435 CAM
        # self.camera_intrinsics = np.array([[381.44378662109375, 0.0, 318.47174072265625],
        #                                     [0.0, 381.1520080566406, 250.35641479492188],
        #                                     [0.0, 0.0, 1.0]])
        # OAK lite pro
        # self.camera_intrinsics = np.array([[1017.5782470703125, 0.0, 612.1245727539062],
        #                                     [0.0, 1017.5782470703125, 388.58673095703125],
        #                                     [0.0, 0.0, 1.0]])


        # robot to camera extrinsic
        # self.camera_extrinsics = np.array([[0, 0, 1, -0.000],
        #                             [-1, 0, 0, -0.000],
        #                             [0, -1, 0, -0.042],
        #                             [0, 0, 0, 1]])
        # FISHEYE
        # 2.5 cm is distance of camera from robot base
        # self.camera_extrinsics = np.array([[0, 0, 1, 0.000],
        #                             [-1, 0, 0, 0.000],
        #                             [0, -1, 0, 0.025],
        #                             [0, 0, 0, 1]])
        # FISHEYE EYE BUNKER2
        # self.camera_extrinsics = np.array([[0, 0, 1, 0.000],
        #                             [-1, 0, 0, 0.000],
        #                             [0, -1, 0, 0.018],
        #                             [0, 0, 0, 1]])

        #                             # FISHEYE BUNKER 2
        # self.camera_intrinsics = np.array([
        #     [319.75407,   1.80869, 319.53298],
        #     [  0.     , 310.83225, 238.80018],
        #     [  0.     ,   0.     ,   1.     ]
        # ], dtype=np.float64)

        # dist_coeffs = np.array([
        #     0.052961, 
        #     -1.747018, 
        #     3.352134, 
        #     -0.019135
        #     ], dtype=np.float64)

        # LIMO ROS AGILEX SIMULATION
        self.camera_intrinsics = np.array([
                                        [381.36246688113556,   0.0, 320.5],
                                        [  0.0,               381.36246688113556, 240.5],
                                        [  0.0,                 0.0,   1.0]
                                    ])
        self.camera_extrinsics = np.array([[0, 0, 1, 0.000],
                                            [-1, 0, 0, 0.000],
                                            [0, -1, 0, 0.03],
                                            [0, 0, 0, 1]])

        # depth anything v2 init
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vits' # or 'vits', 'vitb', 'vitg'
        # max_depth 20 for indoor model, 80 for outdoor model
        self.model = DepthAnythingV2(**{**model_configs[encoder]})
        # package_name = 'depth_anything_v2'
        # package_spec = importlib.util.find_spec(package_name)
        # if package_spec is None:
        #     raise ImportError(f"Package '{package_name}' not found")
        # package_path = os.path.dirname(package_spec.origin)
        # USING DEPTH ANYTHING WITH METRIC DEPTH ESTIMATION INSTEAD OF FOLLOWING NAVID CODEBASE
        self.model.load_state_dict(torch.load(f'/workspace/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth'))
        self.model = self.model.to(self.device).eval()

        # TSDF init
        self.tsdf_cfg = CostMapConfig()
        # self.tsdf_cost_map = TsdfCostMap(self.tsdf_cfg.general, self.tsdf_cfg.tsdf_cost_map)

        # # point cloud
        # self.pseudo_pcd = None
        # self.depth_image = None

        # DEBUG

        self.tsdf_converter = TSDFConverter()

    def _norm_delta_to_ori_trajs(self, trajs):
        delta_tmp = (trajs + 1) / 2
        delta_ori = delta_tmp * (self.delta_max - self.delta_min) + self.delta_min
        trajs_ori = delta_ori.cumsum(dim=1)
        return trajs_ori

    def goal_cost(self, trajs, goal, scale_factor=None):
        import time
        trajs_ori = self._norm_delta_to_ori_trajs(trajs)
        if scale_factor is not None:
            trajs_ori *= scale_factor
        trajs_end_positions = trajs_ori[:, -1, :]

        distances = torch.norm(goal - trajs_end_positions, dim=1)

        gloss = 0.05 * torch.sum(distances)

        if trajs.grad is not None:
            trajs.grad.zero_()

        gloss.backward()
        return trajs.grad

    def generate_scale(self, n):
        scale = torch.linspace(0, 1, steps=n)
        
        squared_scale = scale ** 1
        
        return squared_scale.to(self.device)

    def add_robot_dim(self, world_ps):
        tangent = world_ps[:, 1:, 0:2] - world_ps[:, :-1, 0:2]
        tangent = tangent / torch.norm(tangent, dim=2, keepdim=True)
        normals = tangent[:, :, [1, 0]] * torch.tensor(
            [-1, 1], dtype=torch.float32, device=world_ps.device
        )
        world_ps_inflated = torch.vstack([world_ps[:, :-1, :]] * 3)
        world_ps_inflated[:, :, 0:2] = torch.vstack(
            [
                world_ps[:, :-1, 0:2] + normals * self.robot_width / 2,
                world_ps[:, :-1, 0:2],  # center
                world_ps[:, :-1, 0:2] - normals * self.robot_width / 2,
            ]
        )
        return world_ps_inflated

    def Pos2Ind(self, points, costmap_info):
        """
        Convert trajectory points (world coordinates) into normalized indices
        for torch.grid_sample with respect to OccupancyGrid.
        
        Args:
            points: (B, N, 2) torch tensor of world coordinates (x,y).
            costmap_info: nav_msgs/OccupancyGrid.info
        
        Returns:
            norm_inds: (B, N, 2) torch tensor in [-1,1] range for grid_sample
            grid_inds: (B, N, 2) raw cell indices before normalization
        """
        res = costmap_info.resolution
        origin_x = costmap_info.origin.position.x
        origin_y = costmap_info.origin.position.y
        width = costmap_info.width
        height = costmap_info.height

        device = points.device

        # Convert world coords → grid indices
        H = torch.empty_like(points)
        H[..., 0] = (points[..., 0] - origin_x) / res  # x -> column index
        H[..., 1] = (points[..., 1] - origin_y) / res  # y -> row index

        # Mask points inside bounds
        mask = torch.logical_and(
            (H[..., 0] >= 0) & (H[..., 0] < width),
            (H[..., 1] >= 0) & (H[..., 1] < height),
        )

        # Normalize indices to [-1,1] for grid_sample
        norm_inds = H.clone()
        norm_inds[..., 0] = (H[..., 0] / (width - 1)) * 2 - 1  # x direction
        norm_inds[..., 1] = (H[..., 1] / (height - 1)) * 2 - 1 # y direction

        return norm_inds, H, mask


    def collision_cost(self, trajs, scale_factor=None):
        # DEBUG ####
        self.tsdf_array, costmap_info = self.tsdf_converter.get_tsdf_array()
        self.cost_map = torch.tensor(self.tsdf_array, dtype=torch.float32, device=self.device)
        # DEBUG ####

        if self.cost_map is None:
            return torch.zeros(trajs.shape)
        batch_size, num_p, _ = trajs.shape
        trajs_ori = self._norm_delta_to_ori_trajs(trajs)
        trajs_ori = self.add_robot_dim(trajs_ori)
        if scale_factor is not None:
            trajs_ori *= scale_factor


        # DEBUG ####
        norm_inds, _, _ = self.Pos2Ind(trajs_ori[..., 0:2], costmap_info)

        cost_grid = self.cost_map.T.expand(trajs_ori.shape[0], 1, -1, -1)  

        oloss_M = F.grid_sample(
            cost_grid,
            norm_inds[:, None, :, :],  
            mode="bicubic",
            padding_mode="border",
            align_corners=False
        ).squeeze(1).squeeze(1)
        # DEBUG ####



        oloss_M = oloss_M.to(torch.float32)

        loss = 0.003 * torch.sum(oloss_M, axis=1)
        if trajs.grad is not None:
            trajs.grad.zero_()
        loss.backward(torch.ones_like(loss))
        cost_list = loss[1::3]
        generate_scale = self.generate_scale(trajs.shape[1])
        return generate_scale.unsqueeze(1).unsqueeze(0) * trajs.grad, cost_list

    def get_gradient(self, trajs, alpha=0.3, t=None, goal_pos=None, ACTION_STATS=None, scale_factor=None):
        trajs_in = trajs.detach().requires_grad_(True).to(self.device)
        if goal_pos is not None:
            goal_pos = torch.tensor(goal_pos).to(self.device)
            goal_cost = self.goal_cost(trajs_in, goal_pos, scale_factor=scale_factor)
            cost = goal_cost
            return cost, None
        else:
            collision_cost, cost_list = self.collision_cost(trajs_in, scale_factor=scale_factor)
            cost = collision_cost
        return cost, cost_list




#!/usr/bin/env python3
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from scipy import ndimage
import math
from threading import Event

class TSDFConverter:
    def __init__(self):
        self._costmap_ready = Event()
        self.tsdf_array = None
        self.costmap_info = None

        rospy.Subscriber("/cost_map_local/costmap/costmap", OccupancyGrid, self.costmap_callback)
        rospy.loginfo("TSDF Converter Node Initialized, waiting for costmap...")
        self._costmap_ready.wait()

    def costmap_callback(self, msg):
        """Convert OccupancyGrid to TSDF."""
        self.costmap_info = msg.info
        rospy.logdebug("Costmap info %s", self.costmap_info)
        self.tsdf_array = self.create_tsdf_from_costmap(msg)

        rospy.logdebug("TSDF updated: shape=%s", self.tsdf_array.shape)
        rospy.loginfo("Costmap TSDF generation complete.")
        self._costmap_ready.set()

    
    def create_tsdf_from_costmap(self, costmap_msg,free_thresh=10,occ_thresh=70,sigma_expand=1.0,sigma_smooth=1.0,robot_radius=0.25):
        """
        Convert nav_msgs/OccupancyGrid into a TSDF that accounts for robot footprint.
        
        Args:
            costmap_msg: OccupancyGrid
            free_thresh: max value considered free
            occ_thresh: min value considered occupied
            sigma_expand: Gaussian expansion for obstacles
            sigma_smooth: Gaussian smoothing for TSDF
            robot_radius: radius of robot in meters (footprint inflation)
        """
        width = costmap_msg.info.width
        height = costmap_msg.info.height
        # resolution = costmap_msg.info.resolution
        data = np.array(costmap_msg.data).reshape((height, width))

        obs_map = np.zeros((height, width))
        free_map = np.ones((height, width))

        obs_map[data >= occ_thresh] = 1.0
        free_map[data <= free_thresh] = 0.0

        # SO costmap already have inflation layer might no be needed 
        # inflation_radius_cells = int(np.ceil(robot_radius / resolution))
        # if inflation_radius_cells > 0:
        #     dist_to_obs = ndimage.distance_transform_edt(1 - obs_map)
        #     inflated_obs = dist_to_obs <= inflation_radius_cells
        #     obs_map[inflated_obs] = 1.0

        # they used gaussian filter to smooth? reduce noise?
        obs_map = ndimage.gaussian_filter(obs_map, sigma=sigma_expand)
        free_map = ndimage.gaussian_filter(free_map, sigma=sigma_expand)

        # same thresold ?
        free_map[free_map < 0.5] = 0
        free_map[obs_map > 0.5] = 1.0

        # Using signed distance transform to compute TSDF like orig code
        tsdf_array = ndimage.distance_transform_edt(free_map)
        tsdf_array[tsdf_array > 0.0] = np.log(tsdf_array[tsdf_array > 0.0] + math.e)
        tsdf_array = ndimage.gaussian_filter(tsdf_array, sigma=sigma_smooth)

        return tsdf_array



    def get_tsdf_array(self):
        return self.tsdf_array, self.costmap_info





























class PathOpt:
    def __init__(self):
        self.traj_cache = None

    def angle_between_vectors(self, vec1, vec2):
        dot_product = np.sum(vec1 * vec2, axis=1)
        norm_product = np.linalg.norm(vec1, axis=1) * np.linalg.norm(vec2, axis=1)
        angle = np.arccos(dot_product / norm_product)
        return np.degrees(angle)

    def select_trajectory(self, trajs, l=2, angle_threshold=45, collision_min_idx=None):
        if self.traj_cache is None or len(self.traj_cache) <= l:
            idx = collision_min_idx if collision_min_idx else 0
            self.traj_cache = trajs[idx]
        else:
            directions = trajs[:, l, :]

            historical_directions = self.traj_cache[l]

            historical_directions = np.broadcast_to(historical_directions, directions.shape)

            angle_diffs = self.angle_between_vectors(directions, historical_directions)

            sorted_indices = np.argsort(angle_diffs)

            if angle_diffs[sorted_indices[0]] > angle_threshold:
                idx = 0
                self.traj_cache = trajs[idx]
            else:
                idx =sorted_indices[0]
                self.traj_cache = trajs[idx]

        return trajs[idx], idx