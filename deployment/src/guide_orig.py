import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, distance_transform_edt

import cv2
import matplotlib.pyplot as plt
from depth_anything_v2.dpt import DepthAnythingV2

import rospy
import importlib.util
import os
import open3d as o3d
from tsdf_cost_map import TsdfCostMap
from costmap_cfg import CostMapConfig

from PIL import Image as PILImage

import time

# CONSTANT 
LOG_DIR_VIZ = "../logs_viz"

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
        self.robot_width = 0.3 # TODO robot width 
        self.spatial_resolution = 0.1
        self.max_distance = 10
        self.bev_dist = self.max_distance / self.spatial_resolution
        self.delta_min = from_numpy(ACTION_STATS['min']).to(self.device)
        self.delta_max = from_numpy(ACTION_STATS['max']).to(self.device)
        
        # camera_intrinsics 
        # TODO: Pass in parameters instead of constants
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
        self.camera_intrinsics = np.array([[1017.5782470703125, 0.0, 612.1245727539062],
                                            [0.0, 1017.5782470703125, 388.58673095703125],
                                            [0.0, 0.0, 1.0]])


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
        # oak lite APPROX TODO again
        self.camera_extrinsics = np.array([[0, 0, 1, 0.000],
                                    [-1, 0, 0, 0.000],
                                    [0, -1, 0, 0.011],
                                    [0, 0, 0, 1]])

        # depth anything v2 init
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        encoder = 'vits' # or 'vits', 'vitb', 'vitg'
        self.model = DepthAnythingV2(**model_configs[encoder])
        # package_name = 'depth_anything_v2'
        # package_spec = importlib.util.find_spec(package_name)
        # if package_spec is None:
        #     raise ImportError(f"Package '{package_name}' not found")
        # package_path = os.path.dirname(package_spec.origin)
        self.model.load_state_dict(torch.load(f'/workspace/ViNT_NaviD/Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth'))
        self.model = self.model.to(self.device).eval()

        # TSDF init
        self.tsdf_cfg = CostMapConfig()
        self.tsdf_cost_map = TsdfCostMap(self.tsdf_cfg.general, self.tsdf_cfg.tsdf_cost_map)

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

    def depth_to_pcd_inverse_depth(self, depth_image, camera_intrinsics, camera_extrinsics, resize_factor=1.0, height_threshold=0.5, max_distance=10.0):
        start_time = time.time()
        height, width = depth_image.shape
        print("height: ", height, "width: ", width)
        fx, fy = camera_intrinsics[0, 0] * resize_factor, camera_intrinsics[1, 1] * resize_factor
        cx, cy = camera_intrinsics[0, 2] * resize_factor, camera_intrinsics[1, 2] * resize_factor
        
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        z = depth_image.astype(np.float32)
        z_safe = np.where(z == 0, np.nan, z)
        z = 1 / z_safe
        x = (x - width / 2) * z / fx
        y = (y - height / 2) * z / fy
        non_ground_mask = (z > 0.5) & (z < max_distance)
        x_non_ground = x[non_ground_mask]
        y_non_ground = y[non_ground_mask]
        z_non_ground = z[non_ground_mask]

        points = np.stack((x_non_ground, y_non_ground, z_non_ground), axis=-1).reshape(-1, 3)
        
        extrinsics = camera_extrinsics
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_points = (extrinsics @ homogeneous_points.T).T[:, :3]
        
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(transformed_points)

        end_time = time.time()
        print(f"[depth_to_pcd] Point cloud generation time: {end_time - start_time:.4f} seconds "
            f"({len(point_cloud.points)} points)")

        return point_cloud

    def depth_to_pcd_non_inverse_depth(self, depth_image, camera_intrinsics, camera_extrinsics, resize_factor=1.0, height_threshold=0.5, max_distance=10.0):
        height, width = depth_image.shape
        print("height: ", height, "width: ", width)
        fx, fy = camera_intrinsics[0, 0] * resize_factor, camera_intrinsics[1, 1] * resize_factor
        cx, cy = camera_intrinsics[0, 2] * resize_factor, camera_intrinsics[1, 2] * resize_factor
        
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        z = depth_image.astype(np.float32)
        z_safe = np.where(z == 0, np.nan, z)  # keep invalid pixels NaN

        x = (x - cx) * z_safe / fx
        y = (y - cy) * z_safe / fy

        non_ground_mask = (z > 0.5) & (z < max_distance)
        x_non_ground = x[non_ground_mask]
        y_non_ground = y[non_ground_mask]
        z_non_ground = z[non_ground_mask]

        points = np.stack((x_non_ground, y_non_ground, z_non_ground), axis=-1).reshape(-1, 3)
        
        extrinsics = camera_extrinsics
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_points = (extrinsics @ homogeneous_points.T).T[:, :3]
        import time
        start_time = time.time()
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(transformed_points)
        end_time = time.time()
        print("point cloud generation time: ", end_time - start_time)
        
        return point_cloud

    def depth_to_pcd(self, depth_image, camera_intrinsics, camera_extrinsics, resize_factor=1.0, height_threshold=0.3, max_distance=30.0):

        start_time = time.time()

        height, width = depth_image.shape
        fx, fy = camera_intrinsics[0, 0] * resize_factor, camera_intrinsics[1, 1] * resize_factor
        cx, cy = camera_intrinsics[0, 2] * resize_factor, camera_intrinsics[1, 2] * resize_factor

        # Convert depth to Open3D Image
        depth_o3d = o3d.geometry.Image(depth_image.astype(np.float32))

        # Intrinsics in Open3D format
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )

        # Ensure extrinsics is homogeneous 4x4
        extrinsics = camera_extrinsics.astype(np.float64)

        # Generate point cloud (C++ optimized)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth_o3d,
            intrinsic,
            np.linalg.inv(extrinsics),   # Open3D expects camera pose, so invert extrinsics
            depth_scale=1.0,             # adjust if depth is not in meters
            depth_trunc=max_distance,
            stride=1                     # >1 will downsample for even faster speed
        )

        # Optional: filter out ground
        points = np.asarray(pcd.points)
        mask = points[:, 2] > height_threshold
        pcd = pcd.select_by_index(np.where(mask)[0])

        end_time = time.time()
        print(f"[depth_to_pcd] Point cloud generation time: {end_time - start_time:.4f} seconds "
            f"({len(pcd.points)} points)")

        return pcd

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

    def get_cost_map_via_tsdf(self, img):
        original_width, original_height = img.size
        resize_factor = 1 # original value was 1
        new_size = (int(original_width * resize_factor), int(original_height * resize_factor))
        # import pdb; pdb.set_trace()
        img = img.resize(new_size)

        depth_image = self.model.infer_image(cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR))
        # print("Depth image generated")
        # TODO uncomment to visualize
        # vis_depth(img, depth_image)

        pseudo_pcd = self.depth_to_pcd(depth_image, self.camera_intrinsics, self.camera_extrinsics, resize_factor=resize_factor)
        # open3d save pointcloud
        o3d.io.write_point_cloud(os.path.join(LOG_DIR_VIZ, f'depth_image.ply'), pseudo_pcd)

        self.tsdf_cost_map.LoadPointCloud(pseudo_pcd)
        data, coord = self.tsdf_cost_map.CreateTSDFMap()

        # data contains [tsdf_array, viz_points, ground_array]
        if data is None:
            self.cost_map = None
        else:
            self.cost_map = torch.tensor(data[0]).requires_grad_(False).to(self.device)
            self.viz_points = data[1]
            self.ground_array = data[2]

    def get_cost_map(self, extra_info: bool = True):
        if extra_info:
            return self.cost_map, self.viz_points, self.ground_array
        return self.cost_map


    def collision_cost(self, trajs, scale_factor=None):
        if self.cost_map is None:
            return torch.zeros(trajs.shape)
        batch_size, num_p, _ = trajs.shape
        trajs_ori = self._norm_delta_to_ori_trajs(trajs)
        trajs_ori = self.add_robot_dim(trajs_ori)
        if scale_factor is not None:
            trajs_ori *= scale_factor
        norm_inds, _ = self.tsdf_cost_map.Pos2Ind(trajs_ori)
        cost_grid = self.cost_map.T.expand(trajs_ori.shape[0], 1, -1, -1)
        oloss_M = F.grid_sample(cost_grid, norm_inds[:, None, :, :], mode='bicubic', padding_mode='border', align_corners=False).squeeze(1).squeeze(1)
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