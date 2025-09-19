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
        self.tsdf_cost_map = TsdfCostMap(self.tsdf_cfg.general, self.tsdf_cfg.tsdf_cost_map)

        # point cloud
        self.pseudo_pcd = None
        self.depth_image = None

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

    def depth_to_pcd_orig(self, depth_image, camera_intrinsics, camera_extrinsics, resize_factor=1.0, height_threshold=0.5, max_distance=10.0):
        start_time = time.time()
        height, width = depth_image.shape
        # print("height: ", height, "width: ", width)
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

    def depth_to_pcd(self, rgb_img, depth_img, camera_intrinsics, camera_extrinsics, resize_factor=1.0, height_threshold=0.5, max_distance=10.0):


        # print(type(rgb_img))
        # print(type(depth_img))

        start_time = time.time()

        height, width = depth_img.shape
        fx, fy = camera_intrinsics[0, 0] * resize_factor, camera_intrinsics[1, 1] * resize_factor
        cx, cy = camera_intrinsics[0, 2] * resize_factor, camera_intrinsics[1, 2] * resize_factor

        # Intrinsics in Open3D format
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )

        # # Ensure extrinsics is homogeneous 4x4
        extrinsics = camera_extrinsics.astype(np.float64)

        rgb_od3 = o3d.geometry.Image(rgb_img.astype(np.uint8))

        o3d.io.write_image(os.path.join(LOG_DIR_VIZ, f'03d_rgb_image.png'), rgb_od3)
        print("writing rgb")
        o3d.io.write_image(os.path.join(LOG_DIR_VIZ, f'o3d_depth_image.png'), o3d.geometry.Image(depth_img.astype(np.uint8))) # To save as png we need to convert to uint8
        print("writing depth")

        # Open3D expects depth in uint16 format with depth scale
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_od3, o3d.geometry.Image(depth_img.astype(np.uint16)), depth_scale=1000.0, depth_trunc=max_distance, convert_rgb_to_intensity=False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd, intrinsic, np.linalg.inv(extrinsics), project_valid_depth_only = True)
        # doing np.lineealg.inv so that the point cloud is correctly shifted to be upright

        points = np.asarray(pcd.points)
        end_time = time.time()
        pcd.points = o3d.utility.Vector3dVector(points)

        # print(f"[depth_to_pcd] Point cloud generation time: {end_time - start_time:.4f} seconds "
        #     f"({len(pcd.points)} points)")

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

    def get_cost_map_via_tsdf(self, img_pil: PILImage.Image):
        width, height = img_pil.size
        # resize_factor = 1 # original value was 1
        # new_size = (int(original_width * resize_factor), int(original_height * resize_factor))
        # # import pdb; pdb.set_trace()
        # img = img.resize(new_size)
        # img_cv2 = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        img = np.array(img_pil)

        self.depth_image = self.model.infer_image(img, height)
        # # TODO SEE IF NORMALIZATION IS NEEDED THEY DID NOT DO IT HERE
        # self.depth_image = (self.depth_image - self.depth_image.min()) / (self.depth_image.max() - self.depth_image.min()) * 255.0
        # self.depth_image = self.depth_image.astype(np.uint8)
        # print("Depth image generated")
        # TODO uncomment to visualize
        # vis_depth(img, self.depth_image)

        # My working point cloud function
        # self.pseudo_pcd = self.depth_to_pcd(img_cv2, self.depth_image, self.camera_intrinsics, self.camera_extrinsics, resize_factor=resize_factor)
        # Using depth anything with metric estimation point cloud function

        # From https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/depth_to_pointcloud.py
        # Resize depth prediction to match the original image size
        resized_pred = PILImage.fromarray(self.depth_image).resize((width, height), PILImage.NEAREST)
        # Generate mesh grid and calculate point cloud coordinates
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        print(self.camera_intrinsics)
        focal_length_x = self.camera_intrinsics[0, 0]
        focal_length_y = self.camera_intrinsics[1, 1]
        x = (x - width / 2) / focal_length_x
        y = (y - height / 2) / focal_length_y
        z = np.array(resized_pred)
        points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
        colors = np.array(img).reshape(-1, 3) / 255.0

        # Create the point cloud and save it to the output directory
        self.pseudo_pcd = o3d.geometry.PointCloud()
        self.pseudo_pcd.points = o3d.utility.Vector3dVector(points)
        self.pseudo_pcd.colors = o3d.utility.Vector3dVector(colors)

        print(f"({len(self.pseudo_pcd.points)} points)")

        # open3d save pointcloud
        o3d.io.write_point_cloud(os.path.join(LOG_DIR_VIZ, f'depth_image.ply'), self.pseudo_pcd)
        print(f"debug plc {self.pseudo_pcd.points[1000]}")
        # exit()


        self.tsdf_cost_map.LoadPointCloud(self.pseudo_pcd)
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


    def get_pseudo_pcd(self):
        return self.pseudo_pcd

    def get_depth_img(self):
        return self.depth_image

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