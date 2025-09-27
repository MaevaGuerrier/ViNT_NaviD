# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

# python
import os

import torch
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.ndimage import gaussian_filter, binary_dilation, binary_erosion

# imperative-cost-map
from costmap_cfg import GeneralCostMapConfig, TsdfCostMapConfig


class TsdfCostMap:
    """
    Cost Map based on geometric information
    """

    def __init__(self, cfg_general: GeneralCostMapConfig, cfg_tsdf: TsdfCostMapConfig):
        self._cfg_general = cfg_general
        self._cfg_tsdf = cfg_tsdf
        # set init flag
        self.is_map_ready = False
        # init point clouds
        self.obs_pcd = o3d.geometry.PointCloud()
        self.free_pcd = o3d.geometry.PointCloud()
        return

    def Pos2Ind(self, points):
        start_xy = torch.tensor([self.start_x, self.start_y], dtype=torch.float64, device=points.device).expand(1, 1, -1)
        H = (points - start_xy) / self._cfg_general.resolution
        mask = torch.logical_and((H > 0).all(axis=2), (H < torch.tensor([self.num_x, self.num_y], device=points.device)[None,None,:]).all(axis=2))
        return self.NormInds(H), H

    def NormInds(self, H):
        norm_matrix = torch.tensor([self.num_x/2.0, self.num_y/2.0], dtype=torch.float64, device=H.device)
        H = (H - norm_matrix) / norm_matrix
        return H

    def UpdatePCDwithPs(self, P_obs, P_free, is_downsample=False):
        self.obs_pcd.points = o3d.utility.Vector3dVector(P_obs)
        self.free_pcd.points = o3d.utility.Vector3dVector(P_free)
        if is_downsample:
            self.obs_pcd = self.obs_pcd.voxel_down_sample(self._cfg_general.resolution)
            self.free_pcd = self.free_pcd.voxel_down_sample(self._cfg_general.resolution * 0.85)

        self.obs_points = np.asarray(self.obs_pcd.points)
        self.free_points = np.asarray(self.free_pcd.points)
        # print("number of obs points: %d, free points: %d" % (self.obs_points.shape[0], self.free_points.shape[0]))

    # NOT USED IN THIS CODE AT ALL
    # def ReadPointFromFile(self):
    #     pcd_load = o3d.io.read_point_cloud(os.path.join(self._cfg_general.root_path, self._cfg_general.ply_file))
    #     obs_p, free_p = self.TerrainAnalysis(np.asarray(pcd_load.points))
    #     # print("point cloud loaded, total points: %d" % (np.asarray(pcd_load.points).shape[0]))
    #     self.UpdatePCDwithPs(obs_p, free_p, is_downsample=True)
    #     if self._cfg_tsdf.filter_outliers:
    #         obs_p = self.FilterCloud(self.obs_points)
    #         free_p = self.FilterCloud(self.free_points, outlier_filter=False)
    #         self.UpdatePCDwithPs(obs_p, free_p)
    #     self.UpdateMapParams()
    #     return[]

        
    
    def LoadPointCloud(self, pcd):
        obs_p, free_p = self.TerrainAnalysis(np.asarray(pcd.points))
        self.UpdatePCDwithPs(obs_p, free_p, is_downsample=True)
        if self._cfg_tsdf.filter_outliers:
            obs_p = self.FilterCloud(self.obs_points, outlier_filter=False)
            free_p = self.FilterCloud(self.free_points, outlier_filter=False)
            self.UpdatePCDwithPs(obs_p, free_p)
        self.UpdateMapParams()
        return

    def TerrainAnalysis(self, input_points):
        obs_points = np.zeros(input_points.shape)
        free_poins = np.zeros(input_points.shape)
        obs_idx = 0
        free_idx = 0
        # naive approach with z values
        for p in input_points:
            p_height = p[2] + self._cfg_tsdf.offset_z
            if (p_height > self._cfg_tsdf.ground_height * 1.0) and (
                p_height < self._cfg_tsdf.robot_height * self._cfg_tsdf.robot_height_factor
            ):  # remove ground and ceiling
                obs_points[obs_idx, :] = p
                obs_idx = obs_idx + 1
            elif p_height < self._cfg_tsdf.ground_height:
                free_poins[free_idx, :] = p
                free_idx = free_idx + 1
        return obs_points[:obs_idx, :], free_poins[:free_idx, :]

    def UpdateMapParams(self):
        if self.obs_points.shape[0] == 0:
            print("No points received.")
            return
        max_x, max_y, _ = np.amax(self.obs_points, axis=0) + self._cfg_general.clear_dist
        min_x, min_y, _ = np.amin(self.obs_points, axis=0) - self._cfg_general.clear_dist

        self.num_x = np.ceil((max_x - min_x) / self._cfg_general.resolution / 10).astype(int) * 10
        self.num_y = np.ceil((max_y - min_y) / self._cfg_general.resolution / 10).astype(int) * 10
        self.start_x = (max_x + min_x) / 2.0 - self.num_x / 2.0 * self._cfg_general.resolution
        self.start_y = (max_y + min_y) / 2.0 - self.num_y / 2.0 * self._cfg_general.resolution

        # print("tsdf map initialized, with size: %d, %d" % (self.num_x, self.num_y))
        self.is_map_ready = True

    def zero_single_isolated_ones(self, arr):
        neighbors = ndimage.convolve(arr, weights=np.ones((3, 3)), mode="constant", cval=0)
        isolated_ones = (arr == 1) & (neighbors == 1)
        arr[isolated_ones] = 0
        return arr

    def CreateTSDFMap(self):
        # TODO for debuging and understanding why waypoints takes so much time to get generated
        import time
        start_time = time.time()
        if not self.is_map_ready:
            raise ValueError("create tsdf map fails, no points received.")
        free_map = np.ones([self.num_x, self.num_y])
        obs_map = np.zeros([self.num_x, self.num_y])
        free_I = self.IndexArrayOfPs(self.free_points)
        obs_I = self.IndexArrayOfPs(self.obs_points)
        # create free place map
        for i in obs_I:
            obs_map[i[0], i[1]] = 1.0
        obs_map = self.zero_single_isolated_ones(obs_map)
        obs_map = gaussian_filter(obs_map, sigma=self._cfg_tsdf.sigma_expand)
        for i in free_I:
            if 0 < i[0] < self.num_x and 0 < i[1] < self.num_y:
                try:
                    free_map[i[0], i[1]] = 0
                except:
                    import ipdb;ipdb.set_trace()

        free_map = gaussian_filter(free_map, sigma=self._cfg_tsdf.sigma_expand)

        free_map[free_map < self._cfg_tsdf.free_space_threshold] = 0
        # assign obstacles
        free_map[obs_map > self._cfg_tsdf.obstacle_threshold] = 1.0
        # print("occupancy map generation completed.")
        # Distance Transform
        tsdf_array = ndimage.distance_transform_edt(free_map)

        tsdf_array[tsdf_array > 0.0] = np.log(tsdf_array[tsdf_array > 0.0] + math.e)
        tsdf_array = gaussian_filter(tsdf_array, sigma=self._cfg_general.sigma_smooth)

        viz_points = np.concatenate((self.obs_points, self.free_points), axis=0)

        ground_array = np.ones([self.num_x, self.num_y]) * 0.0
        end_time = time.time()
        print("occupancy map generation time: ", end_time - start_time)
        return [tsdf_array, viz_points, ground_array], [
            float(self.start_x),
            float(self.start_y),
        ]

    def IndexArrayOfPs(self, points):
        indexes = points[:, :2] - np.array([self.start_x, self.start_y])
        indexes = (np.round(indexes / self._cfg_general.resolution)).astype(int)
        return indexes

    def FilterCloud(self, points, outlier_filter=True):
        # crop points
        if any(
            [
                self._cfg_general.x_max,
                self._cfg_general.x_min,
                self._cfg_general.y_max,
                self._cfg_general.y_min,
            ]
        ):
            points_x_idx_upper = (
                (points[:, 0] < self._cfg_general.x_max)
                if self._cfg_general.x_max is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points_x_idx_lower = (
                (points[:, 0] > self._cfg_general.x_min)
                if self._cfg_general.x_min is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points_y_idx_upper = (
                (points[:, 1] < self._cfg_general.y_max)
                if self._cfg_general.y_max is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points_y_idx_lower = (
                (points[:, 1] > self._cfg_general.y_min)
                if self._cfg_general.y_min is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points = points[
                np.vstack(
                    (
                        points_x_idx_lower,
                        points_x_idx_upper,
                        points_y_idx_upper,
                        points_y_idx_lower,
                    )
                ).all(axis=0)
            ]

        if outlier_filter:
            # Filter outlier in points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            cl, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self._cfg_tsdf.nb_neighbors,
                std_ratio=self._cfg_tsdf.std_ratio,
            )
            points = np.asarray(cl.points)

        return points

    def VizCloud(self, pcd):
        o3d.visualization.draw_geometries([pcd])  # visualize point cloud


# EoF


############### what do i need 

# scipy.ndimage.gaussian_filter can be used to compute derivatives of the Gaussian, 
# such as the gradient magnitude or Laplacian, by specifying an order parameter.





# Understand what is the num_x num_y
# ground_array = np.ones([self.num_x, self.num_y]) * 0.0 just do np.zeros 


## GROUND 

# max_x, max_y, _ = np.amax(self.obs_points, axis=0) + self._cfg_general.clear_dist
# min_x, min_y, _ = np.amin(self.obs_points, axis=0) - self._cfg_general.clear_dist

# self.num_x = np.ceil((max_x - min_x) / self._cfg_general.resolution / 10).astype(int) * 10
# self.num_y = np.ceil((max_y - min_y) / self._cfg_general.resolution / 10).astype(int) * 10




## TSDF ARRAY 

# free_map = gaussian_filter(free_map, sigma=self._cfg_tsdf.sigma_expand)
# free_map[free_map < self._cfg_tsdf.free_space_threshold] = 0
# # assign obstacles
# free_map[obs_map > self._cfg_tsdf.obstacle_threshold] = 1.0
# tsdf_array = ndimage.distance_transform_edt(free_map)
# tsdf_array[tsdf_array > 0.0] = np.log(tsdf_array[tsdf_array > 0.0] + math.e)
# tsdf_array = gaussian_filter(tsdf_array, sigma=self._cfg_general.sigma_smooth)



# data[0] -> [tsdf_array, viz_points, ground_array]
#  -> self.cost_map = torch.tensor(data[0]).requires_grad_(False).to(self.device)


# cost_grid = self.cost_map.T.expand(trajs_ori.shape[0], 1, -1, -1)


# viz points 


# self.obs_points = np.asarray(self.obs_pcd.points)
# self.free_points = np.asarray(self.free_pcd.points)




# free map values range b4 Gaussian filter:  0.0 1.0
# free map shape:  (880, 500)
# free map values range:  0.0 1.0
# tsdf map values range:  0.014938984326016421 6.6064603080822994
# tsdf map shape:  (880, 500)
# obs points shape:  (1903, 3)
# obs point value range:  -2.059154557569928 7.2788310050964355
# free points shape:  (1473, 3)
# free point value range:  -3.662443911086536 9.598074913024902
# viz points shape:  (3376, 3)
# ground array shape:  (880, 500)
# ground array value range:  0.0 0.0
# occupancy map generation time:  0.06496119499206543
# cost map shape: torch.Size([880, 500]), norm_inds shape: torch.Size([24, 7, 2])
# cost grid shape (costmap after expand): torch.Size([24, 1, 500, 880])