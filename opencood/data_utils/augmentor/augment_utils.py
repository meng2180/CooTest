# -*- coding: utf-8 -*-
# Author: OpenPCDet

import os
import torch
import shutil
import random
import numpy as np
import opencood.utils.pcd_utils as pcd_utils

from opencood.utils import common_utils
from opencood.tools.atmos_models import LISA



def random_flip_along_x(gt_boxes, points, enable=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
        enable: Only will be valid for intermediate fusion augmentation
    Returns:
    """
    if enable is None:
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 8] = -gt_boxes[:, 8]

    return gt_boxes, points


def random_flip_along_y(gt_boxes, points, enable=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C)
    Returns:
    """
    if enable is None:
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
    if enable:
        gt_boxes[:, 0] = -gt_boxes[:, 0]
        gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
        points[:, 0] = -points[:, 0]

        if gt_boxes.shape[1] > 7:
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes, points


def global_rotation(gt_boxes, points, rot_range, noise_rotation=None):
    """
    Args:
        gt_boxes: (N, 7 + C), [x, y, z, dx, dy, dz, heading, [vx], [vy]]
        points: (M, 3 + C),
        rot_range: [min, max]
        noise_rotation: A fixed rotation, only will be valid in intermediate fusion
    Returns:
    """
    if noise_rotation is None:
        noise_rotation = np.random.uniform(rot_range[0],
                                           rot_range[1])
    points = common_utils.rotate_points_along_z(points[np.newaxis, :, :],
                                                np.array([noise_rotation]))[0]

    gt_boxes[:, 0:3] = \
        common_utils.rotate_points_along_z(gt_boxes[np.newaxis, :, 0:3],
                                           np.array([noise_rotation]))[0]
    gt_boxes[:, 6] += noise_rotation

    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = common_utils.rotate_points_along_z(
            np.hstack((gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))))[
            np.newaxis, :, :],
            np.array([noise_rotation]))[0][:, 0:2]

    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range, noise_scale=None):
    """
    Args:
        gt_boxes: (N, 7), [x, y, z, dx, dy, dz, heading]
        points: (M, 3 + C),
        scale_range: [min, max]
    Returns:
    """
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    if noise_scale is None:
        noise_scale = np.random.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale

    return gt_boxes, points


def snow_op(pcd_path, snow_rate):
    pcd_np = \
            pcd_utils.pcd_to_np(pcd_path)
    atmos_noise = LISA(atm_model='snow')

    atoms_np = atmos_noise.augment(pcd_np, snow_rate)[:, :-1]
    pcd_utils.np_to_pcd_and_save(pcd_path, atoms_np)


def rain_op(pcd_path, rain_rate):
    pcd_np = \
            pcd_utils.pcd_to_np(pcd_path)
    atmos_noise = LISA(atm_model='rain')

    atoms_np = atmos_noise.augment(pcd_np, rain_rate)[:, :-1]
    pcd_utils.np_to_pcd_and_save(pcd_path, atoms_np)


def fog_op(pcd_path, visibility):
    pcd_np =  pcd_utils.pcd_to_np(pcd_path)
    atmos_noise = LISA(atm_model='fog')
    atoms_np = atmos_noise.augment(pcd_np, visibility)
    pcd_utils.np_to_pcd_and_save(pcd_path, atoms_np)
    print(f"v = {visibility}")


def loc_op(pose, tran_x, tran_y, tran_z, yaw):
    # build a translation transformation matrix
    tran_matrix = np.array([[1, 0, 0, tran_x],
                            [0, 1, 0, tran_y],
                            [0, 0, 1, tran_z],
                            [0, 0, 0, 1]])
        
    # build a rotation matrix around the z-axis
    yaw_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                             [np.sin(yaw), np.cos(yaw), 0, 0],
                             [0, 0, 1, 0],
                             [0, 0, 0, 1]])
        
    # apply the transformation(translation, z-rotation) to the original pose
    noise_pose = np.dot(pose, np.dot(yaw_matrix, tran_matrix))
    # print(f"yaw = {yaw}")    
    return noise_pose


def async_op(ego_flag, overhead=0):
    # there is not time delay for ego vehicle
    if ego_flag:
        return 0

    random_overhead = overhead
    time_delay = np.abs(random_overhead)
    
    # todo: current 10hz, we may consider 20hz in the future
    time_delay = time_delay // 100
    # return time_delay if self.async_flag else 0
    return int(time_delay)

# Lossy Communication on global feature
def lossy_op(spatial_features_2d, p):
    feature_max = torch.max(spatial_features_2d)
    feature_min = torch.min(spatial_features_2d)
    feature = spatial_features_2d.clone()
    if feature.size(0) == 2:
        random_data = random.uniform(feature_min, feature_max)
        mask = torch.bernoulli(torch.full_like(feature[1:], p))
        feature[1:][mask.bool()] = random_data

    return feature

# Channelwise Lossy Communication
def chlossy_op(spatial_features_2d, p):
    random_channels = int(spatial_features_2d.size(0) * \
                     spatial_features_2d.size(1) * p)

    feature_max = torch.max(spatial_features_2d)
    feature_min = torch.min(spatial_features_2d)

    index = 0
    channel_indices = np.random.choice(spatial_features_2d.numel(), random_channels, replace=False)

    for num in range(spatial_features_2d.size(0)):
        for channel in range(spatial_features_2d.size(1)):
            if index in channel_indices and num == 1:
                spatial_features_2d[num, channel, :, :] = random.uniform(feature_min, feature_max)
                index += 1
    # print("chossy!")
    return spatial_features_2d


def get_random_numbers(total, data_file_num):
    num_list = [total // data_file_num] * (data_file_num - 1)
    num_list.append(total - sum(num_list))
    return num_list

def copy_selected_data(cav_path, label_path,
                       destination_path, 
                       idx, op_tag, ego_flag):
    op_data_path = os.path.join(destination_path, f"{op_tag}")

    # build async operate data path
    if not os.path.exists(op_data_path):
        os.makedirs(op_data_path)

    if ego_flag:
        new_path = os.path.join(op_data_path, '0')
    else:
        new_path = os.path.join(op_data_path, '1')
    if not os.path.exists(new_path):
        os.makedirs(new_path)

                
    if ego_flag:
        new_path = os.path.join(op_data_path, '0')
    else:
        new_path = os.path.join(op_data_path, '1')

    
    shutil.copy(os.path.join(os.path.dirname(cav_path), os.path.basename(cav_path)),
                os.path.join(new_path, f"{idx}.pcd"))
    shutil.copy(os.path.join(os.path.dirname(label_path), os.path.basename(label_path)),
                os.path.join(new_path, f"{idx}.yaml"))
    
    pcd_file_path = os.path.join(new_path, f"{idx}.pcd")
    
    if op_tag == 'rain' or op_tag == 'snow' or\
        op_tag == 'fog':
        return pcd_file_path
    
                        
