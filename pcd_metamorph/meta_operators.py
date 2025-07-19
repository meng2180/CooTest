import torch
import random
import numpy as np

from third.atmos_models import LISA
from opencood.utils import pcd_utils

def rain_operator(pcd_path, rain_rate):
    """
    Add metamorph operator rain.

    Parameters
    ----------
    pcd_path : str
        Dataset for metamorphose.

    rain_rate : float
        Rain rate.

    Returns
    -------
    """
    def rain_operation(pcd_path, rain_rate):
        pcd_np = \
            pcd_utils.pcd_to_np(pcd_path)
        atmos_noise = LISA(atm_model='rain')

        atoms_np = atmos_noise.augment(pcd_np, rain_rate)[:, :-1]
        pcd_utils.np_to_pcd_and_save(pcd_path, atoms_np)


def snow_operator(pcd_path, snow_rate):
    """
    Add metamorph operator snow.

    Parameters
    ----------
    pcd_path : str
        Dataset for metamorphose.

    snow_rate : float
        Snow rate.

    Returns
    -------
    """
    pcd_np = \
        pcd_utils.pcd_to_np(pcd_path)
    atmos_noise = LISA(atm_model='snow')

    atoms_np = atmos_noise.augment(pcd_np, snow_rate)[:, :-1]
    pcd_utils.np_to_pcd_and_save(pcd_path, atoms_np)


def fog_operator(pcd_path, visibility):
    """
    Add metamorph operator fog.

    Parameters
    ----------
    pcd_path : str
        Dataset for metamorphose.

    visibility : float
        Fog visibility.

    Returns
    -------
    """
    pcd_np = pcd_utils.pcd_to_np(pcd_path)
    atmos_noise = LISA(atm_model='fog')
    atoms_np = atmos_noise.augment(pcd_np, visibility)
    pcd_utils.np_to_pcd_and_save(pcd_path, atoms_np)


def time_delay_operator(ego_flag, overhead=0):
    """
    Add metamorph operator time delay.

    Parameters
    ----------
    ego_flag : bool

    overhead : float
        Time delay overhead.

    Returns
    -------
    """
    # there is not time delay for ego vehicle
    if ego_flag:
        return 0

    random_overhead = overhead
    time_delay = np.abs(random_overhead)

    time_delay = time_delay // 100
    # return time_delay if self.async_flag else 0
    return int(time_delay)


def loc_operator(pose, tran_x, tran_y, tran_z, yaw):
    """
    Add metamorph operator spatial mismatch.

    Parameters
    ----------
    pose : list
        Spatial matrix of data.

    tran_x : float
        The extension of the x-axis is misaligned.

    tran_y : float
        The extension of the y-axis is misaligned.

    tran_z : float
        The extension of the z-axis is misaligned.

    yaw : float
        Heading angle offset.

    Returns
    -------
    """
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


def global_lossy_operator(spatial_features_2d, p):
    """
    Add metamorph operator global lossy.

    Parameters
    ----------
    spatial_features_2d : bool
        Spatial features matrix.

    p : float
        Interference rate.

    Returns
    -------
    """
    feature_max = torch.max(spatial_features_2d)
    feature_min = torch.min(spatial_features_2d)
    feature = spatial_features_2d.clone()
    if feature.size(0) == 2:
        random_data = random.uniform(feature_min, feature_max)
        mask = torch.bernoulli(torch.full_like(feature[1:], p))
        feature[1:][mask.bool()] = random_data

    return feature


def channel_lossy_operator(spatial_features_2d, p):
    """
    Add metamorph operator channel specific lossy.

    Parameters
    ----------
    spatial_features_2d : bool
        Spatial features matrix.

    p : float
        Interference rate.

    Returns
    -------
    """
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

