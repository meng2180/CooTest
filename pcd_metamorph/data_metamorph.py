import torch
import random
import numpy as np

from pcd_metamorph import operator_utils
from pcd_metamorph.meta_config import operator_configs
from third.atmos_models import LISA
from opencood.utils import pcd_utils

class DataMetamorph(object):
    """
    Data Metamorph.

    Parameters
    ----------
    operator : str
       CooTest data metamorph operator in 'RN', 'SW', 'FG', 'CT', 'CL', 'GL', 'SM'.

    Attributes
    ----------
    operator : str
        CooTest data metamorph operator.
    operator_configs : dict
        The operator's configurations.
    data_path : str
        Path of the data.
    operator_param : dict
        The specific parameters of the operator.
    pose_matrix : torch.Tensor
        Spatial matrix of data.
    spatial_features : torch.Tensor
        Spatial features matrix.
    is_ego : bool
        Is ego vehicle data.
    """
    def __init__(self, operator, data_path="", pose_matrix=None, spatial_features=None, is_ego='0'):
        self.operator = operator.lower()
        self.operator_configs = operator_configs
        self.data_path = data_path
        self.operator_param = self.get_operator_param()
        self.pose_matrix = pose_matrix
        self.spatial_features = spatial_features
        self.is_ego = is_ego


    def get_operator_param(self):
        config_param = {}
        key_dict = {
            'rn': 'rain_rate',
            'sw': 'snow_rate',
            'fg': 'visibility',
            'ct': 'latency',
            'cl': 'chlossy_p',
            'gl': 'lossy_p',
        }

        if self.operator == "sm":
            config_param["trans_x"] = self.get_random_value("trans_x")
            config_param["trans_y"] = self.get_random_value("trans_y")
            config_param["trans_z"] = self.get_random_value("trans_z")
            config_param["yaw"] = self.get_random_value("yaw")
        else:
            config_key = key_dict[self.operator]
            config_param[self.operator] = self.get_random_value(config_key)
        return config_param


    def get_random_value(self, config_key):
        config = self.operator_configs[config_key]
        if config.get('type') == 'random':
            value = random.random()
        else:
            min_val, max_val = config['range']
            value = np.random.uniform(min_val, max_val)

        return round(value, config['decimals'])


    def rn_operator(self):
        """
        Add metamorph operator rain.
        """
        rn_rate = self.operator_param['rn']
        pcd_np = pcd_utils.pcd_to_np(self.data_path)
        atmos_noise = LISA(atm_model='rain')
        atmos_np = atmos_noise.augment(pcd_np, rn_rate)[:, :-1]
        pcd_utils.np_to_pcd_and_save(self.data_path, atmos_np)


    def sw_operator(self):
        """
        Add metamorph operator snow.
        """
        sw_rate = self.operator_param['sw']
        pcd_np = pcd_utils.pcd_to_np(self.data_path)
        atmos_noise = LISA(atm_model='snow')
        atmos_np = atmos_noise.augment(pcd_np, sw_rate)[:, :-1]
        pcd_utils.np_to_pcd_and_save(self.data_path, atmos_np)


    def fg_operator(self):
        """
        Add metamorph operator fog.
        """
        visibility = self.operator_param['fg']
        pcd_np = pcd_utils.pcd_to_np(self.data_path)
        atmos_noise = LISA(atm_model='fog')
        atoms_np = atmos_noise.augment(pcd_np, visibility)
        pcd_utils.np_to_pcd_and_save(self.data_path, atoms_np)


    def ct_operator(self):
        """
        Add metamorph operator time delay.
        """
        # there is not time delay for ego vehicle
        if self.is_ego == '0':
            return 0

        latency = self.operator_param['ct']
        time_delay = np.abs(latency)
        time_delay = time_delay // 100
        # return time_delay if self.async_flag else 0
        return int(time_delay)


    def cl_operator(self):
        """
        Add metamorph operator channel specific lossy.
        """
        spatial_features_2d = self.spatial_features
        p = self.operator_param['cl']

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


    def gl_operator(self):
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
        spatial_features_2d = self.spatial_features
        p = self.operator_param['gl']

        feature_max = torch.max(spatial_features_2d)
        feature_min = torch.min(spatial_features_2d)
        feature = spatial_features_2d.clone()
        if feature.size(0) == 2:
            random_data = random.uniform(feature_min, feature_max)
            mask = torch.bernoulli(torch.full_like(feature[1:], p))
            feature[1:][mask.bool()] = random_data

        return feature


    def sm_operator(self):
        """
        Add metamorph operator spatial mismatch.
        """
        trans_x = self.operator_param['trans_x']
        trans_y = self.operator_param['trans_y']
        trans_z = self.operator_param['trans_z']
        yaw = self.operator_param['yaw']

        # build a translation transformation matrix
        tran_matrix = np.array([[1, 0, 0, trans_x],
                                [0, 1, 0, trans_y],
                                [0, 0, 1, trans_z],
                                [0, 0, 0, 1]])

        # build a rotation matrix around the z-axis
        yaw_matrix = np.array([[np.cos(yaw), -np.sin(yaw), 0, 0],
                               [np.sin(yaw), np.cos(yaw), 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]])

        # apply the transformation(translation, z-rotation) to the original pose
        noise_pose = np.dot(self.pose_matrix, np.dot(yaw_matrix, tran_matrix))
        # print(f"yaw = {yaw}")
        return noise_pose


    def execute(self):
        method_name = f"{self.operator}_operator"

        method = getattr(self, method_name, None)
        if method:
            return method()
        else:
            raise ValueError(f"unsupported operator：{self.operator}")
