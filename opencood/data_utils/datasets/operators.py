import opencood.hypes_yaml.yaml_utils as yaml
import os
import random
from collections import OrderedDict

import torch
import numpy as np
from torch.utils.data import Dataset
from opencood.data_utils.datasets import basedataset
import opencood.utils.pcd_utils as pcd_utils
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2, dist_two_pose
from opencood.tools.atmos_models import LISA

# class Operators(basedataset.BaseDataset):
#     def __init__(self, params, visualize, train=True, isSim=False):
#         super().__init__(params, visualize, train, isSim)
#         self.augment_data_number = None
#         self.augment_data_path = None

#     def __getitem__(self, idx):
#         self.base_data_dict = self.retrieve_base_data(idx)

def augment_base_data():
    pass


def snow_op(self, cav_content, timestamp):
    pcd_np = \
            pcd_utils.pcd_to_np(cav_content[timestamp]['lidar'])
    atmos_noise = LISA('snow')

    # random generate rain rate
    rain_rate = round(random.uniform(0, 1), 1)

    atoms_np = atmos_noise.augment(pcd_np, rain_rate)[:, :-1]

def rain_op(self, cav_content, timestamp):
    pcd_np = \
            pcd_utils.pcd_to_np(cav_content[timestamp]['lidar'])
    atmos_noise = LISA('rain')

    # random generate rain rate
    rain_rate = round(random.uniform(0, 10), 1)

    atoms_np = atmos_noise.augment(pcd_np, rain_rate)[:, :-1]
    

def fog_op(self, cav_content, timestamp):
    model_list = ['chu_hogg_fog', 'strong_advection_fog', 'moderate_advection_fog']
    fog_model = random.choice(model_list)
    pcd_np = \
            pcd_utils.pcd_to_np(cav_content[timestamp]['lidar'])
    atmos_noise = LISA(fog_model)
    atoms_np = atmos_noise.augment(pcd_np)



def loc_op():
    print("efewfef")



def async_op():
    print("efewfef")



def lossy_op():
    print("efewfef")


# if __name__ == "__main__" :
#     path='/home/software/V2V4Real/model/PointPillar_V2VNet/config.yaml'
#     params=yaml.load_yaml(path)
#     params['wild_setting']['rot_z']=0.3
#     print(params)
#     yaml.save_yaml(params,path)