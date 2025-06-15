import os
import numpy as np
import torch

from opencood.utils.common_utils import torch_tensor_to_numpy


def get_v2x_gen_dict(selected_cav_base):
    v2x_gen_dict = {}
    # print(selected_cav_base.keys())
    vehicles_info = selected_cav_base['params']['vehicles']

    for object_id, vehicle_info in vehicles_info.items():
        # print(selected_cav_base['cav_id'])
        if selected_cav_base['cav_id'] == 1:    # cp car
            if vehicle_info['ass_id'] != -1:
                v2x_gen_dict[object_id] = {
                    "ego_occlusion_rate": 0,
                    "cp_occlusion_rate": vehicle_info['cp_occ_rate'],
                    "ego_distance": 0,
                    "cp_distance": vehicle_info['cp_distance'],
                    "timestamp": selected_cav_base['timestamp'],
                    "folder_name": selected_cav_base['folder_name']
                }
            else:
                object_id = object_id + 100
                v2x_gen_dict[object_id] = {
                    "ego_occlusion_rate": vehicle_info['ego_occ_rate'],
                    "cp_occlusion_rate": vehicle_info['cp_occ_rate'],
                    "ego_distance": vehicle_info['ego_distance'],
                    "cp_distance": vehicle_info['cp_distance'],
                    "timestamp": selected_cav_base['timestamp'],
                    "folder_name": selected_cav_base['folder_name']
                }
        else:   # ego car
            v2x_gen_dict[object_id] = {
                "ego_occlusion_rate": vehicle_info['ego_occ_rate'],
                "cp_occlusion_rate": vehicle_info['cp_occ_rate'],
                "ego_distance": vehicle_info['ego_distance'],
                "cp_distance": vehicle_info['cp_distance'],
                "timestamp": selected_cav_base['timestamp'],
                "folder_name": selected_cav_base['folder_name']
            }

    return v2x_gen_dict


def get_valid_param_dict(param_dict, valid_ids):
    """
    select v2x gen params in valid ids
    :param param_dict:
    :param valid_ids:
    :return:
    """
    valid_param_dict = {}
    for car_id, value in param_dict.items():
        if car_id in valid_ids:
            valid_param_dict[car_id] = value
    return valid_param_dict


def save_box_tensor(box_tensor, score_tensor, timestamp, save_path):
    det_np = torch_tensor_to_numpy(box_tensor)
    score_np = torch_tensor_to_numpy(score_tensor)

    np.save(os.path.join(save_path, '%04d_det.npy' % timestamp), det_np)
    np.save(os.path.join(save_path, '%04d_score.npy' % timestamp), score_np)


def load_box_tensor(timestamp, save_path):
    box_path = os.path.join(save_path, '%04d_det.npy' % timestamp)
    score_path = os.path.join(save_path, '%04d_score.npy' % timestamp)
    if not os.path.exists(box_path):
        return None, None

    np_box = np.load(box_path)
    np_score = np.load(score_path)

    torch_box = torch.from_numpy(np_box)
    torch_score = torch.from_numpy(np_score)

    return torch_box, torch_score


def get_total_occ_and_dis(ego_gen_param, cp_gen_param):
    op = 0
    lp = 0

    for car_id, param_dict in ego_gen_param.items():
        if param_dict['occlusion_rate'] > 0.05:
            op += 1
        if param_dict['ego_distance'] > 50:
            lp += 1

    # for car_id, param_dict in cp_gen_param.items():
    #     if param_dict['occlusion_rate'] > 0:
    #         op += 1

    return op, lp