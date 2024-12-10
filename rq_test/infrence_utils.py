import os
from collections import OrderedDict

import numpy as np
import torch
import open3d as o3d

from opencood.utils.common_utils import torch_tensor_to_numpy


def inference_no_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.
    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset
    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        if cav_id == 'ego':
            output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_late_fusion(batch_data, model, dataset):
    """
    Model inference for late fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.LateFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()

    for cav_id, cav_content in batch_data.items():
        output_dict[cav_id] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_early_fusion(batch_data, model, dataset):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    output_dict = OrderedDict()
    cav_content = batch_data['ego']

    output_dict['ego'] = model(cav_content)

    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def inference_intermediate_fusion(batch_data, model, dataset, augment_tag, p=0):
    """
    Model inference for early fusion.

    Parameters
    ----------
    batch_data : dict
    model : opencood.object
    dataset : opencood.EarlyFusionDataset

    Returns
    -------
    pred_box_tensor : torch.Tensor
        The tensor of prediction bounding box after NMS.
    gt_box_tensor : torch.Tensor
        The tensor of gt bounding box.
    """
    # return inference_early_fusion(batch_data, model, dataset)

    output_dict = OrderedDict()
    cav_content = batch_data['ego']
    # print(augment_tag)
    
    # model opertate!
    if augment_tag:
        cav_content['augment_op'] = augment_tag
        cav_content['p'] = p
    output_dict['ego'] = model(cav_content)
    
    pred_box_tensor, pred_score, gt_box_tensor = \
        dataset.post_process(batch_data,
                             output_dict)

    return pred_box_tensor, pred_score, gt_box_tensor


def save_prediction_gt(pred_tensor, gt_tensor, pcd, timestamp, save_path):
    """
    Save prediction and gt tensor to txt file.
    """
    # pred_np = torch_tensor_to_numpy(pred_tensor)
    # gt_np = torch_tensor_to_numpy(gt_tensor)
    pcd_np = torch_tensor_to_numpy(pcd)
    # print(pred_tensor)
    # print(pcd_np.shape)

    np.save(os.path.join(save_path, '%04d_pcd.npy' % timestamp), pcd_np)
    # np.save(os.path.join(save_path, '%04d_pred.npy' % timestamp), pred_np)
    # np.save(os.path.join(save_path, '%04d_gt.npy' % timestamp), gt_np)

    # p_pcd = o3d.geometry.PointCloud()
    # p_pcd.points = o3d.utility.Vector3dVector(pcd_np[:, :3])
    # reflectivity = pcd_np[:, 3].astype(np.float32)
    # p_pcd.colors = o3d.utility.Vector3dVector(np.column_stack((reflectivity,reflectivity,reflectivity)))

    # o3d.io.write_point_cloud(os.path.join(save_path, '%04d_pcd.pcd' % timestamp), p_pcd, write_ascii=True)

def save_selected_pcd(original_path, save_path, indices_path):
    select_indices = []
    with open(indices_path, 'r') as file:
        for line in file:
            select_indices.append(int(line.rsplit()))


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

    