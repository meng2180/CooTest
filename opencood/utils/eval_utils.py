import os

import numpy as np
import torch

from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils


def voc_ap(rec, prec):
    """
    VOC 2010 Average Precision.
    """
    rec.insert(0, 0.0)
    rec.append(1.0)
    mrec = rec[:]

    prec.insert(0, 0.0)
    prec.append(0.0)
    mpre = prec[:]

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            i_list.append(i)

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i - 1]) * mpre[i])
    return ap, mrec, mpre


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh,
                    left_range=-float('inf'), right_range=float('inf')):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
    result_stat: dict
        A dictionary contains fp, tp and gt number.
    iou_thresh : float
        The iou thresh.
    right_range : float
        The evaluarion range right bound
    left_range : float
        The evaluation range left bound
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    
    if det_boxes is not None:
        # convert bounding boxes to numpy array
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        gt_boxes = common_utils.torch_tensor_to_numpy(gt_boxes)

        det_polygon_list_origin = list(common_utils.convert_format(det_boxes))
        gt_polygon_list_origin = list(common_utils.convert_format(gt_boxes))
        det_polygon_list = []
        gt_polygon_list = []
        det_score_new = []
        # remove the bbx out of range
        for i in range(len(det_polygon_list_origin)):
            det_polygon = det_polygon_list_origin[i]
            distance = np.sqrt(det_polygon.centroid.x**2 +
                               det_polygon.centroid.y**2)
            if left_range < distance < right_range:
                det_polygon_list.append(det_polygon)
                det_score_new.append(det_score[i])

        for i in range(len(gt_polygon_list_origin)):
            gt_polygon = gt_polygon_list_origin[i]
            distance = np.sqrt(gt_polygon.centroid.x**2 +
                               gt_polygon.centroid.y**2)
            if left_range < distance < right_range:
                gt_polygon_list.append(gt_polygon)

        gt = len(gt_polygon_list)
        det_score_new = np.array(det_score_new)
        # sort the prediction bounding box by score
        score_order_descend = np.argsort(-det_score_new)

        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)

            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            gt_polygon_list.pop(gt_index)
    else:
        gt = gt_boxes.shape[0]
    result_stat[iou_thresh]['fp'] += fp
    result_stat[iou_thresh]['tp'] += tp
    result_stat[iou_thresh]['gt'] += gt


def calculate_ap(result_stat, iou):
    """
    Calculate the average precision and recall, and save them into a txt.

    Parameters
    ----------
    result_stat : dict
        A dictionary contains fp, tp and gt number.
    iou : float
    """
    iou_5 = result_stat[iou]

    fp = iou_5['fp']
    tp = iou_5['tp']
    assert len(fp) == len(tp)

    gt_total = iou_5['gt']

    cumsum = 0
    for idx, val in enumerate(fp):
        fp[idx] += cumsum
        cumsum += val

    cumsum = 0
    for idx, val in enumerate(tp):
        tp[idx] += cumsum
        cumsum += val

    rec = tp[:]
    for idx, val in enumerate(tp):
        rec[idx] = float(tp[idx]) / gt_total

    prec = tp[:]
    for idx, val in enumerate(tp):
        prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])

    ap, mrec, mprec = voc_ap(rec[:], prec[:])

    return ap, mrec, mprec


def eval_final_results(result_stat, save_path, range=""):
    dump_dict = {}
    file_name = 'eval.yaml' if range == "" else range+'_eval.yaml'
    ap_50, mrec_50, mpre_50 = calculate_ap(result_stat, 0.50)
    ap_70, mrec_70, mpre_70 = calculate_ap(result_stat, 0.70)

    dump_dict.update({'ap_50': ap_50,
                      'ap_70': ap_70,
                      'mpre_50': mpre_50,
                      'mrec_50': mrec_50,
                      'mpre_70': mpre_70,
                      'mrec_70': mrec_70,
                      })
    yaml_utils.save_yaml(dump_dict, os.path.join(save_path, file_name))

    print('The range is %s, '
          'The Average Precision at IOU 0.5 is %.3f, '
          'The Average Precision at IOU 0.7 is %.3f' % (range,  ap_50, ap_70))


def coo_method_result(det_boxes, det_score, pred_boxes):
    """
    CooTest data select method
    """
    if det_boxes is not None and pred_boxes is not None:
        det_boxes = common_utils.torch_tensor_to_numpy(det_boxes)
        det_score = common_utils.torch_tensor_to_numpy(det_score)
        pred_boxes = common_utils.torch_tensor_to_numpy(pred_boxes)

        pred_polygon_list = list(common_utils.convert_format(pred_boxes))
        det_polygon_list = list(common_utils.convert_format(det_boxes))
        det_score_list = np.array(det_score).tolist()

        select_param_list = []
        det_boxes_volumes = []

        # calculate det boxes volume
        for box in det_boxes:
            box_lengths = []
            for i in range(3):
                lengths = box[:, i].max(axis=0) - box[:, i].min(axis=0)
                box_lengths.append(lengths)
            box_volume = np.prod(box_lengths)
            det_boxes_volumes.append(box_volume)

        for i, det_box in enumerate(det_boxes):
            overlap_volume = common_utils.compute_intersection_volume(det_box, pred_boxes)

            # guide method
            select_param = (overlap_volume * det_score_list[i]) / \
                           (len(det_polygon_list) * len(pred_polygon_list) * det_boxes_volumes[i])

            select_param_list.append(select_param)

    else:
        print("boxes list is empty!")
        return 0.0

    return sum(select_param_list)
