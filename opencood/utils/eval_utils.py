import os

import numpy as np
import math
import torch
import random
from opencood.utils import common_utils
from opencood.hypes_yaml import yaml_utils
import pickle

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


def caluclate_tp_fp(det_boxes, det_score, gt_boxes, result_stat, iou_thresh, select_flag, frame,
                    left_range=-float('inf'), right_range=float('inf')):
    """
    Calculate the true positive and false positive numbers of the current
    frames.

    Parameters
    ----------
    det_boxes : torch.Tensor
        The detection bounding box, shape (N, 8, 3) or (N, 4, 2).
        检测到的边框矩阵，格式为 (N, 8, 3)
    det_score :torch.Tensor
        The confidence score for each preditect bounding box.
        对应于每个检测结果(N)的置信分数
    gt_boxes : torch.Tensor
        The groundtruth bounding box.
        真实结果的边框矩阵
    result_stat: dict
        A dictionary contains fp, tp and gt number.
        统计结果
    iou_thresh : float
        The iou thresh.
        给定的IoU
    right_range : float
        The evaluarion range right bound
    left_range : float
        The evaluation range left bound
    """
    # fp, tp and gt in the current frame
    fp = []
    tp = []
    fn = len(gt_boxes)

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

        

        # remove the bbx out of rangeb
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
        score_order_descend = np.argsort(-det_score_new)
        # print(det_score_new)
        
        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            # calculate iou of det area and gt area
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)
            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                fp.append(1)
                tp.append(0)
                continue

            fp.append(0)
            tp.append(1)

            gt_index = np.argmax(ious)
            # print(len(ious))
            gt_polygon_list.pop(gt_index)
            # actual_position.pop(gt_index)
        
    else:
        gt = gt_boxes.shape[0]
    # print(f"fp = {fp}")

    if select_flag:
        result_stat[iou_thresh]['tp'].append(tp)
        result_stat[iou_thresh]['fp'].append(fp)
        result_stat[iou_thresh]['gt'].append(gt)
    else:
        result_stat[iou_thresh]['fp'] += fp
        result_stat[iou_thresh]['tp'] += tp
        result_stat[iou_thresh]['gt'] += gt
    
    return fn


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

    if isinstance(iou_5['gt'], list):
        gt_total = sum(iou_5['gt'])
    else:
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

    # print('The range is %s, '
    #       'The Average Precision at IOU 0.5 is %.3f, '
    #       'The Average Precision at IOU 0.7 is %.3f' % (range,  ap_50, ap_70))
    print('The range is %s, '
          'The Average Precision at IOU 0.5 is %.3f' % (range,  ap_50))

    return round(ap_50, 3)
    

def calculate_coop_error(det_boxes, det_score, 
                         pred_boxes, pred_score, 
                         gt_boxes, iou_thresh,
                         left_range=-float('inf'), right_range=float('inf'),
                         score_stat=None):
    det_match_dict = {}
    pred_match_dict = {}
    det_score_stat = {}
    coop_error = 0
    pred_right = 0
    det_right = 0
    score_stat = score_stat

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

        # remove the bbx out of rangeb
        for i in range(len(det_polygon_list_origin)):
            det_polygon = det_polygon_list_origin[i]
            distance = np.sqrt(det_polygon.centroid.x**2 +
                               det_polygon.centroid.y**2)
            if left_range <= distance < right_range:
                det_polygon_list.append(det_polygon)
                det_score_new.append(det_score[i])

        for i in range(len(gt_polygon_list_origin)):
            gt_polygon = gt_polygon_list_origin[i]
            distance = np.sqrt(gt_polygon.centroid.x**2 +
                               gt_polygon.centroid.y**2)
            if left_range <= distance < right_range:
                gt_polygon_list.append(gt_polygon)

        det_score_new = np.array(det_score_new)
        # sort the prediction bounding box by score 检测置信度分数降序排列
        score_order_descend = np.argsort(-det_score_new)
        
        # match prediction and gt bounding box
        for i in range(score_order_descend.shape[0]):
            det_polygon = det_polygon_list[score_order_descend[i]]
            # calculate iou of det area and gt area
            ious = common_utils.compute_iou(det_polygon, gt_polygon_list)   
            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                continue
            gt_index = np.argmax(ious)
            det_match_dict[gt_index] = True
            det_score_stat[gt_index] = det_score_new[score_order_descend[i]]
            det_right += 1
            gt_polygon_list[gt_index] = [(0., 0.), (0., 0.), (0., 0.), (0., 0.)]
    else:
        print("det_boxes is empty!")
        return 0
    
    if pred_boxes is not None:
        # convert bounding boxes to numpy array
        pred_boxes = common_utils.torch_tensor_to_numpy(pred_boxes)
        pred_score = common_utils.torch_tensor_to_numpy(pred_score)

        pred_polygon_list_origin = list(common_utils.convert_format(pred_boxes))
        pred_polygon_list = []
        pred_score_new = []

        # remove the bbx out of rangeb
        for i in range(len(pred_polygon_list_origin)):
            pred_polygon = pred_polygon_list_origin[i]
            distance = np.sqrt(pred_polygon.centroid.x**2 +
                               pred_polygon.centroid.y**2)
            if left_range <= distance < right_range:
                pred_polygon_list.append(pred_polygon)
                pred_score_new.append(pred_score[i])
        gt_polygon_list.clear()
        for i in range(len(gt_polygon_list_origin)):
            gt_polygon = gt_polygon_list_origin[i]
            distance = np.sqrt(gt_polygon.centroid.x**2 +
                               gt_polygon.centroid.y**2)
            if left_range <= distance < right_range:
                gt_polygon_list.append(gt_polygon)

        pred_score_new = np.array(pred_score_new)
        # sort the prediction bounding box by score
        pred_score_order_descend = np.argsort(-pred_score_new)
        
        # match prediction and gt bounding box
        for i in range(pred_score_order_descend.shape[0]):
            pred_polygon = pred_polygon_list[pred_score_order_descend[i]]
            # calculate iou of det area and gt area
            ious = common_utils.compute_iou(pred_polygon, gt_polygon_list)
            if len(gt_polygon_list) == 0 or np.max(ious) < iou_thresh:
                continue
            gt_index = np.argmax(ious)
            pred_match_dict[gt_index] = True
            pred_right += 1
            gt_polygon_list[gt_index] = [(0., 0.), (0., 0.), (0., 0.), (0., 0.)]
    else:
        for key in det_match_dict:
            if det_match_dict[key] == True:
                coop_error += 1
        if coop_error is None:
            coop_error = 0
        if score_stat is not None:
            for key in det_score_stat:
                score = det_score_stat[key]
                if score >= 0 and score < 0.1:
                    score_stat['0.0-0.1'] += 1
                elif score >= 0.1 and score < 0.2:
                    score_stat['0.1-0.2'] += 1
                elif score >= 0.2 and score < 0.3:
                    score_stat['0.2-0.3'] += 1
                elif score >= 0.3 and score < 0.4:
                    score_stat['0.3-0.4'] += 1
                elif score >= 0.4 and score < 0.5:
                    score_stat['0.4-0.5'] += 1
                elif score >= 0.5 and score < 0.6:
                    score_stat['0.5-0.6'] += 1
                elif score >= 0.6 and score < 0.7:
                    score_stat['0.6-0.7'] += 1
                elif score >= 0.7 and score < 0.8:
                    score_stat['0.7-0.8'] += 1
                elif score >= 0.8 and score < 0.9:
                    score_stat['0.8-0.9'] += 1
                elif score >= 0.9 and score <= 1.0:
                    score_stat['0.9-1.0'] += 1
        return coop_error
    
    for i in det_match_dict:
        if i not in pred_match_dict.keys():
            coop_error += 1
            if score_stat is not None:
                score = det_score_stat[i]
                if score >= 0 and score < 0.1:
                    score_stat['0.0-0.1'] += 1
                elif score >= 0.1 and score < 0.2:
                    score_stat['0.1-0.2'] += 1
                elif score >= 0.2 and score < 0.3:
                    score_stat['0.2-0.3'] += 1
                elif score >= 0.3 and score < 0.4:
                    score_stat['0.3-0.4'] += 1
                elif score >= 0.4 and score < 0.5:
                    score_stat['0.4-0.5'] += 1
                elif score >= 0.5 and score < 0.6:
                    score_stat['0.5-0.6'] += 1
                elif score >= 0.6 and score < 0.7:
                    score_stat['0.6-0.7'] += 1
                elif score >= 0.7 and score < 0.8:
                    score_stat['0.7-0.8'] += 1
                elif score >= 0.8 and score < 0.9:
                    score_stat['0.8-0.9'] += 1
                elif score >= 0.9 and score <= 1.0:
                    score_stat['0.9-1.0'] += 1
                print(f"score = {score}")
          
    if coop_error is None:
        coop_error = 0


    return coop_error, len(gt_boxes) - pred_right
    

def calculate_select_param(det_boxes, det_score, pred_boxes):
    if det_boxes is not None and \
        pred_boxes is not None:
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
            select_param = (overlap_volume  * det_score_list[i]) / \
                           (len(det_polygon_list) * len(pred_polygon_list) * det_boxes_volumes[i])

            select_param_list.append(select_param)

    else:
        print("boxes list is empty!")
        return 0.0

    # print(select_param_list)

    return sum(select_param_list)


def method_eval_result(select_params, result_stat,
                       coop_error_stat, model_dir, scale, save_indices=False):
        
    total_result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                         0.7: {'tp': [], 'fp': [], 'gt': 0}}
    select_result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                          0.7: {'tp': [], 'fp': [], 'gt': 0}}
    random_result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                          0.7: {'tp': [], 'fp': [], 'gt': 0}}
    select_error_list = {0.5: [],
                         0.7: []}
    random_error_list = {0.5: [],
                         0.7: []}
    fog_err_list = []
    selected_operate_stat = {'rain': {'num': 0,
                                      'error': 0},
                             'snow': {'num': 0,
                                      'error': 0},
                             'fog': {'num': 0,
                                      'error': 0},
                             'loc': {'num': 0,
                                      'error': 0},
                             'async': {'num': 0,
                                      'error': 0},
                             'lossy': {'num': 0,
                                      'error': 0},
                             'chlossy': {'num': 0,
                                      'error': 0}}
    select_indices = []

    # random select
    num = 996 * 7
    random_select_list = sorted(random.sample(range(num), int(num * scale))) 

    # Normalized data
    max_param = max(select_params)
    min_param = min(select_params)
    nomalized_params_list = [-(x - min_param) / (max_param - min_param) \
                            for x in select_params]
    select_number = int(len(nomalized_params_list) * scale)
    # select_number = 1100
    select_indices = sorted(range(len(nomalized_params_list)),
                                key=lambda i: nomalized_params_list[i],
                                reverse=True)[:select_number]

    for i in select_indices:
        select_result_stat[0.5]['tp'] += result_stat[0.5]['tp'][i]
        select_result_stat[0.5]['fp'] += result_stat[0.5]['fp'][i]
        select_result_stat[0.5]['gt'] += result_stat[0.5]['gt'][i]
        select_error_list[0.5].append(coop_error_stat[0.5][i])

        if i < 996: 
            selected_operate_stat['async']['num'] += 1
            selected_operate_stat['async']['error'] += coop_error_stat[0.5][i]
        elif i >= 996 and i < 996 * 2: 
            selected_operate_stat['chlossy']['num'] += 1
            selected_operate_stat['chlossy']['error'] += coop_error_stat[0.5][i]
        elif i >= 996 * 2 and i < 996 * 3: 
            selected_operate_stat['fog']['num'] += 1
            selected_operate_stat['fog']['error'] += coop_error_stat[0.5][i] 
        elif i >= 996 * 3 and i < 996 * 4: 
            selected_operate_stat['loc']['num'] += 1
            selected_operate_stat['loc']['error'] += coop_error_stat[0.5][i]
        elif i >= 996 * 4 and i < 996 * 5: 
            selected_operate_stat['lossy']['num'] += 1
            selected_operate_stat['lossy']['error'] += coop_error_stat[0.5][i]
        elif i >= 996 * 5 and i < 996 * 6: 
            selected_operate_stat['rain']['num'] += 1
            selected_operate_stat['rain']['error'] += coop_error_stat[0.5][i]
        else: 
            selected_operate_stat['snow']['num'] += 1
            selected_operate_stat['snow']['error'] += coop_error_stat[0.5][i]


    for i in range(len(result_stat[0.5]['gt'])):
        total_result_stat[0.5]['tp'] += result_stat[0.5]['tp'][i]
        total_result_stat[0.5]['fp'] += result_stat[0.5]['fp'][i]
        total_result_stat[0.5]['gt'] += result_stat[0.5]['gt'][i]


    for i in random_select_list:
        random_result_stat[0.5]['tp'] += result_stat[0.5]['tp'][i]
        random_result_stat[0.5]['fp'] += result_stat[0.5]['fp'][i]
        random_result_stat[0.5]['gt'] += result_stat[0.5]['gt'][i]
        random_error_list[0.5].append(coop_error_stat[0.5][i])


    print("---------------------------")
    print(f"scale = {scale}")
    print("---------------------------")
    print(f"select result, {sum(select_error_list[0.5])} | {eval_final_results(select_result_stat, model_dir)}")
    print(f"random result, {sum(random_error_list[0.5])} | {eval_final_results(random_result_stat, model_dir)}")
    print(f"sum result, {sum(coop_error_stat[0.5])} | {eval_final_results(total_result_stat, model_dir)}")
    print(selected_operate_stat)

    # save t1''
    if save_indices:
        save_path = '/media/jlutripper/Samsung_T51/V2Vreal/Retrain/select_indices.txt'
        with open(save_path, 'w') as file:
            for i in select_indices:
                file.write(f"{i}\n")
        

 