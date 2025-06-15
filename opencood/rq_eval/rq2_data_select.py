import os
import numpy as np
import opencood.utils.common_utils as common_utils


def V2X_Gen_method(ego_gen_param, cp_gen_param, false_pred_ids, a=0.5, b=0.5):
    fop = 0  # Fop: occlusion perceptual error
    flp = 0  # Flp: long distance perceptual error
    MAX_LIDAR_RANGE = 200

    # traversal ego param
    for car_id, param_dict in ego_gen_param.items():
        if car_id in false_pred_ids:
            flp += min(param_dict['ego_distance'], MAX_LIDAR_RANGE) / MAX_LIDAR_RANGE * \
                   (1 - min(param_dict['cp_distance'], MAX_LIDAR_RANGE) / MAX_LIDAR_RANGE)

            if car_id in list(cp_gen_param.keys()):
                fop += param_dict['ego_occlusion_rate'] * (1 - cp_gen_param[car_id]['cp_occlusion_rate'])
            else:
                fop += param_dict['ego_occlusion_rate'] * (1 - param_dict['cp_occlusion_rate'])

    # traversal cooperative vehicles
    for car_id, param_dict in cp_gen_param.items():
        # only cooperative vehicle
        if car_id in false_pred_ids and car_id not in list(ego_gen_param.keys()):
            fop += param_dict['ego_occlusion_rate'] * (1 - param_dict['cp_occlusion_rate'])

            flp += min(param_dict['ego_distance'], MAX_LIDAR_RANGE) / MAX_LIDAR_RANGE * \
                (1 - min(param_dict['cp_distance'], MAX_LIDAR_RANGE) / MAX_LIDAR_RANGE)

    select_method_param = a * fop + b * flp

    return select_method_param, fop, flp


def CooTest_method_result(det_boxes, det_score, pred_boxes):
    """
    CooTest data select method
    :param det_boxes:
    :param det_score:
    :param pred_boxes:
    :return:
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

    # print(select_param_list)

    return sum(select_param_list)
