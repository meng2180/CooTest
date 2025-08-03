import os
import torch
import opencood.hypes_yaml.yaml_utils as yaml_utils

from logger import CLogger
from torch.utils.data import DataLoader
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils


def inference(model_path, dataset_path, operator):
    """
    Input the model path, data path, and metamorph operator,
    and output the statistical results after metamorph.

    Parameters
    ----------
    model_path : string
        Object detection model path.

    dataset_path : string
        Dataset path.

    operator : string
        operator for data metamorph.

    Returns
    ----------
    """
    result_stat = {
        0.5: {'tp': [], 'fp': [], 'gt': 0},
        0.7: {'tp': [], 'fp': [], 'gt': 0}
    }

    model_name = os.path.basename(model_path)
    fusion_method = ""

    if model_name == 'late_fusion':
        fusion_method = 'late'
    elif model_name == 'early_fusion':
        fusion_method = 'early'
    else:
        fusion_method = 'intermediate'

    hypes_path = os.path.join(model_path, 'config.yaml')
    hypes = yaml_utils.load_yaml(hypes_path)

    hypes['validate_dir'] = dataset_path

    # print('Dataset Building')
    CLogger.info(f"CooTest: operator - {operator}, fusion method - {fusion_method}, inferencing...")
    opencood_dataset = build_dataset(hypes, visualize=True, train=False, operator=operator)

    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    # print('Creating Model')
    model = train_utils.create_model(hypes)

    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    saved_path = model_path
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    for i, batch_data in enumerate(data_loader):
        # print(i)
        with torch.no_grad():
            torch.cuda.synchronize()
            batch_data = train_utils.to_device(batch_data, device)

            if fusion_method == 'nofusion':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_no_fusion(batch_data,
                                                       model,
                                                       opencood_dataset)
            elif fusion_method == 'late':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_late_fusion(batch_data,
                                                         model,
                                                         opencood_dataset)
            elif fusion_method == 'early':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_early_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
            elif fusion_method == 'intermediate':
                pred_box_tensor, pred_score, gt_box_tensor = \
                    infrence_utils.inference_intermediate_fusion(batch_data,
                                                                 model,
                                                                 opencood_dataset,
                                                                 operator)
            else:
                raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')

            # overall calculating
            eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5)

    # eval_utils.eval_final_results(result_stat, model_path)
    return result_stat

