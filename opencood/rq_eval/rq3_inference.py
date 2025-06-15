import argparse
import statistics
import sys
import os

import torch
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.rq_eval.rq2_data_select import CooTest_method_result, V2X_Gen_method
from opencood.rq_eval.v2x_gen_utils import save_box_tensor, load_box_tensor, get_valid_param_dict, get_total_occ_and_dis


def rq3_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Test dataset dir')
    parser.add_argument("--scale", required=True,
                        help="retrain data scale, 0.1 or 0.15.")
    parser.add_argument("--method", required=True,
                        help="coo-test, v2x_gen or random.")
    parser.add_argument('--isSim', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    opt = parser.parse_args()
    return opt


def main():
    opt = rq3_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate']

    if opt.method == 'ori':
        hypes = yaml_utils.load_yaml("", opt)
        saved_path = opt.model_dir
    else:
        saved_path = os.path.join(opt.model_dir, opt.method, opt.scale)
        yaml_path = os.path.join(saved_path, 'config.yaml')
        hypes = yaml_utils.load_yaml(yaml_path, None)

    # load model label
    hypes['validate_dir'] = opt.dataset_dir

    print('Dataset Building')
    opencood_dataset = build_dataset(hypes, visualize=True, train=False,
                                     isSim=opt.isSim)

    data_loader = DataLoader(opencood_dataset,
                             batch_size=1,
                             num_workers=16,
                             collate_fn=opencood_dataset.collate_batch_test,
                             shuffle=False,
                             pin_memory=False,
                             drop_last=False)

    print('Creating Model')
    model = train_utils.create_model(hypes)
    # we assume gpu is necessary
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('Loading Model from checkpoint')
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation
    result_stat = {
        0.5: {'tp': [], 'fp': [], 'gt': []},
        0.7: {'tp': [], 'fp': [], 'gt': []},
        'occ_error': [],
        'dis_error': [],
        'total_occ': [],
        'total_dis': []
    }

    if opt.fusion_method == 'nofusion':
        # only eval late/early/intermediate fusion
        print("only eval late/early/intermediate fusion!")
        sys.exit()
    else:
        # late/intermediate/early method
        for i, batch_data in enumerate(data_loader):
            with torch.no_grad():
                torch.cuda.synchronize()
                batch_data = train_utils.to_device(batch_data, device)

                if opt.fusion_method == 'late':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_object_ids = \
                        infrence_utils.inference_late_fusion(batch_data,
                                                             model,
                                                             opencood_dataset)
                elif opt.fusion_method == 'early':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_object_ids = \
                        infrence_utils.inference_early_fusion(batch_data,
                                                              model,
                                                              opencood_dataset)
                elif opt.fusion_method == 'intermediate':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_object_ids = \
                        infrence_utils.inference_intermediate_fusion(batch_data,
                                                                     model,
                                                                     opencood_dataset)
                else:
                    raise NotImplementedError('Only early, late and intermediate'
                                              'fusion is supported.')
                # overall calculating
                fp, tp, gt, false_pred_ids = eval_utils.caluclate_tp_fp(pred_box_tensor,
                                                                        pred_score,
                                                                        gt_box_tensor,
                                                                        result_stat,
                                                                        0.5,
                                                                        gt_object_ids=gt_object_ids.copy())

                # print(sum(fp), sum(tp), gt, gt_object_ids, false_pred_ids)
                if gt != len(gt_object_ids):
                    sys.exit()

                # print(batch_data['ego'].keys())
                ego_v2x_gen_dict = batch_data['ego']['v2x_gen']

                timestamp_key = ego_v2x_gen_dict[list(ego_v2x_gen_dict.keys())[0]]['timestamp']
                folder_name = ego_v2x_gen_dict[list(ego_v2x_gen_dict.keys())[0]]['folder_name']
                print('idx =', i, 'data path =', f'{folder_name}/{timestamp_key}')

                if opt.fusion_method == 'late':
                    cp_v2x_gen_dict = batch_data['1']['v2x_gen']
                    cp_v2x_gen_dict = get_valid_param_dict(cp_v2x_gen_dict, batch_data['1']['object_ids'])
                else:
                    cp_v2x_gen_dict = batch_data['ego']['v2x_gen_cp']

                ego_v2x_gen_dict = get_valid_param_dict(ego_v2x_gen_dict, batch_data['ego']['object_ids'])

                occ_error, total_occ = eval_utils.get_occ_error(ego_v2x_gen_dict, cp_v2x_gen_dict,
                                                                false_pred_ids)
                dis_error, total_dis = eval_utils.get_long_distance_error(ego_v2x_gen_dict, cp_v2x_gen_dict,
                                                                          false_pred_ids, 50)

                result_stat['occ_error'].append(occ_error)
                result_stat['dis_error'].append(dis_error)
                result_stat['total_occ'].append(total_occ)
                result_stat['total_dis'].append(total_dis)

    eval_utils.v2x_eval_result(result_stat, opt.model_dir, opt.method)


if __name__ == '__main__':
    main()
