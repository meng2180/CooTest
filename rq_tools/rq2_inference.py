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


def rq2_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Test dataset dir')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--isSim', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    opt = parser.parse_args()
    return opt


def main():
    opt = rq2_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'nofusion']

    if opt.fusion_method == 'nofusion':
        hypes = yaml_utils.load_yaml('model/late_fusion/config.yaml', None)
    else:
        hypes = yaml_utils.load_yaml("", opt)

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
    saved_path = opt.model_dir
    _, model = train_utils.load_saved_model(saved_path, model)
    model.eval()

    # Create the dictionary for evaluation
    result_stat = {
        0.5: {'tp': [], 'fp': [], 'gt': []},
        0.7: {'tp': [], 'fp': [], 'gt': []},
        'occ_error': [],
        'dis_error': [],
        'total_occ': [],
        'total_dis': [],
        'timestamp': []
    }
    select_result_stat = {
        'v2x_gen': [],
        'cootest': []
    }

    total_fop = []
    total_flp = []

    if opt.fusion_method == 'nofusion':
        # nofusion method
        for i, batch_data in enumerate(data_loader):
            print('data idx =', i)

            with torch.no_grad():
                torch.cuda.synchronize()
                batch_data = train_utils.to_device(batch_data, device)

                det_box_tensor, det_score, _, _ = \
                    infrence_utils.inference_no_fusion(batch_data, model, opencood_dataset)

            det_save_path = "/media/jlutripper/My Passport/v2x_dataset/rq2/rq2_det_box"
            if det_box_tensor is not None:
                save_box_tensor(det_box_tensor, det_score, i, det_save_path)
    else:
        # late/intermediate/early method
        for i, batch_data in enumerate(data_loader):
            print('data idx =', i)
            # if i > 100:
            #     break
            with torch.no_grad():
                torch.cuda.synchronize()
                batch_data = train_utils.to_device(batch_data, device)

                det_save_path = "/media/jlutripper/My Passport/v2x_dataset/rq2/rq2_det_box"
                det_box_tensor, det_score = load_box_tensor(i, det_save_path)

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

                if gt != len(gt_object_ids):
                    sys.exit()

                ego_v2x_gen_dict = batch_data['ego']['v2x_gen']

                timestamp_key = ego_v2x_gen_dict[list(ego_v2x_gen_dict.keys())[0]]['timestamp']
                folder_name = ego_v2x_gen_dict[list(ego_v2x_gen_dict.keys())[0]]['folder_name']
                timestamp = [folder_name, timestamp_key]
                result_stat['timestamp'].append(timestamp)
                print(timestamp)

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

                cootest_method_result = CooTest_method_result(det_box_tensor, det_score, pred_box_tensor)
                gen_method_result, fop, flp = V2X_Gen_method(ego_v2x_gen_dict, cp_v2x_gen_dict,
                                                             false_pred_ids, a=0.5, b=0.5)
                select_result_stat['v2x_gen'].append(gen_method_result)
                total_flp.append(flp)
                total_fop.append(fop)
                # print(gen_method_result)
                select_result_stat['cootest'].append(cootest_method_result)
                print('cootest param =', cootest_method_result, ', v2x gen param =', gen_method_result)
                print('occ error =', occ_error, ', dis error =', dis_error)

    # result
    if opt.fusion_method == 'nofusion':
        eval_utils.eval_final_results(result_stat,
                                      opt.model_dir)
    else:
        eval_utils.method_eval_result(select_result_stat, result_stat, opt.model_dir, 0.1, True, opt.dataset_dir,
                                      '/media/jlutripper/My Passport/v2x_dataset/rq2/rq2_select', opt.model_dir.split('/')[-1])
        eval_utils.method_eval_result(select_result_stat, result_stat, opt.model_dir, 0.15, True, opt.dataset_dir,
                                      '/media/jlutripper/My Passport/v2x_dataset/rq2/rq2_select', opt.model_dir.split('/')[-1])
        print("total FOP =", sum(total_fop), "total FLP =", sum(total_flp))
        print("avg FOP =", sum(total_fop) / len(total_fop))
        print("avg FLP =", sum(total_flp) / len(total_flp))
        print("mid FOP =", statistics.median(total_fop))
        print("mid FLP =", statistics.median(total_flp))


if __name__ == '__main__':
    main()
