import os
import torch
import argparse
import opencood.hypes_yaml.yaml_utils as yaml_utils

from torch.utils.data import DataLoader
from rq_test import infrence_utils
from opencood.tools import train_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Test dataset dir')
    parser.add_argument('--data_augment', type=int, default=None,
                        help='select a number of data to use operations for data augmentation')
    parser.add_argument('--rq_command', type=str, default=None, help='rq1 or rq3')
    parser.add_argument('--isSim', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'nofusion']
    
    if opt.fusion_method == 'nofusion':
        hypes = yaml_utils.load_yaml('model/late_fusion/config.yaml', None)
    else:
        hypes = yaml_utils.load_yaml("", opt)

    hypes = yaml_utils.load_yaml(None, opt)

    # set test dataset path
    hypes['validate_dir'] = opt.dataset_dir

    print('Dataset Building')

    opencood_dataset = build_dataset(hypes, visualize=True, train=False,
                                     isSim=opt.isSim, dataAugment=opt.data_augment)

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

    # Create the dictionaries for evaluation
    result_stat = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                   0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_short = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                         0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_middle = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                          0.7: {'tp': [], 'fp': [], 'gt': 0}}
    result_stat_long = {0.5: {'tp': [], 'fp': [], 'gt': 0},
                        0.7: {'tp': [], 'fp': [], 'gt': 0}}

    result_stat_formula = {0.5: {'tp': [], 'fp': [], 'gt': []},
                           0.7: {'tp': [], 'fp': [], 'gt': []}}
    coop_error_dict = {0.5: [], 0.7: []}
    coop_error_dict_short = {0.5: [], 0.7: []}
    coop_error_dict_long = {0.5: [], 0.7: []}
    coop_error_dict_middle = {0.5: [], 0.7: []}
    select_params_list = []
    score_stat = {
        '0.0-0.1': 0,
        '0.1-0.2': 0,
        '0.2-0.3': 0,
        '0.3-0.4': 0,
        '0.4-0.5': 0,
        '0.5-0.6': 0,
        '0.6-0.7': 0,
        '0.7-0.8': 0,
        '0.8-0.9': 0,
        '0.9-1.0': 0
    }
    sum_pred_error = 0

    if opt.fusion_method == 'nofusion':
        # nofusion method
        for i, batch_data in enumerate(data_loader):
            print('data idx =', i)

            with torch.no_grad():
                torch.cuda.synchronize()
                batch_data = train_utils.to_device(batch_data, device)

                det_box_tensor, det_score, _ = \
                    infrence_utils.inference_no_fusion(batch_data, model, opencood_dataset)

            det_save_path = os.path.join(opt.dataset_dir, "rq1/rq1_det_box")
            if det_box_tensor is not None:
                infrence_utils.save_box_tensor(det_box_tensor, det_score, i, det_save_path)
    else:
        for i, batch_data in enumerate(data_loader):
            print(f"id = {i}")
            with torch.no_grad():
                torch.cuda.synchronize()

                det_save_path = os.path.join(opt.dataset_dir, "rq1/rq1_det_box")
                det_box_tensor, det_score = infrence_utils.load_box_tensor(i, det_save_path)

                batch_data = train_utils.to_device(batch_data, device)

                if opt.fusion_method == 'early':
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        infrence_utils.inference_early_fusion(batch_data,
                                                              model,
                                                              opencood_dataset)
                    
                elif opt.fusion_method == 'late':
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        infrence_utils.inference_late_fusion(batch_data,
                                                             model,
                                                             opencood_dataset)

                elif opt.fusion_method == 'intermediate':
                    if opt.rq_command is not None:
                        augment_number = 285
                        if i >= augment_number and i < augment_number * 2:
                            index = i - augment_number
                            p = opencood_dataset.augment_params['chlossy_p'][index]
                            pred_box_tensor, pred_score, gt_box_tensor = \
                                infrence_utils.inference_intermediate_fusion(batch_data,
                                                                             model,
                                                                             opencood_dataset,
                                                                             'chlossy', p)

                        elif i >= augment_number * 4 and i < augment_number * 5:
                            index = i - augment_number * 4
                            p = opencood_dataset.augment_params['lossy_p'][index]
                            pred_box_tensor, pred_score, gt_box_tensor = \
                                infrence_utils.inference_intermediate_fusion(batch_data,
                                                                             model,
                                                                             opencood_dataset,
                                                                             'lossy', p)
                        else:
                            pred_box_tensor, pred_score, gt_box_tensor = \
                                infrence_utils.inference_intermediate_fusion(batch_data,
                                                                             model,
                                                                             opencood_dataset,
                                                                             '')
                    else:
                        pred_box_tensor, pred_score, gt_box_tensor = \
                            infrence_utils.inference_intermediate_fusion(batch_data,
                                                                         model,
                                                                         opencood_dataset,
                                                                         '')
                else:
                    raise NotImplementedError('Only early, late and intermediate'
                                              'fusion is supported.')

                sum_pred_error += eval_utils.caluclate_tp_fp(pred_box_tensor,
                                                             pred_score,
                                                             gt_box_tensor,
                                                             result_stat,
                                                             0.5, False, i)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                           pred_score,
                                           gt_box_tensor,
                                           result_stat_short,
                                           0.5, False, i,
                                           left_range=0, right_range=30)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                           pred_score,
                                           gt_box_tensor,
                                           result_stat_middle,
                                           0.5, False, i,
                                           left_range=30, right_range=50)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                           pred_score,
                                           gt_box_tensor,
                                           result_stat_long,
                                           0.5, False, i,
                                           left_range=50, right_range=100)

                # eval_utils.caluclate_tp_fp(pred_box_tensor,
                #                        pred_score,
                #                        gt_box_tensor,
                #                        result_stat,
                #                        0.7,False,i)
                coop_error_5, _ = eval_utils.calculate_coop_error(det_box_tensor, det_score,
                                                                  pred_box_tensor, pred_score,
                                                                  gt_box_tensor, 0.5, score_stat=score_stat)
                # sum_pred_error += pred_err
                coop_error_short, _ = eval_utils.calculate_coop_error(det_box_tensor, det_score,
                                                                      pred_box_tensor, pred_score,
                                                                      gt_box_tensor, 0.5,
                                                                      left_range=0, right_range=30)
                
                coop_error_middle, _ = eval_utils.calculate_coop_error(det_box_tensor, det_score,
                                                                       pred_box_tensor, pred_score,
                                                                       gt_box_tensor, 0.5,
                                                                       left_range=30, right_range=50)
                
                coop_error_long, _ = eval_utils.calculate_coop_error(det_box_tensor, det_score,
                                                                     pred_box_tensor, pred_score,
                                                                     gt_box_tensor, 0.5,
                                                                     left_range=50, right_range=100)
                
                coop_error_5 = coop_error_short + coop_error_middle + coop_error_long

                coop_error_dict[0.5].append(coop_error_5)
                coop_error_dict_short[0.5].append(coop_error_short)
                coop_error_dict_middle[0.5].append(coop_error_middle)
                coop_error_dict_long[0.5].append(coop_error_long)

                # coop_error_dict[0.7].append(coop_error_7)

                print(f"coop err at 0.5 IoU = {coop_error_5}")


    print(f"data num = {len(coop_error_dict[0.5])}")
    print(f"coop error: avg={sum(coop_error_dict[0.5])}, \
            short = {sum(coop_error_dict_short[0.5])}, \
            middle = {sum(coop_error_dict_middle[0.5])}, \
            long = {sum(coop_error_dict_long[0.5])}")
    # print(coop_error_dict)
    print("predict errors = ", sum_pred_error)


    eval_utils.eval_final_results(result_stat,
                                    opt.model_dir)
    
    eval_utils.eval_final_results(result_stat_short,
                                    opt.model_dir,
                                    "short")
    
    eval_utils.eval_final_results(result_stat_middle,
                                    opt.model_dir,
                                    "middle")
    
    eval_utils.eval_final_results(result_stat_long,
                                    opt.model_dir,
                                    "long")
    
    print(sum(result_stat[0.5]['fp']))


if __name__ == '__main__':
    main()
