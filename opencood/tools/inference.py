import argparse
import os
import time
import random

import torch
import numpy as np
import open3d as o3d
from torch.utils.data import DataLoader

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.visualization import vis_utils
from opencood.utils import eval_utils
from opencood.data_utils import augmentor


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str, required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis', action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--isSim', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy file')
    parser.add_argument('--data_augment', type=int, default=None,
                        help='select a number of data to use operations for data augmentation')
    parser.add_argument('--data_select', type=str, default=None,
                        help='random or formula')
    parser.add_argument('--rq_command', type=str, default=None, help='rq1 or rq3')
    opt = parser.parse_args()
    return opt


def main():
    opt = test_parser()
    assert opt.fusion_method in ['late', 'early', 'intermediate', 'nofusion']
    assert not (opt.show_vis and opt.show_sequence), \
        'you can only visualize ' \
        'the results in single ' \
        'image mode or video mode'

    hypes = yaml_utils.load_yaml(None, opt)
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

    # Create the dictionary for evaluation
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
    coop_error_dict = {0.5: [],  0.7: []}
    coop_error_dict_short = {0.5: [], 0.7: []}
    coop_error_dict_long = {0.5: [], 0.7: []}
    coop_error_dict_middle = {0.5: [], 0.7: []}
    select_params_list = []


    if opt.fusion_method != 'nofusion':
        n_hypes = yaml_utils.load_yaml('../model/nofusion/config.yaml', None)
        n_saved_path = '../model/nofusion/'

        n_opencood_dataset = build_dataset(n_hypes, visualize=True, train=False,
                                     isSim=opt.isSim, dataAugment=opt.data_augment)
        
        n_opencood_dataset.scenario_database = opencood_dataset.scenario_database
        n_opencood_dataset.augment_params = opencood_dataset.augment_params
    
    
        n_data_loader = DataLoader(n_opencood_dataset,
                                    batch_size=1,      
                                    num_workers=16,
                                    collate_fn=n_opencood_dataset.collate_batch_test,
                                    shuffle=False,
                                    pin_memory=False,
                                    drop_last=False)

        print('Creating Model for nofuion detection')
        n_model = train_utils.create_model(n_hypes)
        if torch.cuda.is_available():
            n_model.cuda()
        _, n_model = train_utils.load_saved_model(n_saved_path, n_model)
        n_model.eval()  


    # select by formula
    if opt.fusion_method != 'nofusion' and \
        opt.fusion_method != 'late' and \
        opt.data_select == 'formula':
        for i, (batch_data,  n_batch_data) in enumerate(zip(data_loader, n_data_loader)):
            print(f"id = {i}")
            
            with torch.no_grad():
                torch.cuda.synchronize()
                n_batch_data = train_utils.to_device(n_batch_data, device)
                det_box_tensor, det_score, gt_box_tensor = \
                            infrence_utils.inference_no_fusion(n_batch_data,
                                                               n_model,
                                                               n_opencood_dataset)

                batch_data = train_utils.to_device(batch_data, device)


                if opt.fusion_method == 'early':
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        infrence_utils.inference_early_fusion(batch_data,
                                                          model,
                                                          opencood_dataset)
                    
                elif opt.fusion_method == 'intermediate':
                    if opt.rq_command == 'rq1':
                        augment_number = 285
                    else:
                        augment_number == 996
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
                    raise NotImplementedError('Only early, late and intermediate'
                                          'fusion is supported.')
                
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_formula,
                                       0.5, True, i)

                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_formula,
                                       0.7, True, i)
                
                coop_error_5 = eval_utils.calculate_coop_error(det_box_tensor, det_score,
                                                               pred_box_tensor, pred_score,
                                                               gt_box_tensor, 0.5)
                # coop_error_7 = eval_utils.calculate_coop_error(det_box_tensor, det_score,
                #                                                pred_box_tensor, pred_score,
                #                                                gt_box_tensor, 0.7)

                coop_error_dict[0.5].append(coop_error_5)
                print(f"coop error = {coop_error_5}")
                # coop_error_dict[0.7].append(coop_error_7)

                
                select_param = eval_utils.calculate_select_param(det_box_tensor, det_score,
                                                  pred_box_tensor)
                                
                select_params_list.append(select_param)
    elif opt.data_select == 'formula':
        for i, batch_data in enumerate(data_loader):
            print(f"id = {i}")

            with torch.no_grad():
                torch.cuda.synchronize()
                batch_data = train_utils.to_device(batch_data, device)             
            
                if opt.fusion_method == 'late':
                    det_box_tensor, det_score, gt_box_tensor = \
                        infrence_utils.inference_no_fusion(batch_data,
                                                       model,
                                                       opencood_dataset)
                    
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        infrence_utils.inference_late_fusion(batch_data,
                                                       model,
                                                       opencood_dataset)
                else:
                    raise NotImplementedError('Only early, late and intermediate'
                                              'fusion is supported.')
                
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_formula,
                                       0.5, True, i)

                # eval_utils.caluclate_tp_fp(pred_box_tensor,
                #                        pred_score,
                #                        gt_box_tensor,
                #                        result_stat_formula,
                #                        0.7, True, i)
                
                coop_error_5 = eval_utils.calculate_coop_error(det_box_tensor, det_score,
                                                               pred_box_tensor, pred_score,
                                                               gt_box_tensor, 0.5)
                # coop_error_7 = eval_utils.calculate_coop_error(det_box_tensor, det_score,
                #                                                pred_box_tensor, pred_score,
                #                                                gt_box_tensor, 0.7)

                coop_error_dict[0.5].append(coop_error_5)
                print(f"coop error = {coop_error_5}")
                # coop_error_dict[0.7].append(coop_error_7)

                
                select_param = eval_utils.calculate_select_param(det_box_tensor, det_score,
                                                  pred_box_tensor)
                
                select_params_list.append(select_param)

    # Normalized data
    if opt.data_select == 'formula':
        eval_utils.method_eval_result(select_params_list, result_stat_formula,
                                  coop_error_dict, opt.model_dir, 0.1)
        
        save_flag = True

        eval_utils.method_eval_result(select_params_list, result_stat_formula,
                                  coop_error_dict, opt.model_dir, 0.15, save_flag)
    

    if opt.show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().line_width = 10
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(500):
            vis_aabbs_gt.append(o3d.geometry.TriangleMesh())
            vis_aabbs_pred.append(o3d.geometry.TriangleMesh())
            

    if opt.fusion_method != 'nofusion' and \
            opt.data_select != 'formula':
        for i, (batch_data,  n_batch_data) in enumerate(zip(data_loader, n_data_loader)):
            print(f"id = {i}")
            with torch.no_grad():
                # print(np.random.uniform(0, 1), 1)
                torch.cuda.synchronize()
                n_batch_data = train_utils.to_device(n_batch_data, device)
                det_box_tensor, det_score, gt_tensor = \
                            infrence_utils.inference_no_fusion(n_batch_data,
                                                               n_model,
                                                               n_opencood_dataset)

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
                        if opt.rq_command == 'rq1':
                            augment_number = 285
                        else:
                            augment_number = 996
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
                
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5,False,i)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_short,
                                       0.5,False,i,
                                       left_range=0,right_range=30)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_middle,
                                       0.5,False,i,
                                       left_range=30,right_range=50)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_long,
                                       0.5,False,i,
                                       left_range=50,right_range=100)

                # eval_utils.caluclate_tp_fp(pred_box_tensor,
                #                        pred_score,
                #                        gt_box_tensor,
                #                        result_stat,
                #                        0.7,False,i)
                # coop_error_7 = eval_utils.calculate_coop_error(det_box_tensor, det_score,
                #                                                pred_box_tensor, pred_score,
                #                                                gt_box_tensor, 0.7)
                coop_error_short = eval_utils.calculate_coop_error(det_box_tensor, det_score,
                                                               pred_box_tensor, pred_score,
                                                               gt_box_tensor, 0.5, 
                                                               left_range=0, right_range=30)
                coop_error_middle = eval_utils.calculate_coop_error(det_box_tensor, det_score,
                                                               pred_box_tensor, pred_score,
                                                               gt_box_tensor, 0.5,
                                                               left_range=30, right_range=50)
                coop_error_long = eval_utils.calculate_coop_error(det_box_tensor, det_score,
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

                if opt.show_vis:
                    opencood_dataset.visualize_result(pred_box_tensor,
                                                    gt_box_tensor,
                                                    batch_data['ego'][
                                                        'origin_lidar'][0],
                                                    opt.show_vis,
                                                    '',
                                                    dataset=opencood_dataset)

                if opt.save_npy:
                    npy_save_path = os.path.join(opt.model_dir, 'npy')
                    if not os.path.exists(npy_save_path):
                        os.makedirs(npy_save_path)
                    infrence_utils.save_prediction_gt(pred_box_tensor,
                                                    gt_box_tensor,
                                                    batch_data['ego'][
                                                        'origin_lidar'][0],
                                                    i,
                                                    npy_save_path)

                if opt.show_sequence:
                    pcd, pred_o3d_box, gt_o3d_box = \
                        vis_utils.visualize_inference_sample_dataloader(
                            pred_box_tensor,
                            gt_box_tensor,
                            batch_data['ego']['origin_lidar'][0],
                            vis_pcd,
                            mode='constant'
                        )
                        
                    if i == 0:
                        vis.add_geometry(pcd)
                        vis_utils.linset_assign_list(vis,
                                                    vis_aabbs_pred,
                                                    pred_o3d_box,
                                                    update_mode='add')

                        vis_utils.linset_assign_list(vis,
                                                    vis_aabbs_gt,
                                                    gt_o3d_box,
                                                    update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                vis_aabbs_pred,
                                                pred_o3d_box)
                    vis_utils.linset_assign_list(vis,
                                                vis_aabbs_gt,
                                                gt_o3d_box)
                    vis.update_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.001)
                

    elif opt.data_select != 'formula':
        for i, batch_data in enumerate(data_loader):
            print(f"id = {i}")

            with torch.no_grad():
                torch.cuda.synchronize()
                batch_data = train_utils.to_device(batch_data, device)             
            
                if opt.fusion_method == 'nofusion':
                    pred_box_tensor, pred_score, gt_box_tensor = \
                        infrence_utils.inference_no_fusion(batch_data,
                                                           model,
                                                           opencood_dataset)
                else:
                    raise NotImplementedError('Only early, late and intermediate'
                                              'fusion is supported.')

                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat,
                                       0.5,False,i)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_short,
                                       0.5,False,i,
                                       left_range=0,right_range=30)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_middle,
                                       0.5,False,i,
                                       left_range=30,right_range=50)
                eval_utils.caluclate_tp_fp(pred_box_tensor,
                                       pred_score,
                                       gt_box_tensor,
                                       result_stat_long,
                                       0.5,False,i,
                                       left_range=50,right_range=100)

                if opt.show_sequence:
                    pcd, pred_o3d_box, gt_o3d_box = \
                        vis_utils.visualize_inference_sample_dataloader(
                            pred_box_tensor,
                            gt_box_tensor,
                            batch_data['ego']['origin_lidar'][0],
                            vis_pcd,
                            mode='constant'
                        )
                    if i == 0:
                        vis.add_geometry(pcd)
                        vis_utils.linset_assign_list(vis,
                                                    vis_aabbs_pred,
                                                    pred_o3d_box,
                                                    update_mode='add')

                        vis_utils.linset_assign_list(vis,
                                                    vis_aabbs_gt,
                                                    gt_o3d_box,
                                                    update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                vis_aabbs_pred,
                                                pred_o3d_box)
                    vis_utils.linset_assign_list(vis,
                                                vis_aabbs_gt,
                                                gt_o3d_box)
                    vis.update_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.001)


    if opt.data_select != 'formula':
        print(f"data num = {len(coop_error_dict[0.5])}")
        print(f"coop error: avg={sum(coop_error_dict[0.5])}, \
                            short={sum(coop_error_dict_short[0.5])}, \
                            middle={sum(coop_error_dict_middle[0.5])}, \
                            long={sum(coop_error_dict_long[0.5])}")

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

    if opt.show_sequence:
        vis.destroy_window()


if __name__ == '__main__':
    main()
