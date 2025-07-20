import argparse
import statistics
import sys
import os

import torch
# from Cython.Utility.MemoryView import result
from torch.utils.data import DataLoader
from rq_tools.rq_utils import merge_scene_folders
from rq_tools.rq_inference import rq_inference

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.rq_eval.rq2_data_select import CooTest_method_result, V2X_Gen_method
from opencood.rq_eval.v2x_gen_utils import save_box_tensor, load_box_tensor, get_valid_param_dict, get_total_occ_and_dis


def rq3_parser():
    parser = argparse.ArgumentParser(description='synthetic data generation')
    parser.add_argument('--model_dir', type=str, default=True,
                        help='Continued training path')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Test dataset dir')
    # parser.add_argument('--fusion_method', required=True, type=str,
    #                     default='late',
    #                     help='choose one fusion method fo nofusion, late, early or intermediate')

    opt = parser.parse_args()
    return opt


def main():
    opt = rq3_parser()
    dataset_dir = opt.dataset_dir
    model_dir = opt.model_dir

    # Rain, snow, fog, communication delay,
    # global feature lossy communication, channel-specific lossy communication, spatial dislocation
    OPERATOR_LIST = ['RN', 'SW', 'SG', 'CT', 'CL', 'GL', 'SM']  # operators

    # RQ3: Test Retraining Results
    result_stat_3 = {}
    for OPERATOR_NAME in OPERATOR_LIST:
        rq_inference(model_dir, dataset_dir, OPERATOR_NAME)
        operator_result_stat = eval_utils.eval_final_results(result_stat_3, opt.model_dir)
        result_stat_3 += operator_result_stat

    eval_utils.eval_final_results(result_stat_3, opt.model_dir)

if __name__ == '__main__':
    main()







