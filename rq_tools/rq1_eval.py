import argparse
import statistics
import sys
import os

import torch
from torch.utils.data import DataLoader
from rq_tools.rq_utils import merge_scene_folders

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, infrence_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.rq_eval.rq2_data_select import CooTest_method_result, V2X_Gen_method
from opencood.rq_eval.v2x_gen_utils import save_box_tensor, load_box_tensor, get_valid_param_dict, get_total_occ_and_dis


def rq1_parser():
    """
    1. 读取数据
    2. 分离数据
    3. 统计评估结果
    4. 合并结果
    """
    parser = argparse.ArgumentParser(description='synthetic data generation')
    parser.add_argument('--model_dir', type=str, default=True,
                        help='Continued training path')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Test dataset dir')
    parser.add_argument('--fusion_method', required=True, type=str,
                        default='late',
                        help='choose one fusion method fo nofusion, late, early or intermediate')

    opt = parser.parse_args()
    return opt


def main():
    opt = rq1_parser()
    dataset_dir = opt.dataset_dir
    model_dir = opt.model_dir

    # 雨、雪、雾、通讯延迟、全局特征有损通信、信道特定有损通信、空间错位
    OPERATOR_LIST = ['RN', 'SW', 'SG', 'CT', 'CL', 'GL', 'SM']  # operators

    # merge scene data for rq1
    rq1_dataset_dir = os.path.join(os.path.dirname(dataset_dir), 'rq1')
    rq1_source_dataset_dir = os.path.join(rq1_dataset_dir, 'source', 'merge_data')

    # '_dataset/rq1/source/merge_data/0,1'

    merge_scene_folders(dataset_dir, rq1_source_dataset_dir)
    # TODO: RQ1_1: 使用指定模型为 test 每个数据进行变换，最后统计结果



    # TODO: RQ1_2: 将 test 随机分给七个目标算子，进行变换后统计结果



if __name__ == '__main__':
    main()







