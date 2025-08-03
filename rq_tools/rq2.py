import os
import shutil
import argparse

from logger import CLogger
from rq_tools.rq_utils import split_files_randomly, init_test_files, merge_scene_folders
from rq_tools.rq2_inference import inference
from opencood.utils import eval_utils


def rq2_parser():
    parser = argparse.ArgumentParser(description='synthetic data generation')
    parser.add_argument('--model_dir', type=str, default=True,
                        help='Continued training path')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Test dataset dir')
    opt = parser.parse_args()
    return opt

def main():
    """
    Enter the Dataset Path and Model Path to complete the contents of rq1_1 and rq1_2
    """
    opt = rq2_parser()
    dataset_path = opt.dataset_dir
    model_dir = opt.model_dir
    model = os.path.basename(model_dir)

    OPERATOR_LIST = ['RN', 'SW', 'FG', 'CT', 'CL', 'GL', 'SM']
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset path not exit: {dataset_path}")

    # rebuild dataset path
    dataset_dir = os.path.dirname(dataset_path)
    init_test_dir = os.path.join(dataset_dir, 'rq_test')
    init_test_files(dataset_path, init_test_dir)

    # randomly divide the test dataset into two parts.
    rq2_dataset_dir = os.path.join(dataset_dir, 'rq2')
    SPLIT_LIST = ['test', 'train']
    split_files_randomly(init_test_dir, rq2_dataset_dir, SPLIT_LIST)

    # add operators to train dataset, select to retrain
    rq2_train_dir = os.path.join(dataset_dir, 'train')
    rq2_select_dir = os.path.join(dataset_dir, 'rq2_select', model)

    if os.path.exists(rq2_select_dir):
        shutil.rmtree(rq2_select_dir)
    os.makedirs(rq2_select_dir, exist_ok=True)

    # 1. transform of 7 operators and record cooperative errors
    for OPERATOR_NAME in OPERATOR_LIST:
        # generate the metamorph dataset folders
        target_path = os.path.join(rq2_select_dir, OPERATOR_NAME, 'metamorph_data')
        merge_scene_folders(rq2_train_dir, target_path)

        meta_dataset_path = os.path.dirname(target_path)
        inference(model_dir, meta_dataset_path, OPERATOR_NAME)

    #  2. random or coo-test method choose the retrain data
    #  3. retrain data
    #  4. test models



if __name__ == '__main__':
    main()







