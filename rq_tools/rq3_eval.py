import os
import shutil
import argparse

from logger import CLogger
from rq_tools.rq_utils import merge_scene_folders, split_files_randomly, init_test_files
from rq_tools.rq_inference import inference
from opencood.utils import eval_utils


def rq_parser():
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
    opt = rq_parser()
    dataset_path = opt.dataset_dir
    model_path = opt.model_dir

    OPERATOR_LIST = ['RN', 'SW', 'FG', 'CT', 'CL', 'GL', 'SM']
    model = os.path.basename(model_path)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset path not exit: {dataset_path}")

    # rebuild dataset path
    dataset_dir = os.path.dirname(os.path.dirname(dataset_path))
    rq3_dataset_dir = os.path.join(dataset_dir, 'rq3', model)
    print(rq3_dataset_dir)

    if os.path.exists(rq3_dataset_dir):
        shutil.rmtree(rq3_dataset_dir)
    os.makedirs(rq3_dataset_dir, exist_ok=True)

    for OPERATOR_NAME in OPERATOR_LIST:
        target_path = os.path.join(rq3_dataset_dir, OPERATOR_NAME, 'metamorph_data')

        # generate the metamorph dataset folder
        merge_scene_folders(dataset_path, target_path)
        meta_dataset_path = os.path.dirname(target_path)

        # output the inference results of the data after applying the operator
        result_stat = inference(model_path, meta_dataset_path, OPERATOR_NAME)
        eval_utils.eval_final_results(result_stat, model_path)

if __name__ == '__main__':
    main()







