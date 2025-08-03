import os
import shutil
import argparse

from logger import CLogger
from rq_tools.rq_utils import merge_scene_folders, split_files_randomly, init_test_files
from rq_tools.rq_inference import inference
from opencood.utils import eval_utils


def rq1_parser():
    parser = argparse.ArgumentParser(description='synthetic data generation')
    parser.add_argument('--model_dir', type=str, default=True,
                        help='Continued training path')
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Test dataset dir')
    opt = parser.parse_args()
    return opt


def rq1_1_eval(dataset_path, model_path):
    """
    RQ1_1: Operators are applied to the test set data,
    and the statistical model applies the AP results of each operator.
    """
    OPERATOR_LIST = ['ORI', 'RN', 'SW', 'FG', 'CT', 'CL', 'GL', 'SM']
    model = os.path.basename(model_path)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset path not exit: {dataset_path}")

    # rebuild dataset path
    dataset_dir = os.path.dirname(dataset_path)
    rq1_dataset_dir = os.path.join(dataset_dir, 'rq1_1', model)

    if os.path.exists(rq1_dataset_dir):
        shutil.rmtree(rq1_dataset_dir)
    os.makedirs(rq1_dataset_dir, exist_ok=True)

    for OPERATOR_NAME in OPERATOR_LIST:
        target_path = os.path.join(rq1_dataset_dir, OPERATOR_NAME, 'metamorph_data')

        # generate the metamorph dataset folder
        merge_scene_folders(dataset_path, target_path)
        meta_dataset_path = os.path.dirname(target_path)

        # output the inference results of the data after applying the operator
        result_stat =  inference(model_path, meta_dataset_path, OPERATOR_NAME)
        eval_utils.eval_final_results(result_stat, model_path)


def rq1_2_eval(dataset_path, model_path):
    """
    # RQ1_2: The test set is randomly assigned to each operator,
    and the data are combined after mutation and the AP results are calculated in different detection ranges
    """
    OPERATOR_LIST = ['RN', 'SW', 'FG', 'CT', 'CL', 'GL', 'SM']
    model = os.path.basename(model_path)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset path not exit: {dataset_path}")

    # rebuild dataset path
    dataset_dir = os.path.dirname(dataset_path)
    rq1_dataset_dir = os.path.join(dataset_dir, 'rq1_2', model)

    if os.path.exists(rq1_dataset_dir):
        shutil.rmtree(rq1_dataset_dir)
    os.makedirs(rq1_dataset_dir, exist_ok=True)

    init_test_dir = os.path.join(dataset_dir, 'rq_test')
    init_test_files(dataset_path, init_test_dir)

    # random distribute the test dataset for each operator
    split_files_randomly(init_test_dir, rq1_dataset_dir, OPERATOR_LIST)

    result_stats = {
            0.5: {'tp': [], 'fp': [], 'gt': 0},
            0.7: {'tp': [], 'fp': [], 'gt': 0}
    }

    for OPERATOR_NAME in OPERATOR_LIST:
        meta_dataset_path = os.path.join(rq1_dataset_dir, OPERATOR_NAME)
        result_stat = inference(model_path, meta_dataset_path, OPERATOR_NAME)
        result_stats[0.5]['tp'] += result_stat[0.5]['tp']
        result_stats[0.5]['fp'] += result_stat[0.5]['fp']
        result_stats[0.5]['gt'] += result_stat[0.5]['gt']

    eval_utils.eval_final_results(result_stats, model_path)


def main():
    """
    Enter the Dataset Path and Model Path to complete the contents of rq1_1 and rq1_2
    """
    opt = rq1_parser()
    dataset_dir = opt.dataset_dir
    model_dir = opt.model_dir

    CLogger.info(f"Start building the radar chart data for rq1.")
    # rq1_1_eval(dataset_dir, model_dir)

    CLogger.info(f"Start reasoning about the impact of mutation operators on evaluation for rq1.")
    rq1_2_eval(dataset_dir, model_dir)


if __name__ == '__main__':
    main()







