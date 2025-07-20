import os
import argparse

from rq_tools.rq_utils import merge_scene_folders, split_files_randomly
from rq_tools.rq_inference import rq_inference
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
    OPERATOR_LIST = ['RN', 'SW', 'SG', 'CT', 'CL', 'GL', 'SM', 'ORI']
    model = os.path.basename(model_path)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset path not exit: {dataset_path}")

    # rebuild dataset path
    dataset_dir = os.path.dirname(dataset_path)
    rq1_dataset_dir = os.path.join(dataset_dir, 'rq1_1', model)

    for OPERATOR_NAME in OPERATOR_LIST:
        target_path = os.path.join(rq1_dataset_dir, OPERATOR_NAME, 'metamorph_data')
        merge_scene_folders(dataset_path, target_path)

        # Statistical untransformed data results
        inference_result = rq_inference(model_path, dataset_path, OPERATOR_NAME)
        eval_utils.eval_final_results(inference_result, model_path)


def rq1_2_eval(dataset_path, model_path):
    """
    # RQ1_2: The test set is randomly assigned to each operator,
    and the data are combined after mutation and the AP results are calculated in different detection ranges
    """
    OPERATOR_LIST = ['RN', 'SW', 'SG', 'CT', 'CL', 'GL', 'SM']
    model = os.path.basename(model_path)

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"dataset path not exit: {dataset_path}")


    # merge scene data
    dataset_dir = os.path.dirname(dataset_path)
    rq1_dataset_dir = os.path.join(dataset_dir, 'rq1_2')
    rq1_source_dataset_dir = os.path.join(rq1_dataset_dir, 'ORI', 'source')
    merge_scene_folders(dataset_path, rq1_source_dataset_dir)

    # rebuild dataset path
    dataset_dir = os.path.dirname(dataset_path)
    rq1_dataset_dir = os.path.join(dataset_dir, 'rq1_2', model)

    split_files_randomly(rq1_source_dataset_dir, rq1_dataset_dir)

    for OPERATOR_NAME in OPERATOR_LIST:
        # Statistical untransformed data results
        inference_result = rq_inference(model_path, dataset_path, OPERATOR_NAME)
        eval_utils.eval_final_results(inference_result, model_path)



def main():
    """
    Enter the Dataset Path and Model Path to complete the contents of rq1_1 and rq1_2
    """
    opt = rq1_parser()
    dataset_dir = opt.dataset_dir
    model_dir = opt.model_dir

    rq1_1_eval(dataset_dir, model_dir)

    rq1_2_eval(dataset_dir, model_dir)


if __name__ == '__main__':
    main()







