import os
import argparse
import random
import shutil
import json


def rq2_split_parser():
    parser = argparse.ArgumentParser(description="rq2 command")
    parser.add_argument('--dataset_dir', help="the dataset path")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    """
    1. Half of the sequences are randomly selected and saved as a training set for retrain
    and a test set for testing.
    2. copy selected data to new folder velodyne/labels/predictions.
    """

    cmd_args = rq2_split_parser()
    all_index_list = list(range(1, 1994))

    selected_index_list = random.sample(all_index_list, len(all_index_list) // 2)   # train dataset
    test_index_list = list(set(all_index_list) - set(selected_index_list))          # test dataset

    # save sequences
    with open("rq2/selected_number.json", "w") as file:
        json.dump({
            "selected": selected_index_list,
            "test": test_index_list
        }, file, indent=4)

    dataset_root = cmd_args.dataset
    v2v_test_root = os.path.join(dataset_root, "test")
    rq2_dataset_root = os.path.join(dataset_root, "rq2")

    trans_folder = os.path.join(dataset_root, "rq2/rq2_select")
    test_folder = os.path.join(dataset_root, "rq3/rq3_test")

    for sub_folder in ["0", "1"]:
        for folder_type in ["velodyne", "labels", "predictions", "pcd"]:
            files = os.listdir(os.path.join(v2v_test_root, sub_folder, folder_type))

            for file in files:
                file_number = int(file.split('.')[0])
                if file_number in selected_index_list:
                    target_folder = trans_folder
                else:
                    target_folder = test_folder

                source_path = os.path.join(v2v_test_root, sub_folder, folder_type, file)
                des_folder = os.path.join(target_folder, sub_folder, folder_type)
                des_path = os.path.join(target_folder, sub_folder, folder_type, file)

                if not os.path.exists(des_folder):
                    os.makedirs(des_folder)

                shutil.copy(source_path, des_path)