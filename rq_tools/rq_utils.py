import os
import random
import shutil
from pathlib import Path


def merge_scene_folders(dataset_dir, target_dir):
    """
    Merge the scene data of the test dataset into the new path.

    Parameters
    ----------
    dataset_dir : str
        Test Dataset path.

    target_dir : str
        Target path.

    Returns
    -------
    """
    target_ego_dir = os.path.join(target_dir, '0')
    target_coop_dir = os.path.join(target_dir, '1')
    os.makedirs(target_ego_dir, exist_ok=True)
    os.makedirs(target_coop_dir, exist_ok=True)

    # count for name
    counter_ego = 0
    counter_coop = 0

    for subdir in os.listdir(dataset_dir):
        dataset_subdir = os.path.join(dataset_dir, subdir)
        if not os.path.isdir(dataset_subdir):
            continue

        dataset_ego_dir = os.path.join(dataset_subdir, '0')
        dataset_coop_dir = os.path.join(dataset_subdir, '1')

        if os.path.exists(dataset_ego_dir):
            files = sorted([f for f in os.listdir(dataset_ego_dir) if f.endswith(('.pcd', '.yaml'))])
            file_groups = {}

            for file in files:
                base = os.path.splitext(file)[0]
                if base not in file_groups:
                    file_groups[base] = []
                file_groups[base].append(file)

                # 按原始序号排序后复制
            for base in sorted(file_groups.keys()):
                for file in file_groups[base]:
                    ext = os.path.splitext(file)[1]
                    new_name = f"{counter_ego:06d}{ext}"
                    shutil.copy2(
                        os.path.join(dataset_ego_dir, file),
                        os.path.join(target_ego_dir, new_name)
                    )
                counter_ego += 1

        if os.path.exists(dataset_coop_dir):
            files = sorted([f for f in os.listdir(dataset_coop_dir) if f.endswith(('.pcd', '.yaml'))])
            file_groups = {}

            for file in files:
                base = os.path.splitext(file)[0]
                if base not in file_groups:
                    file_groups[base] = []
                file_groups[base].append(file)

                # 按原始序号排序后复制
            for base in sorted(file_groups.keys()):
                for file in file_groups[base]:
                    ext = os.path.splitext(file)[1]
                    new_name = f"{counter_coop:06d}{ext}"
                    shutil.copy2(
                        os.path.join(dataset_coop_dir, file),
                        os.path.join(target_coop_dir, new_name)
                    )
                counter_coop += 1


def split_files_randomly(dataset_dir, target_dir):
    """
    Randomly divide the files in the source directory into
    seven groups by serial number and copy them to the destination directory.

    Parameters
    ----------
    dataset_dir : str
        Test Dataset path.

    target_dir : str
        Target path.

    Returns
    -------
    """
    OPERATOR_LIST = ['RN', 'SW', 'SG', 'CT', 'CL', 'GL', 'SM']
    for i in OPERATOR_LIST:
        os.makedirs(os.path.join(target_dir, i), exist_ok=True)

    for source_subdir in ['0', '1']:
        source_path = os.path.join(target_dir, source_subdir)
        if not os.path.exists(source_path):
            continue

        file_indices = set()
        for file in os.listdir(source_path):
            if file.endswith(('.pcd', '.yaml')):
                index = file.split('.')[0]
                file_indices.add(index)

        indices = list(file_indices)
        random.shuffle(indices)
        group_size = len(indices) // 7
        groups = [indices[i * group_size: (i + 1) * group_size] for i in range(7)]

        remainder = indices[7 * group_size:]
        for i, index in enumerate(remainder):
            groups[i].append(index)

        for group_idx, group_indices in enumerate(groups):
            target_dir = os.path.join(target_dir, str(group_idx), source_subdir)
            os.makedirs(target_dir, exist_ok=True)

            for index in group_indices:
                for ext in ['.pcd', '.yaml']:
                    source_file = os.path.join(source_path, f"{index}{ext}")
                    if os.path.exists(source_file):
                        shutil.copy2(source_file, target_dir)






