import os
import random
import shutil
from collections import defaultdict


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

            for base in sorted(file_groups.keys()):
                for file in file_groups[base]:
                    ext = os.path.splitext(file)[1]
                    new_name = f"{counter_coop:06d}{ext}"
                    shutil.copy2(
                        os.path.join(dataset_coop_dir, file),
                        os.path.join(target_coop_dir, new_name)
                    )
                counter_coop += 1


def split_files_randomly(dataset_dir, target_dir, split_list):
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
    file_groups = defaultdict(lambda: defaultdict(dict))

    for op in split_list:
        for subdir in ['0', '1']:
            target_path = os.path.join(target_dir, op, 'metamorph_data', subdir)
            if os.path.exists(target_path) and len(os.listdir(target_path)) != 0:
                shutil.rmtree(target_path)
            os.makedirs(target_path, exist_ok=True)

    for subdir in ["0", "1"]:
        subdir_path = os.path.join(dataset_dir, 'ORI', subdir)
        if not os.path.exists(subdir_path):
            continue

        for filename in os.listdir(subdir_path):
            if filename.endswith((".pcd", ".yaml")):
                base_name = os.path.splitext(filename)[0]
                file_type = "pcd" if filename.endswith(".pcd") else "yaml"
                file_groups[base_name][subdir][file_type] = os.path.join(subdir_path, filename)

    all_indices = sorted(file_groups.keys(), key=lambda x: int(x))
    total_groups = len(all_indices)
    print(f"A total of {total_groups} groups of files have been collected and are ready to be evenly distributed into {len(split_list)} directories...")

    if total_groups == 0:
        print("No files!")
        return

    random.shuffle(all_indices)

    num_operators = len(split_list)
    groups_per_op = total_groups // num_operators
    remainder = total_groups % num_operators

    start_idx = 0
    for i, op in enumerate(split_list):
        end_idx = start_idx + groups_per_op + (1 if i < remainder else 0)
        assigned_indices = all_indices[start_idx:end_idx]

        for idx in assigned_indices:
            group = file_groups[idx]

            if "0" in group:
                for file_type, src_path in group["0"].items():
                    filename = os.path.basename(src_path)
                    dest_path = os.path.join(
                        target_dir, op, "metamorph_data", "0", filename
                    )
                    shutil.copy2(src_path, dest_path)

            if "1" in group:
                for file_type, src_path in group["1"].items():
                    filename = os.path.basename(src_path)
                    dest_path = os.path.join(
                        target_dir, op, "metamorph_data", "1", filename
                    )
                    shutil.copy2(src_path, dest_path)

        start_idx = end_idx


def init_test_files(source_dir, target_dir):
    """
    Merge all files in the source directory into the target directory and renumber them in order.

    Parameters
    ----------
        source_dir : str
            Source dataset path.
        target_dir : str
            Target dataset path.
    """
    ori_dir = os.path.join(target_dir, "ORI")
    for subdir in ["0", "1"]:
        dest_path = os.path.join(ori_dir, subdir)
        if os.path.exists(dest_path) and len(os.listdir(dest_path)) != 0:
            return
        os.makedirs(dest_path, exist_ok=True)


    all_groups = defaultdict(list)

    cav_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]
    cav_dirs.sort()

    for cav in cav_dirs:
        cav_path = os.path.join(source_dir, cav)

        for subdir in ["0", "1"]:
            subdir_path = os.path.join(cav_path, subdir)
            if not os.path.exists(subdir_path):
                continue

            file_groups = defaultdict(dict)
            for filename in os.listdir(subdir_path):
                if filename.endswith((".pcd", ".yaml")):
                    base_name = os.path.splitext(filename)[0]
                    file_type = "pcd" if filename.endswith(".pcd") else "yaml"
                    file_groups[base_name][file_type] = os.path.join(subdir_path, filename)

            sorted_basenames = sorted(file_groups.keys(), key=lambda x: int(x))
            for bn in sorted_basenames:
                all_groups[subdir].append(file_groups[bn])

    for subdir in ["0", "1"]:
        for new_idx, group in enumerate(all_groups[subdir]):
            new_basename = f"{new_idx:06d}"

            if "pcd" in group:
                src_pcd = group["pcd"]
                dest_pcd = os.path.join(ori_dir, subdir, f"{new_basename}.pcd")
                shutil.copy2(src_pcd, dest_pcd)

            if "yaml" in group:
                src_yaml = group["yaml"]
                dest_yaml = os.path.join(ori_dir, subdir, f"{new_basename}.yaml")
                shutil.copy2(src_yaml, dest_yaml)



