import os
import shutil


def merge_scene_folders(dataset_dir, target_dir):
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
