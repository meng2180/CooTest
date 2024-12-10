"""
Basedataset class for lidar data pre-processing
"""

import os
import shutil
import random
from collections import OrderedDict

import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset
import opencood.utils.pcd_utils as pcd_utils
import opencood.data_utils.augmentor.augment_utils as Augment
from opencood.data_utils.augmentor.data_augmentor import DataAugmentor
from opencood.hypes_yaml.yaml_utils import load_yaml
from opencood.utils.pcd_utils import downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2, dist_two_pose


class BaseDataset(Dataset):
    """
    Base dataset for all kinds of fusion. Mainly used to assign correct
    index.

    Parameters
    __________
    params : dict
        The dictionary contains all parameters for training/testing.

    visualize : false
        If set to true, the dataset is used for visualization.

    Attributes
    ----------
    scenario_database : OrderedDict
        A structured dictionary contains all file information.

    len_record : list
        The list to record each scenario's data length. This is used to
        retrieve the correct index during training.

    """

    def __init__(self, params, visualize, train=True, isSim=False, dataAugment=None):
        self.params = params
        self.visualize = visualize
        self.train = train
        self.isSim = isSim
        self.dataAugment = dataAugment

        self.pre_processor = None
        self.post_processor = None
        self.data_augmentor = DataAugmentor(params['data_augment'],
                                            train)
        if 'wild_setting' in params:
            self.seed = params['wild_setting']['seed']
            self.async_flag = params['wild_setting']['async']
            self.async_mode = \
                'sim' if 'async_mode' not in params['wild_setting'] \
                    else params['wild_setting']['async_mode']
            self.async_overhead = params['wild_setting']['async_overhead']

            self.loc_err_flag = params['wild_setting']['loc_err']
            self.xyz_noise_std = params['wild_setting']['xyz_std']
            self.ryp_noise_std = params['wild_setting']['ryp_std']
            self.data_size = \
                params['wild_setting']['data_size'] \
                    if 'data_size' in params['wild_setting'] else 0
            self.transmission_speed = \
                params['wild_setting']['transmission_speed']\
                    if 'transmission_speed' in params['wild_setting'] else 27
            self.backbone_delay = \
                params['wild_setting']['backbone_delay'] \
                    if 'backbone_delay' in params['wild_setting'] else 0
            

        else:
            self.async_flag = False
            self.async_overhead = 0 # ms
            self.async_mode = 'sim'
            self.loc_err_flag = False
            self.xyz_noise_std = 0
            self.ryp_noise_std = 0
            self.data_size = 0 # Mb
            self.transmission_speed = 27 # Mbps
            self.backbone_delay = 0 # ms
            self.root_dir = ''

        if self.train:
            root_dir = params['root_dir']
            self.root_dir = params['root_dir']
        else:
            root_dir = params['validate_dir']
            self.root_dir = params['validate_dir']
        if 'max_cav' not in params['train_params']:
            self.max_cav = 7
        else:
            self.max_cav = params['train_params']['max_cav']

        # first load all paths of different scenarios
        self.scenario_folders = sorted([os.path.join(root_dir, x)
                                   for x in os.listdir(root_dir) if
                                   os.path.isdir(os.path.join(root_dir, x))])
        self.scenario_data_number = 0
        self.augment_folders = {}
        self.augment_database = {}
        self.augment_list = {'rain': [],
                             'snow': [],
                             'fog': [],
                             'loc': [],
                             'async': [],
                             'lossy': [],
                             'chlossy': []}
        self.augment_list_t2 = {'rain': [],
                             'snow': [],
                             'fog': [],
                             'loc': [],
                             'async': [],
                             'lossy': [],
                             'chlossy': []}
        self.augment_params = {'rain_rate': [],
                               'snow_rate': [],
                               'visibility': [],
                               'async': [],
                               'tran_x': [],
                               'tran_y': [],
                               'tran_z': [],
                               'yaw': [],
                               'chlossy_p': [],
                               'lossy_p': []}
        self.augmented_data_number = 0
        self.reinitialize()
        # if self.dataAugment is not None:
        if os.path.basename(self.root_dir) == 'augment_data' or \
            self.dataAugment is not None:
            self.augmented_data_init()
        elif os.path.basename(self.root_dir) == 't1':
            self.augmented_data_init()

    def __len__(self):
        return self.len_record[-1]

    def __getitem__(self, idx):
        """
        Abstract method, needs to be define by the children class.
        """
        pass

    def reinitialize(self, augment_flag=False):
        self.scenario_database = OrderedDict()
        self.len_record = []

        # loop over all scenarios
        for (i, scenario_folder) in enumerate(self.scenario_folders):
            self.scenario_database.update({i: OrderedDict()})

            # at least 1 cav should show up
            # at least 1 cav should show up
            if self.train:
                cav_list = [x for x in os.listdir(scenario_folder)
                            if os.path.isdir(
                        os.path.join(scenario_folder, x))]
                random.shuffle(cav_list)
            else:
                cav_list = sorted([x for x in os.listdir(scenario_folder)
                                   if os.path.isdir(
                        os.path.join(scenario_folder, x))])
            assert len(cav_list) > 0

            # roadside unit data's id is always negative, so here we want to
            # make sure they will be in the end of the list as they shouldn't
            # be ego vehicle.
            if int(cav_list[0]) < 0:
                cav_list = cav_list[1:] + [cav_list[0]]

            # loop over all CAV data
            for (j, cav_id) in enumerate(cav_list):
                if j > self.max_cav - 1:
                    print('too many cavs')
                    break
                self.scenario_database[i][cav_id] = OrderedDict()

                # save all yaml files to the dictionary
                cav_path = os.path.join(scenario_folder, cav_id)

                # use the frame number as key, the full path as the values
                # todo: hardcoded to remove additional yamls. no need to worry
                # about this for users.
                yaml_files = \
                    sorted([os.path.join(cav_path, x)
                            for x in os.listdir(cav_path) if
                            x.endswith('.yaml') and 'additional' \
                            not in x and 'camera_gt' not in x])
                timestamps = self.extract_timestamps(yaml_files)

                if cav_id == '0' and not augment_flag:
                    self.scenario_data_number = self.scenario_data_number + int(max(self.extract_timestamps(yaml_files))) + 1
                
                for timestamp in timestamps:
                    self.scenario_database[i][cav_id][timestamp] = \
                        OrderedDict()

                    yaml_file = os.path.join(cav_path,
                                             timestamp + '.yaml')
                    lidar_file = os.path.join(cav_path,
                                              timestamp + '.pcd')
                    camera_files = self.load_camera_files(cav_path, timestamp)                    


                    self.scenario_database[i][cav_id][timestamp]['yaml'] = \
                        yaml_file
                    self.scenario_database[i][cav_id][timestamp]['lidar'] = \
                        lidar_file
                    self.scenario_database[i][cav_id][timestamp]['camera0'] = \
                        camera_files
                # Assume all cavs will have the same timestamps length. Thus
                # we only need to calculate for the first vehicle in the
                # scene.
                if j == 0:
                    self.scenario_database[i][cav_id]['ego'] = True
                    if not self.len_record:
                        self.len_record.append(len(timestamps))
                    else:
                        prev_last = self.len_record[-1]
                        self.len_record.append(prev_last + len(timestamps))
                else:
                    self.scenario_database[i][cav_id]['ego'] = False

        # RQ1: total operation data number = 1993
        # if not augment_flag and self.dataAugment is not None:
        #     if self.dataAugment == 285:
        #         total_list = list(range(1993))
        #         if not augment_flag and self.dataAugment is not None:
        #             keys = list(self.augment_list.keys())
        #             for i, key in enumerate(keys):
        #                 if len(total_list) < 285:
        #                     self.augment_list[key] = total_list
        #                 else:
        #                     for _ in range(285):
        #                         random_index = random.choice(range(len(total_list)))
        #                         self.augment_list[key].append(total_list.pop(random_index))
        #         self.augment_list_t2 = None

        #     # RQ2,3
        #     elif self.dataAugment == 996:
        #         keys = list(self.augment_list.keys())
        #         data_num = self.scenario_data_number
        #         half_list = sorted(random.sample(range(1992), 996))
        #         for i, key in enumerate(keys):
        #             self.augment_list[key] = half_list
        #         for key in keys:
        #             for i in range(1992):
        #                 if i not in self.augment_list[key]:
        #                     self.augment_list_t2[key].append(i)
 
        # if augment_flag:
        #     augment_number = int(self.scenario_data_number / 7)
        augment_keys = list(self.augment_params.keys())



        # for i, key in enumerate(augment_keys):
        #     if self.dataAugment is not None or \
        #         os.path.basename(self.root_dir) == 'augment_data' and augment_flag:
                # for _ in range(augment_number):
                #     if key == 'rain_rate' and len(self.augment_params[key]) < augment_number:
                #         self.augment_params[key].append(round(np.random.uniform(0.1, 10), 1))
                #     elif key == 'snow_rate' and len(self.augment_params[key]) < augment_number:
                #         self.augment_params[key].append(round(np.random.uniform(0.1, 2.4), 1))
                #     elif key == 'visibility' and len(self.augment_params[key]) < augment_number:
                #         self.augment_params[key].append(round(np.random.uniform(200, 1000), 1))
                #     elif key == 'async' and len(self.augment_params[key]) < augment_number:
                #         self.augment_params[key].append(round(np.random.uniform(1, 300), 1))
                #     elif key == 'tran_x' and len(self.augment_params[key]) < augment_number:
                #         self.augment_params[key].append(round(np.random.uniform(-0.2, 0.2), 4))
                #     elif key == 'tran_y' and len(self.augment_params[key]) < augment_number:
                #         self.augment_params[key].append(round(np.random.uniform(-0.2, 0.2), 4))
                #     elif key == 'tran_z' and len(self.augment_params[key]) < augment_number:
                #         self.augment_params[key].append(round(np.random.uniform(-0.2, 0.2), 4))
                #     elif key =='yaw' and len(self.augment_params[key]) < augment_number:
                #         # yaw = 1, rot 60 degree
                #         self.augment_params[key].append(round(np.random.uniform(-0.033, 0.033), 4))
                #     elif key == 'chlossy_p' and len(self.augment_params[key]) < augment_number:
                #         self.augment_params[key].append(round(random.random(), 1))
                #     elif key == 'lossy_p' and len(self.augment_params[key]) < augment_number:
                #         self.augment_params[key].append(round(random.random(), 1))
        if os.path.basename(self.root_dir) == 'augment_data' and augment_flag:
            # save_path = os.path.join(self.root_dir, 'augment_params.txt')
            save_path = '/media/jlutripper/Samsung_T51/V2Vreal/Retrain/augment_params.txt'
            with open(save_path, 'r') as file:
                count = 0
                for line in file:
                    if line.rstrip() in self.augment_params:
                        key = line.rstrip()
                        count = 0
                        continue
                    if count < 996:
                        self.augment_params[key].append(float(line.rstrip()))
                        count += 1


        if os.path.basename(self.root_dir) == 't1' and augment_flag: 

            save_path = os.path.join(self.root_dir, 'augment_params.txt')
            with open(save_path, 'r') as file:
                count = 0
                for line in file:
                    if line.rstrip() in self.augment_params:
                        key = line.rstrip()
                        count = 0
                        continue
                    if count < 996:
                        self.augment_params[key].append(float(line.rstrip()))
                        count += 1
        camera_info = self.scenario_database[i][cav_id][timestamp]['camera0']


    def augmented_data_init(self):
        if os.path.basename(self.root_dir) == 'augment_data':
            augment_data_path = self.root_dir

            self.scenario_folders.clear()

            self.scenario_folders = sorted(os.path.join(augment_data_path, x)
                                        for x in os.listdir(augment_data_path)
                                        if os.path.isdir(os.path.join(augment_data_path, x)))
            self.reinitialize(True)

            # save augment params in save_path(select use)
            # if self.root_dir == '/media/jlutripper/Samsung_T51/V2Vreal/Retrain/retrain/augment_data':
            #     save_path = os.path.join(self.root_dir, 'augment_params.txt')
            #     with open(save_path, 'w') as file:
            #         for key in self.augment_params:
            #             file.write(f"{key}\n")
            #             for param in self.augment_params[key]:
            #                 file.write(f"{param}\n")            
            self.scenario_database[0]['0'].update({'op_tag': 'async'})
            self.scenario_database[0]['1'].update({'op_tag': 'async'})
            self.scenario_database[1]['0'].update({'op_tag': 'chlossy'})
            self.scenario_database[1]['1'].update({'op_tag': 'chlossy'})
            self.scenario_database[2]['0'].update({'op_tag': 'fog'})
            self.scenario_database[2]['1'].update({'op_tag': 'fog'})
            self.scenario_database[3]['0'].update({'op_tag': 'loc_err'})
            self.scenario_database[3]['1'].update({'op_tag': 'loc_err'})
            self.scenario_database[4]['0'].update({'op_tag': 'lossy'})
            self.scenario_database[4]['1'].update({'op_tag': 'lossy'})
            self.scenario_database[5]['0'].update({'op_tag': 'rain'})
            self.scenario_database[5]['1'].update({'op_tag': 'rain'})
            self.scenario_database[6]['0'].update({'op_tag': 'snow'})

            print(f"params num = {len(self.augment_params['async'])}")
            print(f"num = {self.scenario_data_number}") 

        elif os.path.basename(self.root_dir) == 't1':
            augment_data_path = self.root_dir

            self.scenario_folders.clear()

            self.scenario_folders = sorted(os.path.join(augment_data_path, x)
                                        for x in os.listdir(augment_data_path)
                                        if os.path.isdir(os.path.join(augment_data_path, x)))
            
            self.reinitialize(True)

            self.scenario_database[0]['0'].update({'op_tag': 'async'})
            self.scenario_database[0]['1'].update({'op_tag': 'async'})
            self.scenario_database[1]['0'].update({'op_tag': 'chlossy'})
            self.scenario_database[1]['1'].update({'op_tag': 'chlossy'})
            self.scenario_database[2]['0'].update({'op_tag': 'fog'})
            self.scenario_database[2]['1'].update({'op_tag': 'fog'})
            self.scenario_database[3]['0'].update({'op_tag': 'loc_err'})
            self.scenario_database[3]['1'].update({'op_tag': 'loc_err'})
            self.scenario_database[4]['0'].update({'op_tag': 'lossy'})
            self.scenario_database[4]['1'].update({'op_tag': 'lossy'})
            self.scenario_database[5]['0'].update({'op_tag': 'rain'})
            self.scenario_database[5]['1'].update({'op_tag': 'rain'})
            self.scenario_database[6]['0'].update({'op_tag': 'snow'})
            self.scenario_database[6]['1'].update({'op_tag': 'snow'})
            print(f"params num = {len(self.augment_params['async'])}")
            print(f"num = {self.scenario_data_number}")     
        else:  
            path = os.path.dirname(os.path.dirname(self.root_dir))
            rq3_path = os.path.join(path, 'Retrain')
            T1_path = os.path.join(rq3_path, 't1')
            T2_path = os.path.join(rq3_path, 't2')
            if os.path.exists(T1_path):
                shutil.rmtree(T1_path)
                
            if os.path.exists(T2_path):
                shutil.rmtree(T2_path)

            for idx in range(self.scenario_data_number):
                self.augment_data_select(idx)

    def retrieve_base_data(self, idx, cur_ego_pose_flag=True):
        """
        Given the index, return the corresponding data.

        Parameters
        ----------
        idx : int
            Index given by dataloader.

        cur_ego_pose_flag : bool
            Indicate whether to use current timestamp ego pose to calculate
            transformation matrix.

        Returns
        -------
        data : dict
            The dictionary contains loaded yaml params and lidar data for
            each cav.
        """
        # we loop the accumulated length list to see get the scenario index
        if self.dataAugment is not None:
            scenario_index = idx // self.dataAugment
        else:
            for i, ele in enumerate(self.len_record):
                if idx < ele:
                    scenario_index = i
                    break

        scenario_database = self.scenario_database[scenario_index]

        # check the timestamp index
        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]
        
        
        # retrieve the corresponding timestamp key
        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)
        
        # calculate distance to ego for each cav for time delay estimation
        ego_cav_content = \
            self.calc_dist_to_ego(scenario_database, timestamp_key)

        data = OrderedDict()

        # load files for all CAVs
        for cav_id, cav_content in scenario_database.items():
            data[cav_id] = OrderedDict()
            data[cav_id]['ego'] = cav_content['ego']
 
            timestamp_delay = 0

            # add time delay to vehicle parameters
            data[cav_id]['time_delay'] = timestamp_delay
            
            
            if 'op_tag' in cav_content and \
                    cav_content['op_tag'] == 'loc_err':
                data[cav_id]['params'] = self.reform_param(cav_content,
                                                            ego_cav_content,
                                                            timestamp_key,
                                                            timestamp_key,
                                                            cur_ego_pose_flag,
                                                            idx,
                                                            True)
            else:
                data[cav_id]['params'] = self.reform_param(cav_content,
                                                            ego_cav_content,
                                                            timestamp_key,
                                                            timestamp_key,
                                                            cur_ego_pose_flag,
                                                            idx,
                                                            False)
                
            data[cav_id]['lidar_np'] = \
                    pcd_utils.pcd_to_np(cav_content[timestamp_key]['lidar']) 

            data[cav_id]['folder_name'] = \
                cav_content[timestamp_key]['lidar'].split('/')[-3]
            data[cav_id]['index'] = timestamp_index
            data[cav_id]['cav_id'] = int(cav_id)

        return data

    @staticmethod
    def extract_timestamps(yaml_files):
        """
        Given the list of the yaml files, extract the mocked timestamps.

        Parameters
        ----------
        yaml_files : list
            The full path of all yaml files of ego vehicle

        Returns
        -------
        timestamps : list
            The list containing timestamps only.
        """
        timestamps = []

        for file in yaml_files:
            res = file.split('/')[-1]

            timestamp = res.replace('.yaml', '')
            timestamps.append(timestamp)

        return timestamps

    @staticmethod
    def return_timestamp_key(scenario_database, timestamp_index):
        """
        Given the timestamp index, return the correct timestamp key, e.g.
        2 --> '000078'.

        Parameters
        ----------
        scenario_database : OrderedDict
            The dictionary contains all contents in the current scenario.

        timestamp_index : int
            The index for timestamp.

        Returns
        -------
        timestamp_key : str
            The timestamp key saved in the cav dictionary.
        """
        # get all timestamp keys
        timestamp_keys = list(scenario_database.items())[0][1]
        # retrieve the correct index
        timestamp_key = list(timestamp_keys.items())[timestamp_index][0]

        return timestamp_key

    def calc_dist_to_ego(self, scenario_database, timestamp_key):
        """
        Calculate the distance to ego for each cav.
        """
        ego_lidar_pose = None
        ego_cav_content = None
        # Find ego pose first
        for cav_id, cav_content in scenario_database.items():
            if cav_content['ego']:
                ego_cav_content = cav_content
                ego_lidar_pose = \
                    load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
                break

        assert ego_lidar_pose is not None

        # calculate the distance
        for cav_id, cav_content in scenario_database.items():
            cur_lidar_pose = \
                load_yaml(cav_content[timestamp_key]['yaml'])['lidar_pose']
            distance = dist_two_pose(cur_lidar_pose, ego_lidar_pose)
            cav_content['distance_to_ego'] = distance
            scenario_database.update({cav_id: cav_content})

        return ego_cav_content
    

    def time_delay_calculation(self, ego_flag):
        """
        Calculate the time delay for a certain vehicle.

        Parameters
        ----------
        ego_flag : boolean
            Whether the current cav is ego.

        Return
        ------
        time_delay : int
            The time delay quantization.
        """
        # there is not time delay for ego vehicle
        if ego_flag:
            return 0
        # time delay real mode
        if self.async_mode == 'real':
            # noise/time is in ms unit
            overhead_noise = np.random.uniform(0, self.async_overhead)
            tc = self.data_size / self.transmission_speed * 1000
            time_delay = int(overhead_noise + tc + self.backbone_delay)
        elif self.async_mode == 'sim':
            time_delay = np.abs(self.async_overhead)

        # todo: current 10hz, we may consider 20hz in the future
        time_delay = time_delay // 100
        return time_delay if self.async_flag else 0
        
    
    def add_loc_noise(self, pose, xyz_std, ryp_std):
        """
        Add localization noise to the pose.

        Parameters
        ----------
        pose : list
            x,y,z,roll,yaw,pitch

        xyz_std : float
            std of the gaussian noise on xyz

        ryp_std : float
            std of the gaussian noise
        """
        if not self.train:
            np.random.seed(self.seed)

        # build a translation transformation matrix
        tran_matrix = np.array([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]])
        
        # build a rotation matrix around the z-axis
        rot_z_matrix = np.array([[np.cos(0), -np.sin(0), 0, 0],
                                 [np.sin(0), np.cos(0), 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, 1]])
        
        # apply the transformation(translation, z-rotation) to the original pose
        noise_pose = np.dot(tran_matrix, np.dot(rot_z_matrix, pose))
        
        return noise_pose

    def reform_param(self, cav_content, ego_content, timestamp_cur,
                     timestamp_delay, cur_ego_pose_flag, idx, loc_op_flag):
        """
        Reform the data params with current timestamp object groundtruth and
        delay timestamp LiDAR pose.

        Parameters
        ----------
        cav_content : dict
            Dictionary that contains all file paths in the current cav/rsu.

        ego_content : dict
            Ego vehicle content.

        timestamp_cur : str
            The current timestamp.(当前数据在文件中的编号)

        timestamp_delay : str
            The delayed timestamp.(时间戳索引大于延迟时取差值, 否则为0)

        cur_ego_pose_flag : bool
            Whether use current ego pose to calculate transformation matrix.

        Return
        ------
        The merged parameters.
        """
        cur_params = load_yaml(cav_content[timestamp_cur]['yaml'])
        delay_params = load_yaml(cav_content[timestamp_delay]['yaml'])
        # print(cur_params)
        cur_ego_params = load_yaml(ego_content[timestamp_cur]['yaml'])
        delay_ego_params = load_yaml(ego_content[timestamp_delay]['yaml'])

        # we need to calculate the transformation matrix from cav to ego
        # at the delayed timestamp
        delay_cav_lidar_pose = delay_params['lidar_pose']
        delay_ego_lidar_pose = delay_ego_params["lidar_pose"]

        cur_ego_lidar_pose = cur_ego_params['lidar_pose']
        cur_cav_lidar_pose = cur_params['lidar_pose']

        if not cav_content['ego'] and loc_op_flag:
            tran_x = self.augment_params['tran_x'][int(timestamp_cur)]
            tran_y = self.augment_params['tran_y'][int(timestamp_cur)]
            tran_z = self.augment_params['tran_z'][int(timestamp_cur)]
            yaw = self.augment_params['yaw'][int(timestamp_cur)]
            delay_cav_lidar_pose = Augment.loc_op(delay_cav_lidar_pose, tran_x, tran_y, tran_z, yaw)

            cur_cav_lidar_pose = Augment.loc_op(cur_cav_lidar_pose, tran_x, tran_y, tran_z, yaw)
        else:
            delay_cav_lidar_pose = self.add_loc_noise(delay_cav_lidar_pose,
                                                      self.xyz_noise_std,
                                                      self.ryp_noise_std)

            cur_cav_lidar_pose = self.add_loc_noise(cur_cav_lidar_pose,
                                                    self.xyz_noise_std,
                                                    self.ryp_noise_std)

        if cur_ego_pose_flag:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             cur_ego_lidar_pose)
            spatial_correction_matrix = np.eye(4)
        else:
            transformation_matrix = x1_to_x2(delay_cav_lidar_pose,
                                             delay_ego_lidar_pose)
            spatial_correction_matrix = x1_to_x2(delay_ego_lidar_pose,
                                                 cur_ego_lidar_pose)
        # This is only used for late fusion, as it did the transformation
        # in the postprocess, so we want the gt object transformation use
        # the correct one
        gt_transformation_matrix = x1_to_x2(cur_cav_lidar_pose,
                                            cur_ego_lidar_pose)

        # we always use current timestamp's gt bbx to gain a fair evaluation
        delay_params['vehicles'] = cur_params['vehicles']
        delay_params['transformation_matrix'] = transformation_matrix
        delay_params['gt_transformation_matrix'] = \
            gt_transformation_matrix
        delay_params['spatial_correction_matrix'] = spatial_correction_matrix

        return delay_params

    @staticmethod
    def load_camera_files(cav_path, timestamp):
        """
        Retrieve the paths to all camera files.

        Parameters
        ----------
        cav_path : str
            The full file path of current cav.

        timestamp : str
            Current timestamp

        Returns
        -------
        camera_files : list
            The list containing all camera png file paths.
        """
        camera0_file = os.path.join(cav_path,
                                    timestamp + '_camera0.png')
        camera1_file = os.path.join(cav_path,
                                    timestamp + '_camera1.png')
        camera2_file = os.path.join(cav_path,
                                    timestamp + '_camera2.png')
        camera3_file = os.path.join(cav_path,
                                    timestamp + '_camera3.png')
        return [camera0_file, camera1_file, camera2_file, camera3_file]

    def project_points_to_bev_map(self, points, ratio=0.1):
        """
        Project points to BEV occupancy map with default ratio=0.1.

        Parameters
        ----------
        points : np.ndarray
            (N, 3) / (N, 4)

        ratio : float
            Discretization parameters. Default is 0.1.

        Returns
        -------
        bev_map : np.ndarray
            BEV occupancy map including projected points
            with shape (img_row, img_col).

        """
        return self.pre_processor.project_points_to_bev_map(points, ratio)

    def augment(self, lidar_np, object_bbx_center, object_bbx_mask,
                flip=None, rotation=None, scale=None):
        """
        """
        tmp_dict = {'lidar_np': lidar_np,
                    'object_bbx_center': object_bbx_center,
                    'object_bbx_mask': object_bbx_mask,
                    'flip': flip,
                    'noise_rotation': rotation,
                    'noise_scale': scale}
        tmp_dict = self.data_augmentor.forward(tmp_dict)

        lidar_np = tmp_dict['lidar_np']
        object_bbx_center = tmp_dict['object_bbx_center']
        object_bbx_mask = tmp_dict['object_bbx_mask']

        return lidar_np, object_bbx_center, object_bbx_mask

    def collate_batch_train(self, batch):
        """
        Customized collate function for pytorch dataloader during training
        for late fusion dataset.

        Parameters
        ----------
        batch : dict

        Returns
        -------
        batch : dict
            Reformatted batch.
        """
        # during training, we only care about ego.
        output_dict = {'ego': {}}

        object_bbx_center = []
        object_bbx_mask = []
        processed_lidar_list = []
        label_dict_list = []

        if self.visualize:
            origin_lidar = []

        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            object_bbx_center.append(ego_dict['object_bbx_center'])
            object_bbx_mask.append(ego_dict['object_bbx_mask'])
            processed_lidar_list.append(ego_dict['processed_lidar'])
            label_dict_list.append(ego_dict['label_dict'])

            if self.visualize:
                origin_lidar.append(ego_dict['origin_lidar'])

        # convert to numpy, (B, max_num, 7)
        object_bbx_center = torch.from_numpy(np.array(object_bbx_center))
        object_bbx_mask = torch.from_numpy(np.array(object_bbx_mask))

        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(processed_lidar_list)
        label_torch_dict = \
            self.post_processor.collate_batch(label_dict_list)
        output_dict['ego'].update({'object_bbx_center': object_bbx_center,
                                   'object_bbx_mask': object_bbx_mask,
                                   'processed_lidar': processed_lidar_torch_dict,
                                   'label_dict': label_torch_dict})
        if self.visualize:
            origin_lidar = \
                np.array(downsample_lidar_minimum(pcd_np_list=origin_lidar))
            origin_lidar = torch.from_numpy(origin_lidar)
            output_dict['ego'].update({'origin_lidar': origin_lidar})

        return output_dict

    def visualize_result(self, pred_box_tensor,
                         gt_tensor,
                         pcd,
                         show_vis,
                         save_path,
                         dataset=None):
        self.post_processor.visualize(pred_box_tensor,
                                      gt_tensor,
                                      pcd,
                                      show_vis,
                                      save_path,
                                      dataset=dataset)
        
    def augment_data_select(self, idx, cur_ego_pose_flag=True):

        for i, ele in enumerate(self.len_record):
            if idx < ele:
                scenario_index = i
                break
        
        scenario_database = self.scenario_database[scenario_index]

        timestamp_index = idx if scenario_index == 0 else \
            idx - self.len_record[scenario_index - 1]

        timestamp_key = self.return_timestamp_key(scenario_database,
                                                  timestamp_index)
        
        # ego_cav_content = self.calc_dist_to_ego(scenario_database, timestamp_key)

        selected_data = OrderedDict()

        for cav_id, cav_content in scenario_database.items():
            selected_data[cav_id] = OrderedDict()
            selected_data[cav_id]['ego'] = cav_content['ego']

            new_path = os.path.dirname(os.path.dirname(self.root_dir))
            
            # build t1 path
            ts_path = os.path.join(new_path, 'test_simple')
            rq3_path = os.path.join(ts_path, 'Retrain')
            t1_path = os.path.join(rq3_path, 't1')
            if not os.path.exists(t1_path):
                os.makedirs(t1_path)

            # build t2 path
            t2_path = os.path.join(rq3_path, 't2')
            if not os.path.exists(t2_path):
                os.makedirs(t2_path)

            if idx in self.augment_list['async'] and \
                self.dataAugment is not None:
                index = 0
                
                for i, value in enumerate(self.augment_list['async']):
                    if value == idx:
                        index = i  

                time_delay = self.augment_params['async'][index]
                timestamp_delay = \
                    Augment.async_op(cav_content['ego'], time_delay)
                
                    
                                
                if timestamp_index - timestamp_delay <= 0:
                    timestamp_delay = timestamp_index

                timestamp_index_delay = max(0, timestamp_index - timestamp_delay)

                timestamp_key_delay = self.return_timestamp_key(scenario_database,
                                                        timestamp_index_delay)
                
                if cav_content['ego']:
                    ego_cav_path = cav_content[timestamp_key]['lidar']
                    ego_label_path = cav_content[timestamp_key]['yaml']
                    Augment.copy_selected_data(ego_cav_path, ego_label_path,
                                               t1_path, index, 'async', True)

                if not cav_content['ego']:
                    delay_cav_path = cav_content[timestamp_key_delay]['lidar']
                    delay_label_path = cav_content[timestamp_key_delay]['yaml']
                    Augment.copy_selected_data(delay_cav_path, delay_label_path,
                                               t1_path, index, 'async', False)
            elif idx in self.augment_list_t2['async'] and \
                self.dataAugment is not None:     
                index = 0
                
                for i, value in enumerate(self.augment_list_t2['async']):
                    if value == idx:
                        index = i                
                time_delay = self.augment_params['async'][index]
                timestamp_delay = \
                    Augment.async_op(cav_content['ego'], time_delay)
                     
                                
                if timestamp_index - timestamp_delay <= 0:
                    timestamp_delay = timestamp_index

                timestamp_index_delay = max(0, timestamp_index - timestamp_delay)

                timestamp_key_delay = self.return_timestamp_key(scenario_database,
                                                        timestamp_index_delay)
                
                if cav_content['ego']:
                    ego_cav_path = cav_content[timestamp_key]['lidar']
                    ego_label_path = cav_content[timestamp_key]['yaml']
                    Augment.copy_selected_data(ego_cav_path, ego_label_path,
                                               t2_path, index, 'async', True)

                if not cav_content['ego']:
                    delay_cav_path = cav_content[timestamp_key_delay]['lidar']
                    delay_label_path = cav_content[timestamp_key_delay]['yaml']
                    Augment.copy_selected_data(delay_cav_path, delay_label_path,
                                               t2_path, index, 'async', False)
                
            if idx in self.augment_list['loc'] and \
                self.dataAugment is not None:
                index = 0
                for i, value in enumerate(self.augment_list['loc']):
                    if value == idx:
                        index = i  

                if cav_content['ego']:
                    ego_cav_path = cav_content[timestamp_key]['lidar']
                    ego_label_path = cav_content[timestamp_key]['yaml']
                    Augment.copy_selected_data(ego_cav_path, ego_label_path,
                                               t1_path, index, 'loc', True)
                    
                if not cav_content['ego']:
                    coop_cav_path = cav_content[timestamp_key]['lidar']
                    coop_label_path = cav_content[timestamp_key]['yaml']
                    Augment.copy_selected_data(coop_cav_path, coop_label_path,
                                               t1_path, index, 'loc', False)
            elif idx in self.augment_list_t2['loc'] and \
                self.dataAugment is not None:
                index = 0
                for i, value in enumerate(self.augment_list_t2['loc']):
                    if value == idx:
                        index = i  

                if cav_content['ego']:
                    ego_cav_path = cav_content[timestamp_key]['lidar']
                    ego_label_path = cav_content[timestamp_key]['yaml']
                    Augment.copy_selected_data(ego_cav_path, ego_label_path,
                                               t2_path, index, 'loc', True)
                    
                if not cav_content['ego']:
                    coop_cav_path = cav_content[timestamp_key]['lidar']
                    coop_label_path = cav_content[timestamp_key]['yaml']
                    Augment.copy_selected_data(coop_cav_path, coop_label_path,
                                               t2_path, index, 'loc', False)

            if idx in self.augment_list['rain'] and \
                self.dataAugment is not None:

                index = 0
                for i, value in enumerate(self.augment_list['rain']):
                    if value == idx:
                        index = i  

                if cav_content['ego']:
                    ego_cav_path = cav_content[timestamp_key]['lidar']
                    ego_label_path = cav_content[timestamp_key]['yaml']
                    rain_pcd_path = Augment.copy_selected_data(ego_cav_path, ego_label_path,
                                               t1_path, index, 'rain', True)
                if not cav_content['ego']:
                    coop_cav_path = cav_content[timestamp_key]['lidar']
                    coop_label_path = cav_content[timestamp_key]['yaml']
                    rain_pcd_path = Augment.copy_selected_data(coop_cav_path, coop_label_path,
                                               t1_path, index, 'rain', False)
                rain_rate = self.augment_params['rain_rate'][index]
                Augment.rain_op(rain_pcd_path, rain_rate)
            elif idx in self.augment_list_t2['rain'] and \
                self.dataAugment is not None:

                index = 0
                for i, value in enumerate(self.augment_list_t2['rain']):
                    if value == idx:
                        index = i  

                if cav_content['ego']:
                    ego_cav_path = cav_content[timestamp_key]['lidar']
                    ego_label_path = cav_content[timestamp_key]['yaml']
                    rain_pcd_path = Augment.copy_selected_data(ego_cav_path, ego_label_path,
                                               t2_path, index, 'rain', True)
                if not cav_content['ego']:
                    coop_cav_path = cav_content[timestamp_key]['lidar']
                    coop_label_path = cav_content[timestamp_key]['yaml']
                    rain_pcd_path = Augment.copy_selected_data(coop_cav_path, coop_label_path,
                                               t2_path, index, 'rain', False)
                rain_rate = self.augment_params['rain_rate'][index]
                Augment.rain_op(rain_pcd_path, rain_rate)


            if idx in self.augment_list['snow'] and \
                self.dataAugment is not None:
                index = 0
                for i, value in enumerate(self.augment_list['snow']):
                    if value == idx:
                        index = i  

                if cav_content['ego']:
                    ego_cav_path = cav_content[timestamp_key]['lidar']
                    ego_label_path = cav_content[timestamp_key]['yaml']
                    snow_pcd_path = Augment.copy_selected_data(ego_cav_path, ego_label_path,
                                               t1_path, index, 'snow', True)
                if not cav_content['ego']:
                    coop_cav_path = cav_content[timestamp_key]['lidar']
                    coop_label_path = cav_content[timestamp_key]['yaml']
                    snow_pcd_path = Augment.copy_selected_data(coop_cav_path, coop_label_path,
                                               t1_path, index, 'snow', False)
                snow_rate = self.augment_params['snow_rate'][index]
                Augment.snow_op(snow_pcd_path, snow_rate)
            elif idx in self.augment_list_t2['snow'] and \
                self.dataAugment is not None:
                index = 0
                for i, value in enumerate(self.augment_list_t2['snow']):
                    if value == idx:
                        index = i  

                if cav_content['ego']:
                    ego_cav_path = cav_content[timestamp_key]['lidar']
                    ego_label_path = cav_content[timestamp_key]['yaml']
                    snow_pcd_path = Augment.copy_selected_data(ego_cav_path, ego_label_path,
                                               t2_path, index, 'snow', True)
                if not cav_content['ego']:
                    coop_cav_path = cav_content[timestamp_key]['lidar']
                    coop_label_path = cav_content[timestamp_key]['yaml']
                    snow_pcd_path = Augment.copy_selected_data(coop_cav_path, coop_label_path,
                                               t2_path, index, 'snow', False)
                snow_rate = self.augment_params['snow_rate'][index]
                Augment.snow_op(snow_pcd_path, snow_rate)
                    
            if idx in self.augment_list['fog'] and \
                self.dataAugment is not None:

                index = 0
                for i, value in enumerate(self.augment_list['fog']):
                    if value == idx:
                        index = i  

                if cav_content['ego']:
                    ego_cav_path = cav_content[timestamp_key]['lidar']
                    ego_label_path = cav_content[timestamp_key]['yaml']
                    fog_pcd_path = Augment.copy_selected_data(ego_cav_path, ego_label_path,
                                               t1_path, index, 'fog', True)
                if not cav_content['ego']:
                    coop_cav_path = cav_content[timestamp_key]['lidar']
                    coop_label_path = cav_content[timestamp_key]['yaml']
                    fog_pcd_path = Augment.copy_selected_data(coop_cav_path, coop_label_path,
                                               t1_path, index, 'fog', False)
                visibility = self.augment_params['visibility'][index]
                Augment.fog_op(fog_pcd_path, visibility)
            elif idx in self.augment_list_t2['fog'] and \
                self.dataAugment is not None:

                index = 0
                for i, value in enumerate(self.augment_list_t2['fog']):
                    if value == idx:
                        index = i  

                if cav_content['ego']:
                    ego_cav_path = cav_content[timestamp_key]['lidar']
                    ego_label_path = cav_content[timestamp_key]['yaml']
                    fog_pcd_path = Augment.copy_selected_data(ego_cav_path, ego_label_path,
                                               t2_path, index, 'fog', True)
                if not cav_content['ego']:
                    coop_cav_path = cav_content[timestamp_key]['lidar']
                    coop_label_path = cav_content[timestamp_key]['yaml']
                    fog_pcd_path = Augment.copy_selected_data(coop_cav_path, coop_label_path,
                                               t2_path, index, 'fog', False)
                visibility = self.augment_params['visibility'][index]
                Augment.fog_op(fog_pcd_path, visibility)
                   
            if idx in self.augment_list['lossy'] and \
                self.dataAugment is not None:

                index = 0
                for i, value in enumerate(self.augment_list['lossy']):
                    if value == idx:
                        index = i  

                if cav_content['ego']:
                    ego_cav_path = cav_content[timestamp_key]['lidar']
                    ego_label_path = cav_content[timestamp_key]['yaml']
                    Augment.copy_selected_data(ego_cav_path, ego_label_path,
                                               t1_path, index, 'lossy', True)
                if not cav_content['ego']:
                    coop_cav_path = cav_content[timestamp_key]['lidar']
                    coop_label_path = cav_content[timestamp_key]['yaml']
                    Augment.copy_selected_data(coop_cav_path, coop_label_path,
                                               t1_path, index, 'lossy', False)
            elif idx in self.augment_list_t2['lossy'] and \
                self.dataAugment is not None:

                index = 0
                for i, value in enumerate(self.augment_list_t2['lossy']):
                    if value == idx:
                        index = i  

                if cav_content['ego']:
                    ego_cav_path = cav_content[timestamp_key]['lidar']
                    ego_label_path = cav_content[timestamp_key]['yaml']
                    Augment.copy_selected_data(ego_cav_path, ego_label_path,
                                               t2_path, index, 'lossy', True)
                if not cav_content['ego']:
                    coop_cav_path = cav_content[timestamp_key]['lidar']
                    coop_label_path = cav_content[timestamp_key]['yaml']
                    Augment.copy_selected_data(coop_cav_path, coop_label_path,
                                               t2_path, index, 'lossy', False)
                    
            if idx in self.augment_list['chlossy'] and \
                self.dataAugment is not None:

                index = 0
                for i, value in enumerate(self.augment_list['chlossy']):
                    if value == idx:
                        index = i  

                if cav_content['ego']:
                    ego_cav_path = cav_content[timestamp_key]['lidar']
                    ego_label_path = cav_content[timestamp_key]['yaml']
                    Augment.copy_selected_data(ego_cav_path, ego_label_path,
                                               t1_path, index, 'chlossy', True)
                if not cav_content['ego']:
                    coop_cav_path = cav_content[timestamp_key]['lidar']
                    coop_label_path = cav_content[timestamp_key]['yaml']
                    Augment.copy_selected_data(coop_cav_path, coop_label_path,
                                               t1_path, index, 'chlossy', False)
            elif idx in self.augment_list_t2['chlossy'] and \
                self.dataAugment is not None:

                index = 0
                for i, value in enumerate(self.augment_list_t2['chlossy']):
                    if value == idx:
                        index = i  

                if cav_content['ego']:
                    ego_cav_path = cav_content[timestamp_key]['lidar']
                    ego_label_path = cav_content[timestamp_key]['yaml']
                    Augment.copy_selected_data(ego_cav_path, ego_label_path,
                                               t2_path, index, 'chlossy', True)
                if not cav_content['ego']:
                    coop_cav_path = cav_content[timestamp_key]['lidar']
                    coop_label_path = cav_content[timestamp_key]['yaml']
                    Augment.copy_selected_data(coop_cav_path, coop_label_path,
                                               t2_path, index, 'chlossy', False)
            
   