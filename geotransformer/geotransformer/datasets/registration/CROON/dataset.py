import os.path as osp
import random
import csv
import pandas as pd
from secrets import choice
# from mathutils import Matrix, Vector, Quaternion, Euler
import numpy as np
import torch.utils.data
import open3d as o3d

from geotransformer.utils.common import load_pickle
from geotransformer.utils.pointcloud import (
    random_sample_rotation,
    get_transform_from_rotation_translation,
    get_rotation_translation_from_transform,
    eulerAnglesToRotationMatrix,
)
from geotransformer.utils.registration import get_correspondences
from geotransformer.modules.ops import (
    apply_transform, 
    inverse_transform,
)

class Lidar2LidarDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_min_scale=0.8,
        augmentation_max_scale=1.2,
        augmentation_shift=2.0,
        augmentation_rotation=1.0,
        return_corr_indices=False,
        matching_radius=None,
        max_t=10, 
        max_r=180.,

    ):
        super(Lidar2LidarDataset, self).__init__()

        self.dataset_root = '/mnt/lustre/weipengjin/geotransformer/data/road_scene'
        self.subset = subset
        self.point_limit = point_limit

        self.use_augmentation = use_augmentation
        self.augmentation_noise = augmentation_noise
        self.augmentation_min_scale = augmentation_min_scale
        self.augmentation_max_scale = augmentation_max_scale
        self.augmentation_shift = augmentation_shift
        self.augmentation_rotation = augmentation_rotation

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        self.max_r = max_r
        self.max_t = max_t 
        data_file = open('../../data/road_scene/metadata/train.txt','r') 
        self.data_train = data_file.readlines()
        data_file = open('../../data/road_scene/metadata/val.txt','r')
        self.data_val = data_file.readlines()
        data_file = open('../../data/road_scene/metadata/test.txt','r')
        self.data_test = data_file.readlines()     
        self.lidar_type = 'back'

        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError('"matching_radius" is None but "return_corr_indices" is set.')

        # self.metadata = load_pickle(osp.join(self.dataset_root, 'metadata', f'{subset}.pkl'))
        # load or genetate RT matrix 
        if subset == 'train':
            self.train_RT = []
            train_RT_file = osp.join(self.dataset_root, 'metadata',
                                        f'Lidar2Lidar_{subset}_RT_{max_r:.2f}_{max_t:.2f}.csv')

            if osp.exists(train_RT_file):
                print(f'TRAIN SET: Using this file: {train_RT_file}')
                df_train_RT = pd.read_csv(train_RT_file, sep=',')
                for index, row in df_train_RT.iterrows():
                    self.train_RT.append(list(row))
            else:
                print(f'TRAIN SET - Not found: {train_RT_file}')
                print("Generating a new one")
                train_RT_file = open(train_RT_file, 'w')
                train_RT_file = csv.writer(train_RT_file, delimiter=',')
                train_RT_file.writerow(['id', 'pair', 'Yaw', 'Pitch', 'Roll', 'tx', 'ty', 'tz', 'classify_label'])

                for i in range(len(self.data_train)):
                    data_seed = random.randint(1,10)  #大于5数据为配准失败（负样本），小于等于5为配准成功（正样本）
                    if data_seed > 10 :
                        max_angle = self.max_r
                        rotz = np.random.uniform(*(choice([(-max_angle,-5),(5,max_angle)]))) * (3.141592 / 180.0)
                        roty = np.random.uniform(*(choice([(-max_angle,-5),(5,max_angle)]))) * (3.141592 / 180.0)
                        rotx = np.random.uniform(*(choice([(-max_angle,-5),(5,max_angle)]))) * (3.141592 / 180.0)
                        transl_x = np.random.uniform(*(choice([(-self.max_t,-2),(2,self.max_t)])))
                        transl_y = np.random.uniform(*(choice([(-self.max_t,-2),(2,self.max_t)])))
                        transl_z = np.random.uniform(*(choice([(-self.max_t,-2),(2,self.max_t)])))  
                        label = 0
                    else:
                        rotz = np.random.uniform(-180, 180) * (3.141592 / 180.0)
                        roty = np.random.uniform(-180, 180) * (3.141592 / 180.0)
                        rotx = np.random.uniform(-180, 180) * (3.141592 / 180.0)
                        transl_x = np.random.uniform(-10, 10)
                        transl_y = np.random.uniform(-10, 10)
                        transl_z = np.random.uniform(-10, 10)   
                        label = 1      

                    train_RT_file.writerow([str(self.data_train[i][:6]),str(self.data_train[i][:-1]), rotx, roty, rotz, transl_x, transl_y, transl_z, label])
                    self.train_RT.append([str(self.data_train[i][:6]), str(self.data_train[i][:-1]), float(rotx), float(roty), float(rotz), float(transl_x), float(transl_y), float(transl_z), int(label)])

            assert len(self.train_RT) == len(self.data_train), "Something wrong with train RTs"
            self.metadata = self.data_train

        self.val_RT = []
        if subset == 'val':
            # val_RT_file = os.path.join(dataset_dir, 'sequences',
            #                            f'val_RT_seq{val_sequence}_{max_r:.2f}_{max_t:.2f}.csv')
            val_RT_file = osp.join(self.dataset_root, 'metadata',
                                       f'Lidar2Lidar_{subset}_RT_{max_r:.2f}_{max_t:.2f}.csv')
            if osp.exists(val_RT_file):
                print(f'{subset} SET: Using this file: {val_RT_file}')
                df_test_RT = pd.read_csv(val_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.val_RT.append(list(row))
            else:
                print(f'{subset} SET - Not found: {val_RT_file}')
                print("Generating a new one")
                val_RT_file = open(val_RT_file, 'w')
                val_RT_file = csv.writer(val_RT_file, delimiter=',')
                val_RT_file.writerow(['id', 'pair', 'Yaw', 'Pitch', 'Roll', 'tx', 'ty', 'tz', 'classify_label'])
                for i in range(len(self.data_val)):
                    data_seed = random.randint(1,10)  #大于5数据为配准失败（负样本），小于等于5为配准成功（正样本）
                    if data_seed > 10 :
                        max_angle = self.max_r
                        rotz = np.random.uniform(*(choice([(-max_angle,-5),(5,max_angle)]))) * (3.141592 / 180.0)
                        roty = np.random.uniform(*(choice([(-max_angle,-5),(5,max_angle)]))) * (3.141592 / 180.0)
                        rotx = np.random.uniform(*(choice([(-max_angle,-5),(5,max_angle)]))) * (3.141592 / 180.0)
                        transl_x = np.random.uniform(*(choice([(-self.max_t,-2),(2,self.max_t)])))
                        transl_y = np.random.uniform(*(choice([(-self.max_t,-2),(2,self.max_t)])))
                        transl_z = np.random.uniform(*(choice([(-self.max_t,-2),(2,self.max_t)])))  
                        label = 0
                    else:
                        rotz = np.random.uniform(-180, 180) * (3.141592 / 180.0)
                        roty = np.random.uniform(-180, 180) * (3.141592 / 180.0)
                        rotx = np.random.uniform(-180, 180) * (3.141592 / 180.0)
                        transl_x = np.random.uniform(-10, 10)
                        transl_y = np.random.uniform(-10, 10)
                        transl_z = np.random.uniform(-10, 10)    
                        label = 1    
                    val_RT_file.writerow([str(self.data_val[i][:6]),str(self.data_val[i][:-1]), rotx, roty, rotz, transl_x, transl_y, transl_z, label])
                    self.val_RT.append([str(self.data_val[i][:6]), str(self.data_val[i][:-1]), float(rotx), float(roty), float(rotz), float(transl_x), float(transl_y), float(transl_z), int(label)])
            assert len(self.val_RT) == len(self.data_val), "Something wrong with test RTs"     
            self.metadata = self.data_val  

        if subset == 'test':
            self.test_RT = []
            test_RT_file = osp.join(self.dataset_root, 'metadata',
                                        f'Lidar2Lidar_{subset}_RT_{max_r:.2f}_{max_t:.2f}.csv')

            if osp.exists(test_RT_file):
                print(f'TEST SET: Using this file: {test_RT_file}')
                df_test_RT = pd.read_csv(test_RT_file, sep=',')
                for index, row in df_test_RT.iterrows():
                    self.test_RT.append(list(row))
            else:
                print(f'TEST SET - Not found: {test_RT_file}')
                print("Generating a new one")
                test_RT_file = open(test_RT_file, 'w')
                test_RT_file = csv.writer(test_RT_file, delimiter=',')
                test_RT_file.writerow(['id', 'pair', 'Yaw', 'Pitch', 'Roll', 'tx', 'ty', 'tz', 'classify_label'])


                for i in range(len(self.data_test)):
                    data_seed = random.randint(1,10)  #大于5数据为配准失败（负样本），小于等于5为配准成功（正样本）
                    if data_seed > 10 :
                        max_angle = self.max_r
                        rotz = np.random.uniform(*(choice([(-max_angle,-5),(5,max_angle)]))) * (3.141592 / 180.0)
                        roty = np.random.uniform(*(choice([(-max_angle,-5),(5,max_angle)]))) * (3.141592 / 180.0)
                        rotx = np.random.uniform(*(choice([(-max_angle,-5),(5,max_angle)]))) * (3.141592 / 180.0)
                        transl_x = np.random.uniform(*(choice([(-self.max_t,-2),(2,self.max_t)])))
                        transl_y = np.random.uniform(*(choice([(-self.max_t,-2),(2,self.max_t)])))
                        transl_z = np.random.uniform(*(choice([(-self.max_t,-2),(2,self.max_t)])))  
                        label = 0
                    else:
                        rotz = np.random.uniform(-180, 180) * (3.141592 / 180.0)
                        roty = np.random.uniform(-180, 180) * (3.141592 / 180.0)
                        rotx = np.random.uniform(-180, 180) * (3.141592 / 180.0)
                        transl_x = np.random.uniform(-10, 10)
                        transl_y = np.random.uniform(-10, 10)
                        transl_z = np.random.uniform(-10, 10)   
                        label = 1      

                    test_RT_file.writerow([str(self.data_test[i][:6]),str(self.data_test[i][:-1]), rotx, roty, rotz, transl_x, transl_y, transl_z, label])
                    self.test_RT.append([str(self.data_test[i][:6]), str(self.data_test[i][:-1]), float(rotx), float(roty), float(rotz), float(transl_x), float(transl_y), float(transl_z), int(label)])

            assert len(self.test_RT) == len(self.data_test), "Something wrong with train RTs"
            self.metadata = self.data_test
 

    def _augment_point_cloud(self, ref_points, src_points, transform):
        rotation, translation = get_rotation_translation_from_transform(transform)
        # add gaussian noise
        ref_points = ref_points + (np.random.rand(ref_points.shape[0], 3) - 0.5) * self.augmentation_noise
        src_points = src_points + (np.random.rand(src_points.shape[0], 3) - 0.5) * self.augmentation_noise
        # random rotation
        aug_rotation = random_sample_rotation(self.augmentation_rotation)
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)
        # random scaling
        scale = random.random()
        scale = self.augmentation_min_scale + (self.augmentation_max_scale - self.augmentation_min_scale) * scale
        ref_points = ref_points * scale
        src_points = src_points * scale
        translation = translation * scale
        # random shift
        ref_shift = np.random.uniform(-self.augmentation_shift, self.augmentation_shift, 3)
        src_shift = np.random.uniform(-self.augmentation_shift, self.augmentation_shift, 3)
        ref_points = ref_points + ref_shift
        src_points = src_points + src_shift
        translation = -np.matmul(src_shift[None, :], rotation.T) + translation + ref_shift
        # compose transform from rotation and translation
        transform = get_transform_from_rotation_translation(rotation, translation)
        return ref_points, src_points, transform

    def _load_point_cloud(self, file_name):
        # points = np.load(file_name)
        pcd = o3d.io.read_point_cloud(file_name)
        points = np.asarray(pcd.points)
        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[: self.point_limit]
            points = points[indices]
        delete_points = []
        for i in range(points.shape[0]):
            if (points[i][0]>-1 and points[i][0]<1 and points[i][1]>-1 and points[i][1]<1 and points[i][2]>-1 and points[i][2]<1) or \
            points[i][0]>25 or points[i][0]<-25 or points[i][1]>25 or points[i][1]<-25 or points[i][1]>25 or points[i][1]<-25:
                delete_points.append(i)
        for i in range(len(delete_points)-1, 0, -1):
            points = np.delete(points, delete_points[i], axis=0)
        return points

    def __getitem__(self, index):
        data_dict = {}
        data_name = self.metadata[index]
        data_order_number = data_name.split("-")[0]
        self.lidar_type = data_name.split("-")[2][:-1]
        data_dict['seq_id'] = str(data_name)
        data_dict['ref_pcd'] = '../../data/road_scene/unreal_world/'+data_order_number+'/top-'+data_order_number+'.pcd'
        data_dict['src_pcd'] = '../../data/road_scene/unreal_world/'+data_order_number+'/'+self.lidar_type+'-'+data_order_number+'.pcd'

        # gt_R = o3d.geometry.get_rotation_matrix_from_zyx([gt_roll/180*3.1415926535, gt_yaw/180*3.1415926535 , gt_pitch/180*3.1415926535])
        if str(self.lidar_type) == "front":
            # front
            gt_R = o3d.geometry.get_rotation_matrix_from_zyx([0/180*3.1415926535, 48/180*3.1415926535 , 0/180*3.1415926535])
            gt_T = np.array([2.9, 0.15, -1.195])
        elif self.lidar_type == 'back':
            # back
            gt_R = o3d.geometry.get_rotation_matrix_from_zyx([180/180*3.1415926535, 48/180*3.1415926535 , 0/180*3.1415926535])
            gt_T = np.array([-1.8, 0, -0.82])
        elif self.lidar_type == 'left':
            # left
            gt_R = o3d.geometry.get_rotation_matrix_from_zyx([-90/180*3.1415926535, 48/180*3.1415926535 , 0/180*3.1415926535])
            gt_T = np.array([0.25, -0.85, -0.595])
        elif self.lidar_type == 'right':            
            # right
            gt_R = o3d.geometry.get_rotation_matrix_from_zyx([90/180*3.1415926535, 48/180*3.1415926535 , 0/180*3.1415926535])
            gt_T = np.array([0.25, 0.85, -0.595])        

        gt_R = gt_R[np.newaxis,:]
        gt_T = gt_T[np.newaxis,:]
        gt_R = torch.tensor(gt_R)
        gt_T = torch.tensor(gt_T)    
        gt_RT = get_transform_from_rotation_translation(gt_R, gt_T)
        gt_RT = torch.tensor(gt_RT).double()
        transform = gt_RT

        ref_points = self._load_point_cloud(osp.join(self.dataset_root, data_dict['ref_pcd']))
        src_points = self._load_point_cloud(osp.join(self.dataset_root, data_dict['src_pcd']))


        # 1.apply transform
        if self.subset == 'train':
            initial_RT = self.train_RT[index]
            rotz = initial_RT[7]
            roty = initial_RT[6]
            rotx = initial_RT[5]
            transl_x = initial_RT[2]
            transl_y = initial_RT[3]
            transl_z = initial_RT[4]  
            label =  initial_RT[8]          
        if self.subset == 'val':
            initial_RT = self.val_RT[index]
            rotz = initial_RT[7]
            roty = initial_RT[6]
            rotx = initial_RT[5]
            transl_x = initial_RT[2]
            transl_y = initial_RT[3]
            transl_z = initial_RT[4]  
            label =  initial_RT[8]    
        if self.subset == 'test':
            initial_RT = self.test_RT[index]
            rotz = initial_RT[7]
            roty = initial_RT[6]
            rotx = initial_RT[5]
            transl_x = initial_RT[2]
            transl_y = initial_RT[3]
            transl_z = initial_RT[4]  
            label =  initial_RT[8]             
        # 2. get rotation translation and matrix    
        R = eulerAnglesToRotationMatrix([rotx, roty, rotz])
        T = np.array([transl_x, transl_y, transl_z])
        R = R[np.newaxis,:]
        T = T[np.newaxis,:]
        R = torch.tensor(R)
        T = torch.tensor(T)
        RT = get_transform_from_rotation_translation(R, T)
        RT = torch.tensor(RT).double()
        # 3.apply "transform" matrix to registration two points cloud
        src_points = torch.tensor(src_points).double()
        src_points = apply_transform(src_points, transform)

        # new_src_points = src_points.numpy()
        # new_ref_points = ref_points
        # new_calibrated = np.concatenate((new_src_points,new_ref_points),axis=0)
        # pcd_src = o3d.geometry.PointCloud()
        # pcd_src.points = o3d.utility.Vector3dVector(new_src_points)   
        # pcd_src.paint_uniform_color([0, 0, 1.0])
        
        # pcd_ref = o3d.geometry.PointCloud()
        # pcd_ref.points = o3d.utility.Vector3dVector(new_ref_points)
        # pcd_ref.paint_uniform_color([0, 1, 0.0]) 

        # pcd = o3d.geometry.PointCloud()
        # # pcd.points = o3d.utility.Vector3dVector(new_calibrated) 
        # pcd += pcd_src   
        # pcd += pcd_ref             
        # o3d.io.write_point_cloud('/Download/GeoTransformer/GeoTransformer/data/Lidar2Lidar/metadata/results/wrong_'+data_name[:-1]+'.pcd', pcd)
        # print("data_dict['src_pcd']:",data_dict['src_pcd'])

        # 4.genetate a new pose for src point cloud
        RT_inv = inverse_transform(RT).double()
        src_points = apply_transform(src_points, RT_inv)

        transform = np.array(RT)
        src_points = np.array(src_points)
        transform = np.array(transform)

        if self.use_augmentation:
            ref_points, src_points, transform = self._augment_point_cloud(ref_points, src_points, transform)

        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points, transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1), dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1), dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)
        data_dict['classification_label'] = label
        return data_dict

    def __len__(self):
        return len(self.metadata)
