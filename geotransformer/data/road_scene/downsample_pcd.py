import os
import os.path as osp
import open3d as o3d
import numpy as np
import glob
from tqdm import tqdm


def main():
    dataset_path = './unreal_world'
    new_dataset_path = './downsampled/unreal_world'
    all_data = os.listdir(dataset_path)
    print("all_data:",len(all_data))
    for file_name in tqdm(all_data):
        # frame = file_name.split('/')[-1][:-4]
        top_pcd = osp.join(dataset_path,file_name,'top-'+file_name+'.pcd')
        front_pcd = osp.join(dataset_path,file_name,'front-'+file_name+'.pcd')
        back_pcd = osp.join(dataset_path,file_name,'back-'+file_name+'.pcd')
        left_pcd = osp.join(dataset_path,file_name,'left-'+file_name+'.pcd')
        right_pcd = osp.join(dataset_path,file_name,'right-'+file_name+'.pcd')

        new_top_pcd = osp.join(new_dataset_path,file_name,'top-'+file_name+'.pcd')
        new_front_pcd = osp.join(new_dataset_path,file_name,'front-'+file_name+'.pcd')
        new_back_pcd = osp.join(new_dataset_path,file_name,'back-'+file_name+'.pcd')
        new_left_pcd = osp.join(new_dataset_path,file_name,'left-'+file_name+'.pcd')
        new_right_pcd = osp.join(new_dataset_path,file_name,'right-'+file_name+'.pcd')

        if not os.path.exists(os.path.dirname(new_top_pcd)):
            os.makedirs(os.path.dirname(new_top_pcd))

        pcd = o3d.io.read_point_cloud(top_pcd)
        points = np.asarray(pcd.points)
        points = points[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(0.1)
        o3d.io.write_point_cloud(new_top_pcd, pcd)

        pcd = o3d.io.read_point_cloud(front_pcd)
        points = np.asarray(pcd.points)
        points = points[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(0.1)
        o3d.io.write_point_cloud(new_front_pcd, pcd)

        pcd = o3d.io.read_point_cloud(back_pcd)
        points = np.asarray(pcd.points)
        points = points[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(0.1)
        o3d.io.write_point_cloud(new_back_pcd, pcd)

        pcd = o3d.io.read_point_cloud(left_pcd)
        points = np.asarray(pcd.points)
        points = points[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(0.1)
        o3d.io.write_point_cloud(new_left_pcd, pcd)

        pcd = o3d.io.read_point_cloud(right_pcd)
        points = np.asarray(pcd.points)
        points = points[:, :3]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(0.1)
        o3d.io.write_point_cloud(new_right_pcd, pcd)                


if __name__ == '__main__':
    main()
