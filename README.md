# LiDAR2LiDAR
For more calibration codes, please refer to the link <a href="https://github.com/PJLab-ADG/SensorsCalibration" title="SensorsCalibration">SensorsCalibration</a>

## Pipline
![image](https://github.com/OpenCalib/LiDAR2LiDAR/blob/master/pipline.png)

## Network
### Installation
```shell
# Install packages and other dependencies
pip install -r requirements.txt
python setup.py build develop
```
### 3DMatch
```shell
python test.py --snapshot=../../output/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/snapshots/snapshot.pth.tar --benchmark=3DMatch
python eval.py --benchmark=3DMatch --method=svd/--method=ransac
```
### Kitti
```shell
python test.py --snapshot=../../output/geotransformer.Kitti.stage4.gse.k3.max.oacl.stage2.sinkhorn/snapshots/snapshot.pth.tar 
python eval.py --method=svd/--method=ransac
```
### road_scene
```shell
python test.py --snapshot=../../output/geotransformer.CROON.stage4.gse.k3.max.oacl.stage2.sinkhorn/snapshots/snapshot.pth.tar 
python eval.py --method=svd/--method=ransac
```
## octree optimize
### Prerequisites

- Cmake
- Opencv 2.4.13
- PCL 1.9

### Compile
Compile in their respective folders

```shell
# mkdir build
mkdir -p build && cd build
# build
cmake .. && make
```

### Dataset
Because the dataset is relatively large, only test samples are uploaded, the complete data can be download from the link below.
```
Link(链接): https://pan.baidu.com/s/1EhiNVWAD1t96h0to7GTlIA
Extration code(提取码): ctuk
```

### Usage

1. Two input files: 

   `point_cloud_path initial_extrinsic`

- **point_cloud_path**: paths of Lidar point clouds
- **initial_extrinsic**: initial extrinsic parameters

2. Run the test sample:

   The executable file is under the bin folder.

   ```
   ./bin/run_lidar2lidar data/0001/lidar_cloud_path.txt data/0001/initial_extrinsic.txt
   ```

