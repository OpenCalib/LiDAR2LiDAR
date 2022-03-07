# LiDAR2LiDAR
# CROON: Automatic Multi-LiDAR Calibration and Refinement Method in Road Scene
For more calibration codes, please refer to the link <a href="https://github.com/PJLab-ADG/SensorsCalibration" title="SensorsCalibration">SensorsCalibration</a>


## Prerequisites

- Cmake
- Opencv 2.4.13
- PCL 1.9

## Compile
Compile in their respective folders

```shell
# mkdir build
mkdir -p build && cd build
# build
cmake .. && make
```


## Usage

1. Three Input files: 

   `point_cloud_path initial_extrinsic output_dir`

- **point_cloud_path**: paths of Lidar point clouds
- **initial_extrinsic**: initial extrinsic parameters
- **output_dir**: output path


2. Run the test sample:

   The executable file is under the bin folder.

   ```
   cd ./lidar2lidar/auto_calib/
   ./bin/run_lidar2lidar point_cloud_path initial_extrinsic output_dir
   ```

3. Calibration result:

   <img src="./result/refine0.png" width="100%" height="100%" alt="Calibration result" div align=center />
<!--    <img src="./result/refine1.png" width="100%" height="100%" alt="Calibration result" div align=center />
   <img src="./result/refine4.png" width="100%" height="100%" alt="Calibration result" div align=center />
   <img src="./result/refine8.png" width="100%" height="100%" alt="Calibration result" div align=center /> -->

