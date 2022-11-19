import os
import random

all_data = os.listdir('/Download/GeoTransformer/GeoTransformer/data/Lidar2Lidar/downsampled/unreal_world/unreal_world')
random.shuffle(all_data)
print("all_data", len(all_data))
train_data = all_data[:1140]
val_data = all_data[1140:1520]
test_data = all_data[1520:]

train_txt = open('/Download/GeoTransformer/GeoTransformer/data/Lidar2Lidar/metadata/train.txt', 'w')
for i in range(len(train_data)):
    train_txt.write(str(train_data[i])+'-top'+'-front'+'\n')
    train_txt.write(str(train_data[i])+'-top'+'-back'+'\n')
    train_txt.write(str(train_data[i])+'-top'+'-left'+'\n')
    train_txt.write(str(train_data[i])+'-top'+'-right'+'\n')

val_txt = open('/Download/GeoTransformer/GeoTransformer/data/Lidar2Lidar/metadata/val.txt', 'w')
for i in range(len(val_data)):
    val_txt.write(str(val_data[i])+'-top'+'-front'+'\n')
    val_txt.write(str(val_data[i])+'-top'+'-back'+'\n')
    val_txt.write(str(val_data[i])+'-top'+'-left'+'\n')
    val_txt.write(str(val_data[i])+'-top'+'-right'+'\n')    

test_txt = open('/Download/GeoTransformer/GeoTransformer/data/Lidar2Lidar/metadata/test.txt', 'w')
for i in range(len(test_data)):
    test_txt.write(str(test_data[i])+'-top'+'-front'+'\n')
    test_txt.write(str(test_data[i])+'-top'+'-back'+'\n')
    test_txt.write(str(test_data[i])+'-top'+'-left'+'\n')
    test_txt.write(str(test_data[i])+'-top'+'-right'+'\n')      