import sys
import os

home_dir = os.path.expanduser("~")

sys.path.append(os.path.join(home_dir, 'icken-and-chegg/'))

from lib import yolo_utils

# convert numpy annotations to yolo format
yolo_utils.convert_numpy_annots_to_yolo(
    os.path.join(home_dir, 'data', 'annotations', 'current'),
    os.path.join(home_dir, 'data', 'maps'),
    os.path.join(home_dir, 'config', 'yolo_annotations.txt'))

# split annotations
yolo_utils.train_val_test_split(
    os.path.join(home_dir, 'config', 'yolo_annotations.txt'),
    os.path.join(home_dir, 'torch-phoc', 'splits', 'train_files.txt'),
    os.path.join(home_dir, 'torch-phoc', 'splits', 'val_files.txt'),
    os.path.join(home_dir, 'torch-phoc', 'splits', 'test_files.txt'),
    os.path.join(home_dir, 'config', 'yolo_train.txt'),
    os.path.join(home_dir, 'config', 'yolo_val.txt'),
    os.path.join(home_dir, 'config', 'yolo_test.txt'))

# create sliding windows
yolo_utils.generate_sliding_windows(
    os.path.join(home_dir, 'config', 'yolo_train.txt'), 
    os.path.join(home_dir, 'config', 'yolo_608window_train.txt'), 
    os.path.join(home_dir, 'data', 'maps', 'sliding_windows_608'))

yolo_utils.generate_sliding_windows(
    os.path.join(home_dir, 'config', 'yolo_val.txt'), 
    os.path.join(home_dir, 'config', 'yolo_608window_val.txt'), 
    os.path.join(home_dir, 'data', 'maps', 'sliding_windows_608'))

yolo_utils.generate_sliding_windows(
    os.path.join(home_dir, 'config', 'yolo_test.txt'), 
    os.path.join(home_dir, 'config', 'yolo_608window_test.txt'), 
    os.path.join(home_dir, 'data', 'maps', 'sliding_windows_608'))