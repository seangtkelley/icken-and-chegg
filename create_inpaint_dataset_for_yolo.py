import sys
import os

from .lib import yolo_utils

home_dir = os.path.expanduser("~")

# convert numpy annotations to yolo format
yolo_utils.convert_numpy_annots_to_yolo(
    os.path.join(home_dir, 'data', 'annotations', 'current'),
    os.path.join(home_dir, 'data', 'maps'),
    os.path.join(home_dir, 'config', 'yolo_annotations.txt')
)

# inpaint images
yolo_utils.create_inpainted_images(
    os.path.join(home_dir, 'config', 'yolo_annotations.txt'),
    os.path.join(home_dir, 'config', 'inpainted_annotations.txt'),
    os.path.join(home_dir, 'data', 'maps', 'inpainted'),
    5
)

# split annotations
yolo_utils.train_val_test_split(
    os.path.join(home_dir, 'config', 'inpainted_annotations.txt'),
    os.path.join(home_dir, 'torch-phoc', 'splits', 'train_files.txt'),
    os.path.join(home_dir, 'torch-phoc', 'splits', 'val_files.txt'),
    os.path.join(home_dir, 'torch-phoc', 'splits', 'test_files.txt'),
    os.path.join(home_dir, 'config', 'yolo_train_inpainted.txt'),
    os.path.join(home_dir, 'config', 'yolo_val_inpainted.txt'),
    os.path.join(home_dir, 'config', 'yolo_test_inpainted.txt')
)

# create sliding windows
yolo_utils.generate_sliding_windows(
    os.path.join(home_dir, 'config', 'yolo_train_inpainted.txt'), 
    os.path.join(home_dir, 'config', 'yolo_inpainted_608window_train.txt'), 
    os.path.join(home_dir, 'data', 'maps', 'inpainted_sliding_windows_608')
)

yolo_utils.generate_sliding_windows(
    os.path.join(home_dir, 'config', 'yolo_val_inpainted.txt'), 
    os.path.join(home_dir, 'config', 'yolo_inpainted_608window_val.txt'), 
    os.path.join(home_dir, 'data', 'maps', 'inpainted_sliding_windows_608')
)

yolo_utils.generate_sliding_windows(
    os.path.join(home_dir, 'config', 'yolo_test_inpainted.txt'), 
    os.path.join(home_dir, 'config', 'yolo_inpainted_608window_test.txt'), 
    os.path.join(home_dir, 'data', 'maps', 'inpainted_sliding_windows_608')
)