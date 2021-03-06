import os
import sys
import numpy as np
import cv2
import glob
import argparse

home_dir = home = os.path.expanduser("~")

lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib')

ssd_detectors_dir = os.path.join(lib_dir, 'ssd_detectors')
sys.path.append(ssd_detectors_dir)

from tbpp_model import TBPP512, TBPP512_dense
from utils.model_utils import load_weights, calc_memory_usage
from ssd_data import preprocess
from tbpp_utils import PriorUtil
from utils.bboxes import rbox3_to_polygon

sys.path.append(os.path.join(lib_dir, 'cascaded-faster-rcnn', 'evaluation'))
from util import rotate_image, adjust_image_size

parser = argparse.ArgumentParser()
parser.add_argument("--output_dir", help="output dir", default=os.path.join(home_dir, 'sean', 'output', 'tbpp', 'np_preds'))
parser.add_argument("--weights_file", help="file with model weights", default=os.path.join(home_dir, 'sean', 'ssd_detectors', 'checkpoints', '201807091503_dsodtbpp512fl_synthtext', 'weights.018.h5'))
parser.add_argument("--images_dir", help="map images directory", default=os.path.join(home_dir, 'data', 'maps'))
parser.add_argument("--preprocess", help="whether or not to preform same preprocess as done in original implementations (background removal, etc...)")
parser.add_argument("--test_only", help="whether or not to only evaluate test images")
parser.add_argument("--test_split", help="file from torch_phoc with test split", default=os.path.join(home_dir, 'torch-phoc', 'splits', 'test_files.txt'))
parser.add_argument("--confidence", help="confidence threshold for predictions", type=float, default=0.8)
parser.add_argument("--rotate", help="whether or not to rotate image")

options = parser.parse_args()

weights_path = options.weights_file
output_dir = options.output_dir
map_images_dir = options.images_dir
do_preprocess = bool(options.preprocess)
test_only = bool(options.test_only)
test_split_file = options.test_split
confidence_threshold = options.confidence
do_rotate_image = bool(options.rotate)

# TextBoxes++ + DenseNet
model = TBPP512_dense(softmax=False)

load_weights(model, weights_path)

prior_util = PriorUtil(model)

test_filenames = []
if test_only:
    with open(test_split_file) as f:
        test_filenames = [line.replace("\n", "") for line in f.readlines()]

crop_h = 512
crop_w = 512
step = 400

angles = range(-90, 95, 5) if do_rotate_image else [0]

for filepath in glob.glob(os.path.join(map_images_dir, 'D*')):
    filename = filepath.split('/')[-1]

    if test_only:
        if filename.split('.')[0] not in test_filenames:
            continue

    print(filepath)
    map_img = cv2.imread(filepath)
    original_shape = map_img.shape

    preds = []
    confs = []
    for angle in angles:
        rot_img, rot_mat, bounds = rotate_image(map_img, angle, original_shape)
        height = rot_img.shape[0]
        width = rot_img.shape[1]
        current_x = 0; current_y = 0

        while current_y + crop_h < height:
            while current_x + crop_w < width:
                
                crop_img = rot_img[current_y:current_y+crop_h, current_x:current_x+crop_w]
        
                if do_preprocess:
                    crop_img = preprocess(crop_img, (512, 512))
                
                model_output = model.predict(np.array([crop_img]), batch_size=1, verbose=0)
                
                res = prior_util.decode(model_output[0], confidence_threshold, fast_nms=False)
                bboxes = res[:,0:4]
                quades = res[:,4:12]
                rboxes = res[:,12:17]
                conf = res[:,17:]

                for j in range(len(rboxes)):
                    # convert rbox
                    polygon = rbox3_to_polygon(rboxes[j])*512

                    # translate to full image location
                    polygon[:, 0] += current_x
                    polygon[:, 1] += current_y

                    # rotate to orientation when image is not rotated
                    image_center = (original_shape[1] // 2, original_shape[0] // 2)
                    rot_mat = cv2.getRotationMatrix2D(image_center, -1*angle, scale=1.0)

                    # add col for rotation
                    polygon = np.concatenate([polygon, np.ones([polygon.shape[0], 1])], axis=1)

                    # rotate
                    transformed_points = rot_mat.dot(polygon.T).T

                    preds.append( transformed_points[:, :2].astype(int) )
                    confs.append( conf[j][0] )

                current_x += step

            current_x = 0
            current_y += step
        
        print("Found", str(len(rboxes)), "boxes for angle", str(angle))

    np.save(os.path.join(output_dir, filename+'.npy'), preds)
    np.save(os.path.join(output_dir, filename+'_scores.npy'), confs)
