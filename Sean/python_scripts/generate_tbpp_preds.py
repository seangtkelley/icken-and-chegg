import os
import sys
import numpy as np
import cv2
import glob
from optparse import OptionParser

home_dir = os.path.expanduser("~")
ssd_detectors_dir = os.path.join(home_dir, 'sean', 'ssd_detectors')

sys.path.append(ssd_detectors_dir)

from tbpp_model import TBPP512, TBPP512_dense
from utils.model_utils import load_weights, calc_memory_usage
from ssd_data import preprocess
from tbpp_utils import PriorUtil
from utils.bboxes import rbox3_to_polygon

sys.path.append(os.path.join(home_dir, 'sean', 'cascaded-faster-rcnn', 'evaluation'))
from util import rotate_image, adjust_image_size

parser = OptionParser()
parser.add_option("-o", "--output_dir", help="output file", default="~/sean/output/tbpp/np_preds/")
parser.add_option("-w", "--weights_file", help="file with model weights", default="~/sean/ssd_detectors/checkpoints/201807091503_dsodtbpp512fl_synthtext/weights.018.h5")
parser.add_option("-i", "--images_dir", help="map images directory", default="~/data/maps/")
parser.add_option("-p", "--preprocess", help="whether or not to preform same preprocess as done in original implementations (background removal, etc...)", type=int, default=0)
parser.add_option("-t", "--test_split", help="file from torch_phoc with test split", default=None)
parser.add_option("-m", "--confidence", help="confidence threshold for predictions", type=float, default=0.8)
parser.add_option("-r", "--rotate", help="whether or not to rotate image", type=int, default=0)

(options, args) = parser.parse_args()

weights_path = options.weights_file
output_dir = options.output_dir
map_images_dir = options.images_dir
do_preprocess = bool(options.preprocess)
test_split_file = options.test_split
confidence_threshold = options.confidence
rotate_image = bool(options.rotate)

# TextBoxes++ + DenseNet
model = TBPP512_dense(softmax=False)

load_weights(model, weights_path)

prior_util = PriorUtil(model)

test_filenames = []
if test_split_file:
    with open(test_split_file) as f:
        test_filenames = [line.replace("\n", "") for line in f.readlines()]

crop_h = 512
crop_w = 512
step = 400

angles = [0] if rotate_image else range(-90, 95, 5)

for filepath in glob.glob(os.path.join(map_images_dir, 'D*')):
    filename = filepath.split('/')[-1]

    if test_split_file:
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
                conf = res[:,17]

                for j in range(len(rboxes)):
                    # convert rbox
                    polygon = rbox3_to_polygon(rboxes[j])*512

                    # sort points
                    xmin = np.min(polygon[:, 0])
                    xmax = np.max(polygon[:, 0])
                    ymin = np.min(polygon[:, 1])
                    ymax = np.max(polygon[:, 1])

                    # translate to full image location
                    xmin += current_x
                    xmax += current_x
                    ymin += current_y
                    ymax += current_y

                    # recreate polygon
                    polygon = np.reshape([xmin, ymax, xmax, ymax, xmax, ymin, xmin, ymin], (-1,2))

                    # rotate to orientation when image is not rotated
                    image_center = (original_shape[1] // 2, original_shape[0] // 2)
                    rot_mat = cv2.getRotationMatrix2D(image_center, -1*angle, scale=1.0)

                    # add col for rotation
                    polygon = np.concatenate([polygon, np.ones([polygon.shape[0], 1])], axis=1)

                    # rotate
                    transformed_points = rot_mat.dot(polygon.T).T

                    pt1 = [int(transformed_points[0][0]), int(transformed_points[0][1])]
                    pt2 = [int(transformed_points[1][0]), int(transformed_points[1][1])]
                    pt3 = [int(transformed_points[2][0]), int(transformed_points[2][1])]
                    pt4 = [int(transformed_points[3][0]), int(transformed_points[3][1])]

                    preds.append( [pt1, pt2, pt3, pt4] )
                    conf.append( conf[j] )

                current_x += step

            current_x = 0
            current_y += step
        
        print("Found", str(len(rbox3_to_polygon)), "boxes for angle", str(angle))

    np.save(os.path.join(output_dir, filename.split('.')[0]+'.npy'), preds)
    np.save(os.path.join(output_dir, filename.split('.')[0]+'_scores.npy'), confs)
