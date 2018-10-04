import os
import sys
from PIL import Image
import PIL.ImageDraw as ImageDraw
import numpy as np
import time

import argparse

dir_path = os.path.dirname(os.path.realpath(__file__))
home_dir = os.path.expanduser("~")

parser = argparse.ArgumentParser()
parser.add_argument("--test_annots_path", type=str, default=os.path.join(home_dir, 'config', 'yolo_test.txt'), help="Path to input weigths for yolo")
parser.add_argument("--input_size", type=int, default=416, help="Size of one dimension of input images")
parser.add_argument("--save_image", type=int, default=0, help="Save image or not")
parser.add_argument("--model_path", type=str, default=os.path.join(home_dir, 'output', 'keras_yolo3', 'baseline_416_09282018', 'trained_weights_final.h5'), help="Path to input weigths for yolo")
parser.add_argument("--classes", type=str, default=os.path.join(home_dir, 'config', 'word_class.txt'), help="Path to file with class names")
parser.add_argument("--anchors", type=str, default=os.path.join(home_dir, 'keras_yolo3', 'model_data', 'yolo_anchors.txt'), help="Path to yolo anchors file")
parser.add_argument("--iou_thres", type=float, default=0.45, help="IoU Threshold for detection")

args = parser.parse_args()

sys.path.append(os.path.join(home_dir, 'icken-and-chegg'))
sys.path.append(os.path.join(home_dir))

from lib import general_utils
from keras_yolo3.yolo import YOLO


with open(args.test_annots_path) as f:
    lines = f.readlines()

window_size = (args.input_size, args.input_size)

detector = YOLO(model_path=args.model_path,
                anchors_path=args.anchors,
                classes_path=args.classes,
                model_image_size=window_size,
                iou=args.iou_thres)


false_negatives = 0
false_positives = 0
true_positives = 0
chopped_detections = 0
for line in lines:
    line = line.split()
    image = Image.open(line[0])
    iw, ih = image.size
    h, w = window_size
    gt_boxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

    draw = ImageDraw.Draw(image)

    predicted_boxes = []
    for y in range(0, ih, h):
        for x in range(0, iw, w):
            area = (x, y, x+w, y+h)
            window = image.crop(area)

            # predicted bounding boxes
            _, window_predictions = detector.detect_image_silent(window)

            for bb in window_predictions:
                adjusted_bb = [bb['left'] + x, bb['top'] + y, bb['right'] + x, bb['bottom'] + y]
                
                predicted_boxes.append(adjusted_bb)
                
                draw.rectangle(adjusted_bb, outline="blue")


    used_prediction_indices = {}
    for bb in gt_boxes:
        
        preds_for_gt = []
        
        for i in range(len(predicted_boxes)):
            if not used_prediction_indices.get(i, False) and general_utils.box_contains_centroid(bb, general_utils.get_centroid_of_box(predicted_boxes[i])):
                preds_for_gt.append(predicted_boxes[i])
                
                # mark prediction as used
                used_prediction_indices[i] = True
                
        if len(preds_for_gt) == 0: # nothing detected for gt
            false_negatives += 1

            draw.rectangle([bb[0], bb[1], bb[2], bb[3]], outline="red")
        elif len(preds_for_gt) > 1: # multiple boxes for one gt
            chopped_detections += 1

            draw.rectangle([bb[0], bb[1], bb[2], bb[3]], outline="yellow")
        else:
            if general_utils.bb_iou([bb[0], bb[1], bb[2], bb[3]], preds_for_gt[0]) > 0.5:
                true_positives += 1
                draw.rectangle([bb[0], bb[1], bb[2], bb[3]], outline="lime")
            else:
                false_negatives += 1
                draw.rectangle([bb[0], bb[1], bb[2], bb[3]], outline="teal")

    false_positives = len([i for i in range(len(predicted_boxes)) if not used_prediction_indices.get(i, False)])

    precision = true_positives / (true_positives+false_positives)
    recall = true_positives / (true_positives+false_negatives)

    print(" False Negatives:", false_negatives, "\n",
        "Chopped Detections:", chopped_detections, "\n",
        "True Positives:", true_positives, "\n",
        "False Positives:", false_positives, "\n",
        "Precision:", precision, "\n",
        "Recall:", recall)
        
    image.save('first_train_example_detections_with_gt.tiff')