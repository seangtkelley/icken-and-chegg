import cv2
import os
import numpy as np 
import scipy.io as sio
import sys
import argparse

home_dir = os.path.expanduser("~")
sys.path.append(os.path.join(home_dir, 'Documents', 'indystudy', 'cascaded-faster-rcnn', 'evaluation'))

from util import retrieve_regions, vis_detections_pts, compute_union, compute_intersection
from util import vis_detections_bbox, rotate_image, adjust_image_size, convert_bbox_format, filter_predictions

parser = argparse.ArgumentParser()
parser.add_argument("--results", help="file with prediction results")
parser.add_argument("--dir_output", help="output directory", default="./")
parser.add_argument("--images_dir", help="images directory", default="./")

# ex: python icken-and-chegg/Sean/python_scripts/convert_txt_preds_npy.py -r ~/Documents/indystudy/data/output/synthtext_trained_maps_tbpp_0.8_preds.txt -d ~/Documents/indystudy/data/output/np_annots/ -i ~/Documents/indystudy/data/maps/

options = parser.parse_args()

prediction_file = options.results
output_dir = options.dir_output
images_dir = options.images_dir

predicted = retrieve_regions(options.results)

for key in predicted.keys():
    original_image = os.path.join(images_dir, key)
    img = cv2.imread(original_image)

    predictions = predicted[key]
    prediction_by_angle = {}

    for p in predictions:
        if p[4] not in prediction_by_angle:
            prediction_by_angle[p[4]] = [p[:4]]
        else:
            prediction_by_angle[p[4]].append( p[:4] )

    translate = (0,0)
    img_shape = img.shape
    pivot = (img_shape[1] // 2, img_shape[0] // 2)

    all_predictions = []
    all_annotations = []
    for angle in prediction_by_angle:
        for pred in prediction_by_angle[angle]:
            corners = convert_bbox_format(pred, -angle, pivot=pivot)
            all_predictions.append( corners )
    
    cnt1 = []
    for i in range(len(all_predictions)):
        pt1 = [int(all_predictions[i][0]), int(all_predictions[i][1])]
        pt2 = [int(all_predictions[i][2]), int(all_predictions[i][3])]
        pt3 = [int(all_predictions[i][4]), int(all_predictions[i][5])]
        pt4 = [int(all_predictions[i][6]), int(all_predictions[i][7])]

        cnt = np.array([pt1, pt2, pt3, pt4])
        cnt1.append(cnt)
    
    np.save(os.path.join(output_dir, key+'.npy'), cnt1)