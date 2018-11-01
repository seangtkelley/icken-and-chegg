import os
import sys
import numpy as np
import cv2
import glob

home_dir = os.path.expanduser("~")
ssd_detectors_dir = os.path.join(home_dir, 'sean', 'ssd_detectors')

sys.path.append(ssd_detectors_dir)

from tbpp_model import TBPP512, TBPP512_dense
from utils.model_utils import load_weights, calc_memory_usage
from ssd_data import preprocess
from tbpp_utils import PriorUtil

# TextBoxes++ + DenseNet
model = TBPP512_dense(softmax=False)
weights_path = os.path.join(ssd_detectors_dir, 'checkpoints', '201807091503_dsodtbpp512fl_synthtext', 'weights.018.h5')
confidence_threshold = 0.35
plot_name = 'dsodtbpp512fl_sythtext'

load_weights(model, weights_path)
checkdir = os.path.dirname(weights_path)

prior_util = PriorUtil(model)

map_images_dir = os.path.join(home_dir, 'data', 'maps')
do_preprocess = True
preds_output_path = os.path.join(home_dir, 'sean', 'output', 'untrained_tbpp_preds.txt')
preds_output_file = open(preds_output_path, "w+")

crop_h = 512
crop_w = 512
step = 100

for filepath in glob.glob(os.path.join(map_images_dir, 'D*')):
    image = cv2.imread(filepath)
    height = image.shape[0]; width = image.shape[1]
    current_x = 0; current_y = 0
    preds = []
    
    while current_y + crop_h < height:
        while current_x + crop_w < width:
            
            crop_img = image[current_y:current_y+crop_h, current_x:current_x+crop_w]
    
            if do_preprocess:
                crop_img = preprocess(crop_img, (512, 512))
            
            model_output = model.predict(np.array([crop_img]), batch_size=1, verbose=1)
            
            res = prior_util.decode(model_output[0], confidence_threshold, fast_nms=False)
            bboxes = res[:,0:4]
            quades = res[:,4:12]
            rboxes = res[:,12:17]
                
            for j in range(len(bboxes)): # xmin, ymin, xmax, ymax
                # scale bbox
                bbox = bboxes[j]*512
                
                # translate points
                crop_x_min = bbox[0] + current_x
                crop_y_min = bbox[1] + current_y
                crop_x_max = bbox[2] + current_x
                crop_y_max = bbox[3] + current_y
                
                # find width and height
                w = crop_x_max - crop_x_min
                h = crop_y_max - crop_y_min

                preds.append( (crop_x_min, crop_y_min, w, h) )
            
            current_x += step

        current_x = 0
        current_y += step
    
    filename = filepath.split('/')[-1]
    preds_output_file.write(filename + "\n")
    
    preds_output_file.write( "\n".join( [" ".join(map(str, bbox)) for bbox in preds] ))
    
preds_output_file.close()