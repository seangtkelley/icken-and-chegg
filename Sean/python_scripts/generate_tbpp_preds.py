import os
import sys
import numpy as np
import cv2
import glob

home_dir = os.path.expanduser("~")
ssd_detectors_dir = os.path.join(home_dir, 'sean', 'ssd_detectors')

sys.path.append(os.path.join(home_dir, 'icken-and-chegg', 'Sean'))

from lib import tbpp_custom_utils

sys.path.append(ssd_detectors_dir)

from tbpp_model import TBPP512, TBPP512_dense
from utils.model_utils import load_weights, calc_memory_usage
from ssd_data import preprocess
from tbpp_utils import PriorUtil

output_dir = os.path.join(os.sep+'mnt', 'nfs', 'work1', 'elm', 'sgkelley', 'sean', 'output')
checkpoint_dir = os.path.join(output_dir, 'tbpp', 'checkpoints', '201811101211_dsodtbpp512fl_maps')

# TextBoxes++ + DenseNet
model = TBPP512_dense(softmax=False)
weights_path = os.path.join(checkpoint_dir, 'weights.018.h5')
confidence_threshold = 0.35
plot_name = 'dsodtbpp512fl_sythtext'

load_weights(model, weights_path)

prior_util = PriorUtil(model)

do_preprocess = False # custom model trained on normal images

preds_output_path = os.path.join(output_dir, 'tbpp', 'map_trained_tbpp_angle_preds.txt')
preds_output_file = open(preds_output_path, "w+")

annots_path = os.path.join(home_dir, 'sean', 'cascaded-faster-rcnn', 'word-faster-rcnn', 'DataGeneration', 'fold_1', 'cropped_annotations_angles_-90to90step5.txt')

test_split_file = os.path.join(home_dir, 'torch-phoc', 'splits', 'test_files.txt')
test_filenames = []
with open(test_split_file) as f:
    test_filenames = [line.replace("\n", "") for line in f.readlines()]

test_images, test_regions = tbpp_custom_utils.read_generated_annots(annots_path, test_filenames)

filename_cache = ""
preds = []

for i, filepath in enumerate(test_images):

    split_imgname = filepath.split('/')[-1].split(".")[0].split("_")
    dimen_split = split_imgname[2].split('x')
    current_x = int(dimen_split[0])
    current_y = int(dimen_split[1])

    if current_x == 0 and current_y == 0 and len(preds)>0:
        # get previous filename
        filename = test_images[i-1].split('/')[-1]
        split_imgname = filename.split(".")[0].split("_")
        angle = split_imgname[1]

        # only write filename if is completely new image
        if filename_cache == "" or filename_cache != split_imgname[0]:
            preds_output_file.write(split_imgname[0] + ".tiff" + "\n")
            filename_cache = split_imgname[0]

        preds_output_file.write("angle " + angle + "\n")
        preds_output_file.write(str(len(preds)) + "\n")
        preds_output_file.write( "\n".join( [" ".join(map(str, bbox)) for bbox in preds] ))
        preds_output_file.write("\n")

        preds = []

    image = cv2.imread(filepath)
    
    model_output = model.predict(np.array([image]), batch_size=1, verbose=1)
            
    res = prior_util.decode(model_output[0], confidence_threshold, fast_nms=False)
    bboxes = res[:,0:4]
    quades = res[:,4:12]
    rboxes = res[:,12:17]
        
    for j in range(len(bboxes)): # xmin, ymin, xmax, ymax
        # scale bbox
        bbox = bboxes[j]*512 # windows were 512x512
        
        # translate points
        crop_x_min = bbox[0] + current_x
        crop_y_min = bbox[1] + current_y
        crop_x_max = bbox[2] + current_x
        crop_y_max = bbox[3] + current_y
        
        # find width and height
        w = crop_x_max - crop_x_min
        h = crop_y_max - crop_y_min

        preds.append( (crop_x_min, crop_y_min, w, h) )

if len(preds)>0:
    # get previous filename
    filename = test_images[i-1].split('/')[-1]
    split_imgname = filename.split(".")[0].split("_")
    angle = split_imgname[1]

    # only write filename if is completely new image
    if filename_cache == "" or filename_cache != split_imgname[0]:
        preds_output_file.write(split_imgname[0] + ".tiff" + "\n")
        filename_cache = split_imgname[0]

    preds_output_file.write("angle " + angle + "\n")
    preds_output_file.write(str(len(preds)) + "\n")
    preds_output_file.write( "\n".join( [" ".join(map(str, bbox)) for bbox in preds] ))
    
preds_output_file.close()