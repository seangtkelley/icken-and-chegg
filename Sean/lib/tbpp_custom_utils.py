import os
import sys
import cv2
import numpy as np

home_dir = os.path.expanduser("~")
ssd_detectors_dir = os.path.join(home_dir, 'sean', 'ssd_detectors')

sys.path.append(ssd_detectors_dir)

from tbpp_utils import PriorUtil

def read_generated_annots(annots_path, filenames):
    annots = open(annots_path, "r").readlines()
    image_paths = []
    regions = []

    temp_path = None
    temp_regions = []
    for line in annots:
        if line.endswith(".tiff\n"):
            if temp_path != None:
                filename = temp_path.split()[0].split('/')[-1]
                head, _, _ = filename.partition('.')
                name = head.split("_")[0]

                if len(temp_regions) > 0 and name in filenames:
                    image_paths.append(temp_path)
                    regions.append(temp_regions)
            
            temp_path = line.replace("\n", "")
            temp_regions = []
            
        elif len(line.split(" ")) >= 4:
            split_line = line.split(" ")
            x = float(split_line[0]); y = float(split_line[1])
            r_w = float(split_line[2]); r_h = float(split_line[3])

            temp_regions.append( (x, y, r_w, r_h) )

    return image_paths, regions

def generate_data(image_paths, regions, batch_size, prior_util, encode=True):
    h, w = (512, 512)
    mean = np.array([104,117,123])
    num_batches = len(image_paths) // batch_size

    inputs, targets = [], []

    while True:
        idxs = np.arange(len(image_paths))
        np.random.shuffle(idxs)
        idxs = idxs[:num_batches*batch_size]
        for j, i in enumerate(idxs):
            img = cv2.imread(image_paths[i])

            boxes = np.zeros(shape=(len(regions[i]), 4*2))
            k = 0
            for box in regions[k]:
                xmin, ymax, width, height = box
                xmax, ymin = xmin+width, ymax-height 
                boxes[k,:] = np.array([xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax])
                k += 1

            boxes[:,0::2] /= w
            boxes[:,1::2] /= h

            # append classes
            boxes = np.concatenate([boxes, np.ones([boxes.shape[0],1])], axis=1)
            
            img = cv2.resize(img, (w,h), cv2.INTER_LINEAR)
            img = img.astype(np.float32)
            
            img -= mean[np.newaxis, np.newaxis, :]
            #img = img / 25.6
            
            inputs.append(img)
            targets.append(boxes)
            
            #if len(targets) == batch_size or j == len(idxs)-1: # last batch in epoch can be smaller then batch_size
            if len(targets) == batch_size:
                if encode:
                    targets = [prior_util.encode(y) for y in targets]
                    targets = np.array(targets, dtype=np.float32)
                tmp_inputs = np.array(inputs, dtype=np.float32)
                tmp_targets = np.array(targets, dtype=np.float32)
                inputs, targets = [], []
                yield tmp_inputs, tmp_targets
            elif j == len(idxs)-1:
                # forgett last batch
                inputs, targets = [], []
                break
                
        print('NEW epoch')
    print('EXIT generator')