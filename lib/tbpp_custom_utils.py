import os
import sys
import cv2
import numpy as np
import glob

home_dir = os.path.expanduser('~')

lib_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'lib')

ssd_detectors_dir = os.path.join(lib_dir, 'sean', 'ssd_detectors')
sys.path.append(ssd_detectors_dir)

from tbpp_utils import PriorUtil
from ssd_data import preprocess

sys.path.append(os.path.join(lib_dir, 'sean', 'cascaded-faster-rcnn', 'evaluation'))

from util import rotate_image, adjust_image_size

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


def read_raw_annots(annots_dir, filenames):
    images, regions = [], []
    for filepath in glob.glob(os.path.join(annots_dir, '*.npy')):
        filename = filepath.split('/')[-1]
        name = filename.split('.')[0]
        if name not in filenames:
            continue

        data = np.load(filepath).item()
        images.append(name+".tiff")
        temp_regions = []
        for key in data.keys():
            vertices = data[key]['vertices']
            if len(vertices) == 4:
                temp_regions.append(vertices)
        regions.append(temp_regions)
    
    return images, regions
    

def tbpp_raw_generate_data(map_images_dir, image_paths, regions, batch_size, prior_util, encode=True, do_rotate=False, do_preprocess=False):
    crop_h = 512
    crop_w = 512
    step = 400
    angles = range(-90, 95, 5) if do_rotate else [0]

    inputs, targets = [], []

    mean = np.array([104,117,123])

    idxs = np.arange(len(image_paths))
    np.random.shuffle(idxs)
    for _, i in enumerate(idxs):
        filepath = os.path.join(map_images_dir, image_paths[i])

        map_img = cv2.imread(filepath)
        original_shape = map_img.shape

        for angle in angles:
            rot_img, rot_mat, _ = rotate_image(map_img, angle, original_shape)
            height = rot_img.shape[0]
            width = rot_img.shape[1]
            current_x = 0; current_y = 0

            while current_y + crop_h < height:
                while current_x + crop_w < width:
                    
                    crop_img = rot_img[current_y:current_y+crop_h, current_x:current_x+crop_w]
                    if do_preprocess:
                        crop_img = preprocess(crop_img, (512, 512))

                    crop_boxes = []
                    for region in regions:
                        # rotate to orientation when image is not rotated
                        image_center = (original_shape[1] // 2, original_shape[0] // 2)
                        rot_mat = cv2.getRotationMatrix2D(image_center, angle, scale=1.0)

                        # add col for rotation
                        region = np.concatenate([region, np.ones([region.shape[0], 1])], axis=1)

                        # rotate
                        transformed_points = rot_mat.dot(region.T).T

                        pt1 = [int(transformed_points[0][0]), int(transformed_points[0][1])]
                        pt2 = [int(transformed_points[1][0]), int(transformed_points[1][1])]
                        pt3 = [int(transformed_points[2][0]), int(transformed_points[2][1])]
                        pt4 = [int(transformed_points[3][0]), int(transformed_points[3][1])]

                        region = np.array( [pt1, pt2, pt3, pt4] )

                        xmin = np.min(region[:, 0])
                        xmax = np.max(region[:, 0])
                        ymin = np.min(region[:, 1])
                        ymax = np.max(region[:, 1])

                        if xmin > current_x and  xmax < (current_x+crop_w) and ymin < (current_y+crop_h) and ymax > current_y:
                            crop_xmin = xmin - current_x
                            crop_ymin = ymin - current_y
                            crop_xmax = xmax - current_x
                            crop_ymax = ymax - current_y

                            crop_boxes.append( [crop_xmin, crop_ymax, crop_xmax, crop_ymax, crop_xmax, crop_ymin, crop_xmin, crop_ymin] )

                    crop_boxes = np.array(crop_boxes)
                    crop_boxes[:,0::2] /= crop_img.shape[1]
                    crop_boxes[:,1::2] /= crop_img.shape[0]  
            
                    # append classes
                    crop_boxes = np.concatenate([crop_boxes, np.ones([crop_boxes.shape[0],1])], axis=1)
                    
                    crop_img -= mean[np.newaxis, np.newaxis, :]
                    #img = img / 25.6
            
                    inputs.append(crop_img)
                    targets.append(crop_boxes)
            
                    #if len(targets) == batch_size or j == len(idxs)-1: # last batch in epoch can be smaller then batch_size
                    if len(targets) == batch_size:
                        if encode:
                            targets = [prior_util.encode(y) for y in targets]
                            targets = np.array(targets, dtype=np.float32)
                        tmp_inputs = np.array(inputs, dtype=np.float32)
                        tmp_targets = np.array(targets, dtype=np.float32)
                        inputs, targets = [], []
                        yield tmp_inputs, tmp_targets

                    current_x += step

            current_x = 0
            current_y += step
                
        print('NEW epoch')
    print('EXIT generator')


def tbpp_generate_data(image_paths, regions, batch_size, prior_util, encode=True):
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
                boxes[k,:] = np.array([xmin, ymax, xmax, ymax, xmax, ymin, xmin, ymin])
                k += 1

            boxes[:,0::2] /= img.shape[1]
            boxes[:,1::2] /= img.shape[0]

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

def tb_generate_data(image_paths, regions, batch_size, prior_util, encode=True):
    h, w = (300, 300)
    mean = np.array([104,117,123])
    num_batches = len(image_paths) // batch_size

    inputs, targets = [], []

    while True:
        idxs = np.arange(len(image_paths))
        np.random.shuffle(idxs)
        idxs = idxs[:num_batches*batch_size]
        for j, i in enumerate(idxs):
            img = cv2.imread(image_paths[i])

            boxes = np.zeros(shape=(len(regions[i]), 4))
            k = 0
            for box in regions[k]:
                xmin, ymax, width, height = box
                xmax, ymin = xmin+width, ymax-height 
                boxes[k,:] = np.array([xmin, ymin, xmax, ymax])
                k += 1

            boxes[:,0::2] /= img.shape[1]
            boxes[:,1::2] /= img.shape[0]

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