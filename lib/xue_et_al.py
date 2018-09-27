import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def inpaint(img, bounding_boxes):
    
    height, width, channels = img.shape

    # initialize mask with all zeros
    mask = np.zeros((height, width, channels), np.uint8)

    for bb in bounding_boxes:
        # assume word is horizontal FIX
        # only choose horizontal words
        box_width = bb[2] - bb[0]
        box_height = bb[3] - bb[1]
        
        if box_width/box_height > 1: # approximately horizontal words

            # choose random point between vertical edges
            mask_centroid = (bb[2] - bb[0]) * np.random.random_sample() + bb[0]

            # choose random width from 
            # 0.2*(width of box) and 2*(distance from centroid to closest edge)
            closest_dist = min(mask_centroid-bb[0], bb[2]-mask_centroid)
            mask_width = ((2*closest_dist) - (0.2*box_width)) * np.random.random_sample() + (0.2*box_width)

            # fill in mask area with white
            white = [255, 255, 255]
            x_min = int(np.floor(mask_centroid - (mask_width/2)))
            y_min = bb[1]
            x_max = int(np.ceil(mask_centroid + (mask_width/2)))
            y_max = bb[3]
                
        else:
            
            # choose random point between vertical edges
            mask_centroid = (bb[3] - bb[1]) * np.random.random_sample() + bb[1]

            # choose random height from 
            # 0.2*(width of box) and 2*(distance from centroid to closest edge)
            closest_dist = min(mask_centroid-bb[1], bb[3]-mask_centroid)
            mask_height = ((2*closest_dist) - (0.2*box_width)) * np.random.random_sample() + (0.2*box_width)

            # fill in mask area with white
            white = [255, 255, 255]
            x_min = bb[0]
            y_min = int(np.floor(mask_centroid - (mask_height/2)))
            x_max = bb[2]
            y_max = int(np.ceil(mask_centroid + (mask_height/2)))
        
        if np.min([x_min, y_min, x_max, y_max]) >= 0: # no negative indices
                mask[y_min:y_max, x_min:x_max] = white

    # https://docs.opencv.org/3.4.0/df/d3d/tutorial_py_inpainting.html
    mask_gray = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)
    dst = cv.inpaint(img, mask_gray, 3, cv.INPAINT_TELEA)
    
    return dst