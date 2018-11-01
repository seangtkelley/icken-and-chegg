import numpy as np
import os
from PIL import Image
import cv2 as cv
import glob

from . import general_utils
from . import xue_et_al

def convert_numpy_annots_to_yolo(annotations_dir, map_images_dir, annots_file_output_path):
    """
        Inputs:
            - annotations_dir: Directory of .npy annotations
            - map_images_dir: Directory where .tiff map images are stored
            - annots_file_output_path: Path to where file containing annotations will be saved
    """

    annots_file_contents = ''
    for filepath in glob.glob(os.path.join(annotations_dir, 'D*')):
        filename = filepath.split('/')[-1]
        head,sep,tail = filename.partition('.')
        head = head.replace('D', '')

        annot_filename = os.path.join(annotations_dir, 'D'+head+'.npy')
        annot_line = os.path.join(map_images_dir, 'D'+head+'.tiff')
        
        A = np.load(annot_filename).item()

        # the following code shows how to loop through all the items in the numpy dictionary.
        # each dictionary can have at most 3 keys: vertices, name and link_to.

        for j in A.keys():
            # copy the list of vertices for jth dictionary element
            vertices = np.array(A[j]['vertices'])
            
            # find left most, right most, top most, and bottom most verticies
            x_min = np.min(vertices[:, 0])
            x_max = np.max(vertices[:, 0])
            y_min = np.min(vertices[:, 1])
            y_max = np.max(vertices[:, 1])

            # add annotation to line
            annot_line += " " + ",".join([str(int(x_min)), str(int(y_min)), str(int(x_max)), str(int(y_max)), "0"])
            
        annots_file_contents += annot_line + "\n"

    with open(annots_file_output_path, "w") as text_file:
        text_file.write(annots_file_contents)


def create_inpainted_images(original_bb_file_path, annots_file_output_path, image_save_dir, sample_num, save_image=True):
    """
        Inputs:
            - original_bb_file_path: Path to original annotations
            - annots_file_output_path: Path to where file containing annotations will be saved
            - sample_num: Amount of times to sample image for inpainting
            - image_save_dir: Directory where inpainted images will be saved/located
            - save_image: To save local copy of image or not
    """
    with open(original_bb_file_path) as f:
        original_lines = f.readlines()

    annots_file_contents = ''
    for line in original_lines:
        line = line.split()
        if len(line) < 2: # sanity check
            continue
        file_path = line[0]
        image = cv.imread(file_path)
        bounding_box_list = np.array(
            [np.array(
                list(map(int,box.split(',')))
            ) for box in line[1:]]
        )
        
        for i in range(sample_num):
            image_name = line[0].split('/')[-1].split('.')[0]
            image_ext = line[0].split('/')[-1].split('.')[1]
            new_name =  image_name + "_inpaint" + str(i) + '.' + image_ext

            if save_image:
                inpainted_image = xue_et_al.inpaint(image, bounding_box_list)
                cv.imwrite(os.path.join(image_save_dir, new_name), inpainted_image)
            
            annot_line = os.path.join(image_save_dir, new_name)
            annot_line += " " + " ".join(line[1:])

            annots_file_contents += annot_line + "\n"
    
    with open(annots_file_output_path, "w") as text_file:
        text_file.write(annots_file_contents)


def train_val_test_split(original_bb_file_path, train_split_file, val_split_file, test_split_file, train_annots_output, val_annots_output, test_annots_output):
    """
        Inputs:
            - original_bb_file_path: Path to original annotations
            - train_split_file, val_split_file, test_split_file: Paths to files containing image filenames of splits
            - train_annots_output, val_annots_output, test_annots_output: Paths to output split annotations
    """

    with open(original_bb_file_path) as f:
        lines = f.readlines()
        
    with open(train_split_file) as f:
        train_filenames = [line.replace("\n", "") for line in f.readlines()]

    with open(val_split_file) as f:
        val_filenames = [line.replace("\n", "") for line in f.readlines()]
        
    with open(test_split_file) as f:
        test_filenames = [line.replace("\n", "") for line in f.readlines()]

    train_content = ""
    val_content = ""
    test_content = ""
    for line in lines:
        filename = line.split()[0].split('/')[-1]
        head, _, _ = filename.partition('.')
        name = head.split("_")[0]
        
        if name in train_filenames:
            train_content += line
        elif name in val_filenames:
            val_content += line
        elif name in test_filenames:
            test_content += line
            
    with open(train_annots_output, "w") as text_file:
        text_file.write(train_content)
        
    with open(val_annots_output, "w") as text_file:
        text_file.write(val_content)
        
    with open(test_annots_output, "w") as text_file:
        text_file.write(test_content)


def generate_sliding_windows(original_bb_file_path, annots_file_output_path, image_save_dir, max_boxes_per_window=30, window_size=(608, 608), save_image=True):
    """
        Inputs:
            - original_bb_file_path: Path to file of unaltered bounding boxes
            - annots_file_output_path: Path to where file containing sliding window annotations will be saved
            - image_save_dir: Directory where sliding windows will be saved/located
            - max_boxes_per_window: Maximum number of bounding boxes within window
            - window_size: How large the windows should be
            - save_image: To save local copy of cropped window or not
    """
    
    with open(original_bb_file_path) as f:
        original_lines = f.readlines()

    annots_file_contents = ''
    for line in original_lines:
        line = line.split()
        if len(line) < 2: # sanity check
            continue
        file_path = line[0]
        image = Image.open(file_path)
        orig_img_height, orig_img_width = image.size
        crop_height, crop_width = window_size
        bounding_box_list = np.array(
            [np.array(
                list(map(int,box.split(',')))
            ) for box in line[1:]]
        )

        crop_window_count = 0
        
        ## begin window sliding
        for height in range(0, orig_img_height, crop_height):
            for width in range(0, orig_img_width, crop_width):
                crop_area = (height, width, (height + crop_height), (width + crop_width)) ## (y0, x0, y1, x1)
                
                image_name = line[0].split('/')[-1].split('.')[0]
                image_ext = line[0].split('/')[-1].split('.')[1]
                new_name =  image_name + "_window" + str(crop_window_count) + '.' + image_ext
                if save_image:
                    window = image.crop(crop_area)
                    window.save(os.path.join(image_save_dir, new_name))
                
                annot_line = os.path.join(image_save_dir, new_name)
                
                boxes_for_window = np.zeros((max_boxes_per_window, 5))
                if len(bounding_box_list) > 0:
                    boxes_added = 0
                    for bounding_box in bounding_box_list:
                        if boxes_added < max_boxes_per_window and general_utils.box_collision(crop_area, bounding_box[:4]):
                            adjusted_bounding_box = [bounding_box[0] - width, bounding_box[1] - height, bounding_box[2] - width, bounding_box[3] - height, 0]
                            boxes_for_window[boxes_added] = adjusted_bounding_box
                            boxes_added += 1
                            annot_line += " " + ",".join(list(map(str, adjusted_bounding_box)))

                if boxes_added > 0:
                    annots_file_contents += annot_line + "\n"
                crop_window_count += 1
    
    with open(annots_file_output_path, "w") as text_file:
        text_file.write(annots_file_contents)