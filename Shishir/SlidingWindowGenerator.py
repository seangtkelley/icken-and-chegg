import os
from PIL import Image
import numpy as np

from BoundingBoxConverter import BoundingBoxConverter

class SlidingWindowGenerator:

    def __init__(self, annotations_folder_path, local_data_folder_path, bounding_box_train_file_path):
        """
            Inputs:
                - annotations_folder_path: Path to bounding box annotations folder
                - local_data_folder_path: Path to the local images folder
                - bounding_box_train_file_path: Path to file of bounding boxes
        """
        self.annotations_folder_path = annotations_folder_path
        self.local_data_folder_path = local_data_folder_path
        self.bounding_box_train_file_path = bounding_box_train_file_path
        with open(self.bounding_box_train_file_path) as f:
            self.lines = f.readlines()

    def generate_windows(self, max_boxes_per_window=20, window_size=(608, 608), save_image=True):
        """
            Inputs:
                - max_boxes_per_window: Maximum number of bounding boxes within window
                - window_size: How large the windows should be
                - save_image: To save local copy of cropped window or not
        """
        for line in self.lines:
            print(line)
            line = line.split()
            file_path = line[0]
            print(line[0])
            image = Image.open(file_path)
            orig_img_height, orig_img_width = image.size
            crop_height, crop_width = window_size
            bounding_box_list = np.array(
                [np.array(
                    list(map(int,box.split(',')))
                ) for box in line[1:]]
            )

            crop_window_count = 0
            anots_file_contents = ''
            ## begin window sliding
            for height in range(0, orig_img_height, crop_height):
                for width in range(0, orig_img_width, crop_width):
                    crop_area = (height, width, (height + crop_height), (width + crop_width)) ## (y0, x0, y1, x1)
                    image_name = line[0].split('/')[-1].split('.')[0]
                    image_ext = line[0].split('/')[-1].split('.')[1]
                    new_name =  image_name + "_" + str(crop_window_count) + '.' + image_ext
                    if save_image:
                        window = image.crop(crop_area)
                        window.save(os.path.join(self.local_data_folder_path, new_name))
                    
                    anot_line = os.path.join(self.annotations_folder_path, new_name)
                    
                    boxes_for_window = np.zeros((max_boxes_per_window, 5))
                    if len(bounding_box_list) > 0:
                        boxes_added = 0
                        for bounding_box in bounding_box_list:
                            if boxes_added < max_boxes_per_window and self.bounding_box_in_window(crop_area, bounding_box[:4]):
                                adjusted_bounding_box = [bounding_box[0] - width, bounding_box[1] - height, bounding_box[2] - width, bounding_box[3] - height, 0]
                                boxes_for_window[boxes_added] = adjusted_bounding_box
                                boxes_added += 1
                                anot_line += " " + ",".join(list(map(str, adjusted_bounding_box)))

                    if boxes_added > 0:
                        anots_file_contents += anot_line + "\n"
                    crop_window_count += 1
        
        with open('window_train.txt', "w") as text_file:
            text_file.write(anots_file_contents)


    def bounding_box_in_window(self, window, box):
        return not (box[0] > window[2]
            or box[2] < window[0]
            or box[1] > window[3]
            or box[3] < window[1])


converter = BoundingBoxConverter('/Volumes/SHISHIRPC/MapRecognition/annotations/current/', '/Volumes/SHISHIRPC/MapRecognition/maps/')
test_generator = SlidingWindowGenerator("/Volumes/SHISHIRPC/MapRecognition/annotations/current", "/Volumes/SHISHIRPC/MapRecognition/maps", "/Users/ShishirJakati/Desktop/MapRecognition/icken-and-chegg/Shishir/train.txt")

converter.convert('/Volumes/SHISHIRPC/MapRecognition/maps/')
test_generator.generate_windows()