import sys
import os
import cv2
import numpy as np

home_dir = os.path.expanduser("~")

def read_generated_annots(annots_path, filenames):
    annots = open(annots_path, "r").readlines()
    image_paths = []
    regions = []
    angle = None

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

        elif 'angle' in line:
            angle = int(line.split(" ")[-1])
            
        elif len(line.split(" ")) >= 4:
            split_line = line.split(" ")
            x = float(split_line[0]); y = float(split_line[1])
            r_w = float(split_line[2]); r_h = float(split_line[3])

            temp_regions.append( (x, y, r_w, r_h, angle) )

    return image_paths, regions

test_only = True
test_filenames = []
if test_only:
    test_split_file = os.path.join(home_dir, 'Documents', 'indystudy', 'torch-phoc', 'splits', 'test_files.txt')

    with open(test_split_file) as f:
        test_filenames = [line.replace("\n", "") for line in f.readlines()]

images, regions = read_generated_annots('/home/sean/Downloads/synthtext_trained_maps_tbpp_0.8_preds.txt', test_filenames)

image_path = '/home/sean/Documents/indystudy/data/maps/'

for i in range(len(images)):
    image = cv2.imread(image_path+images[i])
    print(image_path+images[i])
    
    for j in range(len(regions[i])):
        xmin, ymax, w, h, angle = regions[i][j]
        xmax, ymin = xmin + w, ymax - h

        # if angle != 0:
        #     continue

        image_center = (image.shape[1] // 2, image.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, -1*angle, scale=1.0)

        pt1 = [int(xmin), int(ymin)]
        pt2 = [int(xmax), int(ymin)]
        pt3 = [int(xmax), int(ymax)]
        pt4 = [int(xmin), int(ymax)]

        points = np.array([np.asarray([pt1[0], pt1[1], 1.0]),
                    np.asarray([pt2[0], pt2[1], 1.0]),
                    np.asarray([pt3[0], pt3[1], 1.0]),
                    np.asarray([pt4[0], pt4[1], 1.0])
                    ])

        transformed_points = rot_mat.dot(points.T).T

        pt1 = [int(transformed_points[0][0]), int(transformed_points[0][1])]
        pt2 = [int(transformed_points[1][0]), int(transformed_points[1][1])]
        pt3 = [int(transformed_points[2][0]), int(transformed_points[2][1])]
        pt4 = [int(transformed_points[3][0]), int(transformed_points[3][1])]

        cnt = np.array([pt1, pt2, pt3, pt4])
        cv2.drawContours(image, [cnt], 0, (255, 0, 0), 5)

    cv2.imwrite('/home/sean/Documents/indystudy/data/output/'+images[i], image)