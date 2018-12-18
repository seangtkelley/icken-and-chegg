import sys
import os
import cv2
import numpy as np
import glob
import argparse

home_dir = '/home/sgkelley/'

parser = argparse.ArgumentParser()
parser.add_argument('--txt', help='read annots from txt file')
parser.add_argument('--annots_path', type=str, default=os.path.join(home_dir, 'sean', 'output', 'tbpp', 'np_preds'), help='path to either annots folder or txt file')
parser.add_argument('--map_images_dir', type=str, default=os.path.join(home_dir, 'data', 'maps'), help='dir where map images are')
parser.add_argument('--output_dir', type=str, default=os.path.join(home_dir, 'sean', 'output', 'output', 'tbpp', 'draws'), help='dir to output images')
parser.add_argument("--test_only", help="whether or not to only evaluate test images")
parser.add_argument("--test_split", help="file from torch_phoc with test split", default=os.path.join(home_dir, 'torch-phoc', 'splits', 'test_files.txt'))

args = parser.parse_args()

def read_txt_preds(annots_path, filenames):
    annots = open(annots_path, "r").readlines()
    image_paths = []
    regions = []
    angle = None

    temp_path = None
    temp_regions, temp_angles = [], []
    for line in annots:
        if line.endswith(".tiff\n"):
            if temp_path != None:
                filename = temp_path.split()[0].split('/')[-1]
                head, _, _ = filename.partition('.')
                name = head.split("_")[0]

                if len(temp_regions) > 0 and (name in filenames or len(filenames) == 0):
                    image_paths.append(temp_path)
                    regions.append(temp_regions)
                    angles.append(temp_angles)
            
            temp_path = line.replace("\n", "")
            temp_regions, temp_angles = [], []

        elif 'angle' in line:
            angle = int(line.split(" ")[-1])
            
        elif len(line.split(" ")) >= 4:
            split_line = line.split(" ")
            xmin = float(split_line[0]); ymax = float(split_line[1])
            r_w = float(split_line[2]); r_h = float(split_line[3])

            xmax, ymin = xmin + r_w, ymax - r_h

            temp_regions.append( np.array( [ [xmin, ymax], [xmax, ymax], [xmax, ymin], [xmin, ymin] ] ))
            temp_angles.append(angle)

    return image_paths, regions, angles

def read_npy_preds(annots_dir, filenames):
    images, regions, angles = [], [], []
    for filepath in glob.glob(os.path.join(annots_dir, '*.tiff.npy')):
        filename = filepath.split('/')[-1]
        name = filename.split('.')[0]
        if name not in filenames and len(filenames) != 0:
            continue

        data = np.load(filename)

        images.append(name+".tiff")
        regions.append(data)
        angles.append([0]*len(data))
    
    return images, regions, angles

test_filenames = []
if args.test_only:
    with open(args.test_split_file) as f:
        test_filenames = [line.replace("\n", "") for line in f.readlines()]

if args.txt:
    images, regions, angles = read_txt_preds(args.annots_path, test_filenames)
else:
    images, regions, angles = read_npy_preds(args.annots_path, test_filenames)

for i in range(len(images)):
    image = cv2.imread(os.path.join(args.map_images_dir, images[i]))
    print(os.path.join(args.map_images_dir, images[i]))
    
    for j in range(len(regions[i])):
        polygon = regions[i][j]
        angle = angles[i][j]

        if angle != 0:
            continue

        image_center = (image.shape[1] // 2, image.shape[0] // 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, -1*angle, scale=1.0)

        xmin = np.min(polygon[:, 0])
        xmax = np.max(polygon[:, 0])
        ymin = np.min(polygon[:, 1])
        ymax = np.max(polygon[:, 1])

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

    cv2.imwrite(os.path.join(args.output_dir, images[i]), image)