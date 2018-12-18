import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry.polygon import Polygon
import glob
from scipy.misc import imread
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

path_to_annotations = '/home/sean/Documents/indystudy/data/annotations/current/'
path_to_maps = '/home/sean/Documents/indystudy/data/map-images/maps/'

for filepath in glob.glob(path_to_annotations+'D*'):
    filename = filepath.split('/')[-1]
    head,sep,tail2 = filename.partition('.')
    head = head.replace('D', '')
    image_filename = path_to_maps+'D'+head+'.tiff'
    anot_filename = path_to_annotations+'D'+head+'.npy'

    A = np.load(anot_filename).item()
    Img = Image.open(image_filename)
    Img1 = Image.open(image_filename)

    draw = ImageDraw.Draw(Img)
    draw1 = ImageDraw.Draw(Img1)

    for j in A.keys():
        # copy the list of vertices for jth dictionary element
        poly = A[j]['vertices']
        # appending the first vertex to the end of the list to from a closed loop
        poly.append(poly[0])
        draw.line(poly, width = 5, fill="red")
        # check for available keys in the dictionary. Here we are checking for the key 'link_to'
        if A[j].get('link_to'):
            # copying the key of the dictionary element which lin_to points to
            link = A[j]['link_to']
            # copying the initial vertex of the linked polygon
            point = A[link]['vertices'][0]
            # drawing line between first vertex of each polygon.
            draw.line([poly[0],point], width = 5, fill="red")
    
    Img.show()