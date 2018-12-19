from multiprocessing.dummy import Pool as ThreadPool 

import numpy as np
import glob
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import os
import cv2
from shapely.geometry import Polygon
from shapely.strtree import STRtree
from scipy.optimize import linear_sum_assignment as lsa
import argparse

home_dir = os.path.expanduser('~')

parser = argparse.ArgumentParser()
parser.add_argument("--train_dir", help="output dir", default=os.path.join(home_dir, 'data', 'annotations', 'current'))
parser.add_argument("--test_dir", help="output dir", default=os.path.join(home_dir, 'sean', 'output', 'tbpp', 'np_preds'))

args = parser.parse_args()

IoU_thresh = 0.5

trains_all = []
test_all = []
P = []
R = []
#step = 0.1 
global_stats = []
global_IoU_vals = []
files = []

def get_results(filename):
	print(filename)
	_dir, _, _file = filename.rpartition('/')
	_image, _, _ext = _file.rpartition('.')
	_annots, _, _im_ext = _image.rpartition('.')

	files.append(_image)
	# print _image

	# grab the train and test annotations
	trains = np.load(os.path.join(args.train_dir, _annots+_+_ext)).item()
	tests = np.load(filename)
	#print filename, maps_dir+_image, args.train_dir+_annots+_+_ext

	# data structure for saving the IoU values
	IoU_vals = np.zeros((len(tests), len(trains.keys())))

	# save the train anots
	train_polys = []
	for i in trains.keys():
		# print(trains[i]['vertices'])
		train_polys.append(Polygon(trains[i]['vertices']))
		pass
	s = STRtree(train_polys)

	# save the test annots
	test_polys = []
	for i in range(len(tests)):
		poly = tests[i]
		poly = poly.tolist()
		# poly.append(poly[0])
		test_poly = Polygon(poly)
		if not test_poly.is_valid:
			continue
		try:
			results = s.query(test_poly)
			for j in range(len(results)):
				_id = train_polys.index(results[j])
				_intersection = train_polys[_id].intersection(test_poly).area
				_union = train_polys[_id].union(test_poly).area
				IoU_vals[i, _id] = _intersection / _union
			test_polys.append(test_poly)
		except Exception:
			continue

	
	# do the linear sum assignment
	_row, _col = lsa(1-IoU_vals)
	assignment_matrix = IoU_vals[_row, _col]
	
	# compute the numbers
	TP = (assignment_matrix >= IoU_thresh).sum()
	FP = (assignment_matrix < IoU_thresh).sum()
	FN = len(trains.keys()) - TP
	return [TP, FP, FN]

# MAIN PART
# compute the IoUs
for dir_name in glob.glob(os.path.join(args.test_dir, '*')):
	print(dir_name)
	
	# threading things
	stats = []
	list_of_files = glob.glob(os.path.join(dir_name, '*.tiff.npy'))
	pool = ThreadPool(len(list_of_files))
	stats = pool.map(get_results, list_of_files)

	global_stats.append(stats)

	# assemble stats
	stats = np.asarray(stats)
	print(stats.shape)
	avg_TP = float(stats[:,0].sum()) / float(stats.shape[0])
	avg_FP = float(stats[:,1].sum()) / float(stats.shape[0])
	avg_FN = float(stats[:,2].sum()) / float(stats.shape[0])

	# compute P, R
	precision = float(avg_TP) / float(avg_TP + avg_FP)
	recall = float(avg_TP) / float(avg_TP + avg_FN)

	P.append(precision)
	R.append(recall)

print('precision: ', P)
print('recall: ', R)
