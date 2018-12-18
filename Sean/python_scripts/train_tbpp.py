import time
import os
import sys
import pickle
import cv2
import argparse

import numpy as np
import tensorflow as tf
import keras

home_dir = '/home/sgkelley/'

sys.path.append(os.path.join(home_dir, 'icken-and-chegg', 'Sean'))
from lib import tbpp_custom_utils

ssd_detectors_dir = os.path.join(home_dir, 'sean', 'ssd_detectors')
sys.path.append(ssd_detectors_dir)

from utils.model_utils import load_weights
from tbpp_model import TBPP512, TBPP512_dense
from tbpp_utils import PriorUtil
from ssd_data import InputGenerator
from ssd_training import Logger
from tbpp_training import TBPPFocalLoss

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)

parser = argparse.ArgumentParser()
parser.add_argument('--use_gen_annots', help='use generated annotations')
parser.add_argument('--vgg', help='use vgg backend (default: densenet)')
parser.add_argument('--annots_path', type=str, default=os.path.join(home_dir, 'data', 'annotations', 'current'), help='path to either annots folder or txt file')
parser.add_argument('--map_images_dir', type=str, default=os.path.join(home_dir, 'data', 'maps'), help='dir where map images are')
parser.add_argument('--output_dir', type=str, default=os.path.join(home_dir, 'sean', 'output'), help='dir to output checkpoints and logs')
parser.add_argument('--train_split_file', type=str, default=os.path.join(home_dir, 'torch-phoc', 'splits', 'train_files.txt'), help='file from torch_phoc with train split')
parser.add_argument('--val_split_file', type=str, default=os.path.join(home_dir, 'torch-phoc', 'splits', 'val_files.txt'), help='file from torch_phoc with val split')
parser.add_argument('--weights_path', type=str, default=os.path.join(home_dir, 'data', 'weights.018.h5'), help='weights for transfer learning')
parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')

args = parser.parse_args()

train_filenames = []
val_filenames = []
with open(args.train_split_file) as f:
    train_filenames = [line.replace("\n", "") for line in f.readlines()]

with open(args.val_split_file) as f:
    val_filenames = [line.replace("\n", "") for line in f.readlines()]

if args.vgg:
    # TextBoxes++
    model = TBPP512(softmax=False)
    freeze = ['conv1_1', 'conv1_2',
            'conv2_1', 'conv2_2',
            'conv3_1', 'conv3_2', 'conv3_3',
            'conv4_1', 'conv4_2', 'conv4_3',
            'conv5_1', 'conv5_2', 'conv5_3',
            ]
    experiment = 'vggtbpp512fl_maps'
else:
    # TextBoxes++ + DenseNet
    model = TBPP512_dense(softmax=False)
    freeze = []
    experiment = 'dsodtbpp512fl_maps'

checkdir = os.path.join(args.output_dir, 'tbpp', 'checkpoints', time.strftime('%Y%m%d%H%M') + '_' + experiment)
if not os.path.exists(checkdir):
    os.makedirs(checkdir)

prior_util = PriorUtil(model)

if args.weights_path is not None:
    load_weights(model, args.weights_path)

for layer in model.layers:
    layer.trainable = not layer.name in freeze

if args.use_gen_annots:
    train_images, train_regions = tbpp_custom_utils.read_generated_annots(args.annots_path, train_filenames)
    val_images, val_regions = tbpp_custom_utils.read_generated_annots(args.annots_path, val_filenames)
else:
    train_images, train_regions = tbpp_custom_utils.read_raw_annots(args.annots_path, train_filenames)
    val_images, val_regions = tbpp_custom_utils.read_raw_annots(args.annots_path, val_filenames)

print('train image count:', len(train_images))
print('val image count:', len(val_images))

epochs = 100
initial_epoch = 18

#optim = keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=0, nesterov=True)
optim = keras.optimizers.Adam(lr=1e-3, beta_1=0.9, beta_2=0.999, epsilon=0.001, decay=0.0)

# weight decay
regularizer = keras.regularizers.l2(5e-4) # None if disabled
# regularizer = None
for l in model.layers:
    if l.__class__.__name__.startswith('Conv'):
        l.kernel_regularizer = regularizer

loss = TBPPFocalLoss()

model.compile(optimizer=optim, loss=loss.compute, metrics=loss.metrics)

if args.use_gen_annots:
    history = model.fit_generator(
            tbpp_custom_utils.tbpp_generate_data(train_images, train_regions, args.batch_size, prior_util),
            steps_per_epoch=int((len(train_images) / float(args.batch_size))), 
            epochs=epochs, 
            verbose=1, 
            callbacks=[
                keras.callbacks.ModelCheckpoint(os.path.join(checkdir, 'weights.{epoch:03d}.h5'), verbose=1, save_weights_only=True),
                Logger(checkdir),
                #LearningRateDecay()
            ], 
            validation_data=tbpp_custom_utils.tbpp_generate_data(val_images, val_regions, args.batch_size, prior_util), 
            validation_steps=int((len(val_images) / float(args.batch_size))), 
            class_weight=None,
            max_queue_size=1, 
            workers=1, 
            #use_multiprocessing=False, 
            initial_epoch=initial_epoch, 
            #pickle_safe=False, # will use threading instead of multiprocessing, which is lighter on memory use but slower
            )
else:
    history = model.fit_generator(
            tbpp_custom_utils.tbpp_raw_generate_data(args.map_images_dir, train_images, train_regions, args.batch_size, prior_util),
            steps_per_epoch=int((len(train_images) / float(args.batch_size))), 
            epochs=epochs, 
            verbose=1, 
            callbacks=[
                keras.callbacks.ModelCheckpoint(os.path.join(checkdir, 'weights.{epoch:03d}.h5'), verbose=1, save_weights_only=True),
                Logger(checkdir),
                #LearningRateDecay()
            ], 
            validation_data=tbpp_custom_utils.tbpp_raw_generate_data(args.map_images_dir, val_images, val_regions, args.batch_size, prior_util), 
            validation_steps=int((len(val_images) / float(args.batch_size))), 
            class_weight=None,
            max_queue_size=1, 
            workers=1, 
            #use_multiprocessing=False, 
            initial_epoch=initial_epoch, 
            #pickle_safe=False, # will use threading instead of multiprocessing, which is lighter on memory use but slower
            )
