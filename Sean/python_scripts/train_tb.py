import time
import os
import sys
import pickle
import cv2

import numpy as np
import tensorflow as tf
import keras

home_dir = os.path.expanduser("~")

sys.path.append(os.path.join(home_dir, 'icken-and-chegg', 'Sean'))
from lib import tbpp_custom_utils

ssd_detectors_dir = os.path.join(home_dir, 'sean', 'ssd_detectors')
sys.path.append(ssd_detectors_dir)

from utils.model_utils import load_weights
from tb_model import TB300
from ssd_utils import PriorUtil
from ssd_utils import load_weights
from tbpp_training import SSDLoss, LearningRateDecay, Logger

config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
sess = tf.Session(config=config)
keras.backend.tensorflow_backend.set_session(sess)

train_annots_path = os.path.join(home_dir, 'sean', 'cascaded-faster-rcnn', 'word-faster-rcnn', 'DataGeneration', 'fold_1', 'cropped_annotations_angles_-90to90step5_fixed.txt')
output_dir = os.path.join(home_dir, 'sean', 'output')

#train_annots_path = os.path.join(home_dir, 'Documents', 'indystudy', 'cascaded-faster-rcnn', 'word-faster-rcnn', 'DataGeneration', 'fold_1', 'cropped_annotations.txt')
#output_dir = os.path.join(home_dir, 'Documents', 'indystudy', 'output')

train_split_file = os.path.join(home_dir, 'torch-phoc', 'splits', 'train_files.txt')
val_split_file = os.path.join(home_dir, 'torch-phoc', 'splits', 'val_files.txt')

#train_split_file = os.path.join(home_dir, 'Documents', 'indystudy', 'torch-phoc', 'splits', 'train_files.txt')
#val_split_file = os.path.join(home_dir, 'Documents', 'indystudy', 'torch-phoc', 'splits', 'val_files.txt')

train_filenames = []
val_filenames = []
with open(train_split_file) as f:
    train_filenames = [line.replace("\n", "") for line in f.readlines()]

with open(val_split_file) as f:
    val_filenames = [line.replace("\n", "") for line in f.readlines()]

train_images, train_regions = tbpp_custom_utils.read_generated_annots(train_annots_path, train_filenames)
val_images, val_regions = tbpp_custom_utils.read_generated_annots(train_annots_path, val_filenames)

print('train image count:', len(train_images))
print('val image count:', len(val_images))

model = TB300()

prior_util = PriorUtil(model)

weights_path = os.path.join(home_dir, 'data', 'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5')
layer_list = [('block1_conv1', 'conv1_1'),
              ('block1_conv2', 'conv1_2'),
              ('block2_conv1', 'conv2_1'),
              ('block2_conv2', 'conv2_2'),
              ('block3_conv1', 'conv3_1'),
              ('block3_conv2', 'conv3_2'),
              ('block3_conv3', 'conv3_3'),
              ('block4_conv1', 'conv4_1'),
              ('block4_conv2', 'conv4_2'),
              ('block4_conv3', 'conv4_3'),
              ('block5_conv1', 'conv5_1'),
              ('block5_conv2', 'conv5_2'),
              ('block5_conv3', 'conv5_3')]
load_weights(model, weights_path, layer_list)

freeze = ['conv1_1', 'conv1_2',
          'conv2_1', 'conv2_2',
          'conv3_1', 'conv3_2', 'conv3_3',
          #'conv4_1', 'conv4_2', 'conv4_3',
          #'conv5_1', 'conv5_2', 'conv5_3',
         ]

for layer in model.layers:
    layer.trainable = not layer.name in freeze

experiment = 'tb300_maps'

epochs = 100
batch_size = 16
initial_epoch = 0

checkdir = os.path.join(output_dir, 'tbpp', 'checkpoints', time.strftime('%Y%m%d%H%M') + '_' + experiment)
if not os.path.exists(checkdir):
    os.makedirs(checkdir)

#optim = keras.optimizers.SGD(lr=1e-3, momentum=0.9, decay=0, nesterov=True)
optim = keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

# weight decay
regularizer = keras.regularizers.l2(5e-4) # None if disabled
for l in model.layers:
    if l.__class__.__name__.startswith('Conv'):
        l.kernel_regularizer = regularizer

loss = SSDLoss(alpha=1.0, neg_pos_ratio=3.0)

model.compile(optimizer=optim, loss=loss.compute, metrics=loss.metrics)

history = model.fit_generator(
        tbpp_custom_utils.tb_generate_data(train_images, train_regions, batch_size, prior_util),
        steps_per_epoch=int((len(train_images) / float(batch_size))), 
        epochs=epochs, 
        verbose=1, 
        callbacks=[
            keras.callbacks.ModelCheckpoint(os.path.join(checkdir, 'weights.{epoch:03d}.h5'), verbose=1, save_weights_only=True),
            Logger(checkdir),
            # learning rate decay usesd with sgd
            # LearningRateDecay(methode='linear', base_lr=1e-3, n_desired=40000, desired=0.1, bias=0.0, minimum=0.1)
        ], 
        validation_data=tbpp_custom_utils.tb_generate_data(val_images, val_regions, batch_size, prior_util), 
        validation_steps=int((len(val_images) / float(batch_size))), 
        class_weight=None,
        max_queue_size=1, 
        workers=1, 
        #use_multiprocessing=False, 
        initial_epoch=initial_epoch, 
        #pickle_safe=False, # will use threading instead of multiprocessing, which is lighter on memory use but slower
        )
