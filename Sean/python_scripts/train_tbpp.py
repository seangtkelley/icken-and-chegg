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
from tbpp_model import TBPP512, TBPP512_dense
from tbpp_utils import PriorUtil
from ssd_data import InputGenerator
from ssd_training import Logger
from tbpp_training import TBPPFocalLoss

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


# TextBoxes++ + DenseNet
model = TBPP512_dense(softmax=False)
freeze = []
batch_size = 8
experiment = 'dsodtbpp512fl_maps'

prior_util = PriorUtil(model)

train_images, train_regions = tbpp_custom_utils.read_generated_annots(train_annots_path, train_filenames)
val_images, val_regions = tbpp_custom_utils.read_generated_annots(train_annots_path, val_filenames)

print('train image count:', len(train_images))
print('val image count:', len(val_images))

epochs = 100
initial_epoch = 0

for layer in model.layers:
    layer.trainable = not layer.name in freeze

checkdir = os.path.join(output_dir, 'tbpp', 'checkpoints', time.strftime('%Y%m%d%H%M') + '_' + experiment)
if not os.path.exists(checkdir):
    os.makedirs(checkdir)

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

history = model.fit_generator(
        tbpp_custom_utils.tbpp_generate_data(train_images, train_regions, batch_size, prior_util),
        steps_per_epoch=int((len(train_images) / float(batch_size))), 
        epochs=epochs, 
        verbose=1, 
        callbacks=[
            keras.callbacks.ModelCheckpoint(os.path.join(checkdir, 'weights.{epoch:03d}.h5'), verbose=1, save_weights_only=True),
            Logger(checkdir),
            #LearningRateDecay()
        ], 
        validation_data=tbpp_custom_utils.tbpp_generate_data(val_images, val_regions, batch_size, prior_util), 
        validation_steps=int((len(val_images) / float(batch_size))), 
        class_weight=None,
        max_queue_size=1, 
        workers=1, 
        #use_multiprocessing=False, 
        initial_epoch=initial_epoch, 
        #pickle_safe=False, # will use threading instead of multiprocessing, which is lighter on memory use but slower
        )
