import numpy as np
import keras
import time
import os
import sys
import pickle
import cv2

home_dir = os.path.expanduser("~")
ssd_detectors_dir = os.path.join(home_dir, 'sean', 'ssd_detectors')

sys.path.append(ssd_detectors_dir)

from utils.model_utils import load_weights

from tbpp_model import TBPP512, TBPP512_dense
from tbpp_utils import PriorUtil
from ssd_data import InputGenerator
from ssd_training import Logger
from tbpp_training import TBPPFocalLoss

train_annots_path = os.path.join(home_dir, 'sean', 'cascaded-faster-rcnn', 'word-faster-rcnn', 'DataGeneration', 'fold_1', 'cropped_annotations.txt')
output_dir = os.path.join(home_dir, 'sean', 'output')

train_split_file = os.path.join(home_dir, 'torch-phoc', 'splits', 'train_files.txt')
val_split_file = os.path.join(home_dir, 'torch-phoc', 'splits', 'val_files.txt')
train_filenames = []
val_filenames = []
with open(train_split_file) as f:
    train_filenames = [line.replace("\n", "") for line in f.readlines()]

with open(val_split_file) as f:
    val_filenames = [line.replace("\n", "") for line in f.readlines()]

def read_annots(annots_path, filenames):
    annots = open(annots_path, "r").readlines()
    image_paths = []
    regions = []

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
            
        elif len(line.split(" ")) == 4:
            split_line = line.split(" ")
            x = float(split_line[0]); y = float(split_line[1])
            r_w = float(split_line[2]); r_h = float(split_line[3])

            temp_regions.append( (x, y, r_w, r_h) )

    return image_paths, regions

def generate_data(image_paths, regions, batch_size):
    h, w = (512, 512)
    mean = np.array([104,117,123])
    num_batches = len(image_paths) // batch_size

    inputs, targets = [], []

    while True:
        idxs = np.arange(len(image_paths))
        np.random.shuffle(idxs)
        idxs = idxs[:num_batches*batch_size]
        for j, i in enumerate(idxs):
            img = cv2.imread(image_paths[i])

            boxes = np.zeros(shape=(4, len(regions[i])))
            i = 0
            for box in regions[i]:
                xmin, ymin, width, height = box
                xmax, ymax = xmin+width, ymin+height 
                boxes[:, i] = np.array([xmin/w, ymin/h, xmax/w, ymax/h]).T
                i += 1

            boxes = np.concatenate([boxes, np.ones([boxes.shape[0],1])], axis=1)
            
            img = cv2.resize(img, (w,h), cv2.INTER_LINEAR)
            img = img.astype(np.float32)
            
            img -= mean[np.newaxis, np.newaxis, :]
            #img = img / 25.6
            
            inputs.append(img)
            targets.append(boxes)
            
            #if len(targets) == batch_size or j == len(idxs)-1: # last batch in epoch can be smaller then batch_size
            if len(targets) == batch_size:
                targets = [prior_util.encode(y) for y in targets]
                targets = np.array(targets, dtype=np.float32)

                tmp_inputs = np.array(inputs, dtype=np.float32)
                tmp_targets = targets
                inputs, targets = [], []
                yield tmp_inputs, tmp_targets
            elif j == len(idxs)-1:
                # forgett last batch
                inputs, targets = [], []
                break
                
        print('NEW epoch')
    print('EXIT generator')


# TextBoxes++ + DenseNet
model = TBPP512_dense(softmax=False)
freeze = []
batch_size = 32
experiment = 'dsodtbpp512fl_maps'

prior_util = PriorUtil(model)

train_images, train_regions = read_annots(train_annots_path, train_filenames)
val_images, val_regions = read_annots(train_annots_path, val_filenames)

print('train image count:', len(train_images))
print('val image count:', len(val_images))

epochs = 100
initial_epoch = 0

for layer in model.layers:
    layer.trainable = not layer.name in freeze

checkdir = os.path.join(output_dir, 'tbpp', 'checkpoints', time.strftime('%Y%m%d%H%M') + '_' + experiment)
if not os.path.exists(checkdir):
    os.makedirs(checkdir)

# with open(checkdir+'/source.py','wb') as f:
#     source = ''.join(['# In[%i]\n%s\n\n' % (i, In[i]) for i in range(len(In))])
#     f.write(source.encode())

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
        generate_data(train_images, train_regions, batch_size),
        steps_per_epoch=int((len(train_images) / float(batch_size))//4), 
        epochs=epochs, 
        verbose=1, 
        callbacks=[
            keras.callbacks.ModelCheckpoint(os.path.join(checkdir, 'weights.{epoch:03d}.h5'), verbose=1, save_weights_only=True),
            Logger(checkdir),
            #LearningRateDecay()
        ], 
        validation_data=generate_data(val_images, val_regions, batch_size), 
        validation_steps=int((len(val_images) / float(batch_size))//4), 
        class_weight=None,
        max_queue_size=1, 
        workers=1, 
        #use_multiprocessing=False, 
        initial_epoch=initial_epoch, 
        #pickle_safe=False, # will use threading instead of multiprocessing, which is lighter on memory use but slower
        )