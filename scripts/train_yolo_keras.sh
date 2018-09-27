#!/bin/bash

python /home/sgkelley/keras_yolo3/train.py --train_annots "/home/sgkelley/config/yolo_608window_train.txt" --val_annots "/home/sgkelley/config/yolo_608window_val.txt" --classes "/home/sgkelley/config/word_class.txt" 

exit