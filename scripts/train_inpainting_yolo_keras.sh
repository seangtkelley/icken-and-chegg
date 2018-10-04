#!/bin/bash

python3 /home/sgkelley/keras-yolo3/train.py --train_annots "/home/sgkelley/config/yolo_inpainted_416window_train.txt" \
                                            --val_annots "/home/sgkelley/config/yolo_inpainted_416window_val.txt" \
                                            --classes "/home/sgkelley/config/word_class.txt" \
                                            --output_dir "/home/sgkelley/output/keras_yolo3/inpainted_416_09282018/" \
                                            --input_size 416

exit
