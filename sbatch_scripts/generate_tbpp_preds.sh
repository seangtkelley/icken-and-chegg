#!/bin/bash

python3 /home/sgkelley/icken-and-chegg/Sean/python_scripts/generate_tbpp_preds.py --output_dir "/home/sgkelley/sean/output/tbpp/np_preds/rboxes_no_angles/synthtext" \
                                                                                --weights_file "/home/sgkelley/sean/ssd_detectors/checkpoints/201807091503_dsodtbpp512fl_synthtext/weights.018.h5"

exit