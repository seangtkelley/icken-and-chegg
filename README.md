# Detection and recognition of texts in cartographic images

### [More info here](https://people.cs.umass.edu/~ray/maps_project.html)

## Setup

1. Clone
2. Install submodules: `git submodule init && git submodule update`

This repository uses [ssd_detectors](https://github.com/mvoelk/ssd_detectors) and [cascaded-faster-rcnn](https://github.com/seangtkelley/cascaded-faster-rcnn).

## Repository Structure

The directory `python_scripts` contains all scripts for training, prediction, and evaluation of TextBoxes++ on the map imagery. The directory `sbatch_scripts` are the corresponding sbatch scripts for running on the Gypsum compute cluster. `lib` containings customing libs as well as the submodules. `notebooks` contains any jupyter notebooks for experimenting.

## Script Details

- `convery_txt_preds_npy`
    - _args_
    - **--results file** with prediction results
    - **--dir_output** output directory for .npy files
    - **--images_dir** directory with 31 maps
- `custom_multithreaded_scorer`
    - _args_
    - **--train_dir** directory with all annotations .npy files
    - **--test_dir** directory with predictions .npy files
- `draw_gt_annots`
    - _args_
    - **path_to_annotations** directory with all annotations .npy files (not cmd arg)
    - **path_to_maps directory** with 31 maps (not cmd arg)
- `draw_preds`
    - _args_
    - **--txt** read annots from txt file
    - **--annots_pathpath** to either annots folder or txt file
    - **--map_images_dir** dir where map images are
    - **--output_dir** dir to output images
    - **--test_only** whether or not to only evaluate test images")
    - **--test_split** file from torch_phoc with test split
- `generate_tbpp_preds`
    - _args_
    - **--output_dir** output dir
    - **--weights_file** file with model weights
    - **--images_dir** map images directory
    - **--preprocess**whether or not to preform same preprocess as done in original implementations (background removal, etc...)
    - **--test_only** whether or not to only evaluate test images
    - **--test_split** file from torch_phoc with test split
    - **--confidence** confidence threshold for predictions
    - **--rotate** whether or not to rotate image
- `train_tbpp`
    - _args_
    - **--use_gen_annots** use generated annotations
    - **--vgg** use vgg backend (default: densenet)
    - **--annots_path** path to either annots folder or txt file
    - **--map_images_dir** dir where map images are
    - **--output_dir** dir to output checkpoints and logs
    - **--train_split_file** file from torch_phoc with train split
    - **--val_split_file** file from torch_phoc with val split
    - **--weights_path** weights for transfer learning
    - **--batch_size** batch size for training