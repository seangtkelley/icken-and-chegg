# Detection and recognition of texts in cartographic images

### [More info here](https://people.cs.umass.edu/~ray/maps_project.html)

## Setup

1. Clone
2. Install submodules: `git submodule init && git submodule update`

## Repository Structure

The directory `python_scripts` contains all scripts for training, prediction, and evaluation of TextBoxes++ on the map imagery. The directory `sbatch_scripts` are the corresponding sbatch scripts for running on the Gypsum compute cluster. `lib` containings customing libs as well as the submodules. `notebooks` contains any jupyter notebooks for experimenting.

## Script Details

- `convery_txt_preds_npy`
- `custom_multithreaded_scorer`
- `draw_gt_annots`
- `draw_preds`
- `generate_tbpp_preds`
- `train_tbpp`