# CILlitbang
## Table of Contents
* [About the Project](#about-the-project)
* [Folder Structure](#folder-structure)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
* [Usage](#usage)
  * [Run the code](#run-the-code)
  * [Reproduce our results](#reproduce-our-results)
    * [Train and predict results](#train-and-predict-results)
    * [Predict using pretrained models](#predict-using-pretrained-models)

## About The Project
This repository contains the source code for the graded semester project for the [Computational Intelligence Lab 2020 lecture](http://da.inf.ethz.ch/teaching/2020/CIL/) at ETH Zurich.
Please follow the instructions below to get started and reproduce our results.
Read the [paper](todo) for more information about our experiments and design decisions.

## Folder Structure
```
├── README.md
├── data                                             
│   ├── test                                          - 94 test images provided for the project
│   │   └── images
│   ├── training                                      - 199 training images (provided and additional data)
│   │   ├── groundtruth
│   │   └── images
│   ├── training_original                             - 90 of the 100 training images provided for the project
│   │   ├── groundtruth
│   │   └── images
│   ├── validation                                    - 22 validation images (provided and additional data)
│   │   ├── groundtruth
│   │   └── images
│   └── validation_original                           - 10 validation images taken from the 100 training images provided for the project
│       ├── groundtruth
│       └── images
├── environment.yml                                   - YAML file for conda environment setup
├── notebook                                          
│   ├── postprocessing.ipynb                          - Jupyter notebook for the visualization of post-processing steps
│   └── test_results_comparison.ipynb                 - Jupyter notebook for the comparison of different predictions
├── out                                               - Created directory. Contains model weights, predictions, config file and submission file after completed run
├── tmp                                               - Created directory. Contains model weights during training
└── src
    ├── argparser.py                                  - Argumentparser for command line arguments
    ├── baselines                                     - Separate baseline architectures
    │   ├── basic_cnn                                 - Baseline from CIL 2020 exercise 9 (results not included in report)
    │   │   ├── tf_aerial_images.py                        - Code to execute for this baseline
    │   │   ├── tf_aerial_images_submission.py             - Creates submission file for this baseline (is called from tf_aerial_images.py)
    │   │   └── training                                   - 100 provided training images for the project 
    │   │       ├── groundtruth
    │   │       └── images
    │   └── keras_segmentation                         - Baseline from https://github.com/divamgupta/image-segmentation-keras (using resnet50_unet model)
    │       ├── keras_seg.py                                - Code to execute for this baseline
    │       └── keras_seg_submission.py                     - Creates submission file for this baseline (is called from keras_seg.py)
    ├── data
    │   ├── combined_prediction.py                     - Creates predictions of test images
    │   ├── data.py                                    - Prepares data for all models except patch-based
    │   ├── data_patch.py                              - Prepares data for patch-based U-Net
    │   ├── helper.py                                  - Helper functions for the other files in this directory
    │   ├── post_processing.py                         - Post-processing functions
    │   └── tensorboard_image.py                       - TensorBoard Callback
    ├── main.py                                        - Main function for all models except patch-based
    ├── main_patch.py                                  - Main function for patch-based U-Net
    ├── models
    │   ├── loss_functions.py                          - Custom loss functions and callback metrics
    │   ├── unet.py                                    - Original U-Net model from https://github.com/zhixuhao/unet (serves as baseline in the report)
    │   ├── unet_dilated_v1.py                         - U-Net dilated v1
    │   ├── unet_dilated_v2.py                         - U-Net dilated v2
    │   ├── unet_dilated_v3.py                         - U-Net dilated v2 with transposed convolutions
    │   ├── unet_dilated_v3_patch.py                   - U-Net dilated v2 with transposed convolutions, adapted for patch-based model
    │   ├── unet_dilated_v4.py                         - U-Net dilated v1 with transposed convolutions
    │   └── unet_patch.py                              - Original U-Net model, adapted for patch-based model
    └── submission
        ├── log_submission.py                          - Prepares submission directory (see 'out' directory)
        ├── mask_to_submission.py                      - Provided code to create submission file from binary mask
        └── submission_to_mask.py                      - Provided code to create binary mask from submission file
```

## Getting Started
### Prerequisites
- Conda
  Please consult the [Conda Installation Guide](https://docs.conda.io/projects/conda/en/latest/user-guide/install/#regular-installation) for more information.
  If you want to run this code on the Leonhard Cluster, please follow the relevant parts of [this tutorial](http://kevinkle.in/jekyll/update/2019/02/28/leonhard.html)
- Create and activate the virtual environment
  ```
  conda env create -f environment.yml
  conda activate tf2
  ```
  Of course, you can give the environment a custom name with the `-- name ` flag.

## Usage
### Run the code
Once the virutal environment is activated you can run the code as follows.
- Go into the `src` directory.
  ```sh
  cd src/
  ```
- Run the program
  ```sh
  python main.py
  ```
- Setting parameters with flags
  - Getting help
    ```sh
    python main.py --help
    ```
  - Example: Start a run with 30 training epochs
    ```sh
    python main.py --epochs=30
    ```
  - However, we recommend using the config template by modifying it.
    Edit the file and set the parameters accordingly. To start a run with a config file as input type
    ```sh
    python main.py -c <PATH_TO_CONFIG>
    ```
 - If you want to run this code on the Leonhard cluster submit your jobs with 8GB of memory. Example:
   ```sh
   bsub -W 8:30 -R "rusage[ngpus_excl_p=1,mem=8192]" "python main.py -c config.cfg"
   ```
### Reproduce our results
Reproducing the results of our paper can be done easily. We provide predefined config files we used for our runs. These include fixed random seeds that worked well in our experiments. While exact reproduction of results is not possible when executing tensorflow code on a GPU, they should still be very similar when taking the same seed in different runs.
#### Train and predict results
Pick the experiment you want to reproduce and select the corresponding config file from XYZ. Follow the above instructions on how to use config files with our code and let the job run.
#### Predict using pretrained models
Pick the experiment you want to reproduce and select the corresponding config file from XYZ. Follow the above instructions on how to use config files with our code. Now modify the config file and set the `train-model` parameter to `False` and set the `model-path` parameter to the model you want to use. Now, let the job run.
