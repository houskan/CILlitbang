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
│   ├── test
│   │   └── images
│   ├── training
│   │   ├── groundtruth
│   │   └── images
│   ├── training_original
│   │   ├── groundtruth
│   │   └── images
│   ├── validation
│   │   ├── groundtruth
│   │   └── images
│   └── validation_original
│       ├── groundtruth
│       └── images
├── environment.yml
├── notebook
│   ├── postprocessing.ipynb
│   └── test_results_comparison.ipynb
└── src
    ├── argparser.py
    ├── baselines
    │   ├── basic_cnn
    │   │   ├── tf_aerial_images.py
    │   │   ├── tf_aerial_images_submission.py
    │   │   └── training
    │   │       ├── groundtruth
    │   │       └── images
    │   └── keras_segmentation
    │       ├── keras_seg.py
    │       └── keras_seg_submission.py
    ├── data
    │   ├── combined_prediction.py
    │   ├── data.py
    │   ├── data_patch.py
    │   ├── helper.py
    │   ├── post_processing.py
    │   └── tensorboard_image.py
    ├── main.py
    ├── main_patch.py
    ├── models
    │   ├── loss_functions.py
    │   ├── unet.py
    │   ├── unet_dilated_v1.py
    │   ├── unet_dilated_v2.py
    │   ├── unet_dilated_v3.py
    │   ├── unet_dilated_v3_patch.py
    │   ├── unet_dilated_v4.py
    │   └── unet_patch.py
    └── submission
        ├── log_submission.py
        ├── mask_to_submission.py
        └── submission_to_mask.py
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
