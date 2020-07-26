# CILlitbang
## Table of Contents
* [About the Project](#about-the-project)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
* [Usage](#usage)
  * [Run the code](#run-the-code)
  * [Reproduce our results](#reproduce-our-results)
   
## About The Project
This is the graded semester project for the graded semester project for the Computational Intelligence Lab 2020 lecture at ETH Zurich.
It contains our approach for road segmentation.

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
If you want to reproduce the results of our paper, we provide predefined config files for the runs. Just set the path to the corresponding config file and run the code. If you want use our trained models: Use the corresponding config file and set the `train` parameter to `False` and set the `model-path` parameter to the model you want to use.
