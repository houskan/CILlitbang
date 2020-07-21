# CILlitbang
## Create the environment
```
conda env create -f environment.yml
conda activate tf2
```
## Run on Leonhard
TODO, copy config.cfg to src then
```sh
cd src
bsub -W 8:30 -R "rusage[ngpus_excl_p=1,mem=8192]" "python main.py -c config.cfg"
```
