This repository is devoted to a research project named "Analysis of the Interaction of Music and Emotions with the Help of EEG"

## Installation guide
Before installing the dependencies, create a virtual environment using pyenv or venv and then run
```
pip install -r requirements.txt
```

This work uses DEAP dataset, to estimate the connectivity, run 
```
python3 connectivity/src/parse_deap.py --method <desired_method> --data_path <path_to_deap_dataset>
```
TODO: how to train a model, how to extract features

## Structure
* ```connectivity``` folder describes the process of acquiring graphs from EEG data using various types of connectivity: structural, functional, effective
* ```graphs``` folder contains an implementation of a [Graphormer](https://arxiv.org/abs/2106.05234) model. This structure of the folder is based on a heavily modified fork of [this repo](https://github.com/victoresque/pytorch-template), code is partially taken from the [official Graphormer implementation](https://github.com/microsoft/Graphormer/tree/main)
* ```tda``` folder contains the code related to the extraction of the topological features of the graph