This repository is devoted to a research project named "Analysis of the Interaction of Music and Emotions with the Help of EEG"

## Installation guide
Before installing the dependencies, create a virtual environment using pyenv or venv and then run
```
pip install -r requirements.txt
```
If you get any problems related to the package ```mne-connectivity```, additionally run ```pip install -U mne-connectivity```

WARNING: While working with connectivity in EEG data, create a separate virtual environment and install the dependencies there, as EEGRAPH currently uses outdated versions of numpy and mne-tools.

TODO: how to train a model, how to extract features, how to build graphs based on eeg data

## Structure
* ```connectivity``` folder describes the process of acquiring graphs from EEG data using various types of connectivity: structural, functional, effective
* ```graphs``` folder contains an implementation of a [Graphormer](https://arxiv.org/abs/2106.05234) model. This structure of the folder is based on a heavily modified fork of [this repo](https://github.com/victoresque/pytorch-template)
* ```tda``` folder contains the code related to the extraction of the topological features of the graph