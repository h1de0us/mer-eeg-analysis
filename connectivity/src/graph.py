import numpy as np
import mne
import networkx as nx
import matplotlib.pyplot as plt
import pickle

from typing import List, Tuple

from mne_connectivity import spectral_connectivity_epochs
from mne import make_fixed_length_epochs

class EEGConnectivityGraph:
    def __init__(self, eeg_data_path):
        x = pickle.load(open(eeg_data_path, "rb"), encoding="latin1")
        data = x["data"] # (n_trials, n_channels, n_samples)
        self.data = data[:, :32, :] # select only the first 32 channels
        n_trials, n_channels, n_samples = data.shape
        self.labels = x["labels"]
        self.sfreq = 128 # 128 Hz, via https://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html
        self.channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 
                         'FC1', 'C3', 'T7', 'CP5', 'CP1', 
                         'P3', 'P7', 'PO3', 'O1', 'Oz', 
                         'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 
                         'F8', 'FC6', 'FC2', 'Cz', 'C4', 
                         'T8', 'CP6', 'CP2', 'P4', 'P8', 
                         'PO4', 'O2']

        print("Data shape: (n_trials, n_channels, n_samples):", n_trials, n_channels, n_samples)

    def compute_connectivity(self, 
                             method : str | List[str]='correlation',
                             **kwargs):
        if method == 'correlation':
            connectivity_matrix = self._compute_correlation(**kwargs)
        else:
            connectivity_matrix = self._compute_connectivity(**kwargs)

        return connectivity_matrix

    def _compute_correlation(self, threshold=0.6):
        correlation_matrix = np.corrcoef(self.data)
        np.fill_diagonal(correlation_matrix, 0)
        correlation_matrix_binary = np.abs(correlation_matrix) > threshold
        return correlation_matrix_binary

    # TODO: apply windowing to compute epochs
    def _compute_connectivity(self, 
                              fmin: float | Tuple[float] = 8,
                              fmax: float | Tuple[float] = 12, 
                              method : str | List[str] = 'coh',
                              mode : str = 'fourier' 
                              ):

        con = spectral_connectivity_epochs( 
            data=self.data,
            method=method, 
            mode=mode,
            sfreq=self.sfreq,
            fmin=fmin, 
            fmax=fmax, 
            faverage=True, 
            verbose=False)

        return con # SpectralConnectivity object

    def create_graph(self, connectivity_matrix, title='EEG Functional Connectivity Network'):
        G = nx.from_numpy_array(connectivity_matrix)
        pos = nx.circular_layout(G)  # Layout nodes in a circle
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=10)
        plt.title(title)
        plt.show()