import numpy as np
import mne
import networkx as nx
import matplotlib.pyplot as plt

from typing import List, Tuple

from mne_connectivity import spectral_connectivity_epochs
from mne import make_fixed_length_epochs

class EEGConnectivityGraph:
    def __init__(self, eeg_data_path):
        self.raw = mne.io.read_raw_edf(eeg_data_path)
        self.sfreq = self.raw.info['sfreq']
        self.data = self.raw.get_data()
        self.epochs = None

        print("Data shape:", self.data.shape)

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
                              window_size: float = 0.5, 
                              overlap: float = 0.25):
        
        epochs = self.compute_epochs(window_size=window_size, overlap=overlap)
        print("Epochs shape:", epochs.get_data().shape)

        con = spectral_connectivity_epochs(
            # data=self.raw, 
            data=epochs,
            method=method, 
            mode='multitaper', 
            sfreq=self.sfreq,
            fmin=fmin, 
            fmax=fmax, 
            faverage=True, 
            verbose=False
        )

        return con # SpectralConnectivity

        # coherence_matrix = con[:, :, 0]  # Use only the first frequency band
        # coherence_matrix_binary = coherence_matrix > threshold
        # return coherence_matrix_binary

    def compute_epochs(self, window_size=0.5, overlap=0.25):
        if self.epochs is None:
            self.epochs = make_fixed_length_epochs(raw=self.raw, duration=window_size, overlap=overlap)
        return self.epochs

    def create_graph(self, connectivity_matrix, title='EEG Functional Connectivity Network'):
        G = nx.from_numpy_array(connectivity_matrix)
        pos = nx.circular_layout(G)  # Layout nodes in a circle
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=10)
        plt.title(title)
        plt.show()