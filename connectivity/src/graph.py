import numpy as np
import mne
import networkx as nx
import matplotlib.pyplot as plt
import pickle

from typing import List, Tuple

from mne_connectivity import spectral_connectivity_epochs, spectral_connectivity_time
from mne import make_fixed_length_epochs

def process_participants(eeg_data_path):
    # for each participant there are 40 videos
    # each video == one EEG Connectivity Graph
    x = pickle.load(open(eeg_data_path, "rb"), encoding="latin1")
    data = x["data"] # (n_videos, n_channels, n_samples = 60 sec * 128 Hz)
    data = data[:, :32, :] # select only the first 32 channels
    sfreq = 128 # 128 Hz, via https://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html
    data = data[:, :, 3 * sfreq:] # remove first 3 seconds
    labels = x["labels"] # (n_videos, 4) valence, arousal, dominance, liking
    # for each video, create an EEG Connectivity Graph
    graphs = []
    for i in range(data.shape[0]):
        graph = EEGConnectivityGraph(data[i], labels[i])
        graphs.append(graph)
    return graphs


class EEGConnectivityGraph:
    def __init__(self, data, label):
        self.data = data
        n_channels, n_samples = data.shape
        self.label = label
        self.sfreq = 128 # 128 Hz, via https://www.eecs.qmul.ac.uk/mmv/datasets/deap/readme.html
        self.channels = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 
                         'FC1', 'C3', 'T7', 'CP5', 'CP1', 
                         'P3', 'P7', 'PO3', 'O1', 'Oz', 
                         'Pz', 'Fp2', 'AF4', 'Fz', 'F4', 
                         'F8', 'FC6', 'FC2', 'Cz', 'C4', 
                         'T8', 'CP6', 'CP2', 'P4', 'P8', 
                         'PO4', 'O2']
        print("Data shape: (n_channels, n_samples):", n_channels, n_samples)

        info = mne.create_info(ch_names=self.channels, sfreq=128, ch_types='eeg', verbose=False)
        self.raw = mne.io.RawArray(self.data, info, verbose=False)
        self.epochs = None
        self.duration = None
        self.overlap = None

    def compute_connectivity(self, 
                             method : str | List[str]='correlation',
                             **kwargs):
        if method == 'correlation':
            connectivity_matrix = self._compute_correlation(**kwargs)
        else:
            connectivity_matrix = self._compute_connectivity(**kwargs)

        return connectivity_matrix
    
    def compute_connectivity_time(self,
                                  method : str | List[str]='correlation',
                                  **kwargs):
        if method == 'correlation':
            return self._compute_correlation(**kwargs)
        else:
            return self._compute_connectivity_time(**kwargs)
        

    def _compute_correlation(self, threshold=0.6):
        correlation_matrix = np.corrcoef(self.data)
        np.fill_diagonal(correlation_matrix, 0)
        correlation_matrix_binary = np.abs(correlation_matrix) > threshold
        return correlation_matrix_binary

    def _compute_epochs(self, duration=3.0, overlap=0.0):
        if self.epochs is None or duration != self.duration or overlap != self.overlap:
            self.epochs = make_fixed_length_epochs(self.raw, duration=duration, overlap=overlap, verbose=False)
            # self.epochs.baseline = (0, duration * self.sfreq)
            self.duration = duration
            self.overlap = overlap
            print("Epochs shape:", self.epochs.get_data().shape)
        return self.epochs
    
    def _compute_connectivity(self, 
                              fmin: float | Tuple[float] = 8,
                              fmax: float | Tuple[float] = 12, 
                              method : str | List[str] = 'coh',
                              mode : str = 'fourier',
                              duration: float = 3.0,
                              overlap: float = 0.0,
                              faverage: bool = True,
                              ):
        
        # self.data.shape == (n_channels, n_samples)
        epochs = self._compute_epochs(duration=duration, overlap=overlap)

        con = spectral_connectivity_epochs( 
            data=epochs.get_data(),
            method=method, 
            mode=mode,
            sfreq=self.sfreq,
            fmin=fmin, 
            fmax=fmax, 
            faverage=faverage, 
            verbose=False)

        return con # SpectralConnectivity object
    

    def _compute_connectivity_time(self, 
                               fmin: float | Tuple[float] = 8,
                               fmax: float | Tuple[float] = 12, 
                               method : str | List[str] = 'coh',
                               mode : str = 'multitaper',
                               duration: float = 3.0,
                               overlap: float = 0.0,
                               faverage: bool = True,
                               ):
    

        con = spectral_connectivity_time(
            data=self.data.reshape(1, self.data.shape[0], self.data.shape[1]),
            freqs=[fmin, fmax],
            method=method, 
            mode=mode,
            sfreq=self.sfreq,
            fmin=fmin, 
            fmax=fmax, 
            verbose=False,
            faverage=faverage
            )

        return con

    def create_graph(self, connectivity_matrix, title='EEG Functional Connectivity Network'):
        G = nx.from_numpy_array(connectivity_matrix)
        pos = nx.circular_layout(G)  # Layout nodes in a circle
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=800, font_size=10)
        plt.title(title)
        plt.show()