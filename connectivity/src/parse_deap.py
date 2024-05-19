
# iterate over data folder:
import os
import argparse
import numpy as np
from tqdm import tqdm

from graph import EEGConnectivityGraph, process_participants

bands_min = [4, 8, 12, 30]
bands_max = [8, 12, 30, 45]

# parse args
parser = argparse.ArgumentParser(description='Compute connectivity measures for DEAP dataset')
parser.add_argument('--method', type=str, default='coh', help='Connectivity method')
parser.add_argument('--data_path', type=str, default='/Users/h1de0us/uni/mer-eeg-analysis/data/deap_filtered', help='Path to the DEAP dataset')
parser.add_argument('--duration', type=float, default=3.0, help='Epoch duration in seconds')
args = parser.parse_args()

eeg_files = [file for file in os.listdir(args.data_path) if file.endswith('.dat')]

for eeg_file in tqdm(eeg_files):
    print(f'Processing {eeg_file}')
    eeg_path = os.path.join(args.data_path, eeg_file)
    graphs = process_participants(eeg_data_path=eeg_path)
    cons = [graph.compute_connectivity(method=args.method, 
                                       fmin=bands_min, 
                                       fmax=bands_max, 
                                       duration=args.duration
                                       ) 
                                       for graph in graphs]
    con_data_by_bands = [con.get_data(output="dense") for con in cons] # (n_channels, n_channels, n_bands)

    # save the connectivity data
    duration_str = str(args.duration)
    for i, con_data in enumerate(con_data_by_bands):
        con_path = eeg_path.replace('.dat', f'_{args.method}_{duration_str}_trial_{i}.npy')
        np.save(con_path, con_data)
