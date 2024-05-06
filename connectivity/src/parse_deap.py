
# iterate over data folder:
import os
import argparse
import numpy as np
from tqdm import tqdm

from graph import EEGConnectivityGraph

bands_min = [0.5, 4, 8, 12, 30]
bands_max = [4, 8, 12, 30, 45]

# parse args
parser = argparse.ArgumentParser(description='Compute connectivity measures for DEAP dataset')
parser.add_argument('--method', type=str, default='coh', help='Connectivity method')
parser.add_argument('--data_path', type=str, default='/Users/h1de0us/uni/mer-eeg-analysis/data/deap_filtered', help='Path to the DEAP dataset')
args = parser.parse_args()

sub_folders = sorted([folder for folder in os.listdir(args.data_path) if folder.startswith('sub')])

for sub_folder in tqdm(sub_folders):
    print(f'Processing {sub_folder}')
    sub_path = os.path.join(args.data_path, sub_folder)
    sub_path = os.path.join(sub_path, 'eeg')
    eeg_files = [file for file in os.listdir(sub_path) if file.endswith('.edf')]
    for eeg_file in eeg_files:
        eeg_path = os.path.join(sub_path, eeg_file)
        eeg_graph = EEGConnectivityGraph(eeg_path)
        con = eeg_graph.compute_connectivity(method=args.method, fmin=bands_min, fmax=bands_max)
        con_data = con.get_data(output="dense")

        # save the connectivity data
        con_path = eeg_path.replace('.edf', '_{}.npy'.format(args.method))
        np.save(con_path, con_data)
