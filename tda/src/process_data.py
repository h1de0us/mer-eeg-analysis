import numpy as np
import gudhi
import os
import argparse
import pandas as pd
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def process_connectivity_matrix(mtx):
    df = pd.DataFrame(columns=['order', 'birth', 'death', 'band'])
    for band in range(mtx.shape[-1]):
        # mtx is a connectivity matrix, but we need distance matrix
        distance_matrix = 1 - np.abs(mtx[:, :, band])
        rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=1.0)
        # calculate only zero- and first-order homologies
        simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)
        persistence = simplex_tree.persistence()
        for entry in persistence[:-1]: # last entry is (0, inf) for each band for each participant
            order, (birth, death) = entry
            df = pd.concat([df, pd.DataFrame([{'order': order, 'birth': birth, 'death': death, 'band': band}])])
    return df


parser = argparse.ArgumentParser(description='Compute barcodes for connectivity matrices from DEAP dataset')
parser.add_argument('--method', type=str, default='coh', help='Used connectivity method')
parser.add_argument('--data_path', type=str, default='/Users/h1de0us/uni/mer-eeg-analysis/data/deap_filtered', help='Path to the connectivity matrices')
parser.add_argument('--duration', type=float, default=3.0, help='Epoch duration in seconds')
parser.add_argument('--n_trials', type=int, default=40, help="Number of trials for each participant, for DEAP use 40")
parser.add_argument('--output', type=str, default='bars.csv', help='Output file')
args = parser.parse_args()

df = pd.DataFrame(columns=['order', 'birth', 'death', 'participant', 'trial'])
participants = [file[:-4] for file in os.listdir(args.data_path) if file.endswith('.dat')]
for participant in tqdm(participants):
    for trial in range(args.n_trials):
        duration_str = str(args.duration)
        prefix = f'{participant}_{args.method}_{duration_str}_trial_{trial}.npy'
        path = os.path.join(args.data_path, prefix)

        mtx = np.load(path)
        current_df = process_connectivity_matrix(mtx)
        current_df['participant'] = participant
        current_df['trial'] = trial
        df = pd.concat([df, current_df])

# statistics of the barcodes
df['bars'] = df['death'] - df['birth']
df['mean_bar'] = df.groupby(['participant', 'order', 'band'])['bars'].transform('mean')
df['std_bar'] = df.groupby(['participant', 'order', 'band'])['bars'].transform('std')
df['max_bar'] = df.groupby(['participant', 'order', 'band'])['bars'].transform('max')
df['min_bar'] = df.groupby(['participant', 'order', 'band'])['bars'].transform('min')
df['mean_birth'] = df.groupby(['participant', 'order', 'band'])['birth'].transform('mean')
df['std_birth'] = df.groupby(['participant', 'order', 'band'])['birth'].transform('std')
df['mean_death'] = df.groupby(['participant', 'order', 'band'])['death'].transform('mean')
df['std_death'] = df.groupby(['participant', 'order', 'band'])['death'].transform('std')
df.to_csv(args.output, index=False)