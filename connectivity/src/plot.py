
# plot the connectivity matrix for all bands
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

bands = {0: 'delta', 1: 'theta', 2: 'alpha', 3: 'beta', 4: 'gamma'}


def plot_connectivity_maps(con_data, n_connections=20):
    # con_data: (n_channels, n_channels, n_freqs)
    fig, axs = plt.subplots(1, con_data.shape[-1], figsize=(20, 5))
    for band in range(con_data.shape[-1]):
        cur = con_data[:, :, band]

        threshold = np.sort(cur, axis=None)[-n_connections]
        cur = cur > threshold

        axs[band].set_title(f'Band {bands[band]}')
        axs[band].imshow(cur, cmap='viridis')

def plot_graphs(con_data, n_connections=20, channel_names=None):
    fig, axs = plt.subplots(1, con_data.shape[-1], figsize=(20, 5))
    for band in range(con_data.shape[-1]):
        cur = con_data[:, :, band]

        threshold = np.sort(cur, axis=None)[-n_connections]
        cur = cur > threshold

        G = nx.from_numpy_array(cur)
        pos = nx.circular_layout(G)  # Layout nodes in a circle
        axs[band].set_title(f'Band {bands[band]}')
        nx.draw(G, pos, ax=axs[band], with_labels=True, labels=channel_names, node_color='skyblue', node_size=800, font_size=10)
