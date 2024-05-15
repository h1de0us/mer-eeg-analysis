import math
import dgl
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.mha import MultiheadAttention

class GraphormerEncoderLayer(nn.Module):
    def __init__(self,
                 n_nodes, 
                 n_heads, 
                 embed_dim,
                 dim_feedforward):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiheadAttention(embed_dim, n_heads)

        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GeLU(),
            nn.Linear(dim_feedforward, embed_dim)
        )

    def forward(self, 
                x,
                spatial_embeddings,
                edge_embeddings):

        x_prime = self.layer_norm1(x)
        x_prime, attention = self.attention(
            x_prime,
            spatial_embeddings,
            edge_embeddings,
        )
        x = x + x_prime

        x_prime = self.layer_norm2(x)
        x_prime = self.ffn(x_prime)
        x = x + x_prime

        return x, attention

        

class GraphormerModel(nn.Module):
    def __init__(self, 
                 n_nodes, 
                 n_layers, 
                 n_heads, 
                 embed_dim,
                 dim_feedforward, 
                 dropout=0.1,
                 batch_first=True,
                 norm_first=True):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        # default node embeddings 
        self.node_encoder = nn.Embedding(n_nodes, embed_dim, bias=False, padding_idx=0)

        # encoding for every node pair: (v_i, v_j) => n_heads numbers (for each attention head)
        self.edge_encoder = nn.Embedding(n_nodes ** 2, n_heads, bias=False, padding_idx=0)

        # centrality encoding assigns each node two real-valued embeddings according to
        # its in-degree and out-degree, respectively (via paper)
        # maximum possible degree is n_nodes - 1
        self.centrality_encoding = nn.Embedding(n_nodes, dim_feedforward)

        # \varphi(v_i, v_j) == -1 if there is no path between v_i and v_j
        # project degrees: b_{\vaprhi} is different for each head, thus (n_nodes => n_heads)
        self.spatial_encoding = nn.Embedding(n_nodes, n_heads, padding_idx=-1)
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        # we apply the layer normalization (LN) before the multi-head self-attention (MHA) 
        # and the feed-forward blocks (FFN) instead of after (via paper)
        self.encoder = nn.ModuleList(GraphormerEncoderLayer(
            n_nodes,
            n_heads,
            embed_dim,
            dim_feedforward,
        ) for _ in range(n_layers))

        self.regressor = nn.Linear(embed_dim, 2) # valence, arousal

    def forward(self, batch):
        outputs = []
        attentions = []
        for adj_matrix in batch:
            output, attention = self._forward(adj_matrix)
            outputs.append(output)
            attentions.append(attention)
        outputs = torch.stack(outputs)
        attentions = torch.stack(attentions)
        return outputs, attentions 

    def _forward(self, adj_matrix):
        # creating dgl-graph from connectivity matrix
        # g = dgl.from_scipy(adj_matrix.to_dense())
        # TODO: скопировать из report
        nx_graph = nx.from_numpy_array(adj_matrix)
        nx_graph = nx_graph.to_directed()
        g = dgl.from_networkx(nx_graph, edge_attrs=['weight'])

        degrees = g.in_degrees()

        # graphs are undirected, so in-degree and out-degree are the same
        centrality_encoding = self.centrality_encoding(degrees) # (n_nodes) -> (n_nodes, dim_feedforward)

        # spatial encoding: computing SPD
        shortest_path_distances = dgl.shortest_dist(g) # -1 if no path
        spatial_encoding = self.spatial_encoding(shortest_path_distances)

        # Edge Encoding
        edge_features = g.edata['weight']
        edge_encoding = self.compute_edge_encoding(edge_features, g) # (n_nodes, n_nodes, n_heads)
        edge_encoding = edge_encoding.permute(2, 0, 1) # (n_heads, n_nodes, n_nodes)

        node_embeddings = self.node_encoder(torch.arange(self.n_nodes)) # (n_nodes, embed_dim)

        # combine encodings
        input_embeddings = self.pos_encoder(node_embeddings) + centrality_encoding

        # Transformer Encoder
        output, attention = self.encoder(input_embeddings, spatial_encoding, edge_encoding)

        # Decode Head
        output = self.regressor(output)

        return output, attention
    
    def compute_edge_encoding(self, graph, edge_features):
        '''
        Compute edge encoding for each edge in the graph
        params:
            edge_features: edge features # (n_edges,)
            adj_matrix: adjacency matrix of the graph # (n_nodes, n_nodes)
            n_nodes: number of nodes in the graph
            dim_feedforward: dimension of the feedforward layer
        '''
        # edge encoding is a bias term for each attention head
        edge_encoding = torch.zeros(self.n_nodes, self.n_nodes, self.n_heads)
        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                # calculating indices of the edges that lie on the path between v_i and v_j
                _, path = dgl.shortest_dist(graph, i, return_paths=True)
                path = path[j] # path from i to j
                # path is a sequence of nodes, len(path) == max_path 
                # -1 is a padding value
                path = path[path >= 0] # remove padding

                edge_embeds = self.edge_encoder(path) # (n_spd, n_heads)
                spd_features = edge_features[path] # (n_spd)
                edge_encoding[i, j] = torch.mean(edge_embeds * spd_features.unsqueeze(-1), dim=0)
        return edge_encoding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)