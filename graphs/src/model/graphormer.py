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
            nn.GELU(),
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
                 device='cpu',
                 batch_first=True,
                 norm_first=True):
        super().__init__()
        self.n_nodes = n_nodes
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.device = device
        # default node embeddings 
        self.node_encoder = nn.Embedding(n_nodes, embed_dim)

        # encoding for every node pair: (v_i, v_j) => n_heads numbers (for each attention head)
        self.edge_encoder = nn.Embedding(n_nodes ** 2, n_heads, padding_idx=0)

        # centrality encoding assigns each node two real-valued embeddings according to
        # its in-degree and out-degree, respectively (via paper)
        # maximum possible degree is n_nodes - 1
        self.centrality_encoding = nn.Embedding(n_nodes, embed_dim)

        # \varphi(v_i, v_j) == -1 if there is no path between v_i and v_j
        # project degrees: b_{\vaprhi} is different for each head, thus (n_nodes => n_heads)
        self.spatial_encoding = nn.Embedding(n_nodes, n_heads, padding_idx=-1)

        # we apply the layer normalization (LN) before the multi-head self-attention (MHA) 
        # and the feed-forward blocks (FFN) instead of after (via paper)
        self.encoder = nn.ModuleList(GraphormerEncoderLayer(
            n_nodes,
            n_heads,
            embed_dim,
            dim_feedforward,
        ) for _ in range(n_layers))

        self.regressor = nn.Linear(embed_dim, 4) # valence, arousal, dominance, liking

    def forward(self, batch):
        outputs = []
        attentions = []
        for adj_matrix in batch['matrices']:
            output, attention = self._forward(adj_matrix.cpu().numpy())
            outputs.append(output)
            attentions.append(attention)
        outputs = torch.stack(outputs)
        attentions = torch.stack(attentions)
        return outputs, attentions 

    def _forward(self, adj_matrix):
        # creating dgl-graph from connectivity matrix

        nx_graph = nx.from_numpy_array(adj_matrix)
        nx_graph = nx_graph.to_directed()
        g = dgl.from_networkx(nx_graph, edge_attrs=['weight'], device=self.device)

        degrees = g.in_degrees()

        # graphs are undirected, so in-degree and out-degree are the same
        centrality_encoding = self.centrality_encoding(degrees) # (n_nodes) -> (n_nodes, dim_feedforward)

        # spatial encoding: computing SPD
        shortest_path_distances = dgl.shortest_dist(g) # -1 if no path
        spatial_encoding = self.spatial_encoding(shortest_path_distances)

        # Edge Encoding
        edge_features = g.edata['weight']
        edge_encoding = self.compute_edge_encoding(g, edge_features) # (n_nodes, n_nodes, n_heads)
        edge_encoding = edge_encoding.permute(2, 0, 1) # (n_heads, n_nodes, n_nodes)

        node_embeddings = self.node_encoder(torch.arange(self.n_nodes, device=self.device)) # (n_nodes, embed_dim)

        # combine encodings
        input_embeddings = node_embeddings + centrality_encoding

        # print all devices
        # print("input_embeddings.device", input_embeddings.device)
        # print("spatial_encoding.device", spatial_encoding.device)
        # print("edge_encoding.device", edge_encoding.device)


        # Transformer Encoder
        attention_total = []
        for layer in self.encoder:
            # print("input_embeddings.shape", input_embeddings.shape)
            input_embeddings, attention = layer(input_embeddings, spatial_encoding, edge_encoding)
            attention_total.append(attention)

        # output.shape == (n_nodes, embed_dim)
        # Decode Head
        output = torch.squeeze(self.regressor(input_embeddings))
        # output.shape = (n_nodes, 4)
        output = torch.mean(output, dim=0) 

        attention_total = torch.stack(attention_total)
        return output, attention_total
    
    def compute_edge_encoding(self, graph, edge_features):
        '''
        Compute edge encoding for each edge in the graph
        params:
            graph: dgl graph
            edge_features: edge features # (n_edges,)
        '''
        # edge encoding is a bias term for each attention head
        edge_encoding = torch.zeros(self.n_nodes, self.n_nodes, self.n_heads)
        for i in range(self.n_nodes):
            for j in range(1, self.n_nodes):
                # calculating indices of the edges that lie on the path between v_i and v_j
                _, path = dgl.shortest_dist(graph, i, return_paths=True)
                path = path[j] # path from i to j
                # path is a sequence of nodes, len(path) == max_path 
                # -1 is a padding value
                path = path[path >= 0] # remove padding

                edge_embeds = self.edge_encoder(path) # (n_spd, n_heads)
                spd_features = edge_features[path] # (n_spd)
                edge_encoding[i, j] = torch.mean(edge_embeds * spd_features.unsqueeze(-1), dim=0)
                edge_encoding[j, i] = edge_encoding[i, j]
        # fill diag with zeros
        for i in range(self.n_nodes):
            edge_encoding[i, i] = 0
        return edge_encoding
