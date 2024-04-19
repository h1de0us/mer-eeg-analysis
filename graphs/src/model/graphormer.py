import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from .modules import GraphormerGraphEncoder

class GraphormerModel(nn.Module):
    def __init__(self,
                num_atoms,
                num_in_degree,
                num_out_degree,
                num_edges,
                num_spatial,
                num_edge_dis,
                edge_type,
                multi_hop_max_dist,
                num_encoder_layers,
                embedding_dim,
                ffn_embedding_dim,
                num_attention_heads,
                dropout,
                attention_dropout,
                activation_dropout,
                encoder_normalize_before,
                pre_layernorm,
                apply_graphormer_init,
                activation_fn,
                max_nodes,
                encoder_embed_dim
                 ):
        self.encoder = GraphormerEncoder(
            num_atoms,
            num_in_degree,
            num_out_degree,
            num_edges,
            num_spatial,
            num_edge_dis,
            edge_type,
            multi_hop_max_dist,
            num_encoder_layers,
            embedding_dim,
            ffn_embedding_dim,
            num_attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            encoder_normalize_before,
            pre_layernorm,
            apply_graphormer_init,
            activation_fn,
            max_nodes,
            encoder_embed_dim,
        )


    def max_nodes(self):
        return self.encoder.max_nodes
    
    def forward(self, x):
        return self.encoder(x)


class GraphormerEncoder(torch.nn.Module):
    def __init__(self,
                num_atoms,
                num_in_degree,
                num_out_degree,
                num_edges,
                num_spatial,
                num_edge_dis,
                edge_type,
                multi_hop_max_dist,
                num_encoder_layers,
                embedding_dim,
                ffn_embedding_dim,
                num_attention_heads,
                dropout,
                attention_dropout,
                activation_dropout,
                encoder_normalize_before,
                pre_layernorm,
                apply_graphormer_init,
                activation_fn,
                max_nodes,
                encoder_embed_dim
                 ):
        super().__init__()
        self.max_nodes = max_nodes

        self.graph_encoder = GraphormerGraphEncoder(
            num_atoms,
            num_in_degree,
            num_out_degree,
            num_edges,
            num_spatial,
            num_edge_dis,
            edge_type,
            multi_hop_max_dist,
            num_encoder_layers,
            embedding_dim,
            ffn_embedding_dim,
            num_attention_heads,
            dropout,
            attention_dropout,
            activation_dropout,
            encoder_normalize_before,
            pre_layernorm,
            apply_graphormer_init,
            activation_fn,
        )

        self.embed_out = None
        self.lm_output_learned_bias = None

        self.masked_lm_pooler = nn.Linear(
            encoder_embed_dim, encoder_embed_dim
        )

        self.lm_head_transform_weight = nn.Linear(
            encoder_embed_dim, encoder_embed_dim
        )
        self.activation_fn = torch.nn.GELU() if activation_fn == "gelu" else torch.nn.ReLU()
        self.layer_norm = LayerNorm(encoder_embed_dim)

        self.lm_output_learned_bias = None

    def reset_output_layer_parameters(self):
        self.lm_output_learned_bias = nn.Parameter(torch.zeros(1))
        if self.embed_out is not None:
            self.embed_out.reset_parameters()

    def forward(self, batched_data, perturb=None, masked_tokens=None, **unused):
        inner_states, graph_rep = self.graph_encoder(
            batched_data,
            perturb=perturb,
        )

        x = inner_states[-1].transpose(0, 1)

        # project masked tokens only
        if masked_tokens is not None:
            raise NotImplementedError

        x = self.layer_norm(self.activation_fn(self.lm_head_transform_weight(x)))

        # project back to size of vocabulary
        if self.embed_out is not None:
            x = self.embed_out(x)
        if self.lm_output_learned_bias is not None:
            x = x + self.lm_output_learned_bias

        return x

    def max_nodes(self):
        """Maximum output length supported by the encoder."""
        return self.max_nodes