import torch
import numpy as np
from torch import nn

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        """
        Args:
            embed_dim: dimensionality of embedding (total)
            num_heads: number of heads (must divide embed_dim)
        """
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    # original implementation uses this initialization
    def _reset_parameters(self):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)

    def forward(self, 
                x, 
                spatial_embeddings=None,
                edge_embeddings=None,
                return_attention=True):
        """
        Args:
            x: torch.Tensor (B, L, D)
            spatial_embeddings: torch.Tensor (B, num_heads, L, L)
            edge_embeddings: torch.Tensor (B, num_heads, L, L)
            return_attention: If specified, returns attention along with outputs
            L equals to n_nodes, D equals to dim_feedforward
        Returns:
            outputs: torch.Tensor (B, L, D)
            attention: Optional[torch.Tensor] (B, num_heads, L, L)

        B is batch size, L is the length of sequence, D is the embedding dimension
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0) # add batch dimension for batch size 1
        B, L, D = x.shape

        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        q = q.permute(0, 2, 1, 3) # (B, num_heads, L, head_dim)

        k = k.reshape(B, L, self.num_heads, self.head_dim)
        k = k.permute(0, 2, 1, 3) # (B, num_heads, L, head_dim)

        v = v.reshape(B, L, self.num_heads, self.head_dim)
        v = v.permute(0, 2, 1, 3) # (B, num_heads, L, head_dim)
        
        d = np.sqrt(q.shape[-1])

        softmax = nn.Softmax(dim=-1)
        softmax_input = torch.matmul(q / d, k.transpose(-2, -1)) # (B, num_heads, L, L)

        # print(spatial_embeddings.shape, edge_embeddings.shape, softmax_input.shape, v.shape)

        # spatial_embeddings.shape is (num_heads, L, L)
        if spatial_embeddings is not None:
            spatial_embeddings = spatial_embeddings.to(x.device)
            # permuting to (L, L, num_heads)
            spatial_embeddings = spatial_embeddings.permute(2, 0, 1)
            spatial_embeddings = spatial_embeddings.unsqueeze(0).expand(B, -1, -1, -1)
            # print("shapes for spatial", softmax_input.shape, spatial_embeddings.shape)
            softmax_input = softmax_input + spatial_embeddings

        # edge_embeddings.shape is (num_heads, L, L)
        if edge_embeddings is not None:
            edge_embeddings = edge_embeddings.to(x.device)
            edge_embeddings = edge_embeddings.unsqueeze(0)
            softmax_input = softmax_input + edge_embeddings

        attention = softmax(softmax_input)
        outputs = attention @ v
#         print(outputs.shape, attention.shape) # (B, num_heads, L, head_dim), (B, num_heads, L, L)
        outputs = outputs.permute(0, 2, 1, 3) # (B, L, num_heads, head_dim)
        outputs = outputs.reshape(B, L, D)
        outputs = self.o_proj(outputs) 

        if return_attention:
            return outputs, attention
        else:
            return outputs