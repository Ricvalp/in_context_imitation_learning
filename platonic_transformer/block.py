import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Union, Callable
import torch.nn.functional as F

# Assumes these modules are in your project structure
from .conv import PlatonicConv
from .linear import PlatonicLinear
from .groups import PLATONIC_GROUPS


class PlatonicBlock(nn.Module):
    """
    A group-equivariant Transformer-style block using Platonic symmetries.

    This block replaces standard self-attention and feed-forward networks with
    equivariant counterparts: PlatonicConv and PlatonicLinear. It operates on
    flattened group feature maps of shape [..., G*C] but internally handles the
    group structure correctly, especially for Layer Normalization.

    Args:
        d_model (int): The total model dimension (G * C_model). Must be divisible
                       by group size and (group_size * nhead).
        nhead (int): The number of base attention heads for the interaction layer.
        dim_feedforward (int): The total dimension of the feed-forward network's
                               hidden layer (G * C_ffn). Must be divisible by G.
        solid_name (str): The name of the Platonic solid ('tetrahedron', 'octahedron',
                          'icosahedron') to define the symmetry group.
        dropout (float): Dropout rate.
        activation (Callable): The activation function for the FFN.
        layer_norm_eps (float): Epsilon for LayerNorm.
        norm_first (bool): If True, applies pre-normalization; otherwise, post-normalization.
        spatial_dims (int): The number of spatial dimensions for positions.
        **kwargs: Additional keyword arguments for the PlatonicConv layer
                  (e.g., freq_sigma, learned_freqs, avg_pool).
    """
    def __init__(self,
                 d_model: int,
                 nhead: int,
                 dim_feedforward: int,
                 solid_name: str,
                 dropout: float = 0.1,
                 activation: Callable[[Tensor], Tensor] = F.gelu,
                 layer_norm_eps: float = 1e-5,
                 norm_first: bool = True,
                 spatial_dims: int = 3,
                 freq_sigma: float = 1.0,
                 learned_freqs: bool = True,
                 mean_aggregation: bool = False,
                 attention: bool = False,
                 attention_type: str = 'equivariant',
                 **kwargs) -> None:
        super().__init__()

        # --- Group and Dimension Setup ---
        self.group = PLATONIC_GROUPS[solid_name.lower()]
        self.num_G = self.group.G
        self.norm_first = norm_first

        # Validate total dimensions against group size and heads
        if d_model % self.num_G != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by group size ({self.num_G}).")
        if dim_feedforward % self.num_G != 0:
            raise ValueError(f"dim_feedforward ({dim_feedforward}) must be divisible by group size ({self.num_G}).")
        if d_model % (nhead) != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_head = {nhead}.")
        
        # Calculate per-group-element dimensions
        self.dim_per_g = d_model // self.num_G
    
        # --- Equivariant Sub-Modules ---
        self.interaction = PlatonicConv(
            in_channels=d_model,
            out_channels=d_model,
            embed_dim=d_model,
            num_heads=nhead,
            solid_name=solid_name,
            spatial_dims=spatial_dims,
            freq_sigma=freq_sigma,
            learned_freqs=learned_freqs,
            mean_aggregation=mean_aggregation,
            attention=attention,
            attention_type=attention_type,
            **kwargs
        )

        # Equivariant Feed-Forward Network
        self.linear1 = PlatonicLinear(d_model, dim_feedforward, solid=solid_name)
        self.linear2 = PlatonicLinear(dim_feedforward, d_model, solid=solid_name)

        # Layer Normalization (acts on the per-group-element channel dimension)
        self.norm1 = nn.LayerNorm(self.dim_per_g, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(self.dim_per_g, eps=layer_norm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn_dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(
        self,
        x: Tensor,
        pos: Tensor,
        batch: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        avg_num_nodes = 1.0
    ) -> Tensor:
        """
        Args:
            x (Tensor): Input feature tensor of shape [..., G*C].
            pos (Tensor): Position tensor of shape [..., D_spatial].
            batch (Optional[Tensor]): For graph mode. Batch index for each element.
            mask (Optional[Tensor]): For dense mode. Boolean mask.

        Returns:
            Tensor: Output feature tensor of the same shape [..., G*C].
        """
        if self.norm_first:
            # 1. Interaction Block (Pre-Norm)
            x_norm = self._normalize(x, self.norm1)
            interaction_out = self._interaction_block(x_norm, pos, batch, mask, avg_num_nodes)
            x = x + interaction_out

            # 2. Feed-Forward Block (Pre-Norm)
            x_norm = self._normalize(x, self.norm2)
            ff_output = self._ff_block(x_norm)
            x = x + ff_output
        else:
            # 1. Interaction Block (Post-Norm)
            interaction_out = self._interaction_block(x, pos, batch, mask, avg_num_nodes)
            x = x + interaction_out
            x = self._normalize(x, self.norm1)

            # 2. Feed-Forward Block (Post-Norm)
            ff_output = self._ff_block(x)
            x = x + ff_output
            x = self._normalize(x, self.norm2)
        
        return x

    def _normalize(self, x: Tensor, norm_layer: nn.LayerNorm) -> Tensor:
        """Helper to apply LayerNorm on the per-group-element dimension."""
        leading_dims = x.shape[:-1]
        # Reshape to expose group axis: [..., G*C] -> [..., G, C]
        x_reshaped = x.view(*leading_dims, self.num_G, self.dim_per_g)
        # Apply normalization
        normed_reshaped = norm_layer(x_reshaped)
        # Reshape back to original convention
        return normed_reshaped.view(*leading_dims, -1)

    def _interaction_block(
        self, x: Tensor, pos: Tensor, batch: Optional[Tensor], mask: Optional[Tensor], avg_num_nodes = 1.0
    ) -> Tensor:
        """Wrapper for the PlatonicConv layer."""
        # x is already normalized and has the correct shape [..., G*C]
        interaction_output = self.interaction(x, pos, batch=batch, mask=mask, avg_num_nodes=avg_num_nodes)
        return self.dropout1(interaction_output)

    def _ff_block(self, x: Tensor) -> Tensor:
        """Equivariant Feed-Forward Network block."""
        # x has shape [..., G*C]
        ff_output = self.linear2(self.ffn_dropout(self.activation(self.linear1(x))))
        return self.dropout2(ff_output)