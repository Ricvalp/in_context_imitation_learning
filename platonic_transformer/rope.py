import torch
import torch.nn as nn
from torch import Tensor
# This assumes the PLATONIC_GROUPS dictionary from the previous problem is available.
# You might need to adjust the import path based on your project structure.
from .groups import PLATONIC_GROUPS


class PlatonicRoPE(nn.Module):
    """
    Group-Equivariant Rotary Position Embedding (RoPE).

    This module extends Rotary Position Embeddings to be equivariant to the discrete
    rotational symmetry groups of the Platonic solids (T, O, I). It operates on
    feature tensors where the head and group dimensions have been merged for seamless
    integration into standard Multi-Head Attention blocks.

    The core principle is to apply the group action to the spatial coordinates `pos`
    before computing the rotary embeddings. For an input with `H` base heads and a
    group of size `G`, this module effectively has `G*H` heads, where each base
    head's features are rotated according to a different group element.

    Args:
        embed_dim (int): The total embedding dimension, must be divisible by num_heads * num_G * 2.
        num_heads (int): The number of base attention heads.
        solid_name (str): The name of the Platonic solid ('tetrahedron', 'octahedron',
                          'icosahedron') to define the symmetry group.
        spatial_dims (int): The number of spatial dimensions for positions (e.g., 3 for x, y, z).
        freq_sigma (float): Standard deviation for sampling initial random frequencies.
        learned_freqs (bool): If True, frequencies are learnable parameters.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        solid_name: str,
        head_dim: int,
        spatial_dims: int = 3,
        freq_sigma: float = 1.0,
        learned_freqs: bool = False,
    ):
        super().__init__()

        # --- Group Setup ---
        try:
            self.group = PLATONIC_GROUPS[solid_name.lower()]
        except KeyError:
            raise ValueError(f"Unknown solid '{solid_name}'. Available options are {list(PLATONIC_GROUPS.keys())}")
        self.num_G = self.group.G
        self.register_buffer('group_elements', self.group.elements.to(torch.float32))

        # --- Dimension Setup ---
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        if self.embed_dim % self.num_G != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by group size ({self.num_G}).")
        self.embed_dim_g = self.embed_dim // self.num_G
        if self.embed_dim_g % self.num_heads != 0:
            raise ValueError(f"embed_dim_g ({self.embed_dim_g}) must be divisible by num_heads ({self.num_heads}).")
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be divisible by 2 for RoPE.")
        self.num_pairs = self.head_dim // 2
        self.spatial_dims = spatial_dims

        # --- Frequency Initialization ---
        # Frequencies are defined per *base* head. The group action is applied to positions.
        freqs = torch.randn(self.num_heads, self.num_pairs, self.spatial_dims) * freq_sigma
        if learned_freqs:
            self.register_parameter("freqs", nn.Parameter(freqs))
        else:
            self.register_buffer("freqs", freqs)

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """
        Apply group-equivariant rotary embeddings to the input tensor.

        Args:
            x (Tensor): Input tensor of shape (..., G, H, D_h). Typically queries or keys.
                        The G*H dimension represents the merged group and head axes.
            pos (Tensor): Position tensor of shape (..., spatial_dims). The leading
                          dimensions '...' must be broadcastable to the input tensor x.

        Returns:
            Tensor: The rotated input tensor `x_rotated` of the same shape (..., G, H, D_h).
        """
        # 1. --- Unpack and Validate Shapes ---
        *leading_dims, G, H, D_h = x.shape
        if G != self.num_G or H != self.num_heads or D_h != self.head_dim:
            raise ValueError(f"Input shape {x.shape} does not match expected shape (..., {self.num_G}, {self.num_heads}, {self.head_dim}).")
        
        # 2. --- Compute Rotated frequencies ---
        freqs_rotated = torch.einsum('ged, hfe -> ghfd', self.group_elements, self.freqs)

        # Compute rotation angles for each rotated position and each base head.
        angles = torch.einsum('...d, ghfd -> ...ghf', pos, freqs_rotated)
        cos_angles = torch.cos(angles)
        sin_angles = torch.sin(angles)

        # 3. --- Apply Rotations to Input Features ---
        # Reshape input features to expose pairs for 2D rotation.
        # Shape: [..., G, H, F, 2]
        x_reshaped = x.view(*leading_dims, self.num_G, self.num_heads, self.num_pairs, 2)
        x0, x1 = x_reshaped.unbind(dim=-1)  # Both have shape [..., G, H, F]

        # Apply the 2D rotation to each pair.
        # The cos/sin angles broadcast across the leading dimensions.
        x_rotated_0 = x0 * cos_angles - x1 * sin_angles
        x_rotated_1 = x0 * sin_angles + x1 * cos_angles
        
        # Stack the rotated pairs back together.
        # Shape: [..., G, H, F, 2]
        x_rotated_pairs = torch.stack([x_rotated_0, x_rotated_1], dim=-1)

        # 4. --- Reshape to Final Output ---
        # Reshape back to the merged (G*H, D_h) convention.
        # Final shape: (..., G, H, D_h)
        x_out = x_rotated_pairs.view(*leading_dims, self.num_G, self.num_heads, self.head_dim)
        
        return x_out