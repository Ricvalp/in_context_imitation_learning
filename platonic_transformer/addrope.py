import torch
import torch.nn as nn
from torch import Tensor
from .groups import PLATONIC_GROUPS


class AddRoPE(nn.Module):
    """
    Group-Equivariant Additive Rotary Position Embedding (AddRoPE).

    This module extends Additive Rotary Position Embeddings to be equivariant to the discrete
    rotational symmetry groups of the Platonic solids (T, O, I).
    
    Unlike standard RoPE which applies rotations multiplicatively, AddRoPE adds the
    rotational positional embeddings to the input features with learnable scaling.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        solid_name: str,
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
        if self.embed_dim % self.num_G != 0:
            raise ValueError(f"embed_dim ({self.embed_dim}) must be divisible by group size ({self.num_G}).")
        self.embed_dim_g = self.embed_dim // self.num_G
        if self.embed_dim_g % self.num_heads != 0:
            raise ValueError(f"embed_dim_g ({self.embed_dim_g}) must be divisible by num_heads ({self.num_heads}).")
        self.head_dim = self.embed_dim_g // self.num_heads
        if self.head_dim % 2 != 0:
            raise ValueError(f"head_dim ({self.head_dim}) must be divisible by 2 for AddRoPE.")
        self.num_pairs = self.head_dim // 2
        self.spatial_dims = spatial_dims

        # --- Frequency Initialization ---
        freqs = torch.randn(self.num_heads, self.num_pairs, self.spatial_dims) * freq_sigma
        if learned_freqs:
            self.register_parameter("freqs", nn.Parameter(freqs))
        else:
            self.register_buffer("freqs", freqs)
            
        # --- AddRoPE-specific parameters ---
        # Initialize weights for scaling the positional embeddings (per head and frequency pair)
        weights = torch.ones(self.num_heads, self.num_pairs)
        self.register_parameter("weights", nn.Parameter(weights * 0.1))  # Start with small weight
        
        # Initialize phase offsets (per head and frequency pair)
        offsets = torch.zeros(self.num_heads, self.num_pairs)
        self.register_parameter("offsets", nn.Parameter(offsets))

    def forward(self, x: Tensor, pos: Tensor) -> Tensor:
        """
        Apply group-equivariant additive rotary embeddings to the input tensor.
        
        Following the same 2D rotation structure as RoPE but using addition:
        Instead of: x_rot = R(θ) @ x
        We compute: x_add = x + weight * R(θ + offset) @ [1, 0]
        """
        # 1. --- Unpack and Validate Shapes ---
        *leading_dims, G, H, D_h = x.shape
        if G != self.num_G or H != self.num_heads or D_h != self.head_dim:
            raise ValueError(f"Input shape {x.shape} does not match expected shape (..., {self.num_G}, {self.num_heads}, {self.head_dim}).")
        
        # 2. --- Compute Rotated frequencies ---
        freqs_rotated = torch.einsum('ged, hfe -> ghfd', self.group_elements, self.freqs)

        # Compute rotation angles for each rotated position and each base head
        angles = torch.einsum('...d, ghfd -> ...ghf', pos, freqs_rotated)
        
        # Add learnable offsets to angles
        # offsets shape: [H, F] -> expand to broadcast with angles [..., G, H, F]
        offsets_expanded = self.offsets.view(1, *(1,) * (len(leading_dims)), 1, H, self.num_pairs)
        angles_with_offsets = angles + offsets_expanded
        
        cos_angles = torch.cos(angles_with_offsets)
        sin_angles = torch.sin(angles_with_offsets)

        # 3. --- Apply AddRoPE to Input Features ---
        # Reshape input features to expose pairs for 2D operations
        # Shape: [..., G, H, F, 2]
        x_reshaped = x.view(*leading_dims, self.num_G, self.num_heads, self.num_pairs, 2)
        x0, x1 = x_reshaped.unbind(dim=-1)  # Both have shape [..., G, H, F]

        # Create positional embedding using 2D rotation matrix applied to unit vector [1, 0]
        # This is equivalent to torch.polar(1, angle) but using explicit rotation matrix
        # R(θ) @ [1, 0] = [cos(θ), sin(θ)]
        pos_emb_0 = cos_angles  # cos component
        pos_emb_1 = sin_angles  # sin component
        
        # Scale by learnable weights
        weights_expanded = self.weights.view(1, *(1,) * (len(leading_dims)), 1, H, self.num_pairs)
        pos_emb_0_weighted = weights_expanded * pos_emb_0
        pos_emb_1_weighted = weights_expanded * pos_emb_1

        # AddRoPE: Add the weighted positional embeddings to input features
        x_with_pos_0 = x0 + pos_emb_0_weighted
        x_with_pos_1 = x1 + pos_emb_1_weighted
        
        # Stack the modified pairs back together
        # Shape: [..., G, H, F, 2]
        x_with_pos_pairs = torch.stack([x_with_pos_0, x_with_pos_1], dim=-1)

        # 4. --- Reshape to Final Output ---
        # Reshape back to the merged format
        # Final shape: (..., G, H, D_h)
        x_out = x_with_pos_pairs.view(*leading_dims, self.num_G, self.num_heads, self.head_dim)
        
        return x_out