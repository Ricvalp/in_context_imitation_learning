import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Tuple

from .block import DensePlatonicBlock
from .groups import PLATONIC_GROUPS
from .linear import PlatonicLinear
from .io import lift, readout_scalars, readout_vectors
from .ape import PlatonicAPE as APE
from .gen_utils import TimestepEmbedder, LabelEmbedder


class DensePlatonicTransformer(nn.Module):
    """Transformer for Platonic-symmetric token sets in dense batches."""

    def __init__(
        self,
        input_dim: int,
        input_dim_vec: int,
        hidden_dim: int,
        output_dim: int,
        output_dim_vec: int,
        nhead: int,
        num_layers: int,
        solid_name: str,
        spatial_dim: int = 3,
        scalar_task_level: str = "graph",
        vector_task_level: str = "node",
        ffn_readout: bool = True,
        mean_aggregation: bool = False,
        dropout: float = 0.1,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: Optional[float] = None,
        attention: bool = False,
        ffn_dim_factor: int = 4,
        rope_sigma: float = 1.0,
        ape_sigma: Optional[float] = None,
        learned_freqs: bool = True,
        freq_init: str = "random",
        use_key: bool = False,
        time_conditioning: bool = False,
        class_conditioning: bool = False,
        use_cls_token: bool = False,
    ) -> None:
        super().__init__()

        if scalar_task_level not in ["node", "graph"]:
            raise ValueError("scalar_task_level must be 'node' or 'graph'.")
        if vector_task_level not in ["node", "graph"]:
            raise ValueError("vector_task_level must be 'node' or 'graph'.")

        # Cache structural metadata shared across the stack.
        self.group = PLATONIC_GROUPS[solid_name.lower()]
        self.num_G = self.group.G
        self.hidden_dim = hidden_dim
        self.scalar_task_level = scalar_task_level
        self.vector_task_level = vector_task_level
        self.output_dim = output_dim
        self.output_dim_vec = output_dim_vec
        self.mean_aggregation = mean_aggregation
        self.use_cls_token = use_cls_token

        # Conditioning helpers.
        self.time_conditioning = time_conditioning
        self.class_conditioning = class_conditioning

        if time_conditioning:
            self.time_embedder = TimestepEmbedder(hidden_size=hidden_dim)
        else:
            self.register_module("time_embedder", None)
        if class_conditioning:
            self.label_embedder = LabelEmbedder(
                output_dim, hidden_size=hidden_dim, dropout_prob=drop_path_rate
            )
        else:
            self.register_module("label_embedder", None)

        # Absolute positional encoding mirrors the original dense path.
        if ape_sigma is not None:
            self.ape = APE(
                hidden_dim, solid_name, ape_sigma, spatial_dim, learned_freqs
            )
        else:
            self.register_buffer("ape", None)

        # Input embedding collapses scalar and vector channels into the hidden width.
        self.x_embedder = PlatonicLinear(
            (input_dim + input_dim_vec * spatial_dim) * self.num_G,
            hidden_dim,
            solid_name,
            bias=False,
        )

        # Stack of dense Platonic blocks.
        dim_feedforward = int(hidden_dim * ffn_dim_factor)
        self.layers = nn.ModuleList(
            [
                # Blocks enable conditioning when either time or class embeddings are requested.
                DensePlatonicBlock(
                    d_model=hidden_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    solid_name=solid_name,
                    dropout=dropout,
                    drop_path_rate=drop_path_rate,
                    layer_scale_init_value=layer_scale_init_value,
                    freq_sigma=rope_sigma,
                    freq_init=freq_init,
                    learned_freqs=learned_freqs,
                    spatial_dims=spatial_dim,
                    mean_aggregation=mean_aggregation,
                    attention=attention,
                    use_key=use_key,
                    conditioning=time_conditioning or class_conditioning,
                )
                for _ in range(num_layers)
            ]
        )

        # Readout layers keep the original equivariant structure.
        if ffn_readout:
            self.scalar_readout = nn.Sequential(
                PlatonicLinear(hidden_dim, hidden_dim, solid_name),
                nn.GELU(),
                PlatonicLinear(hidden_dim, self.num_G * output_dim, solid_name),
            )
            self.vector_readout = nn.Sequential(
                PlatonicLinear(hidden_dim, hidden_dim, solid_name),
                nn.GELU(),
                PlatonicLinear(hidden_dim, hidden_dim, solid_name),
                nn.GELU(),
                PlatonicLinear(
                    hidden_dim, self.num_G * output_dim_vec * spatial_dim, solid_name
                ),
            )
        else:
            self.scalar_readout = PlatonicLinear(
                hidden_dim, self.num_G * output_dim, solid_name
            )
            self.vector_readout = PlatonicLinear(
                hidden_dim, self.num_G * output_dim_vec * spatial_dim, solid_name
            )

        if self.use_cls_token:
            if input_dim > 0:
                cls_init = torch.zeros(1, input_dim)
                self.cls_scalar = nn.Parameter(cls_init)
            else:
                self.register_parameter("cls_scalar", None)
            self.register_buffer("cls_pos", torch.zeros(1, spatial_dim))
        else:
            self.register_parameter("cls_scalar", None)
            self.register_buffer("cls_pos", None)

    def forward(
        self,
        scalars: Optional[Tensor],
        pos: Tensor,
        vec: Optional[Tensor] = None,
        time_conditioning: Optional[Tensor] = None,
        class_conditioning: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            scalars: Scalar token features ``[B, N, C_s]`` or ``None``.
            pos: Token coordinates ``[B, N, D]``.
            vec: Vector-valued token features ``[B, N, C_v, D]`` or ``None``.
            time_conditioning: Optional diffusion timestep embedding ``[B, 1]`` or ``[B, d]``.
            class_conditioning: Optional class embedding ``[B, num_classes]``.
        """

        if scalars is None and vec is None:
            raise ValueError("At least one of `scalars` or `vec` must be provided.")
        if pos.ndim != 3:
            raise ValueError("Position tensor must have shape [B, N, D].")
        if scalars is not None and scalars.ndim != 3:
            raise ValueError("Scalar features must have shape [B, N, C_s].")
        if vec is not None and vec.ndim != 4:
            raise ValueError("Vector features must have shape [B, N, C_v, D].")

        if scalars is not None and scalars.shape[:2] != pos.shape[:2]:
            raise ValueError("Scalar feature batch/sequence dimensions must match positions.")
        if vec is not None and vec.shape[:2] != pos.shape[:2]:
            raise ValueError("Vector feature batch/sequence dimensions must match positions.")

        dense_scalars = scalars
        dense_vec = vec
        dense_pos = pos
        B, N = dense_pos.shape[:2]
        data_token_count = N

        # --- 1. Optionally append a CLS token for ViT-style readout ---
        if self.use_cls_token:
            if dense_scalars is None:
                raise ValueError("CLS token requires scalar features to be provided.")
            cls_scalar = (
                self.cls_scalar.expand(B, 1, -1)
                if self.cls_scalar is not None
                else dense_scalars.new_zeros(B, 1, dense_scalars.shape[-1])
            )  # [B, 1, C_s]
            dense_scalars = torch.cat([cls_scalar, dense_scalars], dim=1)  # [B, N+1, C_s]

            if dense_vec is not None:
                cls_vec = dense_vec.new_zeros(B, 1, dense_vec.shape[2], dense_vec.shape[3])
                dense_vec = torch.cat([cls_vec, dense_vec], dim=1)  # [B, N+1, C_v, D]

            cls_pos = (
                self.cls_pos.expand(B, 1, -1)
                if self.cls_pos is not None
                else dense_pos.new_zeros(B, 1, dense_pos.shape[-1])
            )  # [B, 1, D]
            dense_pos = torch.cat([cls_pos, dense_pos], dim=1)  # [B, N+1, D]

        # --- 2. Lift scalar/vector fields into the group representation ---
        lifted = lift(dense_scalars, dense_vec, self.group)  # [B, N', G*(C_s + C_v*D)]

        # --- 3. Project into hidden space and add absolute position encodings ---
        embedded = self.x_embedder(lifted)  # [B, N', hidden_dim]
        if self.ape is not None:
            embedded = embedded + self.ape(dense_pos)  # [B, N', hidden_dim]

        # --- 4. Prepare optional conditioning signal ---
        conditioning = None
        if self.time_conditioning and time_conditioning is not None:
            conditioning = self.time_embedder(time_conditioning)  # [B, hidden_dim]
        if self.class_conditioning and class_conditioning is not None:
            class_emb = self.label_embedder(class_conditioning)  # [B, hidden_dim]
            conditioning = class_emb if conditioning is None else conditioning + class_emb

        # --- 5. Pass through the stack of dense Platonic blocks ---
        hidden = embedded
        for layer in self.layers:
            hidden = layer(
                x=hidden,
                pos=dense_pos,
                conditioning=conditioning,
                token_normaliser=float(max(data_token_count, 1)),
            )  # [B, N', hidden_dim]

        if self.use_cls_token:
            cls_feature = hidden[:, 0, :]  # [B, hidden_dim]
            tokens = hidden[:, 1:, :]  # [B, N, hidden_dim]
        else:
            cls_feature = None
            tokens = hidden
        token_count = tokens.shape[1]

        # --- 6. Reduce to graph-level summaries when requested ---
        if self.scalar_task_level == "graph":
            scalar_repr = (
                cls_feature
                if cls_feature is not None
                else self._pool_tokens(tokens)
            )  # [B, hidden_dim]
        else:
            scalar_repr = tokens  # [B, N, hidden_dim]

        if self.vector_task_level == "graph":
            vector_repr = (
                cls_feature
                if cls_feature is not None
                else self._pool_tokens(tokens)
            )  # [B, hidden_dim]
        else:
            vector_repr = tokens  # [B, N, hidden_dim]

        # --- 7. Decode back to scalar/vector outputs and split group structure ---
        scalar_logits = self.scalar_readout(scalar_repr)
        vector_logits = self.vector_readout(vector_repr)

        scalar_group = scalar_logits.view(
            *scalar_logits.shape[:-1], self.num_G, self.output_dim
        )
        scalars_out = readout_scalars(scalar_group, self.group)

        vector_group = vector_logits.view(
            *vector_logits.shape[:-1], self.num_G, self.output_dim_vec * self.group.dim
        )
        vectors_out = readout_vectors(vector_group, self.group)

        return scalars_out, vectors_out

    def _pool_tokens(self, tokens: Tensor) -> Tensor:
        """Aggregate token features into a single graph summary."""
        pooled = tokens.sum(dim=1)
        denom = tokens.shape[1]
        if denom == 0:
            return pooled
        scale = torch.tensor(float(denom), device=tokens.device, dtype=tokens.dtype)
        return pooled / scale
