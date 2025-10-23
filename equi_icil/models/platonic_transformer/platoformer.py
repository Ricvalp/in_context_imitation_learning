import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from .block import PlatonicBlock
from .groups import PLATONIC_GROUPS
from .linear import PlatonicLinear
from .io import to_dense_and_mask, pool, lift, to_scalars_vectors
from .ape import PlatonicAPE as APE

from .gen_utils import TimestepEmbedder, LabelEmbedder


class PlatonicTransformer(nn.Module):
    """
    A Transformer architecture equivariant to the symmetries of a specified Platonic solid.

    This model processes point cloud data. It first embeds input node features, then
    "lifts" them into a group-equivariant feature space. A series of PlatonicBlocks
    process these features equivariantly. Finally, for graph-level tasks, it pools
    over the nodes and the group to produce a single invariant prediction. For node-level
    tasks, it pools over the group axis to produce invariant node predictions.

    Args:
        input_dim (int): Dimensionality of the initial node features.
        hidden_dim (int): The per-group-element channel dimension used throughout the model.
        output_dim (int): Dimensionality of the final output.
        nhead (int): Number of attention heads in each PlatonicBlock.
        num_layers (int): Number of PlatonicBlock layers.
        solid_name (str): The name of the Platonic solid ('tetrahedron', 'octahedron',
                          'icosahedron') to define the symmetry group.
        ffn_dim_factor (int): Multiplier for the feed-forward network's hidden dimension,
                              relative to `hidden_dim`.
        scalar_task_level (str): "node" or "graph". Determines the pooling strategy.
        dropout (float): Dropout rate.
        norm_first (bool): If True, use pre-normalization in the blocks.
        drop_path_rate (float): Stochastic depth rate. Default: 0.0.
        layer_scale_init_value (Optional[float]): Initial value for LayerScale. Default: None.
        conditioning_dim (Optional[int]): Size of the diffusion-style conditioning vector passed to each
            PlatonicBlock. If provided, blocks use AdaLayerNorm-style modulation and gating.
        conditioning_mlp_dim (Optional[int]): Hidden dimension for an internal MLP that processes the
            conditioning signal before modulation. If None, conditioning is used as-is.
        **kwargs: Additional keyword arguments for the PlatonicBlock layers
    """

    def __init__(
        self,
        # Basic/essential specification:
        input_dim: int,
        input_dim_vec: int,
        hidden_dim: int,
        output_dim: int,
        output_dim_vec: int,
        nhead: int,
        num_layers: int,
        solid_name: str,
        spatial_dim: int = 3,
        dense_mode: bool = False,  # force dense mode, even if batch is provided
        # Pooling and readout specification:
        scalar_task_level: str = "graph",
        vector_task_level: str = "node",
        ffn_readout: bool = True,
        # Attention block specification:
        mean_aggregation: bool = False,
        dropout: float = 0.1,
        norm_first: bool = True,
        drop_path_rate: float = 0.0,
        layer_scale_init_value: Optional[float] = None,
        attention: bool = False,
        ffn_dim_factor: int = 4,
        # RoPE and APE specification:
        rope_sigma: float = 1.0,  # if None it is not used
        ape_sigma: float = None,  # if None it is not used
        learned_freqs: bool = True,
        freq_init: str = "random",
        use_key: bool = False,
        # Conditioning specification:
        time_conditioning: bool = False,
        class_conditioning: bool = False,
        use_cls_token: bool = False,
    ):
        super().__init__()

        if scalar_task_level not in ["node", "graph"]:
            raise ValueError("scalar_task_level must be 'node' or 'graph'.")

        if vector_task_level not in ["node", "graph"]:
            raise ValueError("vector_task_level must be 'node' or 'graph'.")

        # --- Group and Dimension Setup ---
        self.group = PLATONIC_GROUPS[solid_name.lower()]
        self.num_G = self.group.G
        self.hidden_dim = hidden_dim
        self.scalar_task_level = scalar_task_level
        self.vector_task_level = vector_task_level
        self.dense_mode = dense_mode
        self.output_dim = output_dim
        self.output_dim_vec = output_dim_vec
        self.mean_aggregation = mean_aggregation
        self.use_cls_token = use_cls_token

        # Conditioning setup
        self.time_conditioning = time_conditioning
        self.class_conditioning = class_conditioning
        if self.use_cls_token and not self.dense_mode:
            raise ValueError("CLS token support currently requires dense_mode=True.")

        if time_conditioning:
            self.time_embedder = TimestepEmbedder(hidden_size=hidden_dim)
        if class_conditioning:
            self.label_embedder = LabelEmbedder(
                output_dim, hidden_size=hidden_dim, dropout_prob=drop_path_rate
            )

        # Global position embedding for fixed patching ViTs
        if ape_sigma is not None:
            self.ape = APE(
                hidden_dim, solid_name, ape_sigma, spatial_dim, learned_freqs
            )
        else:
            self.register_buffer("ape", None)

        # --- Modules ---
        # 1. Input Embedding: Applied before lifting to the group.
        # Maps input features to the per-group-element hidden dimension.
        self.x_embedder = PlatonicLinear(
            (input_dim + input_dim_vec * spatial_dim) * self.num_G,
            self.hidden_dim,
            solid_name,
            bias=False,
        )

        # 2. Equivariant Encoder Layers
        # The blocks operate on the total flattened dimension (G * C).
        dim_feedforward = int(self.hidden_dim * ffn_dim_factor)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                PlatonicBlock(
                    d_model=self.hidden_dim,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    solid_name=solid_name,
                    dropout=dropout,
                    norm_first=norm_first,
                    drop_path=drop_path_rate,
                    layer_scale_init_value=layer_scale_init_value,
                    freq_sigma=rope_sigma,
                    freq_init=freq_init,
                    learned_freqs=learned_freqs,
                    spatial_dims=spatial_dim,
                    mean_aggregation=mean_aggregation,
                    attention=attention,
                    use_key=use_key,
                )
            )

        if ffn_readout:
            self.scalar_readout = nn.Sequential(
                PlatonicLinear(self.hidden_dim, self.hidden_dim, solid_name),
                nn.GELU(),
                PlatonicLinear(self.hidden_dim, self.num_G * output_dim, solid_name),
            )

            self.vector_readout = nn.Sequential(
                PlatonicLinear(self.hidden_dim, self.hidden_dim, solid_name),
                nn.GELU(),
                PlatonicLinear(self.hidden_dim, self.hidden_dim, solid_name),
                nn.GELU(),
                PlatonicLinear(
                    self.hidden_dim,
                    self.num_G * output_dim_vec * spatial_dim,
                    solid_name,
                ),
            )
        else:
            self.scalar_readout = PlatonicLinear(
                self.hidden_dim, self.num_G * output_dim, solid_name
            )
            self.vector_readout = PlatonicLinear(
                self.hidden_dim, self.num_G * output_dim_vec * spatial_dim, solid_name
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
        x: Tensor,
        pos: Tensor,
        batch: Optional[torch.Tensor] = None,
        mask: Optional[Tensor] = None,
        vec: Optional[Tensor] = None,
        time_conditioning: Optional[Tensor] = None,
        class_conditioning: Optional[Tensor] = None,
        avg_num_nodes: float = 1.0,
        attention_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Forward pass for the Platonic Transformer.

        Args:
            x (Tensor): Input node features of shape (N, input_dim).
            pos (Tensor): Node positions of shape (N, spatial_dims).
            batch (Tensor): Batch index for each node of shape (N,).
            mask (Tensor, optional): Attention mask of shape (B, N) or (N, N) for dense inputs.
            conditioning (Tensor, optional): Diffusion-style conditioning embeddings of shape
                (B, conditioning_dim), where B is the batch size / number of graphs.

        Returns:
            Tensor: Final predictions. Shape is (B, output_dim) for graph tasks
                    or (N, output_dim) for node tasks.
        """

        # 1. Convert to dense format if needed
        if self.dense_mode:
            self._input_was_dense_format = batch is None
            x, vec, pos, mask = to_dense_and_mask(x, vec, pos, batch)
            batch = None
        else:
            self._input_was_dense_format = False
            mask = None

        if self.use_cls_token:
            if x is None:
                raise ValueError("CLS token requires scalar features to be provided.")
            batch_size = x.shape[0]
            if self.cls_scalar is not None:
                cls_scalar = self.cls_scalar.expand(batch_size, 1, -1)
            else:
                cls_scalar = x.new_zeros(batch_size, 1, x.shape[-1])
            x = torch.cat([cls_scalar, x], dim=1)
            if vec is not None:
                cls_vec = vec.new_zeros(batch_size, 1, vec.shape[2], vec.shape[3])
                vec = torch.cat([cls_vec, vec], dim=1)
            if self.cls_pos is not None:
                pos = torch.cat([self.cls_pos.expand(batch_size, 1, -1), pos], dim=1)
            else:
                pos = torch.cat(
                    [pos.new_zeros(batch_size, 1, pos.shape[-1]), pos], dim=1
                )
            if mask is not None:
                cls_mask = mask.new_ones(batch_size, 1)
                mask = torch.cat([cls_mask, mask], dim=1)
            if attention_mask is not None:
                expanded = torch.zeros(
                    attention_mask.shape[0],
                    attention_mask.shape[-1] + 1,
                    attention_mask.shape[-1] + 1,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                )
                expanded[:, 1:, 1:] = attention_mask
                attention_mask = expanded

        # 2. Lift scalars and vectors, then embed
        x = lift(x, vec, self.group)
        x = self.x_embedder(x)  # [..., N, num_patches * C]
        x = (
            x + self.ape(pos) if self.ape is not None else x
        )  # Add absolute position embedding

        # 3. Embed conditioning vector
        t_embed = c_embed = None
        if self.time_conditioning and time_conditioning is not None:
            t_embed = self.time_embedder(time_conditioning)
        if self.class_conditioning and class_conditioning is not None:
            c_embed = self.label_embedder(class_conditioning)

        # 4. Merge conditioning vectors if necessary
        if t_embed is not None and c_embed is not None:
            conditioning = t_embed + c_embed
        elif t_embed is not None:
            conditioning = t_embed
        else:
            conditioning = None

        # 5. Equivariant Encoder (Platonic Conv Blocks)
        for layer in self.layers:
            x = layer(
                x=x,
                pos=pos,
                batch=batch,
                mask=mask,
                attn_mask=attention_mask,
                conditioning=conditioning,
                avg_num_nodes=avg_num_nodes,
            )

        if self.use_cls_token:
            cls_feature = x[:, 0, :]
            x_no_cls = x[:, 1:, :]
            mask_no_cls = None if mask is None else mask[:, 1:]
        else:
            cls_feature = None
            x_no_cls = x
            mask_no_cls = mask

        # 6. Post-pooling readout
        if self.scalar_task_level == "graph":
            if self.use_cls_token:
                scalar_x = cls_feature
            else:
                scalar_x = pool(
                    x_no_cls,
                    batch,
                    mask_no_cls,
                    avg_num_nodes,
                    self.dense_mode,
                    self.mean_aggregation,
                )
        else:
            if (
                not self._input_was_dense_format
                and self.dense_mode
                and mask_no_cls is not None
            ):
                scalar_x = x_no_cls[mask_no_cls]
            else:
                scalar_x = x_no_cls

        if self.vector_task_level == "graph":
            if self.use_cls_token:
                vector_x = cls_feature
            else:
                vector_x = pool(
                    x_no_cls,
                    batch,
                    mask_no_cls,
                    avg_num_nodes,
                    self.dense_mode,
                    self.mean_aggregation,
                )
        else:
            if (
                not self._input_was_dense_format
                and self.dense_mode
                and mask_no_cls is not None
            ):
                vector_x = x_no_cls[mask_no_cls]
            else:
                vector_x = x_no_cls

        scalar_x = self.scalar_readout(scalar_x)
        vector_x = self.vector_readout(vector_x)

        # 7. Extract the scalar and vector parts
        scalars = to_scalars_vectors(scalar_x, self.output_dim, 0, self.group)[0]
        vectors = to_scalars_vectors(vector_x, 0, self.output_dim_vec, self.group)[1]

        # Return final result
        return scalars, vectors
