import torch
import pytest

from equi_icil.models.platonic_transformer.groups import PLATONIC_GROUPS
from equi_icil.models.platonic_transformer.conv import PlatonicConv
from equi_icil.models.platonic_transformer.block import PlatonicBlock
from equi_icil.models.platonic_transformer.platoformer import PlatonicTransformer

from equi_icil.models.dense_platonic_transformer.attention import DensePlatonicAttention
from equi_icil.models.dense_platonic_transformer.block import DensePlatonicBlock
from equi_icil.models.dense_platonic_transformer.platoformer import DensePlatonicTransformer


def _copy_conv_to_dense(conv: PlatonicConv, dense: DensePlatonicAttention) -> None:
    dense.q_proj.load_state_dict(conv.q_proj.state_dict())
    dense.v_proj.load_state_dict(conv.v_proj.state_dict())
    if conv.k_proj is not None:
        dense.k_proj.load_state_dict(conv.k_proj.state_dict())  # type: ignore[attr-defined]
    if conv.rope_emb is not None:
        dense.rope_emb.load_state_dict(conv.rope_emb.state_dict())  # type: ignore[attr-defined]
    dense.out_proj.load_state_dict(conv.out_proj.state_dict())


def _copy_block_to_dense(block: PlatonicBlock, dense_block: DensePlatonicBlock) -> None:
    _copy_conv_to_dense(block.interaction, dense_block.interaction)
    dense_block.linear1.load_state_dict(block.linear1.state_dict())
    dense_block.linear2.load_state_dict(block.linear2.state_dict())
    dense_block.norm1.load_state_dict(block.norm1.state_dict())
    dense_block.norm2.load_state_dict(block.norm2.state_dict())
    dense_block.dropout1.p = block.dropout1.p
    dense_block.dropout2.p = block.dropout2.p
    dense_block.ffn_dropout.p = block.ffn_dropout.p
    if hasattr(block.drop_path1, "drop_prob") and hasattr(dense_block.drop_path1, "drop_prob"):
        dense_block.drop_path1.drop_prob = block.drop_path1.drop_prob
    if hasattr(block.drop_path2, "drop_prob") and hasattr(dense_block.drop_path2, "drop_prob"):
        dense_block.drop_path2.drop_prob = block.drop_path2.drop_prob
    if block.gamma_1 is not None and dense_block.gamma_1 is not None:
        dense_block.gamma_1.data.copy_(block.gamma_1.data)
    if block.gamma_2 is not None and dense_block.gamma_2 is not None:
        dense_block.gamma_2.data.copy_(block.gamma_2.data)
    if block.conditioning:
        dense_block.adaLN_modulation.load_state_dict(block.adaLN_modulation.state_dict())


def _copy_transformer_to_dense(
    transformer: PlatonicTransformer, dense_transformer: DensePlatonicTransformer
) -> None:
    dense_transformer.x_embedder.load_state_dict(transformer.x_embedder.state_dict())
    if transformer.ape is not None and dense_transformer.ape is not None:
        dense_transformer.ape.load_state_dict(transformer.ape.state_dict())
    if transformer.time_conditioning and dense_transformer.time_conditioning:
        dense_transformer.time_embedder.load_state_dict(
            transformer.time_embedder.state_dict()
        )
    if transformer.class_conditioning and dense_transformer.class_conditioning:
        dense_transformer.label_embedder.load_state_dict(
            transformer.label_embedder.state_dict()
        )
    for src_block, dst_block in zip(transformer.layers, dense_transformer.layers):
        _copy_block_to_dense(src_block, dst_block)
    dense_transformer.scalar_readout.load_state_dict(transformer.scalar_readout.state_dict())
    dense_transformer.vector_readout.load_state_dict(transformer.vector_readout.state_dict())


@pytest.mark.parametrize("attention", [False, True])
def test_dense_attention_matches_original(attention: bool) -> None:
    torch.manual_seed(0)
    solid = "tetrahedron"
    G = PLATONIC_GROUPS[solid].G
    in_channels = G * 2
    out_channels = G * 2
    embed_dim = G * 2
    num_heads = G * 1

    conv = PlatonicConv(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        solid_name=solid,
        attention=attention,
    )
    dense_attn = DensePlatonicAttention(
        in_channels=in_channels,
        out_channels=out_channels,
        embed_dim=embed_dim,
        num_heads=num_heads,
        solid_name=solid,
        attention=attention,
    )
    _copy_conv_to_dense(conv, dense_attn)

    B, N, D = 2, 5, 3
    x = torch.randn(B, N, in_channels)
    pos = torch.randn(B, N, D)
    avg_nodes = float(N)

    conv.eval()
    dense_attn.eval()

    mask = torch.ones(B, N, dtype=torch.bool)

    with torch.no_grad():
        ref = conv(x, pos, batch=None, mask=mask, avg_num_nodes=avg_nodes)
        dense_out = dense_attn(x, pos, token_normaliser=avg_nodes)

    assert torch.allclose(ref, dense_out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("attention", [False, True])
def test_dense_block_matches_original(attention: bool) -> None:
    torch.manual_seed(1)
    solid = "tetrahedron"
    G = PLATONIC_GROUPS[solid].G
    hidden_dim = G * 4
    nhead = G * 1
    ffn_dim = G * 8

    block = PlatonicBlock(
        d_model=hidden_dim,
        nhead=nhead,
        dim_feedforward=ffn_dim,
        solid_name=solid,
        attention=attention,
        conditioning=True,
    )
    dense_block = DensePlatonicBlock(
        d_model=hidden_dim,
        nhead=nhead,
        dim_feedforward=ffn_dim,
        solid_name=solid,
        attention=attention,
        conditioning=True,
    )
    _copy_block_to_dense(block, dense_block)

    B, N, D = 3, 4, 3
    x = torch.randn(B, N, hidden_dim)
    pos = torch.randn(B, N, D)
    conditioning = torch.randn(B, hidden_dim)
    avg_nodes = float(N)

    block.eval()
    dense_block.eval()

    mask = torch.ones(B, N, dtype=torch.bool)

    with torch.no_grad():
        ref = block(
            x,
            pos,
            batch=None,
            mask=mask,
            attn_mask=None,
            conditioning=conditioning,
            avg_num_nodes=avg_nodes,
        )
        dense_out = dense_block(
            x,
            pos,
            conditioning=conditioning,
            token_normaliser=avg_nodes,
        )

    assert torch.allclose(ref, dense_out, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("attention", [False, True])
def test_dense_transformer_matches_original(attention: bool) -> None:
    torch.manual_seed(2)

    solid = "tetrahedron"
    G = PLATONIC_GROUPS[solid].G

    input_dim = 6
    input_dim_vec = 2
    hidden_dim = G * 4
    output_dim = 5
    output_dim_vec = 3
    nhead = G * 1
    num_layers = 2
    spatial_dim = 3

    transformer = PlatonicTransformer(
        input_dim=input_dim,
        input_dim_vec=input_dim_vec,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        output_dim_vec=output_dim_vec,
        nhead=nhead,
        num_layers=num_layers,
        solid_name=solid,
        spatial_dim=spatial_dim,
        dense_mode=True,
        scalar_task_level="graph",
        vector_task_level="graph",
        attention=attention,
        time_conditioning=True,
        class_conditioning=False,
        use_cls_token=False,
        dropout=0.0,
        drop_path_rate=0.0,
    )

    dense_transformer = DensePlatonicTransformer(
        input_dim=input_dim,
        input_dim_vec=input_dim_vec,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        output_dim_vec=output_dim_vec,
        nhead=nhead,
        num_layers=num_layers,
        solid_name=solid,
        spatial_dim=spatial_dim,
        scalar_task_level="graph",
        vector_task_level="graph",
        attention=attention,
        time_conditioning=True,
        class_conditioning=False,
        dropout=0.0,
        drop_path_rate=0.0,
    )

    _copy_transformer_to_dense(transformer, dense_transformer)

    B, N = 2, 5
    scalars = torch.randn(B, N, input_dim)
    vec = torch.randn(B, N, input_dim_vec, spatial_dim)
    pos = torch.randn(B, N, spatial_dim)
    time_cond = torch.randint(0, 1000, (B,))
    avg_nodes = float(N)

    transformer.eval()
    dense_transformer.eval()

    with torch.no_grad():
            ref_scalars, ref_vectors = transformer(
                scalars,
                pos,
                batch=None,
                mask=None,
                vec=vec,
                time_conditioning=time_cond,
                class_conditioning=None,
                avg_num_nodes=avg_nodes,
                attention_mask=None,
            )
            dense_scalars, dense_vectors = dense_transformer(
                scalars,
                pos,
                vec=vec,
                time_conditioning=time_cond,
                class_conditioning=None,
            )

    assert torch.allclose(ref_scalars, dense_scalars, atol=1e-5, rtol=1e-5)
    assert torch.allclose(ref_vectors, dense_vectors, atol=1e-5, rtol=1e-5)
