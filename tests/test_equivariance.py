import torch
import pytest

from equi_icil.models.platonic_transformer.platoformer import PlatonicTransformer
from equi_icil.models.dense_platonic_transformer.platoformer import (
    DensePlatonicTransformer,
)
from equi_icil.models.platonic_transformer.groups import PLATONIC_GROUPS


def _build_original(solid: str):
    torch.manual_seed(0)
    group = PLATONIC_GROUPS[solid]
    model = PlatonicTransformer(
        input_dim=6,
        input_dim_vec=2,
        hidden_dim=group.G * 4,
        output_dim=4,
        output_dim_vec=2,
        nhead=group.G,
        num_layers=2,
        solid_name=solid,
        spatial_dim=3,
        dense_mode=True,
        scalar_task_level="graph",
        vector_task_level="graph",
        attention=True,
        time_conditioning=False,
        class_conditioning=False,
        use_cls_token=False,
        dropout=0.0,
        drop_path_rate=0.0,
    )
    return model


def _build_dense(solid: str):
    torch.manual_seed(0)
    group = PLATONIC_GROUPS[solid]
    model = DensePlatonicTransformer(
        input_dim=6,
        input_dim_vec=2,
        hidden_dim=group.G * 4,
        output_dim=4,
        output_dim_vec=2,
        nhead=group.G,
        num_layers=2,
        solid_name=solid,
        spatial_dim=3,
        scalar_task_level="graph",
        vector_task_level="graph",
        attention=True,
        time_conditioning=False,
        class_conditioning=False,
        use_cls_token=False,
        dropout=0.0,
        drop_path_rate=0.0,
    )
    return model


@pytest.mark.parametrize(
    "model_builder",
    [
        pytest.param(_build_original, id="original"),
        pytest.param(_build_dense, id="dense"),
    ],
)
def test_transformer_equivariance(model_builder):
    solid = "tetrahedron"
    group = PLATONIC_GROUPS[solid]

    model = model_builder(solid)
    model.eval()

    torch.manual_seed(42)
    B, N = 2, 5
    num_scalars = 6
    num_vectors = 2
    scalars = torch.randn(B, N, num_scalars)
    vectors = torch.randn(B, N, num_vectors, 3)
    pos = torch.randn(B, N, 3)

    dense_model = isinstance(model, DensePlatonicTransformer)
    common_kwargs = {
        "vec": vectors,
        "time_conditioning": None,
        "class_conditioning": None,
    }
    if not dense_model:
        common_kwargs["avg_num_nodes"] = float(N)

    with torch.no_grad():
        base_scalars, base_vectors = model(scalars, pos, **common_kwargs)

        for R in group.elements.to(dtype=scalars.dtype):
            pos_rot = torch.einsum("ij,bnj->bni", R, pos)
            vec_rot = torch.einsum("ij,bncj->bnci", R, vectors)

            rotated_kwargs = dict(common_kwargs)
            rotated_kwargs["vec"] = vec_rot
            scalars_rot, vectors_rot = model(scalars, pos_rot, **rotated_kwargs)

            assert torch.allclose(
                scalars_rot, base_scalars, atol=1e-4, rtol=1e-4
            ), "Scalar outputs must remain invariant under group action."

            expected_vectors = torch.einsum(
                "ij,bvj->bvi", R, base_vectors
            )
            assert torch.allclose(
                vectors_rot, expected_vectors, atol=1e-4, rtol=1e-4
            ), "Vector outputs must transform equivariantly under group action."
