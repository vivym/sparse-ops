from typing import Optional, Tuple

import torch


def trilinear_devoxelize(
    points: torch.Tensor,
    *,
    batch_indices: Optional[torch.Tensor] = None,
    voxel_size: torch.Tensor,
    points_range_min: torch.Tensor,
    points_range_max: torch.Tensor,
    voxel_coords: torch.Tensor,
    voxel_features: torch.Tensor,
    voxel_batch_indices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.sparse_ops.trilinear_devoxelize(
        points, batch_indices, voxel_size, points_range_min, points_range_max,
        voxel_coords, voxel_features, voxel_batch_indices,
    )


def main():
    device = torch.device("cuda")

    points = torch.as_tensor(
        [
            [0.6, 0.7, 0.9],
            [0.2, 0.1, 0.3],
            [0.7, 0.6, 0.8],
            [0.1, 0.6, 0.3],
            [0.7, 0.7, 0.1],
        ],
        dtype=torch.float32,
        device=device,
    )
    batch_indices = torch.as_tensor([0, 0, 0, 1, 1], dtype=torch.int64, device=device)

    from .voxelize import voxelize

    (
        voxel_coords, voxel_indices, voxel_batch_indices, voxel_point_indices
    ) = voxelize(
        points,
        batch_indices=batch_indices,
        voxel_size=torch.as_tensor([0.5, 0.5, 0.5], dtype=points.dtype),
        points_range_min=torch.as_tensor([0, 0, 0], dtype=points.dtype),
        points_range_max=torch.as_tensor([1, 1, 1], dtype=points.dtype),
    )
    print("voxel_coords", voxel_coords)
    print("voxel_indices", voxel_indices)
    print("voxel_batch_indices", voxel_batch_indices)
    print("voxel_point_indices", voxel_point_indices)

    voxel_features = torch.randn(
        voxel_coords.shape[0], 6, dtype=torch.float32, device=device
    )

    point_features, indices, weights = trilinear_devoxelize(
        points,
        batch_indices=batch_indices,
        voxel_size=torch.as_tensor([0.5, 0.5, 0.5], dtype=points.dtype),
        points_range_min=torch.as_tensor([0, 0, 0], dtype=points.dtype),
        points_range_max=torch.as_tensor([1, 1, 1], dtype=points.dtype),
        voxel_coords=voxel_coords,
        voxel_features=voxel_features,
        voxel_batch_indices=voxel_batch_indices,
    )
    print("point_features", point_features, point_features.shape)
    print("indices", indices, indices.shape)
    print("weights", weights, weights.shape)


if __name__ == "__main__":
    main()
