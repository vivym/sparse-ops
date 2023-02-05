from dataclasses import dataclass
from typing import Optional, Tuple

import torch

@dataclass
class Voxels:
    coords: torch.Tensor
    batch_indices: torch.Tensor
    features: Optional[torch.Tensor] = None


def voxelize(
    point_coords: torch.Tensor,
    *,
    point_features: Optional[torch.Tensor] = None,
    batch_indices: Optional[torch.Tensor] = None,
    voxel_size: torch.Tensor,
    points_range_min: torch.Tensor,
    points_range_max: torch.Tensor,
    reduction: str = "mean",
) -> Voxels:
    with torch.no_grad():
        (
            voxel_coords, voxel_indices, voxel_batch_indices, voxel_point_indices
        ) = torch.ops.sparse_ops.voxelize(
            point_coords, batch_indices, voxel_size,
            points_range_min, points_range_max,
        )

    if point_features is None:
        voxel_features = None
    else:
        _, num_points_per_voxel = torch.unique_consecutive(
            voxel_indices, return_counts=True
        )

        voxel_features = point_features.view(-1, point_features.shape[-1])
        voxel_features = voxel_features[voxel_point_indices]
        voxel_features = torch.segment_reduce(
            voxel_features, reduce=reduction, lengths=num_points_per_voxel, unsafe=True
        )

    return Voxels(
        coords=voxel_coords,
        batch_indices=voxel_batch_indices,
        features=voxel_features,
    )
