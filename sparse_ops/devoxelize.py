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
    hash_table_load_factor: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.sparse_ops.trilinear_devoxelize(
        points, batch_indices, voxel_size, points_range_min, points_range_max,
        voxel_coords, voxel_features, voxel_batch_indices, hash_table_load_factor,
    )
