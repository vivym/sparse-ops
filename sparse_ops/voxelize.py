from typing import Optional, Tuple

import torch


@torch.no_grad()
def voxelize(
    points: torch.Tensor,
    *,
    batch_indices: Optional[torch.Tensor] = None,
    voxel_size: torch.Tensor,
    points_range_min: torch.Tensor,
    points_range_max: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.sparse_ops.voxelize(
        points, batch_indices, voxel_size, points_range_min, points_range_max
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

    voxel_coords, voxel_indices, voxel_batch_indices, voxel_point_indices = voxelize(
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


if __name__ == "__main__":
    main()
