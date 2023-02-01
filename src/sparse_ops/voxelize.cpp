#include <torch/script.h>
#include "sparse_ops/voxelize.h"

namespace sparse_ops::voxelize {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> voxelize(
    at::Tensor points,
    c10::optional<at::Tensor> batch_indices,
    at::Tensor voxel_size,
    at::Tensor points_range_min,
    at::Tensor points_range_max) {
  if (batch_indices.has_value()) {
    TORCH_CHECK(points.dim() == 2, "The points must be a 2D tensor.");
    TORCH_CHECK(batch_indices.value().dim() == 1,
                "The batch_indices must be a 1D tensor.");
  } else {
    TORCH_CHECK(points.dim() == 3, "The points must be a 3D tensor.");
  }

  TORCH_CHECK(0 < points.size(-1) && points.size(-1) < 9,
              "The number of dimensions must be in [1, ..., 8].");
  TORCH_CHECK(voxel_size.size(0) == points.size(-1),
              "The number of dimensions of voxel_size is invalid.");
  TORCH_CHECK(points_range_min.size(0) == points.size(-1),
              "The number of dimensions of points_range_min is invalid.");
  TORCH_CHECK(points_range_max.size(0) == points.size(-1),
              "The number of dimensions of points_range_max is invalid.");

  TORCH_CHECK(voxel_size.is_cpu(),
              "The voxel_size must be a cpu tensor.");
  TORCH_CHECK(points_range_min.is_cpu(),
              "The points_range_min must be a cpu tensor.");
  TORCH_CHECK(points_range_max.is_cpu(),
              "The points_range_max must be a cpu tensor.");

  TORCH_CHECK(voxel_size.dim() == 1,
              "The voxel_size must be a 1D tensor.");
  TORCH_CHECK(points_range_min.dim() == 1,
              "The points_range_min must be a 1D tensor.");
  TORCH_CHECK(points_range_max.dim() == 1,
              "The points_range_max must be a 1D tensor.");

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("sparse_ops::voxelize", "")
                       .typed<decltype(voxelize)>();
  return op.call(points, batch_indices, voxel_size, points_range_min, points_range_max);
}

TORCH_LIBRARY_FRAGMENT(sparse_ops, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse_ops::voxelize(Tensor points, Tensor? batch_indices, "
      "Tensor voxel_size, Tensor points_range_min, Tensor points_range_max) "
      "-> (Tensor, Tensor, Tensor, Tensor)"));
}

} // namespace sparse_ops::voxelize
