#include <torch/script.h>
#include "sparse_ops/devoxelize.h"

namespace sparse_ops::devoxelize {

std::tuple<at::Tensor, at::Tensor, at::Tensor> trilinear_devoxelize(
    at::Tensor points,
    c10::optional<at::Tensor> batch_indices,
    at::Tensor voxel_size,
    at::Tensor points_range_min,
    at::Tensor points_range_max,
    at::Tensor voxel_coords,
    at::Tensor voxel_features,
    at::Tensor voxel_batch_indices,
    double hash_table_load_factor) {
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
                       .findSchemaOrThrow("sparse_ops::trilinear_devoxelize", "")
                       .typed<decltype(trilinear_devoxelize)>();
  return op.call(
      points, batch_indices, voxel_size, points_range_min, points_range_max,
      voxel_coords, voxel_features, voxel_batch_indices, hash_table_load_factor);
}

at::Tensor trilinear_devoxelize_backward(
    const at::Tensor& grad_outputs,
    const at::Tensor& indices,
    const at::Tensor& weights,
    int64_t num_voxels) {
  TORCH_CHECK(grad_outputs.is_contiguous(), "The grad_outputs must be a contiguous tensor.");
  TORCH_CHECK(grad_outputs.dim() == 2 || grad_outputs.dim() == 3,
              "The grad_outputs must be a 2D or 3D tensor.");

  TORCH_CHECK(indices.is_contiguous(), "The indices must be a contiguous tensor.");
  TORCH_CHECK(indices.dim() == grad_outputs.dim(), "The indices must be a 2D or 3D tensor.");

  TORCH_CHECK(weights.is_contiguous(), "The weights must be a contiguous tensor.");
  TORCH_CHECK(weights.dim() == grad_outputs.dim(), "The weights must be a 2D or 3D tensor.");

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("sparse_ops::trilinear_devoxelize_backward", "")
                       .typed<decltype(trilinear_devoxelize_backward)>();
  return op.call(grad_outputs, indices, weights, num_voxels);
}

TORCH_LIBRARY_FRAGMENT(sparse_ops, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse_ops::trilinear_devoxelize(Tensor points, Tensor? batch_indices, "
      "Tensor voxel_size, Tensor points_range_min, Tensor points_range_max, "
      "Tensor voxel_coords, Tensor voxel_features, Tensor voxel_batch_indices, "
      "float hash_table_load_factor) -> (Tensor, Tensor, Tensor)"));
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse_ops::trilinear_devoxelize_backward(Tensor grad_outputs, "
      "Tensor indices, Tensor weights, int num_voxels) -> Tensor"));
}

} // namespace sparse_ops::devoxelize
