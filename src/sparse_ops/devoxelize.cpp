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
