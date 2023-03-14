#include <torch/script.h>
#include "sparse_ops/voxelize.h"

namespace sparse_ops::voxelize {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> voxelize(
    at::Tensor points,
    c10::optional<at::Tensor> batch_indices,
    at::Tensor voxel_size,
    at::Tensor points_range_min,
    at::Tensor points_range_max) {
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
