#include <torch/autograd.h>
#include "sparse_ops/devoxelize.h"

namespace sparse_ops::devoxelize::autograd {

class TrilinearDevoxelize : public torch::autograd::Function<TrilinearDevoxelize> {
public:
  static torch::autograd::variable_list forward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::Variable& points,
      const c10::optional<torch::autograd::Variable>& batch_indices,
      const torch::autograd::Variable& voxel_size,
      const torch::autograd::Variable& points_range_min,
      const torch::autograd::Variable& points_range_max,
      const torch::autograd::Variable& voxel_coords,
      const torch::autograd::Variable& voxel_features,
      const torch::autograd::Variable& voxel_batch_indices) {
    ctx->saved_data["num_voxels"] = voxel_coords.size(0);

    at::AutoDispatchBelowADInplaceOrView g;
    auto [point_features, indices, weights] = trilinear_devoxelize(
        points, batch_indices, voxel_size, points_range_min, points_range_max,
        voxel_coords, voxel_features, voxel_batch_indices);

    ctx->save_for_backward({indices, weights});

    return {point_features, indices, weights};
  }

  static torch::autograd::variable_list backward(
      torch::autograd::AutogradContext* ctx,
      const torch::autograd::variable_list& grad_outputs) {
    auto grad_output = grad_outputs[0].contiguous();

    const auto& saved_tensors = ctx->get_saved_variables();
    auto indices = saved_tensors[0];
    auto weights = saved_tensors[1];

    auto num_voxels = ctx->saved_data["num_voxels"].toInt();

    auto grad_inputs = trilinear_devoxelize_backward(
        grad_output, indices, weights, num_voxels);

    return {
        at::Tensor(), at::Tensor(), at::Tensor(), at::Tensor(),
        at::Tensor(), at::Tensor(), grad_inputs, at::Tensor()};
  }
};

std::tuple<at::Tensor, at::Tensor, at::Tensor> trilinear_devoxelize_autograd(
    at::Tensor points,
    c10::optional<at::Tensor> batch_indices,
    at::Tensor voxel_size,
    at::Tensor points_range_min,
    at::Tensor points_range_max,
    at::Tensor voxel_coords,
    at::Tensor voxel_features,
    at::Tensor voxel_batch_indices) {
  auto results = TrilinearDevoxelize::apply(
      points, batch_indices, voxel_size, points_range_min, points_range_max,
      voxel_coords, voxel_features, voxel_batch_indices);
  return {results[0], results[1], results[2]};
}

TORCH_LIBRARY_IMPL(sparse_ops, Autograd, m) {
  m.impl(TORCH_SELECTIVE_NAME("sparse_ops::trilinear_devoxelize"),
         TORCH_FN(trilinear_devoxelize_autograd));
}

} // sparse_ops::devoxelize::autograd
