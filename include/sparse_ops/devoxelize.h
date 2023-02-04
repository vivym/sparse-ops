#pragma once

#include <tuple>
#include <torch/types.h>

namespace sparse_ops::devoxelize {

std::tuple<at::Tensor, at::Tensor, at::Tensor> trilinear_devoxelize(
    at::Tensor points,
    c10::optional<at::Tensor> batch_indices,
    at::Tensor voxel_size,
    at::Tensor points_range_min,
    at::Tensor points_range_max,
    at::Tensor voxel_coords,
    at::Tensor voxel_features,
    at::Tensor voxel_batch_indices);

at::Tensor trilinear_devoxelize_backward(
    const at::Tensor& grad_outputs,
    const at::Tensor& indices,
    const at::Tensor& weights,
    int64_t num_voxels);

namespace cuda {

std::tuple<at::Tensor, at::Tensor, at::Tensor> trilinear_devoxelize(
    at::Tensor points,
    c10::optional<at::Tensor> batch_indices,
    at::Tensor voxel_size,
    at::Tensor points_range_min,
    at::Tensor points_range_max,
    at::Tensor voxel_coords,
    at::Tensor voxel_features,
    at::Tensor voxel_batch_indices);

at::Tensor trilinear_devoxelize_backward(
    const at::Tensor& grad_outputs,
    const at::Tensor& indices,
    const at::Tensor& weights,
    int64_t num_voxels);

} // namespace cuda

namespace autograd {

std::tuple<at::Tensor, at::Tensor, at::Tensor> trilinear_devoxelize_autograd(
    at::Tensor points,
    c10::optional<at::Tensor> batch_indices,
    at::Tensor voxel_size,
    at::Tensor points_range_min,
    at::Tensor points_range_max,
    at::Tensor voxel_coords,
    at::Tensor voxel_features,
    at::Tensor voxel_batch_indices);

}

} // namespace sparse_ops::devoxelize
