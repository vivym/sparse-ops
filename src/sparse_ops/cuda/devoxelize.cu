#include <torch/script.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <bcht.hpp>
#include "sparse_ops/devoxelize.h"
#include "sparse_ops/thrust/devoxelize.hpp"
#include "sparse_ops/thrust/misc.hpp"
#include "sparse_ops/thrust_allocator.h"

namespace sparse_ops::devoxelize::cuda {

template <typename T>
using ThrustVector = thrust::device_vector<T, DeviceVectorAllocator<T>>;

template <typename scalar_t, typename index_t>
void trilinear_devoxelize_impl(
    at::Tensor& point_features,
    at::Tensor& indices,
    at::Tensor& weights,
    const at::Tensor& point_coords,
    c10::optional<at::Tensor> batch_indices,
    const at::Tensor& voxel_size,
    const at::Tensor& points_range_min,
    const at::Tensor& points_range_max,
    const at::Tensor& voxel_coords,
    const at::Tensor& voxel_features,
    const at::Tensor& voxel_batch_indices) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(ThrustAllocator()).on(stream);

  if (!batch_indices.has_value()) {
    auto batch_size = point_coords.size(0);
    auto num_points = point_coords.size(1);

    auto indices_options = point_coords.options().dtype(at::kLong);
    auto tmp = at::empty({batch_size * num_points}, indices_options);
    sparse_ops::thrust_impl::generate_batch_indices<index_t>(
        policy, tmp.data_ptr<index_t>(), batch_size, num_points);
    batch_indices.value() = tmp;
  }

  auto num_points = batch_indices.value().size(0);
  auto num_voxels = voxel_coords.size(0);
  auto num_channels = voxel_features.size(-1);

  auto point_features_ptr = point_features.data_ptr<scalar_t>();
  auto indices_ptr = indices.data_ptr<index_t>();
  auto weights_ptr = weights.data_ptr<scalar_t>();

  const auto* const point_coords_ptr = point_coords.data_ptr<scalar_t>();
  const auto* const batch_indices_ptr = batch_indices.value().data_ptr<index_t>();
  const auto* const voxel_coords_ptr = voxel_coords.data_ptr<index_t>();
  const auto* const voxel_features_ptr = voxel_features.data_ptr<scalar_t>();
  const auto* const voxel_batch_indices_ptr = voxel_batch_indices.data_ptr<index_t>();

  FixedVec<scalar_t, 3> voxel_size_vec;
  voxel_size_vec.load(voxel_size.data_ptr<scalar_t>());
  FixedVec<scalar_t, 3> points_range_min_vec;
  points_range_min_vec.load(points_range_min.data_ptr<scalar_t>());
  FixedVec<scalar_t, 3> points_range_max_vec;
  points_range_max_vec.load(points_range_max.data_ptr<scalar_t>());

  const auto voxel_extents = ceil(
      (points_range_max_vec - points_range_min_vec) / voxel_size_vec)
          .template cast<index_t>();
  FixedVec<index_t, 3> voxel_strides;
  voxel_strides[0] = 1;
  for (int i = 1; i < 3; i++) {
    voxel_strides[i] = voxel_strides[i - 1] * voxel_extents[i - 1];
  }
  const index_t batch_stride = voxel_strides[2] * voxel_extents[2];

  using key_type = index_t;
  using value_type = index_t;
  using HashTable = bght::bcht<key_type, value_type>;

  auto invalid_key = std::numeric_limits<key_type>::max();
  auto invalid_value = std::numeric_limits<value_type>::max();

  constexpr double load_factor = 0.9;
  std::size_t capacity = static_cast<double>(num_voxels) / load_factor;

  printf("#1 %d\n", (int)capacity);

  HashTable table(capacity, invalid_key, invalid_value);

  {
    ThrustVector<bght::pair<key_type, value_type>> kv_pairs_vec(num_voxels);

    printf("#2\n");

    thrust_impl::compute_hash_table_kv_pairs<index_t, 3>(
        policy, kv_pairs_vec.data().get(), voxel_coords_ptr, voxel_batch_indices_ptr,
        num_voxels, batch_stride, voxel_strides);

    printf("#3\n");

    TORCH_CHECK(table.insert(kv_pairs_vec.begin(), kv_pairs_vec.end(), stream));
  }

  printf("#4\n");

  thrust_impl::compute_hash_table_queries_and_weights<scalar_t, index_t, 3>(
      policy, indices_ptr, weights_ptr, point_coords_ptr, batch_indices_ptr,
      num_points, voxel_size_vec, points_range_min_vec,
      batch_stride, voxel_strides, voxel_extents);

  printf("#5\n");

  table.find(indices_ptr, indices_ptr + num_points * 8, indices_ptr, stream);

  printf("#6\n");

  thrust_impl::trilinear_interpolate(
      policy, point_features_ptr, indices_ptr, weights_ptr,
      voxel_features_ptr, num_points, num_channels, invalid_value);

  printf("#7\n");
}

std::tuple<at::Tensor, at::Tensor, at::Tensor> trilinear_devoxelize(
    at::Tensor points,
    c10::optional<at::Tensor> batch_indices,
    at::Tensor voxel_size,
    at::Tensor points_range_min,
    at::Tensor points_range_max,
    at::Tensor voxel_coords,
    at::Tensor voxel_features,
    at::Tensor voxel_batch_indices) {
  TORCH_CHECK(points.is_cuda(), "The point_coords must be a CUDA tensor.");
  if (batch_indices.has_value()) {
    TORCH_CHECK(batch_indices.value().is_cuda(),
                "The batch_indices must be a CUDA tensor.");
  }
  TORCH_CHECK(voxel_coords.is_cuda(),
              "The voxel_coords must be a CUDA tensor.");
  TORCH_CHECK(voxel_features.is_cuda(),
              "The voxel_features must be a CUDA tensor.");
  TORCH_CHECK(voxel_batch_indices.is_cuda(),
              "The voxel_batch_indices must be a CUDA tensor.");

  auto num_channels = voxel_features.size(-1);

  auto indices_options = voxel_batch_indices.options();

  at::Tensor point_features, indices, weights;
  if (batch_indices.has_value()) {
    auto num_points = points.size(0);
    point_features = at::empty(
        {num_points, num_channels}, voxel_features.options());
    indices = at::empty({num_points, 8}, indices_options);
    weights = at::empty({num_points, 8}, voxel_features.options());
  } else {
    auto batch_size = points.size(0);
    auto num_points = points.size(1);
    point_features = at::empty(
        {batch_size, num_points, num_channels}, voxel_features.options());
    indices = at::empty({batch_size, num_points, 8}, indices_options);
    weights = at::empty({batch_size, num_points, 8}, voxel_features.options());
  }

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.type(), "sparse_ops::devoxelize::cuda::trilinear_devoxelize", [&] {
    if (batch_indices.has_value()) {
      trilinear_devoxelize_impl<scalar_t, int64_t>(
          point_features, indices, weights, points, batch_indices,
          voxel_size, points_range_min, points_range_max,
          voxel_coords, voxel_features, voxel_batch_indices);
    }
  });

  return {point_features, indices, weights};
}

template <typename scalar_t, typename index_t>
void trilinear_devoxelize_backward_impl(
    at::Tensor& grad_inputs,
    const at::Tensor& grad_outputs,
    const at::Tensor& indices,
    const at::Tensor& weights) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(ThrustAllocator()).on(stream);

  int64_t num_points = 0;
  if (grad_outputs.dim() == 3) {
    num_points = grad_outputs.size(0) * grad_outputs.size(1);
  } else {
    num_points = grad_outputs.size(0);
  }
  int64_t num_channels = grad_outputs.size(-1);

  auto grad_inputs_ptr = grad_inputs.data_ptr<scalar_t>();

  const auto* const grad_outputs_ptr = grad_outputs.data_ptr<scalar_t>();
  const auto* const indices_ptr = indices.data_ptr<index_t>();
  const auto* const weights_ptr = weights.data_ptr<scalar_t>();

  thrust_impl::trilinear_devoxelize_backward<scalar_t, index_t>(
      policy, grad_inputs_ptr, grad_outputs_ptr,
      indices_ptr, weights_ptr, num_points, num_channels);
}

at::Tensor trilinear_devoxelize_backward(
    const at::Tensor& grad_outputs,
    const at::Tensor& indices,
    const at::Tensor& weights,
    int64_t num_voxels) {
  TORCH_CHECK(grad_outputs.is_cuda(), "grad_outputs must be a CUDA tensor");
  TORCH_CHECK(indices.is_cuda(), "indices must be a CUDA tensor");
  TORCH_CHECK(weights.is_cuda(), "weights must be a CUDA tensor");

  auto num_channels = grad_outputs.size(-1);

  auto grad_inputs = at::zeros(
      {num_voxels, num_channels}, grad_outputs.options());

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_outputs.type(), "sparse_ops::devoxelize::cuda::trilinear_devoxelize_backward", [&] {
    trilinear_devoxelize_backward_impl<scalar_t, int64_t>(
        grad_inputs, grad_outputs, indices, weights);
  });

  return grad_inputs;
}

TORCH_LIBRARY_IMPL(sparse_ops, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME(
      "sparse_ops::trilinear_devoxelize"), TORCH_FN(trilinear_devoxelize));
  m.impl(TORCH_SELECTIVE_NAME(
      "sparse_ops::trilinear_devoxelize_backward"),
      TORCH_FN(trilinear_devoxelize_backward));
}

} // namespace sparse_ops::devoxelize::cuda
