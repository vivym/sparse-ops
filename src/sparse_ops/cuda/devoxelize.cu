#include <torch/script.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <cuco/static_map.cuh>
#include <cuda/std/atomic>
#include "sparse_ops/devoxelize.h"
#include "sparse_ops/thrust/devoxelize.hpp"
#include "sparse_ops/thrust/misc.hpp"
#include "sparse_ops/thrust_allocator.h"

namespace sparse_ops::devoxelize::cuda {

template <typename T>
using ThrustVector = thrust::device_vector<T, DeviceVectorAllocator<T>>;

template <typename T>
class cuda_allocator {
 public:
  using value_type = T;  ///< Allocator's value type

  cuda_allocator() = default;

  /**
   * @brief Copy constructor.
   */
  template <class U>
  cuda_allocator(cuda_allocator<U> const&) noexcept
  {
  }

  /**
   * @brief Allocates storage for `n` objects of type `T` using `cudaMalloc`.
   *
   * @param n The number of objects to allocate storage for
   * @return Pointer to the allocated storage
   */
  value_type* allocate(std::size_t n)
  {
    return reinterpret_cast<value_type*>(
        c10::cuda::CUDACachingAllocator::raw_alloc(n));
  }

  /**
   * @brief Deallocates storage pointed to by `p`.
   *
   * @param p Pointer to memory to deallocate
   */
  // void deallocate(value_type* p, std::size_t) { CUCO_CUDA_TRY(cudaFree(p)); }

  void deallocate(value_type* p, std::size_t) {
    c10::cuda::CUDACachingAllocator::raw_delete(p);
  }
};

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
    const at::Tensor& voxel_batch_indices,
    double hash_table_load_factor) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(ThrustAllocator()).on(stream);

  if (!batch_indices.has_value()) {
    auto batch_size = point_coords.size(0);
    auto num_points = point_coords.size(1);

    auto indices_options = point_coords.options().dtype(at::kLong);
    auto tmp = at::empty({batch_size * num_points}, indices_options);
    sparse_ops::thrust_impl::generate_batch_indices<index_t>(
        policy, tmp.data_ptr<index_t>(), batch_size, num_points);
    batch_indices = tmp;
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

  auto invalid_key = std::numeric_limits<key_type>::max();
  auto invalid_value = std::numeric_limits<value_type>::max();

  std::size_t capacity = static_cast<double>(num_voxels) / hash_table_load_factor;

  cuco::static_map<key_type, value_type, ::cuda::thread_scope_device, cuda_allocator<char>> map{
    capacity, cuco::empty_key{invalid_key}, cuco::empty_value{invalid_value}};

  {
    ThrustVector<thrust::tuple<key_type, value_type>> kv_pairs_vec(num_voxels);
    thrust_impl::compute_hash_table_kv_pairs<index_t, 3>(
        policy, kv_pairs_vec.data().get(), voxel_coords_ptr, voxel_batch_indices_ptr,
        num_voxels, batch_stride, voxel_strides);

    map.insert(kv_pairs_vec.begin(), kv_pairs_vec.end());
  }

  thrust_impl::compute_hash_table_queries_and_weights<scalar_t, index_t, 3>(
      policy, indices_ptr, weights_ptr, point_coords_ptr, batch_indices_ptr,
      num_points, voxel_size_vec, points_range_min_vec,
      batch_stride, voxel_strides, voxel_extents);

  map.find(indices_ptr, indices_ptr + num_points * 8, indices_ptr);

  thrust_impl::trilinear_interpolate(
      policy, point_features_ptr, indices_ptr, weights_ptr,
      voxel_features_ptr, num_points, num_channels, invalid_value);
}

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
  TORCH_CHECK(points.is_cuda(), "The point_coords must be a CUDA tensor.");
  TORCH_CHECK(points.is_contiguous(), "The points must be a contiguous tensor.");

  if (batch_indices.has_value()) {
    TORCH_CHECK(points.dim() == 2, "The points must be a 2D tensor.");
    TORCH_CHECK(batch_indices.value().dim() == 1,
                "The batch_indices must be a 1D tensor.");
    TORCH_CHECK(batch_indices.value().is_cuda(),
                "The batch_indices must be a CUDA tensor.");
    TORCH_CHECK(batch_indices.value().is_contiguous(),
                "The batch_indices must be a contiguous tensor.");
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

  TORCH_CHECK(voxel_size.is_contiguous(),
              "The voxel_size must be a contiguous tensor.");
  TORCH_CHECK(points_range_min.is_contiguous(),
              "The points_range_min must be a contiguous tensor.");
  TORCH_CHECK(points_range_max.is_contiguous(),
              "The points_range_max must be a contiguous tensor.");

  TORCH_CHECK(voxel_coords.is_cuda(),
              "The voxel_coords must be a CUDA tensor.");
  TORCH_CHECK(voxel_features.is_cuda(),
              "The voxel_features must be a CUDA tensor.");
  TORCH_CHECK(voxel_batch_indices.is_cuda(),
              "The voxel_batch_indices must be a CUDA tensor.");

  TORCH_CHECK(voxel_coords.is_contiguous(),
              "The voxel_coords must be a contiguous tensor.");
  TORCH_CHECK(voxel_features.is_contiguous(),
              "The voxel_features must be a contiguous tensor.");
  TORCH_CHECK(voxel_batch_indices.is_contiguous(),
              "The voxel_batch_indices must be a contiguous tensor.");

  TORCH_CHECK(voxel_coords.dim() == 2,
              "The voxel_coords must be a 2D tensor.");
  TORCH_CHECK(voxel_features.dim() == 2,
              "The voxel_features must be a 2D tensor.");
  TORCH_CHECK(voxel_batch_indices.dim() == 1,
              "The voxel_batch_indices must be a 1D tensor.");

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
    trilinear_devoxelize_impl<scalar_t, int64_t>(
        point_features, indices, weights, points, batch_indices,
        voxel_size, points_range_min, points_range_max,
        voxel_coords, voxel_features, voxel_batch_indices, hash_table_load_factor);
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
  TORCH_CHECK(grad_outputs.is_contiguous(),
              "The grad_outputs must be a contiguous tensor.");
  TORCH_CHECK(grad_outputs.dim() == 2 || grad_outputs.dim() == 3,
              "The grad_outputs must be a 2D or 3D tensor.");

  TORCH_CHECK(indices.is_contiguous(),
              "The indices must be a contiguous tensor.");
  TORCH_CHECK(indices.dim() == grad_outputs.dim(),
              "The indices must be a 2D or 3D tensor.");

  TORCH_CHECK(weights.is_contiguous(),
              "The weights must be a contiguous tensor.");
  TORCH_CHECK(weights.dim() == grad_outputs.dim(),
              "The weights must be a 2D or 3D tensor.");

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
