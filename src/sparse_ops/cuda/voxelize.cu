#include <torch/script.h>
#include <thrust/execution_policy.h>
#include "sparse_ops/voxelize.h"
#include "sparse_ops/thrust/voxelize.hpp"
#include "sparse_ops/thrust_allocator.h"

namespace sparse_ops::voxelize::cuda {

template <typename scalar_t, typename index_t>
void voxelize_sparse(
    at::Tensor& voxel_coords,
    at::Tensor& voxel_indices,
    at::Tensor& voxel_batch_indices,
    at::Tensor& voxel_point_indices,
    const at::Tensor& points,
    const at::Tensor& batch_indices,
    const at::Tensor& voxel_size,
    const at::Tensor& points_range_min,
    const at::Tensor& points_range_max) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(ThrustAllocator()).on(stream);

  auto voxel_coords_ptr = voxel_coords.data_ptr<index_t>();
  auto voxel_indices_ptr = voxel_indices.data_ptr<index_t>();
  auto voxel_batch_indices_ptr = voxel_batch_indices.data_ptr<index_t>();
  auto voxel_point_indices_ptr = voxel_point_indices.data_ptr<index_t>();

  const auto* const points_ptr = points.data_ptr<scalar_t>();
  const auto* const batch_indices_ptr = batch_indices.data_ptr<index_t>();

  switch (points.size(1)) {
#define CASE(NDIM)                                                        \
  case NDIM: {                                                            \
    const FixedVec<scalar_t, NDIM> voxel_size_vec(                        \
        voxel_size.data_ptr<scalar_t>());                                 \
    const FixedVec<scalar_t, NDIM> points_range_min_vec(                  \
        points_range_min.data_ptr<scalar_t>());                           \
    const FixedVec<scalar_t, NDIM> points_range_max_vec(                  \
        points_range_max.data_ptr<scalar_t>());                           \
    auto num_voxels = thrust_impl::voxelize<scalar_t, index_t, NDIM>(     \
        policy, voxel_coords_ptr, voxel_indices_ptr,                      \
        voxel_batch_indices_ptr, voxel_point_indices_ptr,                 \
        points_ptr, batch_indices_ptr, points.size(0),                    \
        voxel_size_vec, points_range_min_vec, points_range_max_vec);      \
    voxel_coords = voxel_coords.slice(0, 0, num_voxels);                  \
    voxel_batch_indices = voxel_batch_indices.slice(0, 0, num_voxels);    \
  } break;
  CASE(1)
  CASE(2)
  CASE(3)
  CASE(4)
  CASE(5)
  CASE(6)
  CASE(7)
  CASE(8)
#undef CASE
  }
}

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> voxelize(
    at::Tensor points,
    c10::optional<at::Tensor> batch_indices,
    at::Tensor voxel_size,
    at::Tensor points_range_min,
    at::Tensor points_range_max) {
  TORCH_CHECK(points.is_cuda(), "The points must be a CUDA tensor.");

  if (batch_indices.has_value()) {
    TORCH_CHECK(batch_indices.value().is_cuda(), "The batch_indices must be a CUDA tensor.");
  }

  auto num_points = batch_indices.has_value()
                  ? points.size(0)
                  : points.size(0) * points.size(1);
  auto point_dim = points.size(-1);

  auto indices_options = points.options().dtype(at::kLong);
  at::Tensor voxel_coords = at::empty({num_points, point_dim}, indices_options);
  at::Tensor voxel_indices = at::empty({num_points}, indices_options);
  at::Tensor voxel_batch_indices = at::empty({num_points}, indices_options);
  at::Tensor voxel_point_indices = at::empty({num_points}, indices_options);

  AT_DISPATCH_FLOATING_TYPES(points.type(), "sparse_ops::voxelize::cuda::voxelize", [&] {
    if (batch_indices.has_value()) {
      voxelize_sparse<scalar_t, int64_t>(
          voxel_coords, voxel_indices, voxel_batch_indices, voxel_point_indices,
          points.contiguous(), batch_indices.value().contiguous(), voxel_size.contiguous(),
          points_range_min.contiguous(), points_range_max.contiguous());
    } else {
    }
  });

  return {voxel_coords, voxel_indices, voxel_batch_indices, voxel_point_indices};
}

TORCH_LIBRARY_IMPL(sparse_ops, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("sparse_ops::voxelize"), TORCH_FN(voxelize));
}

} // namespace sparse_ops::voxelize::cuda
