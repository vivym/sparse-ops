#include <torch/script.h>
#include <thrust/execution_policy.h>
#include "sparse_ops/voxelize.h"
#include "sparse_ops/thrust/voxelize.hpp"
#include "sparse_ops/thrust/misc.hpp"
#include "sparse_ops/thrust_allocator.h"

namespace sparse_ops::voxelize::cuda {

template <typename scalar_t, typename index_t>
void voxelize_impl(
    at::Tensor& voxel_coords,
    at::Tensor& voxel_indices,
    at::Tensor& voxel_batch_indices,
    at::Tensor& voxel_point_indices,
    const at::Tensor& points,
    c10::optional<at::Tensor> batch_indices,
    const at::Tensor& voxel_size,
    const at::Tensor& points_range_min,
    const at::Tensor& points_range_max) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(ThrustAllocator<char>()).on(stream);

  auto num_points = batch_indices.has_value()
                  ? points.size(0)
                  : points.size(0) * points.size(1);

  auto voxel_coords_ptr = voxel_coords.data_ptr<index_t>();
  auto voxel_indices_ptr = voxel_indices.data_ptr<index_t>();
  auto voxel_batch_indices_ptr = voxel_batch_indices.data_ptr<index_t>();
  auto voxel_point_indices_ptr = voxel_point_indices.data_ptr<index_t>();

  if (!batch_indices.has_value()) {
    auto batch_size = points.size(0);
    auto num_points = points.size(1);

    auto indices_options = points.options().dtype(at::kLong);
    auto tmp = at::empty({batch_size, num_points}, indices_options);
    sparse_ops::thrust_impl::generate_batch_indices<index_t>(
        policy, tmp.data_ptr<index_t>(), batch_size, num_points);
    batch_indices = tmp;
  }

  const auto* const points_ptr = points.data_ptr<scalar_t>();
  const auto* const batch_indices_ptr = batch_indices.value().data_ptr<index_t>();

  switch (points.size(-1)) {
#define CASE(NDIM)                                                        \
  case NDIM: {                                                            \
    FixedVec<scalar_t, NDIM> voxel_size_vec;                              \
    voxel_size_vec.load(voxel_size.data_ptr<scalar_t>());                 \
    FixedVec<scalar_t, NDIM> points_range_min_vec;                        \
    points_range_min_vec.load(points_range_min.data_ptr<scalar_t>());     \
    FixedVec<scalar_t, NDIM> points_range_max_vec;                        \
    points_range_max_vec.load(points_range_max.data_ptr<scalar_t>());     \
    auto num_voxels = thrust_impl::voxelize<scalar_t, index_t, NDIM>(     \
        policy, voxel_coords_ptr, voxel_indices_ptr,                      \
        voxel_batch_indices_ptr, voxel_point_indices_ptr,                 \
        points_ptr, batch_indices_ptr, num_points,                        \
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

  auto num_points = batch_indices.has_value()
                  ? points.size(0)
                  : points.size(0) * points.size(1);
  auto point_dim = points.size(-1);

  auto indices_options = points.options().dtype(at::kLong);
  auto voxel_coords = at::empty({num_points, point_dim}, indices_options);
  auto voxel_indices = at::empty({num_points}, indices_options);
  auto voxel_batch_indices = at::empty({num_points}, indices_options);
  auto voxel_point_indices = at::empty({num_points}, indices_options);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(points.type(), "sparse_ops::voxelize::cuda::voxelize", [&] {
    voxelize_impl<scalar_t, int64_t>(
        voxel_coords, voxel_indices, voxel_batch_indices, voxel_point_indices,
        points, batch_indices, voxel_size, points_range_min, points_range_max);
  });

  return {voxel_coords, voxel_indices, voxel_batch_indices, voxel_point_indices};
}

TORCH_LIBRARY_IMPL(sparse_ops, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("sparse_ops::voxelize"), TORCH_FN(voxelize));
}

} // namespace sparse_ops::voxelize::cuda
