#include <thrust/iterator/counting_iterator.h>
#include <thrust/for_each.h>
#include <thrust/memory.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/unique.h>

#include "sparse_ops/voxelize.h"

namespace sparse_ops::voxelize::thrust_impl {

template <typename scalar_t, typename index_t, int kDim, typename policy_t>
void compute_voxel_coords_and_indices(
    const policy_t& policy,
    index_t* voxel_coords_ptr,
    index_t* voxel_indices_ptr,
    const scalar_t* const points_ptr,
    const scalar_t* const batch_indices_ptr,
    int64_t num_points,
    const FixedVec<index_t, kDim>& voxel_size,
    const FixedVec<index_t, kDim>& points_range_min,
    const FixedVec<index_t, kDim>& points_range_max,
    int64_t batch_stride,
    const FixedVec<index_t, kDim>& voxel_strides) {
  thrust::for_each(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(num_points),
      [=] __host__ __device__ (index_t i) {
        FixedVec<scalar_t, kDim> p(points_ptr + i * kDim);
        auto voxel_coords = (p - points_range_min) / voxel_size;

        auto voxel_indices = batch_indices_ptr[i] * batch_stride + \
                             voxel_coords.dot(voxel_strides);

        voxel_coords.store(voxel_coords_ptr + i * kDim);
        voxel_indices_ptr[i] = voxel_indices;
      });
}

template <typename scalar_t, typename index_t, int kDim, typename policy_t>
int64_t voxelize(
    const policy_t& policy,
    index_t* voxel_coords_ptr,
    index_t* voxel_point_indices_ptr,
    index_t* voxel_indices_ptr,
    index_t* voxel_batch_indices_ptr,
    const scalar_t* const points_ptr,
    const index_t* const batch_indices_ptr,
    int64_t num_points,
    const FixedVec<scalar_t, kDim>& voxel_size,
    const FixedVec<scalar_t, kDim>& points_range_min,
    const FixedVec<scalar_t, kDim>& points_range_max) {
  const auto voxel_extents = (points_range_max - points_range_min) / voxel_size;
  const FixedVec<index_t, kDim> voxel_strides;
  voxel_strides[0] = 1;
  for (int i = 1; i < kDim; i++) {
    voxel_strides[i] = voxel_strides[i - 1] * voxel_extents[i];
  }
  const auto batch_stride = voxel_strides[kDim - 1] * voxel_extents[kDim - 1];

  index_t* tmp_voxel_coords_ptr = \
      thrust::malloc<index_t>(policy, num_points * kDim);

  index_t* tmp_voxel_indices_ptr = \
      thrust::malloc<index_t>(policy, num_points);

  compute_voxel_coords_and_indices(
      policy, tmp_voxel_coords_ptr, tmp_voxel_indices_ptr,
      points_ptr, batch_indices_ptr, num_points, voxel_size,
      points_range_min, points_range_max, batch_stride, voxel_strides);

  thrust::sequence(policy, voxel_point_indices_ptr, voxel_point_indices_ptr + num_points);

  thrust::sort_by_key(
      policy, tmp_voxel_indices_ptr, tmp_voxel_indices_ptr + num_points,
      voxel_point_indices_ptr);

  thrust::gather(
      policy, voxel_point_indices_ptr, voxel_point_indices_ptr + num_points,
      tmp_voxel_indices_ptr, voxel_indices_ptr);

  index_t* unique_voxel_point_indices_ptr = \
      thrust::malloc<index_t>(policy, num_points);

  auto new_end = thrust::unique_by_key(
      policy, voxel_indices_ptr, voxel_indices_ptr + num_points,
      unique_voxel_point_indices_ptr);
  int64_t num_voxels = thrust::distance(new_end.first, new_end.second);

  thrust::gather(
      policy,
      unique_voxel_point_indices_ptr, unique_voxel_point_indices_ptr + num_voxels,
      reinterpret_cast<FixedVec<index_t, kDim>*>(tmp_voxel_coords_ptr),
      reinterpret_cast<FixedVec<index_t, kDim>*>(voxel_coords_ptr));

  thrust::gather(
      policy, unique_voxel_point_indices_ptr, unique_voxel_point_indices_ptr + num_voxels,
      batch_indices_ptr, voxel_batch_indices_ptr);

  return num_voxels;
}

} // sparse_ops::voxelize::thrust_impl
