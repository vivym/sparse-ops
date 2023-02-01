#pragma once

#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>
#include <thrust/for_each.h>
#include <thrust/memory.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "sparse_ops/fixed_vec.h"
#include "sparse_ops/voxelize.h"

namespace sparse_ops::voxelize::thrust_impl {

template <typename scalar_t, typename index_t, int kDim, typename policy_t>
inline
void compute_voxel_coords_and_indices(
    const policy_t& policy,
    index_t* voxel_coords_ptr,
    index_t* voxel_indices_ptr,
    const scalar_t* const points_ptr,
    const index_t* const batch_indices_ptr,
    const int64_t num_points,
    const FixedVec<scalar_t, kDim>& voxel_size,
    const FixedVec<scalar_t, kDim>& points_range_min,
    const FixedVec<scalar_t, kDim>& points_range_max,
    const int64_t batch_stride,
    const FixedVec<index_t, kDim>& voxel_strides) {
  thrust::for_each(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(num_points),
      [=] __host__ __device__ (index_t i) {
        FixedVec<scalar_t, kDim> p(points_ptr + i * kDim);
        if ((p >= points_range_min && p <= points_range_max).all()) {
          auto voxel_coord = ((p - points_range_min) / voxel_size)
                                  .template cast<index_t>();
          auto voxel_index = batch_indices_ptr[i] * batch_stride +
                             voxel_coord.dot(voxel_strides);
          voxel_coord.store(voxel_coords_ptr + i * kDim);
          voxel_indices_ptr[i] = voxel_index;
        } else {
          voxel_indices_ptr[i] = std::numeric_limits<index_t>::max();
        }
      });
}

template <typename scalar_t, typename index_t, int kDim, typename policy_t>
inline
int64_t voxelize(
    const policy_t& policy,
    index_t* voxel_coords_ptr,
    index_t* voxel_indices_ptr,
    index_t* voxel_batch_indices_ptr,
    index_t* voxel_point_indices_ptr,
    const scalar_t* const points_ptr,
    const index_t* const batch_indices_ptr,
    int64_t num_points,
    const FixedVec<scalar_t, kDim>& voxel_size,
    const FixedVec<scalar_t, kDim>& points_range_min,
    const FixedVec<scalar_t, kDim>& points_range_max) {
  const auto voxel_extents = ceil(
      (points_range_max - points_range_min) / voxel_size)
          .template cast<index_t>();
  FixedVec<index_t, kDim> voxel_strides;
  voxel_strides[0] = 1;
  for (int i = 1; i < kDim; i++) {
    voxel_strides[i] = voxel_strides[i - 1] * voxel_extents[i - 1];
  }
  const index_t batch_stride = voxel_strides[kDim - 1] * voxel_extents[kDim - 1];

  auto tmp_voxel_coords_ptr =
      thrust::malloc<index_t>(policy, num_points * kDim);

  auto tmp_voxel_indices_ptr =
      thrust::malloc<index_t>(policy, num_points);

  printf("#1\n");

  compute_voxel_coords_and_indices<scalar_t, index_t, kDim>(
      policy, tmp_voxel_coords_ptr.get(), tmp_voxel_indices_ptr.get(),
      points_ptr, batch_indices_ptr, num_points, voxel_size,
      points_range_min, points_range_max, batch_stride, voxel_strides);

  printf("#1 done\n");

  thrust::device_vector<index_t> d_coords(tmp_voxel_coords_ptr.get(), tmp_voxel_coords_ptr.get() + num_points * kDim);
  thrust::host_vector<index_t> h_coords = d_coords;
  //  thrust::copy(policy, d_coords.begin(), d_coords.end(), h_coords.begin());

  thrust::device_vector<index_t> d_indices(tmp_voxel_indices_ptr.get(), tmp_voxel_indices_ptr.get() + num_points);
  thrust::host_vector<index_t> h_indices = d_indices;
  // thrust::copy(policy, d_indices.begin(), d_indices.end(), h_indices.begin());

  for (int i = 0; i < num_points; ++i) {
    printf("h_coords: %d, %d, %d\n", (int)h_coords[i * kDim + 0], (int)h_coords[i * kDim + 1], (int)h_coords[i * kDim + 2]);
  }
  for (int i = 0; i < num_points; ++i) {
    printf("tmp_voxel_indices_ptr: %d\n", (int)h_indices[i]);
  }

  printf("#2\n");

  thrust::sequence(policy, voxel_point_indices_ptr, voxel_point_indices_ptr + num_points);

  printf("#3\n");

  thrust::sort_by_key(
      policy, tmp_voxel_indices_ptr, tmp_voxel_indices_ptr + num_points,
      voxel_point_indices_ptr);

  // TODO: filter invalid points (out of range)

  {
    thrust::device_vector<index_t> d_v(voxel_point_indices_ptr, voxel_point_indices_ptr + num_points);
    thrust::host_vector<index_t> h_v = d_v;
    for (int i = 0; i < num_points; ++i) {
      printf("voxel_point_indices_ptr: %d\n", (int)h_v[i]);
    }
  }

  printf("#4\n");

  thrust::gather(
      policy, voxel_point_indices_ptr, voxel_point_indices_ptr + num_points,
      tmp_voxel_indices_ptr, voxel_indices_ptr);

  {
    thrust::device_vector<index_t> d_v(voxel_indices_ptr, voxel_indices_ptr + num_points);
    thrust::host_vector<index_t> h_v = d_v;
    for (int i = 0; i < num_points; ++i) {
      printf("voxel_indices_ptr: %d\n", (int)h_v[i]);
    }
  }

  printf("#5\n");

  auto unique_voxel_point_indices_ptr =
      thrust::malloc<index_t>(policy, num_points);

  auto new_end = thrust::unique_by_key_copy(
      policy, voxel_indices_ptr, voxel_indices_ptr + num_points, voxel_point_indices_ptr,
      voxel_indices_ptr, unique_voxel_point_indices_ptr.get());
  auto num_voxels = thrust::distance(new_end.first, new_end.second);
  num_voxels = num_points;

  printf("#6 %d\n", (int)num_voxels);

  {
    thrust::device_vector<index_t> d_v(voxel_point_indices_ptr, voxel_point_indices_ptr + num_points);
    thrust::host_vector<index_t> h_v = d_v;
    for (int i = 0; i < num_points; ++i) {
      printf("voxel_point_indices_ptr: %d\n", (int)h_v[i]);
    }
  }

  {
    thrust::device_vector<index_t> d_v(unique_voxel_point_indices_ptr.get(), unique_voxel_point_indices_ptr.get() + num_points);
    thrust::host_vector<index_t> h_v = d_v;
    for (int i = 0; i < num_points; ++i) {
      printf("unique_voxel_point_indices_ptr: %d\n", (int)h_v[i]);
    }
  }

  thrust::gather(
      policy,
      unique_voxel_point_indices_ptr, unique_voxel_point_indices_ptr + num_voxels,
      reinterpret_cast<FixedVec<index_t, kDim>*>(tmp_voxel_coords_ptr.get()),
      reinterpret_cast<FixedVec<index_t, kDim>*>(voxel_coords_ptr));

  printf("#7\n");

  thrust::gather(
      policy, unique_voxel_point_indices_ptr, unique_voxel_point_indices_ptr + num_voxels,
      batch_indices_ptr, voxel_batch_indices_ptr);

  printf("#8\n");

  // TODO: thrust::free

  return num_voxels;
}

} // sparse_ops::voxelize::thrust_impl
