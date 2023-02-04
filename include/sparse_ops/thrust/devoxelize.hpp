#pragma once

#include <thrust/transform.h>
#include <thrust/for_each.h>
#include <thrust/iterator/counting_iterator.h>

#include "sparse_ops/fixed_vec.h"
#include "sparse_ops/devoxelize.h"
#include "sparse_ops/torch_utils.h"

namespace sparse_ops::devoxelize::thrust_impl {

template <typename index_t, int kDim, typename policy_t, typename pair_t>
inline
void compute_hash_table_kv_pairs(
    const policy_t& policy,
    pair_t* kv_pairs_ptr,
    const index_t* const voxel_coords_ptr,
    const index_t* const voxel_batch_indices_ptr,
    const int64_t num_voxels,
    const int64_t batch_stride,
    const FixedVec<index_t, kDim>& voxel_strides) {
  thrust::transform(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(num_voxels),
      kv_pairs_ptr,
      [=] __host__ __device__ (index_t i) {
        FixedVec<index_t, kDim> voxel_coord;
        voxel_coord.load(voxel_coords_ptr + i * kDim);
        const index_t voxel_index = voxel_batch_indices_ptr[i] * batch_stride +
                                    voxel_coord.dot(voxel_strides);
        return pair_t{voxel_index, i};
      });
}

template <typename scalar_t, typename index_t, int kDim, typename policy_t>
inline
void compute_hash_table_queries_and_weights(
    const policy_t& policy,
    index_t* queries_ptr,
    scalar_t* weights_ptr,
    const scalar_t* const point_coords_ptr,
    const index_t* const batch_indices_ptr,
    const int64_t num_points,
    const FixedVec<scalar_t, kDim>& voxel_size,
    const FixedVec<scalar_t, kDim>& points_range_min,
    const int64_t batch_stride,
    const FixedVec<index_t, kDim>& voxel_strides,
    const FixedVec<index_t, kDim>& voxel_extents) {
  thrust::for_each(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(num_points),
      [=] __host__ __device__ (index_t i) {
        FixedVec<scalar_t, kDim> p;
        p.load(point_coords_ptr + i * kDim);

        p = (p - points_range_min) / voxel_size;

        const auto p_lo = floor(p);
        const auto p_hi = ceil(p);
        const auto d_1 = p - p_lo;
        const auto d_0 = static_cast<scalar_t>(1.0) - d_1;

        const auto wgt000 = d_0[0] * d_0[1] * d_0[2];
        const auto wgt001 = d_0[0] * d_0[1] * d_1[2];
        const auto wgt010 = d_0[0] * d_1[1] * d_0[2];
        const auto wgt011 = d_0[0] * d_1[1] * d_1[2];
        const auto wgt100 = d_1[0] * d_0[1] * d_0[2];
        const auto wgt101 = d_1[0] * d_0[1] * d_1[2];
        const auto wgt110 = d_1[0] * d_1[1] * d_0[2];
        const auto wgt111 = d_1[0] * d_1[1] * d_1[2];

        printf("p: %f, %f, %f; p_lo: %f, %f, %f; p_hi: %f, %f, %f\n",
               (float)p[0], (float)p[1], (float)p[2],
               (float)p_lo[0], (float)p_lo[1], (float)p_lo[2],
               (float)p_hi[0], (float)p_hi[1], (float)p_hi[2]);
        printf("d_0: %f, %f, %f; d_1: %f, %f, %f\n",
               (float)d_0[0], (float)d_0[1], (float)d_0[2],
               (float)d_1[0], (float)d_1[1], (float)d_1[2]);

        weights_ptr[i * 8 + 0] = wgt000;
        weights_ptr[i * 8 + 1] = wgt001;
        weights_ptr[i * 8 + 2] = wgt010;
        weights_ptr[i * 8 + 3] = wgt011;
        weights_ptr[i * 8 + 4] = wgt100;
        weights_ptr[i * 8 + 5] = wgt101;
        weights_ptr[i * 8 + 6] = wgt110;
        weights_ptr[i * 8 + 7] = wgt111;

        const index_t offset_x = p_hi[0] >= voxel_extents[0]
                               ? 0
                               : voxel_strides[0];
        const index_t offset_y = p_hi[1] >= voxel_extents[1]
                               ? 0
                               : voxel_strides[1];
        const index_t offset_z = p_hi[2] >= voxel_extents[2]
                               ? 0
                               : voxel_strides[2];

        const auto idx000 = batch_indices_ptr[i] * batch_stride +
                            p_lo.template cast<index_t>().dot(voxel_strides);
        const auto idx001 = idx000 + offset_z;
        const auto idx010 = idx000 + offset_y;
        const auto idx011 = idx010 + offset_z;
        const auto idx100 = idx000 + offset_x;
        const auto idx101 = idx100 + offset_z;
        const auto idx110 = idx100 + offset_y;
        const auto idx111 = idx110 + offset_z;

        queries_ptr[i * 8 + 0] = idx000;
        queries_ptr[i * 8 + 1] = idx001;
        queries_ptr[i * 8 + 2] = idx010;
        queries_ptr[i * 8 + 3] = idx011;
        queries_ptr[i * 8 + 4] = idx100;
        queries_ptr[i * 8 + 5] = idx101;
        queries_ptr[i * 8 + 6] = idx110;
        queries_ptr[i * 8 + 7] = idx111;
      });
}

template <typename scalar_t, typename index_t, typename policy_t>
inline
void trilinear_interpolate(
    const policy_t& policy,
    scalar_t* point_features_ptr,
    const index_t* const indices_ptr,
    const scalar_t* const weights_ptr,
    const scalar_t* const voxel_features_ptr,
    const int64_t num_points,
    const int64_t num_channels,
    const index_t invalid_index) {
  thrust::for_each(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(num_points * num_channels),
      [=] __host__ __device__ (index_t i) {
        const auto point_idx = i / num_channels;
        const auto channel_idx = i % num_channels;

        const auto idx000 = indices_ptr[point_idx * 8 + 0];
        const auto idx001 = indices_ptr[point_idx * 8 + 1];
        const auto idx010 = indices_ptr[point_idx * 8 + 2];
        const auto idx011 = indices_ptr[point_idx * 8 + 3];
        const auto idx100 = indices_ptr[point_idx * 8 + 4];
        const auto idx101 = indices_ptr[point_idx * 8 + 5];
        const auto idx110 = indices_ptr[point_idx * 8 + 6];
        const auto idx111 = indices_ptr[point_idx * 8 + 7];

        const auto wgt000 = weights_ptr[point_idx * 8 + 0];
        const auto wgt001 = weights_ptr[point_idx * 8 + 1];
        const auto wgt010 = weights_ptr[point_idx * 8 + 2];
        const auto wgt011 = weights_ptr[point_idx * 8 + 3];
        const auto wgt100 = weights_ptr[point_idx * 8 + 4];
        const auto wgt101 = weights_ptr[point_idx * 8 + 5];
        const auto wgt110 = weights_ptr[point_idx * 8 + 6];
        const auto wgt111 = weights_ptr[point_idx * 8 + 7];

        scalar_t point_feature = 0;
        if (idx000 != invalid_index) {
          point_feature +=
              voxel_features_ptr[idx000 * num_channels + channel_idx] * wgt000;
        }
        if (idx001 != invalid_index) {
          point_feature +=
              voxel_features_ptr[idx001 * num_channels + channel_idx] * wgt001;
        }
        if (idx010 != invalid_index) {
          point_feature +=
              voxel_features_ptr[idx010 * num_channels + channel_idx] * wgt010;
        }
        if (idx011 != invalid_index) {
          point_feature +=
              voxel_features_ptr[idx011 * num_channels + channel_idx] * wgt011;
        }
        if (idx100 != invalid_index) {
          point_feature +=
              voxel_features_ptr[idx100 * num_channels + channel_idx] * wgt100;
        }
        if (idx101 != invalid_index) {
          point_feature +=
              voxel_features_ptr[idx101 * num_channels + channel_idx] * wgt101;
        }
        if (idx110 != invalid_index) {
          point_feature +=
              voxel_features_ptr[idx110 * num_channels + channel_idx] * wgt110;
        }
        if (idx111 != invalid_index) {
          point_feature +=
              voxel_features_ptr[idx111 * num_channels + channel_idx] * wgt111;
        }

        point_features_ptr[i] = point_feature;
      });
}

template <typename scalar_t, typename index_t, typename policy_t>
inline
void trilinear_devoxelize_backward(
    const policy_t& policy,
    scalar_t* grad_inputs_ptr,
    const scalar_t* const grad_outputs_ptr,
    const index_t* const indices_ptr,
    const scalar_t* const weights_ptr,
    const int64_t num_points,
    const int64_t num_channels) {
  auto invalid_index = std::numeric_limits<index_t>::max();

  thrust::for_each(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(num_points * num_channels),
      [=] __host__ __device__ (index_t i) {
        const auto point_idx = i / num_channels;
        const auto channel_idx = i % num_channels;

        const auto idx000 = indices_ptr[point_idx * 8 + 0];
        const auto idx001 = indices_ptr[point_idx * 8 + 1];
        const auto idx010 = indices_ptr[point_idx * 8 + 2];
        const auto idx011 = indices_ptr[point_idx * 8 + 3];
        const auto idx100 = indices_ptr[point_idx * 8 + 4];
        const auto idx101 = indices_ptr[point_idx * 8 + 5];
        const auto idx110 = indices_ptr[point_idx * 8 + 6];
        const auto idx111 = indices_ptr[point_idx * 8 + 7];

        const auto wgt000 = weights_ptr[point_idx * 8 + 0];
        const auto wgt001 = weights_ptr[point_idx * 8 + 1];
        const auto wgt010 = weights_ptr[point_idx * 8 + 2];
        const auto wgt011 = weights_ptr[point_idx * 8 + 3];
        const auto wgt100 = weights_ptr[point_idx * 8 + 4];
        const auto wgt101 = weights_ptr[point_idx * 8 + 5];
        const auto wgt110 = weights_ptr[point_idx * 8 + 6];
        const auto wgt111 = weights_ptr[point_idx * 8 + 7];

        const auto grad_output = grad_outputs_ptr[i];

        if (idx000 != invalid_index) {
          atomicAdd(
              grad_inputs_ptr + idx000 * num_channels + channel_idx,
              grad_output * wgt000);
        }
        if (idx001 != invalid_index) {
          atomicAdd(
              grad_inputs_ptr + idx001 * num_channels + channel_idx,
              grad_output * wgt001);
        }
        if (idx010 != invalid_index) {
          atomicAdd(
              grad_inputs_ptr + idx010 * num_channels + channel_idx,
              grad_output * wgt010);
        }
        if (idx011 != invalid_index) {
          atomicAdd(
              grad_inputs_ptr + idx011 * num_channels + channel_idx,
              grad_output * wgt011);
        }
        if (idx100 != invalid_index) {
          atomicAdd(
              grad_inputs_ptr + idx100 * num_channels + channel_idx,
              grad_output * wgt100);
        }
        if (idx101 != invalid_index) {
          atomicAdd(
              grad_inputs_ptr + idx101 * num_channels + channel_idx,
              grad_output * wgt101);
        }
        if (idx110 != invalid_index) {
          atomicAdd(
              grad_inputs_ptr + idx110 * num_channels + channel_idx,
              grad_output * wgt110);
        }
        if (idx111 != invalid_index) {
          atomicAdd(
              grad_inputs_ptr + idx111 * num_channels + channel_idx,
              grad_output * wgt111);
        }
      });
}

} // sparse_ops::devoxelize::thrust_impl
