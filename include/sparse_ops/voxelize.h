#pragma once

#include <tuple>
#include <torch/types.h>
#include "sparse_ops/fixed_vec.h"

namespace sparse_ops::voxelize {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> voxelize(
    at::Tensor points,
    c10::optional<at::Tensor> batch_indices,
    at::Tensor voxel_size,
    at::Tensor points_range_min,
    at::Tensor points_range_max);

namespace cuda {

std::tuple<at::Tensor, at::Tensor, at::Tensor, at::Tensor> voxelize(
    at::Tensor points,
    c10::optional<at::Tensor> batch_indices,
    at::Tensor voxel_size,
    at::Tensor points_range_min,
    at::Tensor points_range_max);

} // namespace cuda

namespace thrust_impl {

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
    const FixedVec<scalar_t, kDim>& points_range_max);

} // namespace thrust_impl

} // namespace sparse_ops::voxelize
