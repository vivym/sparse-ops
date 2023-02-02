#pragma once

#include <thrust/reduce.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include "sparse_ops/fixed_vec.h"
#include "sparse_ops/reduce.h"

namespace sparse_ops::reduce::thrust_impl {

template <typename scalar_t, typename index_t, int kDim, typename op_t, typename policy_t>
inline
int64_t reduce_by_key(
    const policy_t& policy,
    scalar_t* results_ptr,
    const scalar_t* const values_ptr,
    const index_t* const keys_ptr,
    int64_t num_values,
    op_t op) {
  auto new_end = thrust::reduce_by_key(
      policy, keys_ptr, keys_ptr + num_values,
      reinterpret_cast<const FixedVec<scalar_t, kDim>*>(values_ptr),
      thrust::make_discard_iterator(),
      reinterpret_cast<FixedVec<scalar_t, kDim>*>(results_ptr),
      thrust::equal_to<index_t>(), op);

  return thrust::distance(
      reinterpret_cast<FixedVec<scalar_t, kDim>*>(results_ptr), new_end.second);
}

} // sparse_ops::reduce::thrust_impl
