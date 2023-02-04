#pragma once

#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>

namespace sparse_ops::thrust_impl {

template <typename index_t, typename policy_t>
inline
void generate_batch_indices(
    const policy_t& policy,
    index_t* batch_indices_ptr,
    int64_t batch_size,
    int64_t num_points) {
  thrust::transform(
      policy,
      thrust::counting_iterator<index_t>(0),
      thrust::counting_iterator<index_t>(batch_size * num_points),
      batch_indices_ptr,
      [=] __host__ __device__ (index_t i) {
        return i / num_points;
      });
}

} // sparse_ops::thrust_impl
