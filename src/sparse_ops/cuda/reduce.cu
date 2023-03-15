#include <torch/script.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include "sparse_ops/reduce.h"
#include "sparse_ops/thrust/reduce.hpp"
#include "sparse_ops/thrust_allocator.h"
#include "sparse_ops/fixed_vec.h"

namespace sparse_ops::reduce::cuda {

template <typename scalar_t, int kDim>
struct plus {
  using vec_t = FixedVec<scalar_t, kDim>;

  __host__ __device__
  constexpr vec_t operator()(const vec_t &lhs, const vec_t &rhs) const {
    return lhs + rhs;
  }
};

template <typename scalar_t, int kDim>
struct minimum {
  using vec_t = FixedVec<scalar_t, kDim>;

  __host__ __device__
  constexpr vec_t operator()(const vec_t &lhs, const vec_t &rhs) const {
    vec_t res;
    for (int i = 0; i < kDim; ++i) {
      res[i] = min(lhs[i], rhs[i]);
    }
    return res;
  }
};

template <typename scalar_t, int kDim>
struct maximum {
  using vec_t = FixedVec<scalar_t, kDim>;

  __host__ __device__
  constexpr vec_t operator()(const vec_t &lhs, const vec_t &rhs) const {
    vec_t res;
    for (int i = 0; i < kDim; ++i) {
      res[i] = max(lhs[i], rhs[i]);
    }
    return res;
  }
};

template <typename scalar_t, typename index_t>
void reduce_by_key_impl(
    at::Tensor& results, at::Tensor values, at::Tensor keys, int64_t op) {
  auto stream = at::cuda::getCurrentCUDAStream().stream();
  auto policy = thrust::cuda::par(ThrustAllocator<char>()).on(stream);

  int64_t num_values = values.size(0);

  auto results_ptr = results.data_ptr<scalar_t>();

  const auto* const values_ptr = values.data_ptr<scalar_t>();
  const auto* const keys_ptr = keys.data_ptr<index_t>();

  switch (values.size(1)) {
#define CASE(NDIM)                                                                        \
  case NDIM: {                                                                            \
    int64_t num_segments = 0;                                                             \
    switch (op) {                                                                         \
    case 0:                                                                               \
      num_segments = thrust_impl::reduce_by_key<scalar_t, index_t, NDIM>(                 \
          policy, results_ptr, values_ptr, keys_ptr, num_values,                          \
          plus<scalar_t, NDIM>());                                                        \
      break;                                                                              \
    case 1:                                                                               \
      num_segments = thrust_impl::reduce_by_key<scalar_t, index_t, NDIM>(                 \
          policy, results_ptr, values_ptr, keys_ptr, num_values,                          \
          minimum<scalar_t, NDIM>());                                                     \
      break;                                                                              \
    case 2:                                                                               \
      num_segments = thrust_impl::reduce_by_key<scalar_t, index_t, NDIM>(                 \
          policy, results_ptr, values_ptr, keys_ptr, num_values,                          \
          maximum<scalar_t, NDIM>());                                                     \
      break;                                                                              \
    }                                                                                     \
    results = results.slice(0, 0, num_segments);                                          \
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

at::Tensor reduce_by_key(at::Tensor values, at::Tensor keys, int64_t op) {
  TORCH_CHECK(values.is_contiguous(),
              "The values must be a is_contiguous tensor.");
  TORCH_CHECK(keys.is_contiguous(),
              "The keys must be a is_contiguous tensor.");

  TORCH_CHECK(values.is_cuda(), "The values must be a CUDA tensor.");
  TORCH_CHECK(keys.is_cuda(), "The keys must be a CUDA tensor.");

  TORCH_CHECK(values.dim() == 2, "The values must be a 2D tensor.");
  TORCH_CHECK(keys.dim() == 1, "The keys must be a 1D tensor.");

  TORCH_CHECK(0 < values.size(1) && values.size(1) < 9,
              "The number of channels must be in [1, ..., 8].");
  TORCH_CHECK(values.size(0) == keys.size(0),
              "values.size(0) != keys.size(0)");

  auto results = at::empty_like(values);

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(values.type(), "sparse_ops::reduce::cuda::reduce_by_key", [&] {
    reduce_by_key_impl<scalar_t, int64_t>(results, values, keys, op);
  });

  return results;
}

TORCH_LIBRARY_IMPL(sparse_ops, CUDA, m) {
  m.impl(TORCH_SELECTIVE_NAME("sparse_ops::reduce_by_key"), TORCH_FN(reduce_by_key));
}

} // namespace sparse_ops::reduce::cuda
