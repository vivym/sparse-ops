#pragma once

#include <tuple>
#include <torch/types.h>

namespace sparse_ops::reduce {

at::Tensor reduce_by_key(at::Tensor values, at::Tensor keys, int64_t op);

namespace cuda {

at::Tensor reduce_by_key(at::Tensor values, at::Tensor keys, int64_t op);

} // namespace cuda

} // namespace sparse_ops::reduce
