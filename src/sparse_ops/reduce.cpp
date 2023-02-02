#include <torch/script.h>
#include "sparse_ops/reduce.h"

namespace sparse_ops::reduce {

at::Tensor reduce_by_key(at::Tensor values, at::Tensor keys, int64_t binary_op) {
  TORCH_CHECK(values.dim() == 2, "The values must be a 2D tensor.");
  TORCH_CHECK(keys.dim() == 1, "The keys must be a 1D tensor.");

  TORCH_CHECK(0 < values.size(1) && values.size(1) < 9,
              "The number of channels must be in [1, ..., 8].");
  TORCH_CHECK(values.size(0) == keys.size(0), "values.size(0) != keys.size(0)");

  static auto op = c10::Dispatcher::singleton()
                       .findSchemaOrThrow("sparse_ops::reduce_by_key", "")
                       .typed<decltype(reduce_by_key)>();
  return op.call(values, keys, binary_op);
}

TORCH_LIBRARY_FRAGMENT(sparse_ops, m) {
  m.def(TORCH_SELECTIVE_SCHEMA(
      "sparse_ops::reduce_by_key(Tensor values, Tensor keys, int op) -> Tensor"));
}

} // namespace sparse_ops::reduce
