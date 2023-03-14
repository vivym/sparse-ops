#include <torch/script.h>
#include "sparse_ops/reduce.h"

namespace sparse_ops::reduce {

at::Tensor reduce_by_key(at::Tensor values, at::Tensor keys, int64_t binary_op) {
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
