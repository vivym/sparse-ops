#pragma once
#include <cuda_fp16.h>
#include <torch/script.h>

__device__ __forceinline__
void atomicAdd(c10::Half* address, c10::Half val) {
  atomicAdd(reinterpret_cast<__half*>(address), static_cast<__half>(val));
}
