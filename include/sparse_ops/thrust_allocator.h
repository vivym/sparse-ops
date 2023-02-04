#pragma once

#include <thrust/device_malloc_allocator.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace sparse_ops {

#define UNUSED(x) (void)(x)

/// Allocator for Thrust to re-route its internal device allocations
/// to the THC allocator
class ThrustAllocator {
public:
  using value_type = char;

  char* allocate(std::size_t n) {
    return reinterpret_cast<char*>(
        c10::cuda::CUDACachingAllocator::raw_alloc(n));
  }

  void deallocate(value_type* p, std::size_t n) {
    UNUSED(n);
    c10::cuda::CUDACachingAllocator::raw_delete(p);
  }
};

template <typename T>
class DeviceVectorAllocator : public thrust::device_malloc_allocator<T> {
public:
  using super_t = thrust::device_malloc_allocator<T>;
  using pointer = typename super_t::pointer;
  using size_type = typename super_t::size_type;

  pointer allocate(size_type n) {
    T* raw_ptr = reinterpret_cast<T*>(
        c10::cuda::CUDACachingAllocator::raw_alloc(n * sizeof(T)));
    return pointer(raw_ptr);
  }

  void deallocate(pointer p, size_type n) {
    UNUSED(n);
    c10::cuda::CUDACachingAllocator::raw_delete(p.get());
  }
};

#undef UNUSED

} // namespace sparse_ops
