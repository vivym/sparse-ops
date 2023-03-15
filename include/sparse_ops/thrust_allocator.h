#pragma once

#include <thrust/device_malloc_allocator.h>
#include <c10/cuda/CUDACachingAllocator.h>

namespace sparse_ops {

/// Allocator for Thrust to re-route its internal device allocations
/// to the THC allocator
template <typename T>
class ThrustAllocator {
public:
  using value_type = T;

  ThrustAllocator() = default;

  /**
   * @brief Copy constructor.
   */
  template <class U>
  ThrustAllocator(ThrustAllocator<U> const&) noexcept {}

  value_type* allocate(std::size_t n) {
    return reinterpret_cast<value_type*>(
        c10::cuda::CUDACachingAllocator::raw_alloc(sizeof(value_type) * n));
  }

  void deallocate(value_type* p, std::size_t) {
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

  void deallocate(pointer p, size_type) {
    c10::cuda::CUDACachingAllocator::raw_delete(p.get());
  }
};

} // namespace sparse_ops
