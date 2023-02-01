#pragma once

#include <cmath>
#include <array>

#ifdef __CUDACC__
#define FN_SPECIFIERS inline __host__ __device__
#else
#define FN_SPECIFIERS inline
#endif

namespace sparse_ops {

template <typename scalar_t, std::size_t kDim>
class FixedVec : public std::array<scalar_t, kDim> {
public:
  FN_SPECIFIERS
  FixedVec() = default;

  FN_SPECIFIERS
  explicit FixedVec(const scalar_t* const ptr) {
    for (int i = 0; i < kDim; ++i) {
      this->operator[](i) = ptr[i];
    }
  }

  FN_SPECIFIERS
  void store(scalar_t* ptr) const {
    for (int i = 0; i < kDim; ++i) {
      ptr[i] = this->operator[](i);
    }
  }

  template <typename target_scalar_t>
  FN_SPECIFIERS
  FixedVec<target_scalar_t, kDim> cast() const {
    FixedVec<target_scalar_t, kDim> res;
    for (int i = 0; i < kDim; ++i) {
      res[i] = target_scalar_t(this->operator[](i));
    }
    return res;
  }

  FN_SPECIFIERS
  FixedVec<scalar_t, kDim> abs() const {
    FixedVec<scalar_t, kDim> res;
    for (int i = 0; i < kDim; ++i) {
      res[i] = std::abs(this->operator[](i));
    }
    return res;
  }

  FN_SPECIFIERS
  scalar_t dot(const FixedVec<scalar_t, kDim>& rhs) const {
    scalar_t res = 0;
    for (int i = 0; i < kDim; ++i) {
      res += this->operator[](i) * rhs[i];
    }
    return res;
  }

  FN_SPECIFIERS
  bool all() const {
    bool res = true;
    for (int i = 0; i < kDim && res; ++i) {
      res = res &&  this->operator[](i);
    }
    return res;
  }

  FN_SPECIFIERS
  bool any() const {
    for (int i = 0; i < kDim; ++i) {
      if (this->operator[](i)) {
        return true;
      }
    }
    return false;
  }
};

template <std::size_t kDim>
FN_SPECIFIERS FixedVec<float, kDim> floor(const FixedVec<float, kDim>& v) {
  FixedVec<float, kDim> res;
  for (int i = 0; i < kDim; ++i) {
    res[i] = floorf(v[i]);
  }
  return res;
}

template <std::size_t kDim>
FN_SPECIFIERS FixedVec<double, kDim> floor(const FixedVec<double, kDim>& v) {
  FixedVec<double, kDim> res;
  for (int i = 0; i < kDim; ++i) {
    res[i] = std::floor(v[i]);
  }
  return res;
}

template <std::size_t kDim>
FN_SPECIFIERS FixedVec<float, kDim> ceil(const FixedVec<float, kDim>& v) {
  FixedVec<float, kDim> res;
  for (int i = 0; i < kDim; ++i) {
    res[i] = ceilf(v[i]);
  }
  return res;
}

template <std::size_t kDim>
FN_SPECIFIERS FixedVec<double, kDim> ceil(const FixedVec<double, kDim>& v) {
  FixedVec<double, kDim> res;
  for (int i = 0; i < kDim; ++i) {
    res[i] = std::ceil(v[i]);
  }
  return res;
}

template <typename scalar_t, std::size_t kDim>
FN_SPECIFIERS FixedVec<scalar_t, kDim> operator- (const FixedVec<scalar_t, kDim>& v) {
  FixedVec<scalar_t, kDim> res;
  for (int i = 0; i < kDim; ++i) {
    res[i] = -v[i];
  }
  return res;
}

template <typename scalar_t, std::size_t kDim>
FN_SPECIFIERS FixedVec<scalar_t, kDim> operator! (const FixedVec<scalar_t, kDim>& v) {
  FixedVec<scalar_t, kDim> res;
  for (int i = 0; i < kDim; ++i) {
    res[i] = !v[i];
  }
  return res;
}

#define DEFINE_OPERATOR(op, opas)                                               \
  template <typename scalar_t, std::size_t kDim>                                \
  FN_SPECIFIERS                                                                 \
  FixedVec<scalar_t, kDim> operator op(const FixedVec<scalar_t, kDim>& lhs,     \
                                       const FixedVec<scalar_t, kDim>& rhs) {   \
    FixedVec<scalar_t, kDim> res;                                               \
    for (int i = 0; i < kDim; ++i) {                                            \
      res[i] = lhs[i] op rhs[i];                                                \
    }                                                                           \
    return res;                                                                 \
  }                                                                             \
                                                                                \
  template <typename scalar_t, std::size_t kDim>                                \
  FN_SPECIFIERS                                                                 \
  void operator opas(FixedVec<scalar_t, kDim>& lhs,                             \
                     const FixedVec<scalar_t, kDim>& rhs) {                     \
    for (int i = 0; i < kDim; ++i) {                                            \
      lhs[i] opas rhs[i];                                                       \
    }                                                                           \
  }                                                                             \
                                                                                \
  template <typename scalar_t, std::size_t kDim>                                \
  FN_SPECIFIERS                                                                 \
  FixedVec<scalar_t, kDim> operator op(const FixedVec<scalar_t, kDim>& lhs,     \
                                       const scalar_t rhs) {                    \
    FixedVec<scalar_t, kDim> res;                                               \
    for (int i = 0; i < kDim; ++i) {                                            \
      res[i] = lhs[i] op rhs;                                                   \
    }                                                                           \
    return res;                                                                 \
  }                                                                             \
                                                                                \
  template <typename scalar_t, std::size_t kDim>                                \
  FN_SPECIFIERS                                                                 \
  void operator opas(FixedVec<scalar_t, kDim>& lhs,                             \
                     const scalar_t rhs) {                                      \
    for (int i = 0; i < kDim; ++i) {                                            \
      lhs[i] opas rhs;                                                          \
    }                                                                           \
  }                                                                             \
                                                                                \
  template <typename scalar_t, std::size_t kDim>                                \
  FN_SPECIFIERS                                                                 \
  FixedVec<bool, kDim> operator op(const scalar_t lhs,                          \
                                   const FixedVec<scalar_t, kDim>& rhs) {       \
    FixedVec<bool, kDim> res;                                                   \
    for (int i = 0; i < kDim; ++i) {                                            \
      res[i] = lhs op rhs[i];                                                   \
    }                                                                           \
    return res;                                                                 \
  }

DEFINE_OPERATOR(+, +=)
DEFINE_OPERATOR(-, -=)
DEFINE_OPERATOR(*, *=)
DEFINE_OPERATOR(/, /=)

#undef DEFINE_OPERATOR

#define DEFINE_OPERATOR(op)                                                     \
  template <typename scalar_t, std::size_t kDim>                                \
  FN_SPECIFIERS                                                                 \
  FixedVec<bool, kDim> operator op(const FixedVec<scalar_t, kDim>& lhs,         \
                                   const FixedVec<scalar_t, kDim>& rhs) {       \
    FixedVec<bool, kDim> res;                                                   \
    for (int i = 0; i < kDim; ++i) {                                            \
      res[i] = lhs[i] op rhs[i];                                                \
    }                                                                           \
    return res;                                                                 \
  }                                                                             \
                                                                                \
  template <typename scalar_t, std::size_t kDim>                                \
  FN_SPECIFIERS                                                                 \
  FixedVec<bool, kDim> operator op(const FixedVec<scalar_t, kDim>& lhs,         \
                                   const scalar_t rhs) {                        \
    FixedVec<bool, kDim> res;                                                   \
    for (int i = 0; i < kDim; ++i) {                                            \
      res[i] = lhs[i] op rhs;                                                   \
    }                                                                           \
    return res;                                                                 \
  }                                                                             \
                                                                                \
  template <typename scalar_t, std::size_t kDim>                                \
  FN_SPECIFIERS                                                                 \
  FixedVec<bool, kDim> operator op(const scalar_t lhs,                          \
                                   const FixedVec<scalar_t, kDim>& rhs) {       \
    FixedVec<bool, kDim> res;                                                   \
    for (int i = 0; i < kDim; ++i) {                                            \
      res[i] = lhs op rhs[i];                                                   \
    }                                                                           \
    return res;                                                                 \
  }

DEFINE_OPERATOR(<)
DEFINE_OPERATOR(<=)
DEFINE_OPERATOR(>)
DEFINE_OPERATOR(>=)
DEFINE_OPERATOR(==)
DEFINE_OPERATOR(!=)
DEFINE_OPERATOR(&&)
DEFINE_OPERATOR(||)

#undef DEFINE_OPERATOR

} // sparse_ops
