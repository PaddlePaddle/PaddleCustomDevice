// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "paddle/phi/common/complex.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/broadcast_function.h"
#include "paddle/phi/kernels/funcs/compare_functors.h"

#ifdef PADDLE_WITH_XPU_KP
#include "paddle/phi/backends/xpu/xpu_context.h"
#else
#include <thrust/fill.h>

#include <vector>

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/kernels/compare_kernel.h"
#include "paddle/phi/kernels/funcs/elementwise_base.h"
#include "paddle/phi/kernels/gpu/reduce.h"
#include "paddle/phi/kernels/legacy/compare_kernel.h"
#include "paddle/phi/kernels/primitive/functor_primitives.h"
#endif

namespace phi {

template <typename T, typename Context, typename Functor>
inline void CompareRawKernelImpl(const Context& ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y,
                                 int axis,
                                 DenseTensor* out) {
  ctx.template Alloc<bool>(out);
  out->set_type(phi::DataType::BOOL);
  if (out->numel() == 0) return;
  std::vector<const DenseTensor*> ins{&x, &y};
  std::vector<DenseTensor*> outs{out};
  funcs::BroadcastKernel<bool>(ctx, ins, &outs, Functor(), axis);
}

template <typename T, typename Context>
void LessThanRawKernel(const Context& ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       int axis,
                       DenseTensor* out) {
  CompareRawKernelImpl<T, Context, funcs::LessThanFunctor<T>>(
      ctx, x, y, axis, out);
}

template <typename T, typename Context>
void LessEqualRawKernel(const Context& ctx,
                        const DenseTensor& x,
                        const DenseTensor& y,
                        int axis,
                        DenseTensor* out) {
  CompareRawKernelImpl<T, Context, funcs::LessEqualFunctor<T>>(
      ctx, x, y, axis, out);
}

template <typename T, typename Context>
void GreaterThanRawKernel(const Context& ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          int axis,
                          DenseTensor* out) {
  CompareRawKernelImpl<T, Context, funcs::GreaterThanFunctor<T>>(
      ctx, x, y, axis, out);
}

template <typename T, typename Context>
void GreaterEqualRawKernel(const Context& ctx,
                           const DenseTensor& x,
                           const DenseTensor& y,
                           int axis,
                           DenseTensor* out) {
  CompareRawKernelImpl<T, Context, funcs::GreaterEqualFunctor<T>>(
      ctx, x, y, axis, out);
}

template <typename T, typename Context>
void EqualRawKernel(const Context& ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    int axis,
                    DenseTensor* out) {
  CompareRawKernelImpl<T, Context, funcs::EqualFunctor<T>>(
      ctx, x, y, axis, out);
}

template <typename T, typename Context>
void NotEqualRawKernel(const Context& ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       int axis,
                       DenseTensor* out) {
  CompareRawKernelImpl<T, Context, funcs::NotEqualFunctor<T>>(
      ctx, x, y, axis, out);
}

template <typename T>
struct BitwiseAdd {
  // Bitwise add operator, returns <tt>a + b</tt>
  inline T initial() { return static_cast<T>(true); }

  __host__ __device__ __forceinline__ T operator()(const T& a,
                                                   const T& b) const {
    return a & b;
  }
};

#define DEFINE_CUDA_COMPARE_KERNEL(name)                      \
  template <typename T, typename Context>                     \
  void name##Kernel(const Context& ctx,                       \
                    const DenseTensor& x,                     \
                    const DenseTensor& y,                     \
                    DenseTensor* out) {                       \
    if (out->IsSharedWith(x)) {                               \
      auto x_origin = x;                                      \
      name##RawKernel<T, Context>(ctx, x_origin, y, -1, out); \
    } else {                                                  \
      name##RawKernel<T, Context>(ctx, x, y, -1, out);        \
    }                                                         \
  }

DEFINE_CUDA_COMPARE_KERNEL(LessThan)
DEFINE_CUDA_COMPARE_KERNEL(LessEqual)
DEFINE_CUDA_COMPARE_KERNEL(GreaterThan)
DEFINE_CUDA_COMPARE_KERNEL(GreaterEqual)
DEFINE_CUDA_COMPARE_KERNEL(Equal)
DEFINE_CUDA_COMPARE_KERNEL(NotEqual)
#undef DEFINE_CUDA_COMPARE_KERNEL

#ifndef PADDLE_WITH_XPU_KP
template <typename T, typename Context, typename Functor>
inline void CompareAllKernelImpl(const Context& ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y,
                                 DenseTensor* out) {
  bool* out_data = ctx.template Alloc<bool>(out);

  if (x.dims() != y.dims()) {
    thrust::device_ptr<bool> out_dev_ptr(out_data);
    thrust::fill(out_dev_ptr, out_dev_ptr + 1, false);
    return;
  }

  DenseTensor tmp;
  tmp.Resize(x.dims());
  ctx.template Alloc<bool>(&tmp);

  std::vector<const DenseTensor*> ins{&x, &y};
  std::vector<DenseTensor*> outs{&tmp};
  funcs::ElementwiseKernel<bool>(ctx, ins, &outs, Functor());

  // Reduce by 'bitwise and' operator
  std::vector<int> reduce_dims;
  reduce_dims.resize(tmp.dims().size());
  for (int i = 0; i < reduce_dims.size(); ++i) {
    reduce_dims[i] = i;
  }
  funcs::ReduceKernel<bool, bool, BitwiseAdd, kps::IdentityFunctor<bool>>(
      ctx, tmp, out, kps::IdentityFunctor<bool>(), reduce_dims);
}

template <typename T, typename Context>
void EqualAllKernel(const Context& ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  CompareAllKernelImpl<T, Context, funcs::EqualFunctor<T>>(ctx, x, y, out);
}
#endif

}  // namespace phi



// PD_REGISTER_KERNEL(equal_all,
//                    KPS,
//                    ALL_LAYOUT,
//                    phi::EqualAllKernel,
//                    bool,
//                    int,
//                    int64_t,
//                    float,
//                    double) {
//   kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
// }

#define PD_REGISTER_COMPARE_KERNEL(name, func)            \
  PD_CUSTOM_KERNEL_REGISTER(name,                         \
                            iluvatar_gpu,                 \
                            ALL_LAYOUT,                   \
                            phi::func##Kernel,            \
                            bool,                         \
                            int,                          \
                            uint8_t,                      \
                            int8_t,                       \
                            int16_t,                      \
                            int64_t,                      \
                            float,                        \
                            phi::dtype::float16,          \
                            phi::dtype::bfloat16) {       \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL); \
  }

#define PD_REGISTER_COMPLEX_COMPARE_KERNEL(name, func)    \
  PD_CUSTOM_KERNEL_REGISTER(name,                         \
                            iluvatar_gpu,                 \
                            ALL_LAYOUT,                   \
                            phi::func##Kernel,            \
                            bool,                         \
                            int,                          \
                            uint8_t,                      \
                            int8_t,                       \
                            int16_t,                      \
                            int64_t,                      \
                            phi::dtype::complex<float>,   \
                            float,                        \
                            phi::dtype::float16,          \
                            phi::dtype::bfloat16) {       \
    kernel->OutputAt(0).SetDataType(phi::DataType::BOOL); \
  }

PD_REGISTER_COMPARE_KERNEL(less_than, LessThan)
PD_REGISTER_COMPARE_KERNEL(less_equal, LessEqual)
PD_REGISTER_COMPARE_KERNEL(greater_than, GreaterThan)
PD_REGISTER_COMPARE_KERNEL(greater_equal, GreaterEqual)

PD_REGISTER_COMPLEX_COMPARE_KERNEL(equal, Equal)
PD_REGISTER_COMPLEX_COMPARE_KERNEL(not_equal, NotEqual)