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

#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/impl/elementwise_kernel_impl.h"
#include "paddle/phi/kernels/legacy/elementwise_add_kernel.h"
#include "paddle/phi/kernels/legacy/elementwise_divide_kernel.h"
#include "paddle/phi/kernels/legacy/elementwise_kernel.h"
#include "paddle/phi/kernels/legacy/elementwise_multiply_kernel.h"
#include "paddle/phi/kernels/legacy/elementwise_subtract_kernel.h"

namespace phi {

#define DEFINE_CUDA_ELEMENTWISE_OP(name)                             \
  template <typename T, typename Context>                            \
  void name##RawKernel(const Context& dev_ctx,                       \
                       const DenseTensor& x,                         \
                       const DenseTensor& y,                         \
                       int axis,                                     \
                       DenseTensor* out) {                           \
    std::vector<const DenseTensor*> inputs = {&x, &y};               \
    std::vector<DenseTensor*> outputs = {out};                       \
    dev_ctx.template Alloc<T>(out);                                  \
    funcs::BroadcastKernel<T>(                                       \
        dev_ctx, inputs, &outputs, funcs::name##Functor<T>(), axis); \
  }

DEFINE_CUDA_ELEMENTWISE_OP(Add)
DEFINE_CUDA_ELEMENTWISE_OP(Divide)
DEFINE_CUDA_ELEMENTWISE_OP(Multiply)
DEFINE_CUDA_ELEMENTWISE_OP(Subtract)
DEFINE_CUDA_ELEMENTWISE_OP(Maximum)
DEFINE_CUDA_ELEMENTWISE_OP(Minimum)
DEFINE_CUDA_ELEMENTWISE_OP(Remainder)
DEFINE_CUDA_ELEMENTWISE_OP(FloorDivide)
DEFINE_CUDA_ELEMENTWISE_OP(ElementwisePow)


template <typename T, typename Context>
void SubtractKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    out->Resize(out->dims());
    dev_ctx.template Alloc<T>(out);
    return;
  }
  phi::SubtractRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void MultiplyKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    out->Resize(out->dims());
    dev_ctx.template Alloc<T>(out);
    return;
  }
  phi::MultiplyRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void DivideKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const DenseTensor& y,
                  DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    out->Resize(out->dims());
    dev_ctx.template Alloc<T>(out);
    return;
  }
  phi::DivideRawKernel<T, Context>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void MultiPrecisionAddKernelImpl(const Context& dev_ctx,
                                 const DenseTensor& x,
                                 const DenseTensor& y,
                                 DenseTensor* out) {
  std::vector<const DenseTensor*> inputs = {&x, &y};
  std::vector<DenseTensor*> outputs = {out};
  if (y.dtype() == phi::DataType::BFLOAT16) {
    funcs::BroadcastKernel<T>(
        dev_ctx,
        inputs,
        &outputs,
        funcs::MultiPrecisionAddFunctor<T, phi::bfloat16>(),
        -1);
  } else if (y.dtype() == phi::DataType::FLOAT16) {
    funcs::BroadcastKernel<T>(
        dev_ctx,
        inputs,
        &outputs,
        funcs::MultiPrecisionAddFunctor<T, phi::float16>(),
        -1);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Unsupported x dtype:%s, y dtype:%s for add(x, y) operation",
        phi::DataTypeToString(x.type()),
        phi::DataTypeToString(y.type())));
  }
}

template <typename T, typename Context>
void AddKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const DenseTensor& y,
               DenseTensor* out) {
  if (x.numel() == 0 || y.numel() == 0) {
    out->Resize(out->dims());
    dev_ctx.template Alloc<T>(out);
    return;
  }
#ifdef PADDLE_WITH_CUDA
  if (x.dtype() == phi::DataType::FLOAT32 &&
      (y.dtype() == phi::DataType::BFLOAT16 ||
       y.dtype() == phi::DataType::FLOAT16)) {
    MultiPrecisionAddKernelImpl<float, Context>(dev_ctx, x, y, out);
  } else {
#endif
    phi::AddRawKernel<T, Context>(dev_ctx, x, y, -1, out);
#ifdef PADDLE_WITH_CUDA
  }
#endif
}

template <typename T, typename Context>
void GradAddKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  phi::AddRawKernel<T>(dev_ctx, x, y, -1, out);
}

template <typename T, typename Context>
void MaximumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  int axis = -1;
  MaximumRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void MinimumKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const DenseTensor& y,
                   DenseTensor* out) {
  int axis = -1;
  MinimumRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void RemainderKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  int axis = -1;
  RemainderRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void FloorDivideKernel(const Context& dev_ctx,
                       const DenseTensor& x,
                       const DenseTensor& y,
                       DenseTensor* out) {
  int axis = -1;
  FloorDivideRawKernel<T>(dev_ctx, x, y, axis, out);
}

// Create the definition of Heaviside
template <typename T, typename Context>
void HeavisideKernel(const Context& dev_ctx,
                     const DenseTensor& x,
                     const DenseTensor& y,
                     DenseTensor* out) {
  std::vector<const DenseTensor*> inputs = {&x, &y};
  std::vector<DenseTensor*> outputs = {out};
  dev_ctx.template Alloc<T>(out);
  funcs::BroadcastKernel<T>(
      dev_ctx, inputs, &outputs, funcs::ElementwiseHeavisideFunctor<T>());
}

template <typename T, typename Context>
void ElementwisePowKernel(const Context& dev_ctx,
                          const DenseTensor& x,
                          const DenseTensor& y,
                          DenseTensor* out) {
  int axis = -1;
  ElementwisePowRawKernel<T>(dev_ctx, x, y, axis, out);
}

template <typename T, typename Context>
void CopySignKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    DenseTensor* out) {
  std::vector<const DenseTensor*> inputs = {&x, &y};
  std::vector<DenseTensor*> outputs = {out};
  dev_ctx.template Alloc<T>(out);
  funcs::BroadcastKernel<T>(
      dev_ctx, inputs, &outputs, funcs::CopySignFunctor<T>());
}

}  // namespace phi

PD_CUSTOM_KERNEL_REGISTER(maximum,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::MaximumKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(minimum,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::MinimumKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(remainder,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::RemainderKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(floor_divide,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FloorDivideKernel,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(elementwise_pow,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::ElementwisePowKernel,
                          float,
                          int,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_CUSTOM_KERNEL_REGISTER(copysign,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::CopySignKernel,
                          bool,
                          uint8_t,
                          int8_t,
                          int16_t,
                          int,
                          int64_t,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;
using complex64 = ::phi::dtype::complex<float>;

PD_CUSTOM_KERNEL_REGISTER(fmax,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FMaxKernel,
                          float,
                          int,
                          float16,
                          bfloat16,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(fmin,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::FMinKernel,
                          float,
                          int,
                          float16,
                          bfloat16,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(heaviside,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::HeavisideKernel,
                          float,
                          int,
                          float16,
                          bfloat16,
                          int64_t) {}

PD_CUSTOM_KERNEL_REGISTER(add,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::AddKernel,
                          float,
                          int16_t,
                          int,
                          bool,
                          uint8_t,
                          int8_t,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          complex64) {}

PD_CUSTOM_KERNEL_REGISTER(grad_add,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::GradAddKernel,
                          float,
                          int16_t,
                          int,
                          bool,
                          uint8_t,
                          int8_t,
                          int64_t,
                          phi::dtype::float16,
                          phi::dtype::bfloat16,
                          complex64) {}

PD_CUSTOM_KERNEL_REGISTER(divide,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::DivideKernel,
                          float,
                          int8_t,
                          uint8_t,
                          int16_t,
                          int,
                          int64_t,
                          bool,
                          float16,
                          bfloat16,
                          complex64) {}

PD_CUSTOM_KERNEL_REGISTER(multiply,
                          iluvatar_gpu,
                          ALL_LAYOUT,
                          phi::MultiplyKernel,
                          float,
                          int,
                          int64_t,
                          bool,
                          float16,
                          complex64,
                          bfloat16) {}