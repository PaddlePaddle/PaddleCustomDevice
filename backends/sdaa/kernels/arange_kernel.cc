// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T>
void GetSize(T start, T end, T step, int64_t* size) {
  PADDLE_ENFORCE_NE(
      step,
      0,
      phi::errors::InvalidArgument("The step of range op should not be 0."));

  if (start < end) {
    PADDLE_ENFORCE_GT(
        step,
        0,
        phi::errors::InvalidArgument(
            "The step should be greater than 0 while start < end."));
  }

  if (start > end) {
    PADDLE_ENFORCE_LT(step,
                      0,
                      phi::errors::InvalidArgument(
                          "The step should be less than 0 while start > end."));
  }

  *size = std::is_integral<T>::value
              ? ((std::abs(end - start) + std::abs(step) - 1) / std::abs(step))
              : std::ceil(std::abs((end - start) / step));
}

template <typename T, typename Context>
void doArangeTensor(const Context& dev_ctx,
                    const T& start,
                    const T& end,
                    const T& step,
                    phi::DenseTensor* out) {
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);

  std::vector<int> out_dims = phi::vectorize<int>(out->dims());

  tecodnnTensorDescriptor_t out_Desc = sdaa_ops::GetTecodnnTensorDesc(
      out_dims, phi::CppTypeToDataType<T>::Type(), TensorFormat::Undefined);
  TECODNN_CHECK(
      tecodnnArange(tecodnnHandle, &start, &end, &step, out_Desc, out->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

template <typename T, typename Context>
void ArangeTensorKernel(const Context& dev_ctx,
                        const phi::DenseTensor& start_t,
                        const phi::DenseTensor& end_t,
                        const phi::DenseTensor& step_t,
                        phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA ArangeTensorKernel";

  T start_value = phi::GetValue<T, Context>(dev_ctx, start_t);
  T end_value = phi::GetValue<T, Context>(dev_ctx, end_t);
  T step_value = phi::GetValue<T, Context>(dev_ctx, step_t);

  int64_t size = 0;
  GetSize(start_value, end_value, step_value, &size);

  out->Resize(phi::make_ddim({size}));
  dev_ctx.template Alloc<T>(out);

  doArangeTensor<T, Context>(dev_ctx, start_value, end_value, step_value, out);
}

// template <typename T, typename Context>
// void ArangeKernel(const Context& dev_ctx,
//                   const phi::Scalar& start,
//                   const phi::Scalar& end,
//                   const phi::Scalar& step,
//                   phi::DenseTensor* out) {
//   T start_value = start.to<T>();
//   T end_value = end.to<T>();
//   T step_value = step.to<T>();

//   int64_t size = 0;
//   GetSize(start_value, end_value, step_value, &size);

//   out->Resize(phi::make_ddim({size}));
//   dev_ctx.template Alloc<T>(out);

//   doArangeTensor<T, Context>(dev_ctx, start_value, end_value, step_value,
//   out);
// }

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(arange_tensor,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ArangeTensorKernel,
                          float,
                          int,
                          int64_t) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
}

// TODO(teco): unregister arange for code coverage since this kernel would
// not be called except in pir, register it back when pir supports custom device
// PD_REGISTER_PLUGIN_KERNEL(arange,
//                           sdaa,
//                           ALL_LAYOUT,
//                           custom_kernel::ArangeKernel,
//                           float,
//                           int,
//                           int64_t) {}
