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

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void doArgMaxMinTensor(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       int axis,
                       bool arg_max,
                       phi::DenseTensor* out) {
  VLOG(4) << "call tecodnn argmax/argmin kernel";

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());
  if (!out_dims.size()) {
    out_dims = {1};
  }

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      x_dims, x.dtype(), TensorFormat::Undefined);
  tecodnnTensorDescriptor_t out_Desc = sdaa_ops::GetTecodnnTensorDesc(
      out_dims, out->dtype(), TensorFormat::Undefined);

  if (arg_max) {
    TECODNN_CHECK(tecodnnArgmax(
        tecodnnHandle, axis, x_Desc, x.data(), out_Desc, out->data()));
  } else {
    TECODNN_CHECK(tecodnnArgmin(
        tecodnnHandle, axis, x_Desc, x.data(), out_Desc, out->data()));
  }

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

template <typename T, typename Context>
void ArgMaxMin(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::Scalar& axis,
               bool keepdims,
               bool flatten,
               phi::DataType dtype,
               bool arg_max,
               phi::DenseTensor* out) {
  int axis_ = axis.to<int>();
  if (x.numel() == 0) return;

  PADDLE_ENFORCE_EQ(
      (dtype == phi::DataType::INT64 || dtype == phi::DataType::INT32),
      true,
      phi::errors::InvalidArgument(
          "The attribute of dtype in argmax op must be [%s] or [%s], "
          "but received [%s]",
          phi::DataType::INT64,
          phi::DataType::INT32,
          dtype));

  if (dtype == phi::DataType::INT64) {
    dev_ctx.template Alloc<int64_t>(out);
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument(
        "Tecodnn only support the output's dtype is int64."));
  }

  if (axis_ < 0) {
    auto x_dims = x.dims();
    axis_ += x_dims.size();
  }

  if (flatten) {
    phi::DenseTensor flatten_x(x);
    flatten_x.Resize(phi::make_ddim({x.numel()}));
    // if flatten, the axis_ is 0
    axis_ = 0;
    custom_kernel::doArgMaxMinTensor<T, Context>(
        dev_ctx, flatten_x, axis_, arg_max, out);
  } else {
    custom_kernel::doArgMaxMinTensor<T, Context>(
        dev_ctx, x, axis_, arg_max, out);
  }
}

template <typename T, typename Context>
void ArgMaxKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  phi::DataType dtype,
                  phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA ArgMaxKernel";
  custom_kernel::ArgMaxMin<T, Context>(
      dev_ctx, x, axis, keepdims, flatten, dtype, true, out);
}

template <typename T, typename Context>
void ArgMinKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::Scalar& axis,
                  bool keepdims,
                  bool flatten,
                  phi::DataType dtype,
                  phi::DenseTensor* out) {
  VLOG(4) << "CALL SDAA ArgMinKernel";
  custom_kernel::ArgMaxMin<T, Context>(
      dev_ctx, x, axis, keepdims, flatten, dtype, false, out);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(argmax,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ArgMaxKernel,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(argmin,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::ArgMinKernel,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}
