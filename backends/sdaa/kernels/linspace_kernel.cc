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
T GetValueOfExpectedType(const Context& ctx, const phi::DenseTensor& x) {
  switch (x.dtype()) {
    case DataType::FLOAT32:
      return static_cast<T>(phi::GetValue<float, Context>(ctx, x));
    case DataType::FLOAT64:
      return static_cast<T>(phi::GetValue<double, Context>(ctx, x));
    case DataType::INT32:
      return static_cast<T>(phi::GetValue<int32_t, Context>(ctx, x));
    case DataType::INT64:
      return static_cast<T>(phi::GetValue<int64_t, Context>(ctx, x));
    case DataType::FLOAT16:
      return static_cast<T>(
          phi::GetValue<phi::dtype::float16, Context>(ctx, x));
    default:
      PADDLE_THROW(phi::errors::Unimplemented(
          "Data type (%s) is not supported when casting data type.",
          x.dtype()));
  }
}

template <typename T, typename Context>
void LinspaceKernel(const Context& dev_ctx,
                    const phi::DenseTensor& start,
                    const phi::DenseTensor& stop,
                    const phi::DenseTensor& number,
                    phi::DataType dtype,
                    phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA LinspaceKernel";

  T start_value = GetValueOfExpectedType<T, Context>(dev_ctx, start);
  T stop_value = GetValueOfExpectedType<T, Context>(dev_ctx, stop);
  using MT = typename sdaa_ops::MPTypeTrait<T>::Type;
  auto mt_start = static_cast<MT>(start_value);
  auto mt_stop = static_cast<MT>(stop_value);
  int32_t num = GetValueOfExpectedType<int32_t, Context>(dev_ctx, number);
  PADDLE_ENFORCE_GT(
      num,
      0,
      phi::errors::InvalidArgument("The num of linspace op should be larger "
                                   "than 0, but received num is %d",
                                   num));

  out->Resize(phi::make_ddim({num}));
  dev_ctx.template Alloc<T>(out);
  std::vector<int> out_dims = phi::vectorize<int>(out->dims());
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnTensorDescriptor_t out_Desc =
      sdaa_ops::GetTecodnnTensorDesc(out_dims, dtype, TensorFormat::Undefined);
  TECODNN_CHECK(tecodnnLinspace(
      tecodnnHandle, &mt_start, &mt_stop, out_Desc, out->data()));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(out_Desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(linspace,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LinspaceKernel,
                          float,
                          int32_t,
                          int64_t,
                          double,
                          phi::dtype::float16) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
}
