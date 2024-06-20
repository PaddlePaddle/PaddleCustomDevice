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

#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename Context>
void AbsGrad(const Context& dev_ctx,
             const phi::DenseTensor& x,
             const phi::DenseTensor& dout,
             phi::DenseTensor* dx) {
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  int num = static_cast<int>(x.numel());
  std::vector<int> dims = {1, 1, 1, num};
  tecodnnTensorDescriptor_t x_Desc =
      sdaa_ops::GetTecodnnTensorDesc(dims, x.dtype(), TensorFormat::NHWC);

  const float alpha = 1.0f, beta = 0.0f;
  TECODNN_CHECK(tecodnnAbsGrad(tecodnnHandle,
                               &alpha,
                               x_Desc,
                               x.data(),
                               x_Desc,
                               dout.data(),
                               &beta,
                               x_Desc,
                               dx->data()));

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
}

template <typename T, typename Context>
void AbsKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA AbsKernel";
  dev_ctx.template Alloc<T>(out);

  sdaa_ops::doUnaryOpTensor(dev_ctx, x, 1.0, UnaryOpMode::FABS, out);
}

template <typename T, typename Context>
void AbsGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  VLOG(4) << "Call SDAA AbsGradKernel";
  dev_ctx.template Alloc<T>(dx);

  AbsGrad<Context>(dev_ctx, x, dout, dx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(abs,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::AbsKernel,
                          float,
                          phi::dtype::float16,
                          int64_t) {
  kernel->InputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

PD_REGISTER_PLUGIN_KERNEL(abs_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::AbsGradKernel,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(1).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
