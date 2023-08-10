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

#include "kernels/funcs/mlu_baseop.h"
#include "paddle/phi/common/type_traits.h"

namespace custom_kernel {

// just test1

template <typename T, typename Context>
void AbsKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);
  MLUCnnl::Abs(dev_ctx,
               input_desc.get(),
               GetBasePtr(&x),
               output_desc.get(),
               GetBasePtr(out));
}

template <typename T, typename Context>
void AbsGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& dout,
                   phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlOpTensorDesc mul_op_desc(
      CNNL_OP_TENSOR_MUL, ToCnnlDataType<T>(), CNNL_NOT_PROPAGATE_NAN);

  Tensor sign_x;
  sign_x.Resize(x.dims());
  dev_ctx.template Alloc<T>(&sign_x);

  MLUCnnl::Sign(dev_ctx,
                input_desc.get(),
                GetBasePtr(&x),
                input_desc.get(),
                GetBasePtr(&sign_x));
  MLUCnnl::OpTensor(dev_ctx,
                    mul_op_desc.get(),
                    input_desc.get(),
                    GetBasePtr(&sign_x),
                    input_desc.get(),
                    GetBasePtr(&dout),
                    input_desc.get(),
                    GetBasePtr(dx),
                    ToCnnlDataType<T>());
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(abs,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::AbsKernel,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(0).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}

PD_REGISTER_PLUGIN_KERNEL(abs_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::AbsGradKernel,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(1).SetDataType(phi::dtype::ToReal(kernel_key.dtype()));
}
