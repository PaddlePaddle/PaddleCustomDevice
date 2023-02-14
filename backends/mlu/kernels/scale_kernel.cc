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
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::Scalar& in_scale,
                 float bias,
                 bool bias_after_scale,
                 phi::DenseTensor* out) {
  // cnnl require input, scale, bias with same type. And all in device side.
  auto scale = in_scale.to<T>();
  phi::DenseTensor scale_tensor;
  scale_tensor.Resize({1});
  dev_ctx.template Alloc<T>(&scale_tensor);

  MLUCnnlTensorDesc scale_desc(scale_tensor);
  MLUCnnl::Fill(dev_ctx,
                CNNL_POINTER_MODE_HOST,
                &scale,
                scale_desc.get(),
                GetBasePtr(&scale_tensor));

  phi::DenseTensor bias_tensor;
  bias_tensor.Resize({1});
  dev_ctx.template Alloc<T>(&bias_tensor);

  MLUCnnlTensorDesc bias_desc(bias_tensor);
  MLUCnnl::Fill(dev_ctx,
                CNNL_POINTER_MODE_HOST,
                &bias,
                bias_desc.get(),
                GetBasePtr(&bias_tensor));

  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);

  const int axis = std::max(x.dims().size() - 1, 0);
  if (bias_after_scale) {
    MLUCnnl::Scale(dev_ctx,
                   axis,
                   input_desc.get(),
                   GetBasePtr(&x),
                   scale_desc.get(),
                   GetBasePtr(&scale_tensor),
                   bias_desc.get(),
                   GetBasePtr(&bias_tensor),
                   output_desc.get(),
                   GetBasePtr(out));
  } else {
    phi::DenseTensor new_bias_tensor;
    new_bias_tensor.Resize({1});
    dev_ctx.template Alloc<T>(&new_bias_tensor);

    MLUCnnlTensorDesc new_bias_desc(new_bias_tensor);

    MLUCnnlOpTensorDesc mul_op_desc(
        CNNL_OP_TENSOR_MUL, ToCnnlDataType(x.dtype()), CNNL_NOT_PROPAGATE_NAN);
    MLUCnnl::OpTensor(dev_ctx,
                      mul_op_desc.get(),
                      scale_desc.get(),
                      GetBasePtr(&scale_tensor),
                      bias_desc.get(),
                      GetBasePtr(&bias_tensor),
                      new_bias_desc.get(),
                      GetBasePtr(&new_bias_tensor),
                      ToCnnlDataType(x.dtype()));
    MLUCnnl::Scale(dev_ctx,
                   axis,
                   input_desc.get(),
                   GetBasePtr(&x),
                   scale_desc.get(),
                   GetBasePtr(&scale_tensor),
                   new_bias_desc.get(),
                   GetBasePtr(&new_bias_tensor),
                   output_desc.get(),
                   GetBasePtr(out));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scale,
                          CustomMLU,
                          ALL_LAYOUT,
                          custom_kernel::ScaleKernel,
                          phi::dtype::float16,
                          float) {}
