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
  phi::DenseTensor scale_tensor;
  scale_tensor.Resize({1});
  phi::DenseTensor bias_tensor;
  bias_tensor.Resize({1});
  if (x.dtype() == DataType::FLOAT16) {
    auto scale = in_scale.to<T>();
    dev_ctx.template Alloc<T>(&scale_tensor);
    dev_ctx.template Alloc<T>(&bias_tensor);

    MLUCnnlTensorDesc scale_desc(scale_tensor);
    MLUCnnl::Fill(dev_ctx,
                  CNNL_POINTER_MODE_HOST,
                  &scale,
                  scale_desc.get(),
                  GetBasePtr(&scale_tensor));

    MLUCnnlTensorDesc bias_desc(bias_tensor);
    T new_bias = static_cast<T>(bias);
    if (!bias_after_scale) {
      new_bias *= scale;
    }
    MLUCnnl::Fill(dev_ctx,
                  CNNL_POINTER_MODE_HOST,
                  &new_bias,
                  bias_desc.get(),
                  GetBasePtr(&bias_tensor));

  } else {
    auto scale = in_scale.to<float>();
    dev_ctx.template Alloc<float>(&scale_tensor);
    dev_ctx.template Alloc<float>(&bias_tensor);

    MLUCnnlTensorDesc scale_desc(scale_tensor);
    MLUCnnl::Fill(dev_ctx,
                  CNNL_POINTER_MODE_HOST,
                  &scale,
                  scale_desc.get(),
                  GetBasePtr(&scale_tensor));

    MLUCnnlTensorDesc bias_desc(bias_tensor);
    if (!bias_after_scale) {
      bias *= scale;
    }
    MLUCnnl::Fill(dev_ctx,
                  CNNL_POINTER_MODE_HOST,
                  &bias,
                  bias_desc.get(),
                  GetBasePtr(&bias_tensor));
  }
  MLUCnnlTensorDesc scale_desc(scale_tensor);
  MLUCnnlTensorDesc bias_desc(bias_tensor);

  dev_ctx.template Alloc<T>(out);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc output_desc(*out);

  const int axis = std::max(x.dims().size() - 1, 0);
  if (x.dtype() == DataType::INT64 || x.dtype() == DataType::INT32) {
    Tensor x_temp_tensor;
    x_temp_tensor.Resize(x.dims());
    dev_ctx.template Alloc<float>(&x_temp_tensor);

    MLUCnnlTensorDesc x_temp_tensor_desc(x_temp_tensor);

    cnnlCastDataType_t cast_type =
        GetCastDataType(x.dtype(), DataType::FLOAT32);
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  input_desc.get(),
                  GetBasePtr(&x),
                  x_temp_tensor_desc.get(),
                  GetBasePtr(&x_temp_tensor));

    Tensor out_temp_tensor;
    out_temp_tensor.Resize(out->dims());
    dev_ctx.template Alloc<float>(&out_temp_tensor);
    MLUCnnlTensorDesc out_temp_tensor_desc(out_temp_tensor);

    MLUCnnl::Scale(dev_ctx,
                   axis,
                   x_temp_tensor_desc.get(),
                   GetBasePtr(&x_temp_tensor),
                   scale_desc.get(),
                   GetBasePtr(&scale_tensor),
                   bias_desc.get(),
                   GetBasePtr(&bias_tensor),
                   out_temp_tensor_desc.get(),
                   GetBasePtr(&out_temp_tensor));
    cnnlCastDataType_t cast_x_type =
        GetCastDataType(DataType::FLOAT32, x.dtype());

    MLUCnnl::Cast(dev_ctx,
                  cast_x_type,
                  out_temp_tensor_desc.get(),
                  GetBasePtr(&out_temp_tensor),
                  output_desc.get(),
                  GetBasePtr(out));

  } else {
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
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scale,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::ScaleKernel,
                          phi::dtype::float16,
                          float,
                          int,
                          int64_t) {}
