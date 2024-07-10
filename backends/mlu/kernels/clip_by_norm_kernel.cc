// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/elementwise_utils.h"
#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {
template <typename T, typename Context>
void ClipByNormKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      float max_norm,
                      phi::DenseTensor* out) {
  PADDLE_ENFORCE_NOT_NULL(&x,
                          phi::errors::InvalidArgument(
                              "Input(X) of ClipByNormOp should not be null. "
                              "Please check if it is created correctly."));
  phi::DenseTensor square_sum;
  phi::DenseTensorMeta square_sum_meta = {x.dtype(), phi::DDim({1})};
  square_sum.set_meta(square_sum_meta);
  dev_ctx.template Alloc<T>(&square_sum);

  MLUCnnlTensorDesc input_desc(x);
  MLUCnnlTensorDesc square_sum_desc(square_sum);

  // L2Loss
  MLUCnnl::L2Loss(
      dev_ctx, input_desc.get(), GetBasePtr(&x), GetBasePtr(&square_sum));

  // do mul
  phi::DenseTensor scale_tensor;
  scale_tensor.Resize({1});
  dev_ctx.template Alloc<T>(&scale_tensor);

  phi::DenseTensor bias_tensor;
  bias_tensor.Resize({1});
  dev_ctx.template Alloc<T>(&bias_tensor);

  MLUCnnlTensorDesc scale_desc(scale_tensor);
  MLUCnnlTensorDesc bias_desc(bias_tensor);
  FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(2.0f), &scale_tensor);
  FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0.0f), &bias_tensor);

  MLUCnnl::Scale(dev_ctx,
                 0,
                 square_sum_desc.get(),
                 GetBasePtr(&square_sum),
                 scale_desc.get(),
                 GetBasePtr(&scale_tensor),
                 bias_desc.get(),
                 GetBasePtr(&bias_tensor),
                 square_sum_desc.get(),
                 GetBasePtr(&square_sum));

  // sqrt
  phi::DenseTensor x_norm;
  phi::DenseTensorMeta x_norm_meta = {x.dtype(), phi::DDim({1})};
  x_norm.set_meta(x_norm_meta);
  dev_ctx.template Alloc<T>(&x_norm);

  MLUCnnlTensorDesc x_norm_desc(x_norm);
  cnnlComputationPreference_t prefer = CNNL_COMPUTATION_HIGH_PRECISION;
  MLUCnnl::Sqrt(dev_ctx,
                prefer,
                square_sum_desc.get(),
                GetBasePtr(&square_sum),
                x_norm_desc.get(),
                GetBasePtr(&x_norm));

  phi::DenseTensor x_norm_t;
  phi::DenseTensorMeta x_norm_t_meta = {
      x_norm.dtype(), x_norm.dims(), x_norm.layout()};
  x_norm_t.set_meta(x_norm_t_meta);

  // sync copy
  dev_ctx.Wait();
  TensorCopy(dev_ctx, x_norm, true, &x_norm_t, phi::CPUPlace());
  auto x_norm_v = static_cast<float>(*(x_norm_t.data<T>()));

  dev_ctx.template Alloc<T>(out);
  if (x_norm_v <= max_norm) {
    TensorCopy(dev_ctx, x, false, out);
  } else {
    auto epsilon = x_norm_v <= static_cast<float>(1e-30)
                       ? static_cast<float>(1e-6)
                       : static_cast<float>(0);

    float scaling = max_norm / (x_norm_v + epsilon);
    auto scale_t = static_cast<T>(scaling);
    phi::DenseTensor scaling_tensor;
    scaling_tensor.Resize({1});
    dev_ctx.template Alloc<T>(&scaling_tensor);
    MLUCnnlTensorDesc scaling_tensor_desc(scaling_tensor);
    MLUCnnl::Fill(dev_ctx,
                  CNNL_POINTER_MODE_HOST,
                  &scale_t,
                  scaling_tensor_desc.get(),
                  GetBasePtr(&scaling_tensor));

    auto data_type = ToCnnlDataType<T>();
    MLUCnnlTensorDesc out_desc(*out);

    // compute out = scaling_tensor * x
    MLUOpTensorKernel<T>(
        dev_ctx, scaling_tensor, x, -1, CNNL_OP_TENSOR_MUL, out);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(clip_by_norm,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::ClipByNormKernel,
                          float,
                          phi::dtype::float16) {}
