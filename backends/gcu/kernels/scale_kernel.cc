// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/common_ops/common_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_op_runner.h"
#include "paddle/phi/core/dense_tensor.h"

namespace custom_kernel {

template <typename T, typename Context>
void GcuScaleKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::Scalar& in_scale,
                    const phi::Scalar& in_bias,
                    bool bias_after_scale,
                    phi::DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "scale", scale);

    phi::DenseTensor scale_tensor;
    scale_tensor.Resize(phi::make_ddim({}));
    scale_tensor = full_like(dev_ctx, scale_tensor, in_scale.to<float>());

    phi::DenseTensor bias_tensor;
    bias_tensor.Resize(phi::make_ddim({}));
    bias_tensor = full_like(dev_ctx, bias_tensor, in_bias.to<float>());

    auto tmp_x = x;
    phi::DenseTensor tmp_out;
    if (tmp_x.dtype() == phi::DataType::INT64) {
      tmp_x = cast(dev_ctx, tmp_x, phi::DataType::INT32);
    }

    if (bias_after_scale) {
      // Out = scale ∗ X + bias
      auto mul_out = mul_compute(dev_ctx, scale_tensor, tmp_x);
      tmp_out = add_compute(dev_ctx, mul_out, bias_tensor);
    } else {
      // Out = scale ∗ (X + bias)
      auto add_out = add_compute(dev_ctx, tmp_x, bias_tensor);
      tmp_out = mul_compute(dev_ctx, add_out, scale_tensor);
    }
    if (out->dtype() != tmp_out.dtype()) {
      tmp_out = cast(dev_ctx, tmp_out, out->dtype());
    }
    if (out->dims() != tmp_out.dims()) {
      tmp_out = reshape(dev_ctx, tmp_out, phi::vectorize(out->dims()));
    }
    *out = tmp_out;
    PADDLE_GCU_KERNEL_END("scale", scale);
  } else {
    dev_ctx.template Alloc<T>(out);
    phi::DenseTensor scale_tensor;
    scale_tensor.Resize({1});
    FillGcuTensorWithConstant<float>(
        &scale_tensor, dev_ctx, in_scale.to<float>());

    phi::DenseTensor bias_tensor;
    bias_tensor.Resize({1});
    FillGcuTensorWithConstant<float>(
        &bias_tensor, dev_ctx, in_bias.to<float>());

    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};

    input_names["ScaleTensor"] = {"scale_tensor"};
    input_names["bias"] = {"bias_tensor"};
    inputs["ScaleTensor"] = {&scale_tensor};
    inputs["bias"] = {&bias_tensor};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    // attrs["scale"] = in_scale.to<float>();
    // attrs["bias"] = bias;
    attrs["bias_after_scale"] = bias_after_scale;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "scale", dev_ctx);
  }
}

template <typename T, typename Context>
void ScaleSrKernel(const Context& dev_ctx,
                   const phi::SelectedRows& x,
                   const phi::Scalar& scale,
                   const phi::Scalar& bias,
                   bool bias_after_scale,
                   phi::SelectedRows* out) {
  // if (x.value().Holder() != out->value().Holder() ||
  if (x.value().data() != out->value().data()) {
    out->set_rows(x.rows());
    out->set_height(x.height());
  }
  GcuScaleKernel<T, Context>(
      dev_ctx, x.value(), scale, bias, bias_after_scale, out->mutable_value());
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scale,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::GcuScaleKernel,
                          phi::dtype::float16,
                          float,
                          int,
                          int64_t) {}

// PD_REGISTER_PLUGIN_KERNEL(scale_sr,
//                           gcu,
//                           ALL_LAYOUT,
//                           custom_kernel::ScaleSrKernel,
//                           float,
//                           double,
//                           phi::dtype::float16,
//                           uint8_t,
//                           int8_t,
//                           int16_t,
//                           int,
//                           int64_t) {}
