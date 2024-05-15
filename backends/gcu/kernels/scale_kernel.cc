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

#include "common/gcu_op_runner.h"
#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {
namespace {
// topsaten binary not support int tensor and float scale/bais now
const std::unordered_map<phi::DataType, phi::DataType> kBinaryDtypeTrans = {
    {phi::DataType::INT64, phi::DataType::FLOAT32},
    {phi::DataType::INT32, phi::DataType::FLOAT32},
    {phi::DataType::FLOAT64, phi::DataType::FLOAT32},
};
}  // namespace

template <typename T, typename Context>
void ScaleKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 const phi::Scalar& in_scale,
                 const phi::Scalar& in_bias,
                 bool bias_after_scale,
                 phi::DenseTensor* out) {
  PADDLE_GCU_KERNEL_TRACE("scale");
  dev_ctx.template Alloc<T>(out);
  if (LaunchAOTKernel()) {
    auto input_scale = in_scale;
    if (in_scale.dtype() == phi::DataType::FLOAT64) {
      input_scale = phi::Scalar(static_cast<float>(in_scale.to<double>()));
    }
    auto input_bias = in_bias;
    if (in_scale.dtype() == phi::DataType::FLOAT64) {
      input_bias = phi::Scalar(static_cast<float>(in_bias.to<double>()));
    }
    phi::DenseTensor input_x =
        MaybeCreateOrTrans(dev_ctx, x, kBinaryDtypeTrans);
    phi::DenseTensor output =
        MaybeCreateOrTrans(dev_ctx, *out, kBinaryDtypeTrans, false);
    if (bias_after_scale) {
      // Out = scale ∗ X + bias
      LAUNCH_TOPSATENOP(
          topsatenAdd, dev_ctx, output, input_bias, input_x, input_scale);
    } else {
      // Out = scale ∗ (X + bias)
      auto tmp_scalar = phi::Scalar(1.0f);
      phi::DenseTensor tmp_out = TensorEmpty(dev_ctx, output.meta());
      LAUNCH_TOPSATENOP(
          topsatenAdd, dev_ctx, tmp_out, input_x, input_bias, tmp_scalar);
      LAUNCH_TOPSATENOP(topsatenMul, dev_ctx, output, input_scale, tmp_out);
    }
    MaybeTransResult(dev_ctx, output, out);

  } else {  // kernel impl base on JIT
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

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(scale,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::ScaleKernel,
                          phi::dtype::float16,
                          float,
                          int,
                          int64_t) {}
