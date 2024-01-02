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

#include "common/common.h"
#include "common/utils.h"
#include "kernels/common_ops/common_ops.h"
#include "kernels/common_ops/elementwise_ops.h"
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_name_list.h"
#include "kernels/funcs/gcu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void BatchNormKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& running_mean,
                     const phi::DenseTensor& running_var,
                     const paddle::optional<phi::DenseTensor>& scale,
                     const paddle::optional<phi::DenseTensor>& bias,
                     bool is_test,
                     float momentum,
                     float epsilon,
                     const std::string& data_layout_str,
                     bool use_global_stats,
                     bool trainable_stats,
                     phi::DenseTensor* y,
                     phi::DenseTensor* mean_out,
                     phi::DenseTensor* variance_out,
                     phi::DenseTensor* saved_mean,
                     phi::DenseTensor* saved_variance,
                     phi::DenseTensor* reserve_space) {
  PADDLE_ENFORCE_EQ(data_layout_str == "NCHW" || data_layout_str == "NHWC",
                    true,
                    phi::errors::InvalidArgument(
                        "The 'data_layout' attribute must be NCHW or NHWC. "
                        "But recevived 'data_layout' is [%s].",
                        data_layout_str));
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size() >= 2 && x_dims.size() <= 5,
      true,
      phi::errors::InvalidArgument(
          "The size of input's dimensions should be between 2 and 5"
          "But received: the size of input's dimensions is [%d]",
          x_dims.size()));

  int C = 1;
  if (x_dims.size() == 2) {
    C = x_dims[1];
  } else {
    C = (data_layout_str == "NCHW") ? x_dims[1] : x_dims[x_dims.size() - 1];
  }

  auto* scale_ptr = scale.get_ptr();
  auto* bias_ptr = bias.get_ptr();

  phi::DenseTensor new_scale;
  phi::DenseTensor new_bias;
  if (scale_ptr) {
    new_scale = scale.get();
  } else {
    new_scale.Resize({C});
    FillGcuTensorWithConstant<T>(&new_scale, dev_ctx, static_cast<T>(1));
  }

  if (bias_ptr) {
    new_bias = bias.get();
  } else {
    new_bias.Resize({C});
    FillGcuTensorWithConstant<T>(&new_bias, dev_ctx, static_cast<T>(0));
  }

  dev_ctx.template Alloc<T>(y);
  dev_ctx.template Alloc<T>(saved_mean);
  dev_ctx.template Alloc<T>(saved_variance);

  phi::DenseTensor mean_out_tmp;
  mean_out_tmp.set_meta(mean_out->meta());
  dev_ctx.template Alloc<T>(&mean_out_tmp);

  phi::DenseTensor variance_out_tmp;
  variance_out_tmp.set_meta(variance_out->meta());
  dev_ctx.template Alloc<T>(&variance_out_tmp);

  if (UseScatterMemory()) {
    std::string op_name =
        (is_test ? kNativeBatchNormInference : kNativeBatchNormTraining);

    PADDLE_GCU_KERNEL_START(dev_ctx, op_name, batch_norm);
    // inputs
    auto x_tmp = x;

    if (data_layout_str == "NCHW") {
      x_tmp =
          ConvertNCHWToNHWC(static_cast<const phi::CustomContext&>(dev_ctx), x);
    }
    auto x_gcu = GetHlirTensorV2(x_tmp, x.dims());
    auto scale_gcu = GetHlirTensor(new_scale);
    auto bias_gcu = GetHlirTensor(new_bias);
    auto running_mean_gcu = GetHlirTensor(running_mean);
    auto running_var_gcu = GetHlirTensor(running_var);

    // outputs
    auto y_tmp = *y;
    if (data_layout_str == "NCHW") {
      auto layout = phi::DataLayout::NHWC;
      auto src_layout = LayoutToVector(y_tmp.layout());
      auto dst_layout = LayoutToVector(layout);
      auto tensor_dims = phi::vectorize(y_tmp.dims());
      std::vector<int64_t> out_permute_dims;
      std::vector<int64_t> out_convert_dims;
      LayoutConvertDims(tensor_dims,
                        src_layout,
                        dst_layout,
                        out_permute_dims,
                        out_convert_dims);

      phi::DenseTensor output_tensor;
      phi::DenseTensorMeta meta(
          y_tmp.dtype(), phi::make_ddim(out_convert_dims), layout);
      output_tensor.set_meta(meta);
      dev_ctx.Alloc(&output_tensor, y_tmp.dtype());
      y_tmp = output_tensor;
    }
    auto y_gcu = GetHlirTensorV2(y_tmp, y->dims());

    hlir::DispatchParam params;
    params.inputs = {
        x_gcu, scale_gcu, bias_gcu, running_mean_gcu, running_var_gcu};
    params.outputs = {y_gcu};
    params.metadata.setValue("bn_mode", int64_t(1));
    params.metadata.setValue("epsilon", epsilon);

    if (!is_test) {
      auto saved_mean_gcu = GetHlirTensor(*saved_mean);
      auto saved_variance_gcu = GetHlirTensor(*saved_variance);
      params.outputs.push_back(saved_mean_gcu);
      params.outputs.push_back(saved_variance_gcu);
      params.metadata.setValue("exponential_average_factor",
                               static_cast<double>(momentum));
    }
    params.stream = static_cast<topsStream_t>(dev_ctx.stream());
    AOTOPS_DEBUG(op_name, params);
    GCUOPS_TRACE_START(batch_norm);
    auto func_ptr = GetOpFuncPtr(op_name, params);
    if (func_ptr) {
      auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
      PADDLE_ENFORCE(
          pass,
          phi::errors::InvalidArgument("dispatch %s failed!", op_name.c_str()));
    } else {
      PADDLE_ENFORCE(false,
                     phi::errors::InvalidArgument("not find aot func for %s",
                                                  op_name.c_str()));
    }
    FreeDispatchParam(params);
    GCUOPS_TRACE_END(batch_norm);
    GcuOpStreamSync(dev_ctx);

    if (data_layout_str == "NCHW") {
      *y = ConvertNHWCToNCHW(static_cast<const phi::CustomContext&>(dev_ctx),
                             y_tmp);
    }

    if (!is_test) {
      auto& x_dims = x_tmp.dims();
      int feature_index = 3;
      int64_t N = 1;
      N = x_dims.at(0) * x_dims.at(1) * x_dims.at(2);
      auto channel_num = x_dims.at(feature_index);

      std::vector<T> v_momentum(channel_num, static_cast<T>(momentum));
      std::vector<T> v_momentum_sub(channel_num, static_cast<T>(1 - momentum));

      phi::DenseTensor v_momentum_tensor;
      v_momentum_tensor.Resize(phi::make_ddim({channel_num}));
      custom_kernel::TensorFromVector(
          dev_ctx, v_momentum, dev_ctx, &v_momentum_tensor);

      phi::DenseTensor v_momentum_sub_tensor;
      v_momentum_sub_tensor.Resize(phi::make_ddim({channel_num}));
      custom_kernel::TensorFromVector(
          dev_ctx, v_momentum_sub, dev_ctx, &v_momentum_sub_tensor);

      std::vector<T> ratio = {static_cast<T>((N) / (N - 1))};

      phi::DenseTensor ratio_tensor;
      ratio_tensor.Resize(phi::make_ddim({1}));
      custom_kernel::TensorFromVector(dev_ctx, ratio, dev_ctx, &ratio_tensor);

      // auto running_mean =
      //     running_mean * v_momentum_tensor + saved_mean *
      //     v_momentum_sub_tensor;

      auto tmp1 = mul_compute(static_cast<const phi::CustomContext&>(dev_ctx),
                              running_mean,
                              v_momentum_tensor);
      auto tmp2 = mul_compute(static_cast<const phi::CustomContext&>(dev_ctx),
                              *saved_mean,
                              v_momentum_sub_tensor);
      add_compute(static_cast<const phi::CustomContext&>(dev_ctx),
                  tmp1,
                  tmp2,
                  &mean_out_tmp);

      // auto running_variance =
      //     running_var * v_momentum_tensor +
      //     saved_variance * ratio_tensor * v_momentum_sub_tensor;
      tmp1 = mul_compute(static_cast<const phi::CustomContext&>(dev_ctx),
                         running_var,
                         v_momentum_tensor);
      tmp2 = mul_compute(static_cast<const phi::CustomContext&>(dev_ctx),
                         *saved_variance,
                         ratio_tensor);
      tmp2 = mul_compute(static_cast<const phi::CustomContext&>(dev_ctx),
                         tmp2,
                         v_momentum_sub_tensor);
      add_compute(static_cast<const phi::CustomContext&>(dev_ctx),
                  tmp1,
                  tmp2,
                  &variance_out_tmp);
    } else {
      *mean_out = running_mean;
      *variance_out = running_var;
      if (saved_mean && saved_mean->numel() > 0) {
        *saved_mean = zeros_like(dev_ctx, running_mean);
      }
      if (saved_variance && saved_variance->numel() > 0) {
        *saved_mean = zeros_like(dev_ctx, running_var);
      }
    }
    PADDLE_GCU_KERNEL_END(op_name, batch_norm);

  } else {
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Mean"] = {"running_mean"};
    input_names["Variance"] = {"running_var"};
    input_names["Bias"] = {"bias"};
    input_names["Scale"] = {"scale"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["Mean"] = {const_cast<DenseTensor*>(&running_mean)};
    inputs["Variance"] = {const_cast<DenseTensor*>(&running_var)};
    inputs["Bias"] = {&new_bias};
    inputs["Scale"] = {&new_scale};

    TensorNameMap output_names;
    output_names["Y"] = {"y"};
    output_names["MeanOut"] = {"mean_out"};
    output_names["VarianceOut"] = {"variance_out"};
    output_names["SavedMean"] = {"saved_mean"};
    output_names["SavedVariance"] = {"saved_variance"};

    TensorValueMap outputs;
    outputs["Y"] = {y};
    outputs["MeanOut"] = {&mean_out_tmp};
    outputs["VarianceOut"] = {&variance_out_tmp};
    outputs["SavedMean"] = {saved_mean};
    outputs["SavedVariance"] = {saved_variance};

    GcuAttributeMap attrs;
    attrs["is_test"] = is_test;
    attrs["momentum"] = momentum;
    attrs["epsilon"] = epsilon;
    attrs["data_layout"] = data_layout_str;
    attrs["use_global_stats"] = use_global_stats;
    attrs["trainable_statistics"] = trainable_stats;

    if (saved_mean && saved_mean->numel() > 0) {
      attrs["init_saved_mean"] = true;
    } else {
      attrs["init_saved_mean"] = false;
    }
    if (saved_variance && saved_variance->numel() > 0) {
      attrs["init_saved_variance"] = true;
    } else {
      attrs["init_saved_variance"] = false;
    }

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "batch_norm",
              dev_ctx);
  }

  TensorCopy(dev_ctx, mean_out_tmp, false, mean_out);
  TensorCopy(dev_ctx, variance_out_tmp, false, variance_out);
}

template <typename T, typename Context>
void BatchNormGradKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& scale,
    const paddle::optional<phi::DenseTensor>& bias,
    const paddle::optional<phi::DenseTensor>& mean,
    const paddle::optional<phi::DenseTensor>& variance,
    const phi::DenseTensor& saved_mean,
    const phi::DenseTensor& saved_variance,
    const paddle::optional<phi::DenseTensor>& reserve_space,
    const phi::DenseTensor& y_grad,
    float momentum,
    float epsilon,
    const std::string& data_layout_str,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics,
    phi::DenseTensor* x_grad,
    phi::DenseTensor* scale_grad,
    phi::DenseTensor* bias_grad) {
  const auto& x_dims = x.dims();
  int C = 1;
  if (x_dims.size() == 2) {
    C = x_dims[1];
  } else {
    C = (data_layout_str == "NCHW") ? x_dims[1] : x_dims[x_dims.size() - 1];
  }

  auto* scale_ptr = scale.get_ptr();
  auto* bias_ptr = bias.get_ptr();

  phi::DenseTensor new_scale;
  phi::DenseTensor new_bias;
  if (scale_ptr) {
    new_scale = scale.get();
  } else {
    new_scale.Resize({C});
    FillGcuTensorWithConstant<T>(&new_scale, dev_ctx, static_cast<T>(1));
  }

  if (bias_ptr) {
    new_bias = bias.get();
  } else {
    new_bias.Resize({C});
    FillGcuTensorWithConstant<T>(&new_bias, dev_ctx, static_cast<T>(0));
  }

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "batch_norm_grad", batch_norm_grad);
    auto x_tmp = x;
    auto y_grad_tmp = y_grad;
    if (data_layout_str == "NCHW") {
      x_tmp =
          ConvertNCHWToNHWC(static_cast<const phi::CustomContext&>(dev_ctx), x);
      y_grad_tmp = ConvertNCHWToNHWC(
          static_cast<const phi::CustomContext&>(dev_ctx), y_grad);
    }
    auto x_gcu = GetHlirTensorV2(x_tmp, x.dims());
    auto y_grad_gcu = GetHlirTensorV2(y_grad_tmp, y_grad.dims());
    auto scale_gcu = GetHlirTensor(new_scale);
    auto saved_mean_gcu = GetHlirTensor(saved_mean);
    auto saved_variance_gcu = GetHlirTensor(saved_variance);
    hlir::DispatchParam params;
    params.inputs = {
        x_gcu, y_grad_gcu, scale_gcu, saved_mean_gcu, saved_variance_gcu};
    phi::DenseTensor x_grad_tmp;
    if (x_grad) {
      dev_ctx.template Alloc<T>(x_grad);
      x_grad_tmp = *x_grad;
      if (data_layout_str == "NCHW") {
        auto layout = phi::DataLayout::NHWC;
        auto src_layout = LayoutToVector(x_grad_tmp.layout());
        auto dst_layout = LayoutToVector(layout);
        auto tensor_dims = phi::vectorize(x_grad_tmp.dims());
        std::vector<int64_t> out_permute_dims;
        std::vector<int64_t> out_convert_dims;
        LayoutConvertDims(tensor_dims,
                          src_layout,
                          dst_layout,
                          out_permute_dims,
                          out_convert_dims);

        phi::DenseTensor output_tensor;
        phi::DenseTensorMeta meta(
            x_grad_tmp.dtype(), phi::make_ddim(out_convert_dims), layout);
        output_tensor.set_meta(meta);
        dev_ctx.Alloc(&output_tensor, x_grad_tmp.dtype());
        x_grad_tmp = output_tensor;
      }
      auto x_grad_gcu = GetHlirTensorV2(x_grad_tmp, x_grad->dims());
      params.outputs.push_back(x_grad_gcu);
    }

    phi::DenseTensor scale_grad_tmp;
    phi::DenseTensor bias_grad_tmp;
    if (scale_grad) {
      dev_ctx.template Alloc<T>(scale_grad);
      scale_grad_tmp = *scale_grad;
    } else {
      scale_grad_tmp.set_meta(new_scale.meta());
      dev_ctx.template Alloc<T>(&scale_grad_tmp);
    }

    auto scale_grad_gcu = GetHlirTensor(scale_grad_tmp);
    params.outputs.push_back(scale_grad_gcu);
    if (bias_grad) {
      dev_ctx.template Alloc<T>(bias_grad);
      bias_grad_tmp = *bias_grad;
    } else {
      bias_grad_tmp.set_meta(new_bias.meta());
      dev_ctx.template Alloc<T>(&bias_grad_tmp);
    }
    auto bias_grad_gcu = GetHlirTensor(bias_grad_tmp);
    params.outputs.push_back(bias_grad_gcu);

    params.metadata.setValue("bn_mode", static_cast<int64_t>(1));
    params.metadata.setValue("epsilon", epsilon);

    params.stream = static_cast<topsStream_t>(dev_ctx.stream());
    AOTOPS_DEBUG(kNativeBatchNormBackward, params);
    GCUOPS_TRACE_START(batch_norm_grad);
    auto func_ptr = GetOpFuncPtr(kNativeBatchNormBackward, params);
    if (func_ptr) {
      auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
      PADDLE_ENFORCE(pass,
                     phi::errors::InvalidArgument("dispatch %s failed!",
                                                  kNativeBatchNormBackward));
    } else {
      PADDLE_ENFORCE(false,
                     phi::errors::InvalidArgument("not find aot func for %s",
                                                  kNativeBatchNormBackward));
    }
    FreeDispatchParam(params);
    GCUOPS_TRACE_END(batch_norm_grad);
    GcuOpStreamSync(dev_ctx);

    if (x_grad) {
      if (data_layout_str == "NCHW") {
        *x_grad = ConvertNHWCToNCHW(
            static_cast<const phi::CustomContext&>(dev_ctx), x_grad_tmp);
      }
    }
    PADDLE_GCU_KERNEL_END("batch_norm_grad", batch_norm_grad);
  } else {
    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["SavedMean"] = {"saved_mean"};
    input_names["SavedVariance"] = {"saved_variance"};
    input_names["Bias"] = {"bias"};
    input_names["Scale"] = {"scale"};
    input_names[GradVarName("Y")] = {"y_grad"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&x)};
    inputs["SavedMean"] = {const_cast<DenseTensor*>(&saved_mean)};
    inputs["SavedVariance"] = {const_cast<DenseTensor*>(&saved_variance)};
    inputs["Bias"] = {&new_bias};
    inputs["Scale"] = {&new_scale};
    inputs[GradVarName("Y")] = {const_cast<DenseTensor*>(&y_grad)};

    TensorNameMap output_names;
    TensorValueMap outputs;
    if (x_grad) {
      dev_ctx.template Alloc<T>(x_grad);
      output_names[GradVarName("X")] = {"x_grad"};
      outputs[GradVarName("X")] = {x_grad};
    }
    if (scale_grad) {
      dev_ctx.template Alloc<T>(scale_grad);
      output_names[GradVarName("Scale")] = {"scale_grad"};
      outputs[GradVarName("Scale")] = {scale_grad};
    }
    if (bias_grad) {
      dev_ctx.template Alloc<T>(bias_grad);
      output_names[GradVarName("Bias")] = {"bias_grad"};
      outputs[GradVarName("Bias")] = {bias_grad};
    }

    GcuAttributeMap attrs;
    attrs["is_test"] = is_test;
    attrs["momentum"] = momentum;
    attrs["epsilon"] = epsilon;
    attrs["data_layout"] = data_layout_str;
    attrs["use_global_stats"] = use_global_stats;
    attrs["trainable_statistics"] = trainable_statistics;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "batch_norm_grad",
              dev_ctx);
  }
}

template <typename T, typename Context>
void BatchNormInferKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& mean,
                          const phi::DenseTensor& variance,
                          const phi::DenseTensor& scale,
                          const phi::DenseTensor& bias,
                          float momentum,
                          float epsilon,
                          const std::string& data_layout_str,
                          phi::DenseTensor* y,
                          phi::DenseTensor* mean_out,
                          phi::DenseTensor* variance_out) {
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size() >= 2 && x_dims.size() <= 5,
      true,
      phi::errors::InvalidArgument(
          "The size of input's dimensions should be between 2 and 5"
          "But received: the size of input's dimensions is [%d]",
          x_dims.size()));

  dev_ctx.template Alloc<T>(y);

  TensorNameMap input_names;
  input_names["X"] = {"x"};
  input_names["Mean"] = {"mean"};
  input_names["Variance"] = {"var"};
  input_names["Bias"] = {"bias"};
  input_names["Scale"] = {"scale"};

  TensorValueMap inputs;
  inputs["X"] = {const_cast<DenseTensor*>(&x)};
  inputs["Mean"] = {const_cast<DenseTensor*>(&mean)};
  inputs["Variance"] = {const_cast<DenseTensor*>(&variance)};
  inputs["Bias"] = {const_cast<DenseTensor*>(&bias)};
  inputs["Scale"] = {const_cast<DenseTensor*>(&scale)};

  TensorNameMap output_names;
  output_names["Y"] = {"y"};

  TensorValueMap outputs;
  outputs["Y"] = {y};

  GcuAttributeMap attrs;
  attrs["epsilon"] = epsilon;
  attrs["data_layout"] = data_layout_str;

  GcuRunner(input_names,
            inputs,
            output_names,
            outputs,
            attrs,
            "batch_norm_infer",
            dev_ctx);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(batch_norm,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormKernel,
                          float,
                          double,
                          phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);   // mean
    kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);   // variance
    kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);   // scale
    kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);   // bias
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // mean_out
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // variance_out
    kernel->OutputAt(3).SetDataType(phi::DataType::FLOAT32);  // saved_mean
    kernel->OutputAt(4).SetDataType(phi::DataType::FLOAT32);  // saved_variance
  }
}

PD_REGISTER_PLUGIN_KERNEL(batch_norm_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormGradKernel,
                          float,
                          double,
                          phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);  // x_grad
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // scale_grad
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // bias_grad
  }
}

PD_REGISTER_PLUGIN_KERNEL(batch_norm_infer,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormInferKernel,
                          float,
                          double,
                          phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // mean_out
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // variance_out
  }
}
