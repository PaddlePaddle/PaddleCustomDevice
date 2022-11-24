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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

using Tensor = phi::DenseTensor;
using DDim = phi::DDim;

using DataLayout = phi::DataLayout;

template <typename T>
class NormDataType;

template <>
class NormDataType<phi::dtype::float16> {
 public:
  // The scaling param type is float for HALF and FLOAT tensors
  using ScalingParamType = const float;
  using BatchNormParamType = float;
};

template <>
class NormDataType<float> {
 public:
  using ScalingParamType = const float;
  using BatchNormParamType = float;
};

template <typename T>
using NormDataType = NormDataType<T>;
template <typename T>
using LayerNormParamType = typename NormDataType<T>::BatchNormParamType;

template <typename T, typename Context>
void LayerNormNPUKernel(const Context& dev_ctx,
                        const phi::DenseTensor& x,
                        const paddle::optional<phi::DenseTensor>& scale_opt,
                        const paddle::optional<phi::DenseTensor>& bias_opt,
                        float epsilon,
                        int begin_norm_axis,
                        bool is_test,
                        phi::DenseTensor* out,
                        phi::DenseTensor* mean,
                        phi::DenseTensor* variance) {
  using U = LayerNormParamType<T>;
  auto* scale = scale_opt.get_ptr();
  auto* bias = bias_opt.get_ptr();
  const auto& x_dims = x.dims();
  std::vector<int> axes;
  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int right = static_cast<int>(matrix_dim[1]);

  // The shape of scale and bias should be equal to x.shape[begin_norm_axis:],
  // required by npu.
  for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
    axes.push_back(x_dims[i]);
  }

  auto stream = dev_ctx.stream();

  Tensor default_scale;
  if (!scale) {
    phi::DenseTensorMeta default_scale_meta = {x.dtype(), phi::make_ddim(axes)};
    default_scale.set_meta(default_scale_meta);
    dev_ctx.template Alloc<T>(&default_scale);

    Tensor value;
    phi::DenseTensorMeta value_meta = {x.dtype(), phi::make_ddim({1})};
    value.set_meta(value_meta);
    dev_ctx.template Alloc<T>(&value);

    FillNpuTensorWithConstant<T>(&value, dev_ctx, static_cast<T>(1.0));

    NpuOpRunner runner;
    runner.SetType("Fill")
        .AddInput(dev_ctx, std::move(axes))
        .AddInput(value)
        .AddOutput(default_scale);
    runner.Run(stream);
    scale = &default_scale;
  } else {
    const_cast<Tensor*>(scale)->Resize(phi::make_ddim(axes));
  }

  Tensor default_bias;
  if (!bias) {
    phi::DenseTensorMeta default_bias_meta = {x.dtype(), phi::make_ddim(axes)};
    default_bias.set_meta(default_bias_meta);
    dev_ctx.template Alloc<T>(&default_bias);

    Tensor value;
    phi::DenseTensorMeta value_meta = {x.dtype(), phi::make_ddim({1})};
    value.set_meta(value_meta);
    dev_ctx.template Alloc<T>(&value);

    FillNpuTensorWithConstant<T>(&value, dev_ctx, static_cast<T>(0));

    NpuOpRunner runner;
    runner.SetType("Fill")
        .AddInput(dev_ctx, std::move(axes))
        .AddInput(value)
        .AddOutput(default_bias);
    runner.Run(stream);
    bias = &default_bias;
  } else {
    const_cast<Tensor*>(bias)->Resize(phi::make_ddim(axes));
  }

  // cast scale from LayerNormParamType to T if needed
  Tensor cast_scale;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      scale->dtype() == phi::DataType::FLOAT32) {
    cast_scale.Resize(scale->dims());
    dev_ctx.template Alloc<T>(&cast_scale);

    auto dst_dtype = ConvertToNpuDtype(x.dtype());
    const auto& runner_cast_scale =
        NpuOpRunner("Cast",
                    {*scale},
                    {cast_scale},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_scale.Run(stream);
  } else {
    cast_scale = *scale;
  }

  // cast bias from LayerNormParamType to T if needed
  Tensor cast_bias;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      bias->dtype() == phi::DataType::FLOAT32) {
    cast_bias.Resize(bias->dims());
    dev_ctx.template Alloc<T>(&cast_bias);

    auto dst_dtype = ConvertToNpuDtype(x.dtype());
    const auto& runner_cast_bias =
        NpuOpRunner("Cast",
                    {*bias},
                    {cast_bias},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_bias.Run(stream);
  } else {
    cast_bias = *bias;
  }

  dev_ctx.template Alloc<T>(out);

  // mean should be of  U type
  Tensor* tmp_mean = mean;
  Tensor cast_mean;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      (scale->dtype() == phi::DataType::FLOAT32 ||
       bias->dtype() == phi::DataType::FLOAT32)) {
    cast_mean.Resize(mean->dims());
    dev_ctx.template Alloc<T>(&cast_mean);
    tmp_mean = &cast_mean;
    dev_ctx.template Alloc<U>(mean);
  } else {
    dev_ctx.template Alloc<T>(mean);
  }

  // same for variance
  Tensor* tmp_variance = variance;
  Tensor cast_variance;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      (scale->dtype() == phi::DataType::FLOAT32 ||
       bias->dtype() == phi::DataType::FLOAT32)) {
    cast_variance.Resize(variance->dims());
    dev_ctx.template Alloc<T>(&cast_variance);
    tmp_variance = &cast_variance;
    dev_ctx.template Alloc<U>(variance);
  } else {
    dev_ctx.template Alloc<T>(variance);
  }

  const auto& runner = NpuOpRunner("LayerNorm",
                                   {x, cast_scale, cast_bias},
                                   {*out, *tmp_mean, *tmp_variance},
                                   {{"begin_norm_axis", begin_norm_axis},
                                    {"begin_params_axis", begin_norm_axis},
                                    {"epsilon", epsilon}});
  runner.Run(stream);

  // cast back from FLOAT16 to FLOAT32
  if (x.dtype() == phi::DataType::FLOAT16 &&
      mean->dtype() == phi::DataType::FLOAT32) {
    auto dst_dtype = ConvertToNpuDtype(mean->dtype());
    const auto& runner_cast_mean =
        NpuOpRunner("Cast",
                    {*tmp_mean},
                    {*mean},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_mean.Run(stream);
  }
  // same for variance
  if (x.dtype() == phi::DataType::FLOAT16 &&
      variance->dtype() == phi::DataType::FLOAT32) {
    auto dst_dtype = ConvertToNpuDtype(variance->dtype());
    const auto& runner_cast_variance =
        NpuOpRunner("Cast",
                    {*tmp_variance},
                    {*variance},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_variance.Run(stream);
  }

  // revert shape of scale and bias
  // TODO(zhiqiu): better implementation, use tmp tensor to avoid write input
  // tensor.
  const_cast<Tensor*>(scale)->Resize(phi::make_ddim({right}));
  const_cast<Tensor*>(bias)->Resize(phi::make_ddim({right}));
}

template <typename T, typename Context>
void LayerNormGradNPUKernel(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const paddle::optional<phi::DenseTensor>& scale_opt,
                            const paddle::optional<phi::DenseTensor>& bias,
                            const phi::DenseTensor& mean,
                            const phi::DenseTensor& variance,
                            const phi::DenseTensor& out_grad,
                            float epsilon,
                            int begin_norm_axis,
                            bool is_test,
                            phi::DenseTensor* x_grad,
                            phi::DenseTensor* scale_grad,
                            phi::DenseTensor* bias_grad) {
  using U = LayerNormParamType<T>;
  const auto& x_dims = x.dims();
  auto* scale = scale_opt.get_ptr();

  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int right = static_cast<int>(matrix_dim[1]);

  std::vector<int> axes;
  for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
    axes.push_back(x_dims[i]);
  }

  auto stream = dev_ctx.stream();

  // No need to compute any gradient, jusr return
  if (!x_grad && !scale_grad && !bias_grad) {
    return;
  }

  // The rank of mean should be equal to x, required by npu.
  std::vector<int> new_shape;
  for (auto i = 0; i < begin_norm_axis; ++i) {
    new_shape.push_back(x_dims[i]);
  }
  for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
    new_shape.push_back(1);
  }

  auto mean_dims = mean.dims();
  const_cast<Tensor*>(&mean)->Resize(phi::make_ddim({new_shape}));
  const_cast<Tensor*>(&variance)->Resize(phi::make_ddim({new_shape}));

  Tensor default_scale;
  if (!scale) {
    phi::DenseTensorMeta default_scale_meta = {x.dtype(), phi::make_ddim(axes)};
    default_scale.set_meta(default_scale_meta);
    dev_ctx.template Alloc<T>(&default_scale);

    Tensor value;
    phi::DenseTensorMeta value_meta = {x.dtype(), phi::make_ddim({1})};
    value.set_meta(value_meta);
    dev_ctx.template Alloc<T>(&value);

    FillNpuTensorWithConstant<T>(&value, dev_ctx, static_cast<T>(1.0));

    NpuOpRunner runner;
    runner.SetType("Fill")
        .AddInput(dev_ctx, std::move(axes))
        .AddInput(value)
        .AddOutput(default_scale);
    runner.Run(stream);
    scale = &default_scale;
  } else {
    const_cast<Tensor*>(scale)->Resize(phi::make_ddim(axes));
  }

  // cast scale from LayerNormParamType to T if needed
  Tensor cast_scale;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      scale->dtype() == phi::DataType::FLOAT32) {
    cast_scale.Resize(scale->dims());
    dev_ctx.template Alloc<T>(&cast_scale);

    auto dst_dtype = ConvertToNpuDtype(x.dtype());
    const auto& runner_cast_scale =
        NpuOpRunner("Cast",
                    {*scale},
                    {cast_scale},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_scale.Run(stream);
  } else {
    cast_scale = *scale;
  }

  // cast mean from LayerNormParamType to T if needed
  Tensor cast_mean;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      mean.dtype() == phi::DataType::FLOAT32) {
    cast_mean.Resize(mean.dims());
    dev_ctx.template Alloc<T>(&cast_mean);
    auto dst_dtype = ConvertToNpuDtype(x.dtype());
    const auto& runner_cast_mean =
        NpuOpRunner("Cast",
                    {mean},
                    {cast_mean},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_mean.Run(stream);
  } else {
    cast_mean = mean;
  }

  // cast variance from LayerNormParamType to T if needed
  Tensor cast_variance;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      variance.dtype() == phi::DataType::FLOAT32) {
    cast_variance.Resize(variance.dims());
    dev_ctx.template Alloc<T>(&cast_variance);
    auto dst_dtype = ConvertToNpuDtype(x.dtype());
    const auto& runner_cast_variance =
        NpuOpRunner("Cast",
                    {variance},
                    {cast_variance},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_variance.Run(stream);
  } else {
    cast_variance = variance;
  }

  Tensor x_grad_, scale_grad_, bias_grad_;
  x_grad = (x_grad == nullptr) ? &x_grad_ : x_grad;
  scale_grad = (scale_grad == nullptr) ? &scale_grad_ : scale_grad;
  bias_grad = (bias_grad == nullptr) ? &bias_grad_ : bias_grad;

  x_grad->Resize(x.dims());
  scale_grad->Resize(phi::make_ddim(axes));
  bias_grad->Resize(phi::make_ddim(axes));

  dev_ctx.template Alloc<T>(x_grad);
  // scale_grad should be of  U type
  Tensor* tmp_scale_grad = scale_grad;
  Tensor cast_scale_grad;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      (mean.dtype() == phi::DataType::FLOAT32 ||
       variance.dtype() == phi::DataType::FLOAT32)) {
    cast_scale_grad.Resize(scale_grad->dims());
    dev_ctx.template Alloc<T>(&cast_scale_grad);
    tmp_scale_grad = &cast_scale_grad;
    dev_ctx.template Alloc<U>(scale_grad);
  } else {
    dev_ctx.template Alloc<T>(scale_grad);
  }

  // same for bias_grad
  Tensor* tmp_bias_grad = bias_grad;
  Tensor cast_bias_grad;
  if (x.dtype() == phi::DataType::FLOAT16 &&
      (mean.dtype() == phi::DataType::FLOAT32 ||
       variance.dtype() == phi::DataType::FLOAT32)) {
    cast_bias_grad.Resize(bias_grad->dims());
    dev_ctx.template Alloc<T>(&cast_bias_grad);
    tmp_bias_grad = &cast_bias_grad;
    dev_ctx.template Alloc<U>(bias_grad);
  } else {
    dev_ctx.template Alloc<T>(bias_grad);
  }

  const auto& runner =
      NpuOpRunner("LayerNormGrad",
                  {out_grad, x, cast_variance, cast_mean, cast_scale},
                  {*x_grad, *tmp_scale_grad, *tmp_bias_grad},
                  {});
  runner.Run(stream);

  // cast back from FLOAT16 to FLOAT32
  if (x.dtype() == phi::DataType::FLOAT16 &&
      scale_grad->dtype() == phi::DataType::FLOAT32) {
    auto dst_dtype = ConvertToNpuDtype(scale_grad->dtype());
    const auto& runner_cast_scale_grad =
        NpuOpRunner("Cast",
                    {*tmp_scale_grad},
                    {*scale_grad},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_scale_grad.Run(stream);
  }
  // same for bias_grad
  if (x.dtype() == phi::DataType::FLOAT16 &&
      bias_grad->dtype() == phi::DataType::FLOAT32) {
    auto dst_dtype = ConvertToNpuDtype(bias_grad->dtype());
    const auto& runner_cast_bias_grad =
        NpuOpRunner("Cast",
                    {*tmp_bias_grad},
                    {*bias_grad},
                    {{"dst_type", static_cast<int>(dst_dtype)}});
    runner_cast_bias_grad.Run(stream);
  }

  const_cast<Tensor*>(&mean)->Resize(mean_dims);
  const_cast<Tensor*>(&variance)->Resize(mean_dims);
  const_cast<Tensor*>(scale)->Resize(phi::make_ddim({right}));
  scale_grad->Resize(phi::make_ddim({right}));
  bias_grad->Resize(phi::make_ddim({right}));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(layer_norm,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LayerNormNPUKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(layer_norm_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::LayerNormGradNPUKernel,
                          float,
                          phi::dtype::float16) {}
