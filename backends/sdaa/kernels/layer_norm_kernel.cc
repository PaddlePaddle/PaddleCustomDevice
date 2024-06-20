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

#include <vector>

#include "kernels/funcs/sdaa_baseop.h"
#include "kernels/funcs/sdaa_funcs.h"
#include "paddle/phi/extension.h"
#include "runtime/runtime.h"
#include "tecodnn.h"  // NOLINT

namespace custom_kernel {

template <typename T, typename Context>
void LayerNormKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const paddle::optional<phi::DenseTensor>& scale_opt,
                     const paddle::optional<phi::DenseTensor>& bias_opt,
                     float epsilon,
                     int begin_norm_axis,
                     phi::DenseTensor* out,
                     phi::DenseTensor* mean,
                     phi::DenseTensor* variance) {
  VLOG(4) << "Call SDAA LayerNormKernel";

  // check argument
  PADDLE_ENFORCE_GT(
      epsilon,
      0.,
      phi::errors::InvalidArgument("epsilon should be greater than zero. "
                                   "But received epsilon = %f",
                                   static_cast<float>(epsilon)));

  // allocate memory for outputs
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<float>(mean);
  dev_ctx.template Alloc<float>(variance);

  // get axis settings for scale & bias
  // The shape of scale and bias should be equal to x.shape[begin_norm_axis:]
  auto* scale = scale_opt.get_ptr();  // gamma
  auto* bias = bias_opt.get_ptr();    // beta
  const auto& x_dims = x.dims();
  std::vector<int64_t> axes;
  for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
    axes.push_back(x_dims[i]);
  }
  std::vector<int64_t> pre_axes;
  for (auto i = 0; i < begin_norm_axis; ++i) {
    pre_axes.push_back(x_dims[i]);
  }
  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int right = static_cast<int>(matrix_dim[1]);

  // set scale to all ones if its none
  phi::DenseTensor default_scale;
  if (!scale) {
    phi::DenseTensorMeta default_scale_meta = {x.dtype(), phi::make_ddim(axes)};
    default_scale.set_meta(default_scale_meta);
    dev_ctx.template Alloc<float>(&default_scale);
    sdaa_ops::doFillTensor<float>(
        dev_ctx, static_cast<float>(1.0), DataType::FLOAT32, &default_scale);
    scale = &default_scale;  // shallow copy to scale
  } else {
    const_cast<phi::DenseTensor*>(scale)->Resize(phi::make_ddim(axes));
  }

  // set bias to all zeros if its none
  phi::DenseTensor default_bias;
  if (!bias) {
    phi::DenseTensorMeta default_bias_meta = {x.dtype(), phi::make_ddim(axes)};
    default_bias.set_meta(default_bias_meta);
    dev_ctx.template Alloc<float>(&default_bias);
    sdaa_ops::doFillTensor<float>(
        dev_ctx, static_cast<float>(0.0), DataType::FLOAT32, &default_bias);
    bias = &default_bias;  // shallow copy to bias
  } else {
    const_cast<phi::DenseTensor*>(bias)->Resize(phi::make_ddim(axes));
  }

  // calculate row and col according to input's shape and axis
  int row = std::accumulate(
      std::begin(pre_axes), std::end(pre_axes), 1, std::multiplies<int>());
  int col = std::accumulate(std::begin(axes),
                            std::end(axes),
                            1,
                            std::multiplies<int>());  // multiply integers

  // set data layouts
  std::vector<int> xy_dims = {row, col, 1, 1};
  std::vector<int> mean_dims = {row, 1, 1, 1};
  std::vector<int> gamma_dims = {1, col, 1, 1};

  // set descriptors
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnLayerMode_t lnMode = TECODNN_LAYER_NORM_0;

  tecodnnTensorDescriptor_t x_Desc =
      sdaa_ops::GetTecodnnTensorDesc(xy_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t y_Desc =
      sdaa_ops::GetTecodnnTensorDesc(xy_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t mean_Desc = sdaa_ops::GetTecodnnTensorDesc(
      mean_dims, DataType::FLOAT32, TensorFormat::NHWC);
  tecodnnTensorDescriptor_t rstd_Desc = sdaa_ops::GetTecodnnTensorDesc(
      mean_dims, DataType::FLOAT32, TensorFormat::NHWC);
  tecodnnTensorDescriptor_t gamma_Desc = sdaa_ops::GetTecodnnTensorDesc(
      gamma_dims, DataType::FLOAT32, TensorFormat::NHWC);
  tecodnnTensorDescriptor_t beta_Desc = sdaa_ops::GetTecodnnTensorDesc(
      gamma_dims, DataType::FLOAT32, TensorFormat::NHWC);

  // execute layer-norm forward
  TECODNN_CHECK(tecodnnLayerNormForward(tecodnnHandle,
                                        lnMode,
                                        epsilon,
                                        x_Desc,
                                        x.data(),
                                        gamma_Desc,
                                        scale->data(),
                                        beta_Desc,
                                        bias->data(),
                                        y_Desc,
                                        out->data(),
                                        mean_Desc,
                                        mean->data(),
                                        rstd_Desc,
                                        variance->data()));

  // resize scale and bias
  const_cast<phi::DenseTensor*>(scale)->Resize(phi::make_ddim({right}));
  const_cast<phi::DenseTensor*>(bias)->Resize(phi::make_ddim({right}));

  // destroy descriptors
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(mean_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(rstd_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(gamma_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(beta_Desc));
}

template <typename T, typename Context>
void LayerNormGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const paddle::optional<phi::DenseTensor>& scale_opt,
                         const paddle::optional<phi::DenseTensor>& bias,
                         const phi::DenseTensor& mean,
                         const phi::DenseTensor& variance,
                         const phi::DenseTensor& out_grad,
                         float epsilon,
                         int begin_norm_axis,
                         phi::DenseTensor* x_grad,
                         phi::DenseTensor* scale_grad,
                         phi::DenseTensor* bias_grad) {
  VLOG(4) << "Call SDAA LayerNormGradKernel";

  // check argument
  PADDLE_ENFORCE_GT(
      epsilon,
      0.,
      phi::errors::InvalidArgument("epsilon should be greater than zero. "
                                   "But received epsilon = %f",
                                   static_cast<float>(epsilon)));

  // At this case, no need to compute any gradient, jusr return
  if (!x_grad && !scale_grad && !bias_grad) {
    return;
  }

  // get axis settings for scale & bias
  const auto& x_dims = x.dims();
  auto* scale = scale_opt.get_ptr();
  std::vector<int64_t> axes;
  for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
    axes.push_back(x_dims[i]);
  }
  std::vector<int64_t> pre_axes;
  for (auto i = 0; i < begin_norm_axis; ++i) {
    pre_axes.push_back(x_dims[i]);
  }
  auto matrix_dim = phi::flatten_to_2d(x_dims, begin_norm_axis);
  int right = static_cast<int>(matrix_dim[1]);

  // get axis settings for mean & variance
  std::vector<int> new_shape;
  for (auto i = 0; i < begin_norm_axis; ++i) {
    new_shape.push_back(x_dims[i]);
  }
  for (auto i = begin_norm_axis; i < x_dims.size(); ++i) {
    new_shape.push_back(1);  // fill the rest dims with 1
  }

  // The rank of mean should be equal to x
  auto mean_dims = mean.dims();  // save its original dims
  const_cast<phi::DenseTensor*>(&mean)->Resize(phi::make_ddim({new_shape}));
  const_cast<phi::DenseTensor*>(&variance)->Resize(phi::make_ddim({new_shape}));

  // set scale to all ones if its none
  phi::DenseTensor default_scale;
  if (!scale) {
    phi::DenseTensorMeta default_scale_meta = {x.dtype(), phi::make_ddim(axes)};
    default_scale.set_meta(default_scale_meta);
    dev_ctx.template Alloc<float>(&default_scale);
    sdaa_ops::doFillTensor<float>(
        dev_ctx, static_cast<float>(1.0), DataType::FLOAT32, &default_scale);
    scale = &default_scale;  // shallow copy to scale
  } else {
    const_cast<phi::DenseTensor*>(scale)->Resize(phi::make_ddim(axes));
  }

  // set and allocate memory for outputs
  phi::DenseTensor x_grad_, scale_grad_, bias_grad_;
  x_grad = (x_grad == nullptr) ? &x_grad_ : x_grad;
  scale_grad = (scale_grad == nullptr) ? &scale_grad_ : scale_grad;
  bias_grad = (bias_grad == nullptr) ? &bias_grad_ : bias_grad;
  x_grad->Resize(x.dims());
  scale_grad->Resize(phi::make_ddim(axes));
  bias_grad->Resize(phi::make_ddim(axes));
  dev_ctx.template Alloc<T>(x_grad);
  dev_ctx.template Alloc<float>(scale_grad);
  dev_ctx.template Alloc<float>(bias_grad);

  // calculate row and col according to input's shape and axis
  int row = std::accumulate(
      std::begin(pre_axes), std::end(pre_axes), 1, std::multiplies<int>());
  int col = std::accumulate(std::begin(axes),
                            std::end(axes),
                            1,
                            std::multiplies<int>());  // multiply integers

  // set data layouts
  std::vector<int> xy_dims = {row, col, 1, 1};
  std::vector<int> var_dims = {row, 1, 1, 1};
  std::vector<int> gamma_dims = {1, col, 1, 1};

  // set descriptors
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnLayerMode_t lnMode = TECODNN_LAYER_NORM_0;

  tecodnnTensorDescriptor_t x_Desc =
      sdaa_ops::GetTecodnnTensorDesc(xy_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t y_Desc =
      sdaa_ops::GetTecodnnTensorDesc(xy_dims, x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t mean_Desc = sdaa_ops::GetTecodnnTensorDesc(
      var_dims, DataType::FLOAT32, TensorFormat::NHWC);
  tecodnnTensorDescriptor_t rstd_Desc = sdaa_ops::GetTecodnnTensorDesc(
      var_dims, DataType::FLOAT32, TensorFormat::NHWC);
  tecodnnTensorDescriptor_t gamma_Desc = sdaa_ops::GetTecodnnTensorDesc(
      gamma_dims, DataType::FLOAT32, TensorFormat::NHWC);
  tecodnnTensorDescriptor_t beta_Desc = sdaa_ops::GetTecodnnTensorDesc(
      gamma_dims, DataType::FLOAT32, TensorFormat::NHWC);

  // execute layer-norm backward
  TECODNN_CHECK(tecodnnLayerNormBackward(tecodnnHandle,
                                         lnMode,
                                         y_Desc,
                                         out_grad.data(),
                                         x_Desc,
                                         x.data(),
                                         mean_Desc,
                                         mean.data(),
                                         rstd_Desc,
                                         variance.data(),
                                         gamma_Desc,
                                         scale->data(),
                                         x_Desc,
                                         x_grad->data(),
                                         gamma_Desc,
                                         scale_grad->data(),
                                         beta_Desc,
                                         bias_grad->data()));

  // resize tensors
  const_cast<phi::DenseTensor*>(&mean)->Resize(mean_dims);
  const_cast<phi::DenseTensor*>(&variance)->Resize(mean_dims);
  const_cast<phi::DenseTensor*>(scale)->Resize(phi::make_ddim({right}));
  scale_grad->Resize(phi::make_ddim({right}));
  bias_grad->Resize(phi::make_ddim({right}));

  // destroy descriptors
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(mean_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(rstd_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(gamma_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(beta_Desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(layer_norm,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LayerNormKernel,
                          float,
                          phi::dtype::float16) {
  kernel->OutputAt(1).SetDataType(phi::DataType::UNDEFINED);
  kernel->OutputAt(2).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(layer_norm_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::LayerNormGradKernel,
                          float,
                          phi::dtype::float16) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);
  }
}
