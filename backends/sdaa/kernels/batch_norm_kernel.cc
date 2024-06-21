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

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

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
  VLOG(4) << "Call SDAA BatchNormInferKernel";

  // allocate memory for outputs
  dev_ctx.template Alloc<T>(y);
  // input and output mean/variance must share the memory
  *mean_out = mean;
  *variance_out = variance;
  dev_ctx.template Alloc<float>(mean_out);
  dev_ctx.template Alloc<float>(variance_out);

  sdaa_ops::BatchNormFunc(dev_ctx,
                          x,
                          mean,
                          variance,
                          scale,
                          bias,
                          momentum,
                          epsilon,
                          false,  // inference
                          data_layout_str,
                          y,
                          mean_out,
                          variance_out,
                          nullptr,
                          nullptr);
}

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
  // if training is False, use global mean and std
  bool test_mode = is_test && (!trainable_stats);
  // current tecodnnAPI does not support parameter use_global_stats=True
  bool training = !test_mode && !use_global_stats;
  const auto& x_dims = x.dims();
  auto* Scale = scale.get_ptr();
  auto* Bias = bias.get_ptr();

  phi::DenseTensor new_scale, new_bias;
  const auto data_layout = common::StringToDataLayout(data_layout_str);

  int C;
  if (x_dims.size() == 2) {
    C = x_dims[1];
  } else {
    C = data_layout == phi::DataLayout::kNCHW ? x_dims[1]
                                              : x_dims[x_dims.size() - 1];
  }

  if (Scale) {
    new_scale = scale.get();
  } else {
    new_scale.Resize({C});
    dev_ctx.template Alloc<T>(&new_scale);
    sdaa_ops::doFillTensor(dev_ctx, static_cast<T>(1), x.dtype(), &new_scale);
  }

  if (Bias) {
    new_bias = bias.get();
  } else {
    new_bias.Resize({C});
    dev_ctx.template Alloc<T>(&new_bias);
    sdaa_ops::doFillTensor(dev_ctx, static_cast<T>(0), x.dtype(), &new_bias);
  }

  if (!training) {
    custom_kernel::BatchNormInferKernel<T>(dev_ctx,
                                           x,
                                           running_mean,
                                           running_var,
                                           new_scale,
                                           new_bias,
                                           momentum,
                                           epsilon,
                                           data_layout_str,
                                           y,
                                           mean_out,
                                           variance_out);
    return;
  }

  VLOG(4) << "Call SDAA BatchNormKernel";

  // allocate memory for outputs
  dev_ctx.template Alloc<T>(y);
  dev_ctx.template Alloc<float>(saved_mean);
  dev_ctx.template Alloc<float>(saved_variance);
  // input and output mean/variance must share the memory
  *mean_out = running_mean;
  *variance_out = running_var;
  dev_ctx.template Alloc<float>(mean_out);
  dev_ctx.template Alloc<float>(variance_out);

  sdaa_ops::BatchNormFunc(dev_ctx,
                          x,
                          running_mean,
                          running_var,
                          new_scale,
                          new_bias,
                          momentum,
                          epsilon,
                          true,  // training
                          data_layout_str,
                          y,
                          mean_out,
                          variance_out,
                          saved_mean,
                          saved_variance);
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
    const phi::DenseTensor& saved_inv_variance,
    const paddle::optional<phi::DenseTensor>& reserve_space,
    const phi::DenseTensor& d_y,
    float momentum,
    float epsilon,
    const std::string& data_layout_str,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics,
    phi::DenseTensor* d_x,
    phi::DenseTensor* d_scale,
    phi::DenseTensor* d_bias) {
  VLOG(4) << "Call SDAA BatchNormGradKernel";
  use_global_stats = is_test || use_global_stats;
  // check arguments
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      (x_dims.size() == 4UL || x_dims.size() == 3UL || x_dims.size() == 2UL),
      true,
      phi::errors::InvalidArgument(
          "The input tensor X's dimension must equal to 2, 3 or 4. "
          " But got X's shape = [%s], X's dimension = [%d].",
          x_dims.to_str(),
          x_dims.size()));

  PADDLE_ENFORCE_GT(
      epsilon,
      0.,
      phi::errors::InvalidArgument("epsilon should be greater than zero. "
                                   "But received epsilon = %f",
                                   static_cast<float>(epsilon)));

  // data layout = NHWC or NCHW ? tecodnn API only supports NHWC layout
  phi::DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  bool need_trans = data_layout == phi::DataLayout::kNCHW;
  int N, H, W, C, D;
  sdaa_ops::ExtractNCWHD(x_dims, data_layout, &N, &C, &H, &W, &D);

  auto* Scale = scale.get_ptr();
  auto* Bias = bias.get_ptr();

  phi::DenseTensor new_scale, new_bias;

  if (Scale) {
    new_scale = scale.get();
  } else {
    new_scale.Resize({C});
    dev_ctx.template Alloc<T>(&new_scale);
    sdaa_ops::doFillTensor(dev_ctx, static_cast<T>(1), x.dtype(), &new_scale);
  }

  if (Bias) {
    new_bias = bias.get();
  } else {
    new_bias.Resize({C});
    dev_ctx.template Alloc<T>(&new_bias);
    sdaa_ops::doFillTensor(dev_ctx, static_cast<T>(0), x.dtype(), &new_bias);
  }

  // allocate memory for outputs
  dev_ctx.template Alloc<T>(d_x);
  phi::DenseTensor scale_grad_tmp, bias_grad_tmp;
  scale_grad_tmp.Resize(new_scale.dims());
  bias_grad_tmp.Resize(new_bias.dims());
  dev_ctx.template Alloc<float>(&scale_grad_tmp);
  dev_ctx.template Alloc<float>(&bias_grad_tmp);
  if (d_scale == nullptr) {
    d_scale = &scale_grad_tmp;
  } else {
    dev_ctx.template Alloc<float>(d_scale);
  }
  if (d_bias == nullptr) {
    d_bias = &bias_grad_tmp;
  } else {
    dev_ctx.template Alloc<float>(d_bias);
  }

  // scale bias mean var dims
  std::vector<int> sbmv_NHWC_dims = {1, 1, 1, C};
  const float alpha = 1.0f, beta = 0.0f;
  const float alphaDataDiff = 1.0f, betaDataDiff = 0.0f;
  const float alphaParamDiff = 1.0f, betaParamDiff = 0.0f;

  // since the tecodnnBatchNormBackward func only supports 4-D tensor,
  // when tensor dims=3, a dimensional complement is required.
  phi::DenseTensor x_temp(x), dy_temp(d_y), dx_temp(*d_x);
  if (x_dims.size() < 4) {
    if (need_trans) {
      x_temp.Resize(phi::make_ddim({N, C, H, W}));
      dy_temp.Resize(phi::make_ddim({N, C, H, W}));
      dx_temp.Resize(phi::make_ddim({N, C, H, W}));
    } else {
      x_temp.Resize(phi::make_ddim({N, H, W, C}));
      dy_temp.Resize(phi::make_ddim({N, H, W, C}));
      dx_temp.Resize(phi::make_ddim({N, H, W, C}));
    }
  }

  phi::DenseTensor x_NHWC, dy_NHWC, dx_NHWC;
  phi::DDim x_NHWC_dims, dy_NHWC_dims, dx_NHWC_dims;

  if (need_trans) {
    x_NHWC_dims = sdaa_ops::doDimPermute(x_temp, Convert_TF::NCHW2NHWC);
    dy_NHWC_dims = sdaa_ops::doDimPermute(dy_temp, Convert_TF::NCHW2NHWC);
    dx_NHWC_dims = sdaa_ops::doDimPermute(dx_temp, Convert_TF::NCHW2NHWC);
    x_NHWC.Resize(x_NHWC_dims);
    dy_NHWC.Resize(dy_NHWC_dims);
    dx_NHWC.Resize(dx_NHWC_dims);
    dev_ctx.template Alloc<T>(&x_NHWC);
    dev_ctx.template Alloc<T>(&dy_NHWC);
    dev_ctx.template Alloc<T>(&dx_NHWC);

    sdaa_ops::doTransformTensor(
        dev_ctx, x_temp, Convert_TF::NCHW2NHWC, &x_NHWC);
    sdaa_ops::doTransformTensor(
        dev_ctx, dy_temp, Convert_TF::NCHW2NHWC, &dy_NHWC);
  } else {
    x_NHWC = x_temp;
    dy_NHWC = dy_temp;
    dx_NHWC = dx_temp;
  }

  if (use_global_stats) {
    PADDLE_ENFORCE_EQ(
        mean.get_ptr() != nullptr,
        true,
        phi::errors::InvalidArgument("mean not support NULL in sdaa device."));
    PADDLE_ENFORCE_EQ(variance.get_ptr() != nullptr,
                      true,
                      phi::errors::InvalidArgument(
                          "variance not support NULL in sdaa device."));
    PADDLE_ENFORCE_EQ(
        scale.get_ptr() != nullptr,
        true,
        phi::errors::InvalidArgument("scale not support NULL in sdaa device."));
    // 1. compuate inv var
    phi::DenseTensor inv_var;
    phi::DenseTensor running_mean = mean.get();

    const auto* running_variance = variance.get_ptr();
    phi::DenseTensor add_res, sqrt_res;
    phi::DDim C_dims = {C};

    add_res.Resize(C_dims);
    sqrt_res.Resize(C_dims);
    inv_var.Resize(C_dims);
    dev_ctx.Alloc(&add_res, running_variance->dtype());
    dev_ctx.Alloc(&sqrt_res, running_variance->dtype());
    dev_ctx.Alloc(&inv_var, running_variance->dtype());
    sdaa_ops::doUnaryOpTensor(
        dev_ctx, *running_variance, epsilon, UnaryOpMode::ADD_A, &add_res);
    sdaa_ops::doUnaryOpTensor(
        dev_ctx, add_res, 1.0, UnaryOpMode::SQRT, &sqrt_res);
    sdaa_ops::doReciprocalTensor(dev_ctx, sqrt_res, &inv_var);

    phi::DenseTensor dy_sum, dy_mul_x_sub_mean_mul_invstd_sum, scale_inv_var;
    phi::DenseTensor dy_NHWC_fp32, x_NHWC_fp32;
    scale_inv_var.Resize(C_dims);
    dev_ctx.Alloc(&scale_inv_var, new_scale.dtype());
    sdaa_ops::doElementMul(dev_ctx, new_scale, inv_var, -1, &scale_inv_var);

    // compute by fp32
    // d_bias: sum of dy's NHW
    // d_sum: sum of dy * (x - mean) ) * inv_var
    dy_sum = *d_bias;
    dy_mul_x_sub_mean_mul_invstd_sum = *d_scale;

    if (!std::is_same<T, float>::value) {
      // cast dy_NHWC to fp32
      dy_NHWC_fp32.Resize(dy_NHWC.dims());
      dev_ctx.template Alloc<float>(&dy_NHWC_fp32);
      sdaa_ops::doCastTensor(dev_ctx, dy_NHWC, &dy_NHWC_fp32);
      // cast x_NHWC to fp32
      x_NHWC_fp32.Resize(x_NHWC.dims());
      dev_ctx.template Alloc<float>(&x_NHWC_fp32);
      sdaa_ops::doCastTensor(dev_ctx, x_NHWC, &x_NHWC_fp32);
    } else {
      dy_NHWC_fp32 = dy_NHWC;
      x_NHWC_fp32 = x_NHWC;
    }

    // 2. compute sum of d_y by axis NHW
    sdaa_ops::doSumTensor(dev_ctx, dy_NHWC_fp32, {0, 1, 2}, &dy_sum);

    // 3. compute dy_mul_x_sub_mean_mul_invstd_sum
    phi::DenseTensor x_sub_mean, invstd_mul_dy, intermediate_res;
    x_sub_mean.set_meta(x_NHWC_fp32.meta());
    invstd_mul_dy.set_meta(dy_NHWC_fp32.meta());
    intermediate_res.set_meta(dy_NHWC_fp32.meta());
    dev_ctx.Alloc(&x_sub_mean, dy_NHWC_fp32.dtype());
    dev_ctx.Alloc(&invstd_mul_dy, dy_NHWC_fp32.dtype());
    dev_ctx.Alloc(&intermediate_res, dy_NHWC_fp32.dtype());

    sdaa_ops::doElementSub(dev_ctx, x_NHWC_fp32, running_mean, -1, &x_sub_mean);
    sdaa_ops::doElementMul(dev_ctx, dy_NHWC_fp32, inv_var, -1, &invstd_mul_dy);
    sdaa_ops::doElementMul(
        dev_ctx, x_sub_mean, invstd_mul_dy, -1, &intermediate_res);
    sdaa_ops::doSumTensor(dev_ctx,
                          intermediate_res,
                          {0, 1, 2},
                          &dy_mul_x_sub_mean_mul_invstd_sum);

    // 4. compute dx
    if (d_x) {
      if (!std::is_same<T, float>::value) {
        phi::DenseTensor dx_NHWC_fp32;
        dx_NHWC_fp32.Resize(dx_NHWC.dims());
        dev_ctx.Alloc(&dx_NHWC_fp32, phi::DataType::FLOAT32);
        sdaa_ops::doElementMul(
            dev_ctx, dy_NHWC_fp32, scale_inv_var, -1, &dx_NHWC_fp32);

        if (need_trans) {
          sdaa_ops::doTransformTensor(
              dev_ctx, dx_NHWC_fp32, Convert_TF::NHWC2NCHW, &dx_temp);
        } else {
          sdaa_ops::doCastTensor(dev_ctx, dx_NHWC_fp32, &dx_temp);
        }
      } else {
        sdaa_ops::doElementMul(
            dev_ctx, dy_NHWC_fp32, scale_inv_var, -1, &dx_NHWC);
        if (need_trans) {
          sdaa_ops::doTransformTensor(
              dev_ctx, dx_NHWC, Convert_TF::NHWC2NCHW, &dx_temp);
        }
      }
    }
    return;

  } else {
    tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
    tecodnnBatchNormMode_t bnMode = TECODNN_BATCHNORM_SPATIAL;

    tecodnnTensorDescriptor_t x_NHWC_Desc = sdaa_ops::GetTecodnnTensorDesc(
        phi::vectorize<int>(x_NHWC.dims()), x_NHWC.dtype(), TensorFormat::NHWC);
    tecodnnTensorDescriptor_t dy_NHWC_Desc =
        sdaa_ops::GetTecodnnTensorDesc(phi::vectorize<int>(dy_NHWC.dims()),
                                       dy_NHWC.dtype(),
                                       TensorFormat::NHWC);
    tecodnnTensorDescriptor_t sbmv_NHWC_Desc = sdaa_ops::GetTecodnnTensorDesc(
        sbmv_NHWC_dims, new_scale.dtype(), TensorFormat::NHWC);

    // excute BNB directly
    size_t workSpaceSizeInBytes = 0;
    TECODNN_CHECK(tecodnnGetBatchNormalizationBackwardWorkspaceSize(
        bnMode, sbmv_NHWC_Desc, &workSpaceSizeInBytes));
    phi::DenseTensor workspace;
    if (workSpaceSizeInBytes != 0)
      workspace.Resize({static_cast<int64_t>(workSpaceSizeInBytes)});
    dev_ctx.Alloc(&workspace, phi::DataType::INT8);
    TECODNN_CHECK(tecodnnBatchNormalizationBackward(tecodnnHandle,
                                                    bnMode,
                                                    &alphaDataDiff,
                                                    &betaDataDiff,
                                                    &alphaParamDiff,
                                                    &betaParamDiff,
                                                    x_NHWC_Desc,
                                                    x_NHWC.data(),
                                                    dy_NHWC_Desc,
                                                    dy_NHWC.data(),
                                                    x_NHWC_Desc,
                                                    dx_NHWC.data(),
                                                    sbmv_NHWC_Desc,
                                                    new_scale.data(),
                                                    d_scale->data(),
                                                    d_bias->data(),
                                                    epsilon,
                                                    saved_mean.data(),
                                                    saved_inv_variance.data(),
                                                    workspace.data(),
                                                    workSpaceSizeInBytes));

    // destroy descriptors
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_NHWC_Desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(dy_NHWC_Desc));
    TECODNN_CHECK(tecodnnDestroyTensorDescriptor(sbmv_NHWC_Desc));
  }

  if (need_trans) {
    sdaa_ops::doTransformTensor(
        dev_ctx, dx_NHWC, Convert_TF::NHWC2NCHW, &dx_temp);
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(batch_norm,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormKernel,
                          phi::dtype::float16,
                          float) {
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

PD_REGISTER_PLUGIN_KERNEL(batch_norm_infer,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormInferKernel,
                          phi::dtype::float16,
                          float) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // mean_out
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // variance_out
  }
}

PD_REGISTER_PLUGIN_KERNEL(batch_norm_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormGradKernel,
                          phi::dtype::float16,
                          float) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->OutputAt(0).SetDataType(phi::DataType::FLOAT32);  // x_grad
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // scale_grad
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // bias_grad
  }
}
