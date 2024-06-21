// BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
// reserved.
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution.
// Neither the name of the copyright holder nor the names of its contributors
// may be used to endorse or promote products derived from this software
// without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
// WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
// OF SUCH DAMAGE.

#include "kernels/funcs/sdaa_baseop.h"
#include "kernels/funcs/slice_utils.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

template <typename T, typename Context>
void SyncBatchNormKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& mean,
                         const phi::DenseTensor& variance,
                         const phi::DenseTensor& scale,
                         const phi::DenseTensor& bias,
                         bool is_test,
                         float momentum,
                         float epsilon_f,
                         const std::string& data_layout_str,
                         bool use_global_stats,
                         bool trainable_statistics,
                         phi::DenseTensor* y,
                         phi::DenseTensor* mean_out,
                         phi::DenseTensor* variance_out,
                         phi::DenseTensor* saved_mean,
                         phi::DenseTensor* saved_variance,
                         phi::DenseTensor* reserve_space) {
  VLOG(4) << "CALL SDAA SyncBatchNormKernel";

  PADDLE_ENFORCE_EQ(use_global_stats,
                    false,
                    phi::errors::InvalidArgument(
                        "sync_batch_norm doesn't support "
                        "to set use_global_stats True. Please use batch_norm "
                        "in this case."));

  const auto& x_dims = x.dims();

  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      4,
      phi::errors::InvalidArgument(
          "The Input dim size should be equal 4 in SDAA device."));

  using MPDType = typename sdaa_ops::MPTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(y);
  dev_ctx.template Alloc<MPDType>(mean_out);
  dev_ctx.template Alloc<MPDType>(variance_out);
  dev_ctx.template Alloc<MPDType>(saved_mean);
  dev_ctx.template Alloc<MPDType>(saved_variance);

  bool test_mode = is_test && (!trainable_statistics);
  if (test_mode) {  // inference
    sdaa_ops::BatchNormFunc(dev_ctx,
                            x,
                            mean,
                            variance,
                            scale,
                            bias,
                            momentum,
                            epsilon_f,
                            false,  // inference
                            data_layout_str,
                            y,
                            mean_out,
                            variance_out,
                            nullptr,
                            nullptr);
    return;
  }

  const DataLayout layout = common::StringToDataLayout(data_layout_str);
  int N, H, W, C, D;
  sdaa_ops::ExtractNCWHD(x_dims, layout, &N, &C, &H, &W, &D);

  phi::DenseTensor trans_x, trans_y;
  phi::DDim trans_x_dims, trans_y_dims;
  const bool need_transpose =
      ((layout == DataLayout::kNCHW && x_dims.size() != 2) ||
       x_dims.size() == 5);

  if (need_transpose) {
    trans_x_dims = sdaa_ops::doDimPermute(x, Convert_TF::NCHW2NHWC);
    trans_y_dims = sdaa_ops::doDimPermute(*y, Convert_TF::NCHW2NHWC);
    trans_x.Resize(trans_x_dims);
    trans_y.Resize(trans_y_dims);
    dev_ctx.template Alloc<T>(&trans_x);
    dev_ctx.template Alloc<T>(&trans_y);

    sdaa_ops::doTransformTensor(dev_ctx, x, Convert_TF::NCHW2NHWC, &trans_x);
  } else {
    trans_x = x;
    trans_y = *y;
  }

  // calculate local_mean and local_square_mean
  std::vector<int> mean_dims = {1, 1, 1, C};

  // because tecodnnCustomSyncBNMeanVar() API requires
  // that the first addr of the two output parameters be 64B aligned.
  int C_temp = ceil(C / 16) * 16;
  const int data_num = C_temp + C;
  phi::DenseTensor status;
  status.Resize(phi::make_ddim({data_num}));
  dev_ctx.template Alloc<MPDType>(&status);

  auto local_mean = custom_kernel::Slice(
      status, static_cast<int64_t>(0), static_cast<int64_t>(C));
  auto local_square_mean = custom_kernel::Slice(
      status, static_cast<int64_t>(C_temp), static_cast<int64_t>(C_temp + C));

  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnsyncBatchNormMode_t syncBNMode = TECODNN_BATCHNORM_SPATIAL_SBN;
  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(trans_x.dims()), trans_x.dtype(), TensorFormat::NHWC);
  tecodnnTensorDescriptor_t meanvar_Desc = sdaa_ops::GetTecodnnTensorDesc(
      mean_dims, DataType::FLOAT32, TensorFormat::NHWC);
  TECODNN_CHECK(tecodnnCustomSyncBNMeanVar(tecodnnHandle,
                                           syncBNMode,
                                           x_Desc,
                                           trans_x.data(),
                                           static_cast<double>(epsilon_f),
                                           meanvar_Desc,
                                           local_mean.data(),
                                           local_square_mean.data()));

  auto comm =
      static_cast<tcclComm_t>(phi::detail::GetCCLComm(dev_ctx.GetPlace(), 0));
  auto stream = reinterpret_cast<CustomSDAAStream_t>(dev_ctx.stream());

  int count = 1;

  if (comm) {
    // sync before communication
    lastCommStream::Instance().update(
        reinterpret_cast<sdaaStream_t>(stream->pStream));

    TCCL_CHECK(tcclCommCount(comm, &count));

    auto tccl_dt = sdaa_ops::ToTcclDataType(status.dtype());

    // AllReduce communication to get the sum of all local_mean and
    // local_square_mean.
    TCCL_CHECK(tcclAllReduce(status.data(),
                             status.data(),
                             status.numel(),
                             tccl_dt,
                             tcclSum,
                             comm,
                             reinterpret_cast<sdaaStream_t>(stream->pStream)));

    // sync after communication
    dev_ctx.Wait();
  }

  // calculate saved_mean and saved_variance
  // 1. calculate saved_mean

  sdaa_ops::doScaleTensor(dev_ctx,
                          local_mean,
                          1 / static_cast<float>(count),
                          0.f,
                          false,
                          false,
                          saved_mean);

  // 2. calculate saved_inv_variance
  phi::DenseTensor global_square_mean, saved_mean_pow, var, sqrt_var,
      saved_variance_tmp;
  global_square_mean.Resize(mean.dims());
  dev_ctx.template Alloc<MPDType>(&global_square_mean);
  saved_mean_pow.Resize(mean.dims());
  dev_ctx.template Alloc<MPDType>(&saved_mean_pow);
  var.Resize(mean.dims());
  dev_ctx.template Alloc<MPDType>(&var);
  sqrt_var.Resize(mean.dims());
  dev_ctx.template Alloc<MPDType>(&sqrt_var);
  saved_variance_tmp.Resize(mean.dims());
  dev_ctx.template Alloc<MPDType>(&saved_variance_tmp);

  // 2.1 calculate global_square_mean
  sdaa_ops::doScaleTensor(dev_ctx,
                          local_square_mean,
                          1 / static_cast<float>(count),
                          0.f,
                          false,
                          false,
                          &global_square_mean);

  // 2.2 calculate saved_mean_pow
  sdaa_ops::doUnaryOpTensor(
      dev_ctx, *saved_mean, 2.f, UnaryOpMode::POW, &saved_mean_pow);

  // 2.3 calculate variance = global_square_mean - saved_mean_pow
  sdaa_ops::doElementSub(dev_ctx, global_square_mean, saved_mean_pow, 0, &var);

  // 2.4 calculate saved_inv_variance
  sdaa_ops::doUnaryOpTensor(
      dev_ctx, var, epsilon_f, UnaryOpMode::ADD_A, &sqrt_var);
  sdaa_ops::doUnaryOpTensor(
      dev_ctx, sqrt_var, 1.0f, UnaryOpMode::SQRT, &saved_variance_tmp);

  sdaa_ops::doReciprocalTensor(dev_ctx, saved_variance_tmp, saved_variance);

  // apply batch normalization for each channel and calculate mean_out,
  // variance_out
  const float alpha = 1.0f, beta = 0.0f;
  tecodnnTensorDescriptor_t y_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(trans_y.dims()), y->dtype(), TensorFormat::NHWC);
  TECODNN_CHECK(
      tecodnnCustomSyncBatchNormalizationForward(tecodnnHandle,
                                                 syncBNMode,
                                                 &alpha,
                                                 &beta,
                                                 x_Desc,
                                                 trans_x.data(),
                                                 y_Desc,
                                                 trans_y.data(),
                                                 meanvar_Desc,
                                                 scale.data(),
                                                 bias.data(),
                                                 static_cast<double>(momentum),
                                                 mean_out->data(),
                                                 variance_out->data(),
                                                 static_cast<double>(epsilon_f),
                                                 meanvar_Desc,
                                                 saved_mean->data(),
                                                 var.data(),
                                                 saved_variance->data()));

  if (need_transpose) {
    sdaa_ops::doTransformTensor(dev_ctx, trans_y, Convert_TF::NHWC2NCHW, y);
  }

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(meanvar_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(y_Desc));
}

template <typename T, typename Context>
void SyncBatchNormGradKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const phi::DenseTensor& scale,
    const phi::DenseTensor& bias,
    const phi::DenseTensor& saved_mean,
    const phi::DenseTensor& saved_variance,
    const paddle::optional<phi::DenseTensor>& reserve_space,
    const phi::DenseTensor& y_grad,
    float momentum,
    float epsilon_f,
    const std::string& data_layout_str,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics,
    phi::DenseTensor* x_grad,
    phi::DenseTensor* scale_grad,
    phi::DenseTensor* bias_grad) {
  VLOG(4) << "CALL SDAA SyncBatchNormGradKernel";

  const DataLayout layout = common::StringToDataLayout(data_layout_str);
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_EQ(
      x_dims.size(),
      4,
      phi::errors::InvalidArgument(
          "The Input dim size should be equal 4 in SDAA device."));

  int N, C, H, W, D;
  sdaa_ops::ExtractNCWHD(x_dims, layout, &N, &C, &H, &W, &D);

  PADDLE_ENFORCE_EQ(scale.dims()[0],
                    C,
                    phi::errors::InvalidArgument(
                        "Expected first dim for input parameter(scale) of "
                        "OP(sync_batch_norm) be (%d), but given (%d).",
                        C,
                        scale.dims()[0]));
  PADDLE_ENFORCE_EQ(scale.dims().size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Expected rank for input parameter(scale) of "
                        "OP(sync_batch_norm) be (1), but given (%d).",
                        scale.dims().size()));

  using MPDType = typename sdaa_ops::MPTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(x_grad);

  phi::DenseTensor d_scale, d_bias;
  void* scale_grad_ptr = nullptr;
  void* bias_grad_ptr = nullptr;
  if (scale_grad && bias_grad) {
    dev_ctx.template Alloc<MPDType>(scale_grad);
    dev_ctx.template Alloc<MPDType>(bias_grad);
    scale_grad_ptr = scale_grad->data();
    bias_grad_ptr = bias_grad->data();
  } else {
    d_scale.Resize(scale.dims());
    d_bias.Resize(bias.dims());
    dev_ctx.template Alloc<MPDType>(&d_scale);
    dev_ctx.template Alloc<MPDType>(&d_bias);
    scale_grad_ptr = d_scale.data();
    bias_grad_ptr = d_bias.data();
  }

  phi::DenseTensor trans_x, trans_dy, trans_dx;
  phi::DDim trans_x_dims, trans_dy_dims, trans_dx_dims;

  const bool need_transpose =
      ((layout == DataLayout::kNCHW && x_dims.size() != 2) ||
       x_dims.size() == 5);

  if (need_transpose) {
    trans_x_dims = sdaa_ops::doDimPermute(x, Convert_TF::NCHW2NHWC);
    trans_dy_dims = sdaa_ops::doDimPermute(y_grad, Convert_TF::NCHW2NHWC);
    trans_dx_dims = sdaa_ops::doDimPermute(*x_grad, Convert_TF::NCHW2NHWC);
    trans_x.Resize(trans_x_dims);
    trans_dy.Resize(trans_dy_dims);
    trans_dx.Resize(trans_dx_dims);
    dev_ctx.template Alloc<T>(&trans_x);
    dev_ctx.template Alloc<T>(&trans_dy);
    dev_ctx.template Alloc<T>(&trans_dx);

    sdaa_ops::doTransformTensor(dev_ctx, x, Convert_TF::NCHW2NHWC, &trans_x);
    sdaa_ops::doTransformTensor(
        dev_ctx, y_grad, Convert_TF::NCHW2NHWC, &trans_dy);
  } else {
    trans_x = x;
    trans_dy = y_grad;
    trans_dx = *x_grad;
  }

  phi::DenseTensor sum_dy_and_sum_dy_xmu;
  sum_dy_and_sum_dy_xmu.Resize(phi::make_ddim({2 * C}));
  dev_ctx.template Alloc<MPDType>(&sum_dy_and_sum_dy_xmu);

  auto sum_dy = custom_kernel::Slice(
      sum_dy_and_sum_dy_xmu, static_cast<int64_t>(0), static_cast<int64_t>(C));
  auto sum_dy_xmu = custom_kernel::Slice(sum_dy_and_sum_dy_xmu,
                                         static_cast<int64_t>(C),
                                         static_cast<int64_t>(2 * C));

  // calculate scale_grad, bias_grad, local_sum_dy and local_sum_dy_xmu
  std::vector<int> meanVar_dims = {1, 1, 1, C};
  tecodnnHandle_t tecodnnHandle = GetHandleFromCTX(dev_ctx);
  tecodnnsyncBatchNormMode_t syncBNMode = TECODNN_BATCHNORM_SPATIAL_SBN;

  tecodnnTensorDescriptor_t x_Desc = sdaa_ops::GetTecodnnTensorDesc(
      phi::vectorize<int>(trans_x.dims()), trans_x.dtype(), TensorFormat::NHWC);

  tecodnnTensorDescriptor_t dy_Desc =
      sdaa_ops::GetTecodnnTensorDesc(phi::vectorize<int>(trans_dy.dims()),
                                     trans_dy.dtype(),
                                     TensorFormat::NHWC);
  tecodnnTensorDescriptor_t meanVar_Desc = sdaa_ops::GetTecodnnTensorDesc(
      meanVar_dims, saved_mean.dtype(), TensorFormat::NHWC);
  TECODNN_CHECK(
      tecodnnCustomSyncBNScaleBiasBackward_1(tecodnnHandle,
                                             syncBNMode,
                                             x_Desc,
                                             trans_x.data(),
                                             dy_Desc,
                                             trans_dy.data(),
                                             static_cast<double>(epsilon_f),
                                             meanVar_Desc,
                                             saved_mean.data(),
                                             saved_variance.data(),
                                             meanVar_Desc,
                                             scale_grad_ptr,
                                             bias_grad_ptr,
                                             meanVar_Desc,
                                             sum_dy.data(),
                                             sum_dy_xmu.data()));

  auto comm =
      static_cast<tcclComm_t>(phi::detail::GetCCLComm(dev_ctx.GetPlace(), 0));
  auto stream = reinterpret_cast<CustomSDAAStream_t>(dev_ctx.stream());

  int count = 1;
  if (comm) {
    // sync before communication
    lastCommStream::Instance().update(
        reinterpret_cast<sdaaStream_t>(stream->pStream));

    TCCL_CHECK(tcclCommCount(comm, &count));
    auto tccl_dt = sdaa_ops::ToTcclDataType(sum_dy_and_sum_dy_xmu.dtype());
    TCCL_CHECK(tcclAllReduce(sum_dy_and_sum_dy_xmu.data(),
                             sum_dy_and_sum_dy_xmu.data(),
                             sum_dy_and_sum_dy_xmu.numel(),
                             tccl_dt,
                             tcclSum,
                             comm,
                             reinterpret_cast<sdaaStream_t>(stream->pStream)));

    // sync after communication
    dev_ctx.Wait();
  }

  // calculate sum_dy_and_sum_dy_xmu / count
  // GPU execute this operation in CUDA kernel.
  sdaa_ops::doScaleTensor(dev_ctx,
                          sum_dy_and_sum_dy_xmu,
                          1 / static_cast<float>(count),
                          0.f,
                          true,
                          false,
                          &sum_dy_and_sum_dy_xmu);

  if (x_grad) {
    // calculate the gradient of x
    const float alphaDataDiff = 1.0f, betaDataDiff = 0.0f;
    const float alphaParamDiff = 1.0f, betaParamDiff = 0.0f;

    TECODNN_CHECK(tecodnnCustomSyncBatchNormalizationBackward(
        tecodnnHandle,
        syncBNMode,
        &alphaDataDiff,
        &betaDataDiff,
        &alphaParamDiff,
        &betaParamDiff,
        x_Desc,
        trans_x.data(),
        dy_Desc,
        trans_dy.data(),
        x_Desc,
        trans_dx.data(),
        static_cast<double>(epsilon_f),
        meanVar_Desc,
        saved_mean.data(),
        saved_variance.data(),
        meanVar_Desc,
        scale.data(),
        meanVar_Desc,
        sum_dy.data(),
        sum_dy_xmu.data()));

    if (need_transpose) {
      sdaa_ops::doTransformTensor(
          dev_ctx, trans_dx, Convert_TF::NHWC2NCHW, x_grad);
    }
  }

  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(x_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(dy_Desc));
  TECODNN_CHECK(tecodnnDestroyTensorDescriptor(meanVar_Desc));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sync_batch_norm,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SyncBatchNormKernel,
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

PD_REGISTER_PLUGIN_KERNEL(sync_batch_norm_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SyncBatchNormGradKernel,
                          phi::dtype::float16,
                          float) {
  if (kernel_key.dtype() == phi::DataType::FLOAT16) {
    kernel->InputAt(1).SetDataType(phi::DataType::FLOAT32);   // mean
    kernel->InputAt(2).SetDataType(phi::DataType::FLOAT32);   // variance
    kernel->InputAt(3).SetDataType(phi::DataType::FLOAT32);   // scale
    kernel->InputAt(4).SetDataType(phi::DataType::FLOAT32);   // bias
    kernel->OutputAt(1).SetDataType(phi::DataType::FLOAT32);  // scale_grad
    kernel->OutputAt(2).SetDataType(phi::DataType::FLOAT32);  // bias_grad
  }
}
