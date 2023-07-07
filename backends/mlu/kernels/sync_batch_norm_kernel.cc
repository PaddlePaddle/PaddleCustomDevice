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

#define GET_LAYOUT_OFFSET 2
#define NO_USE_COMM 0

static std::vector<cnnlTensorLayout_t> supported_input_layout = {
    CNNL_LAYOUT_NC, CNNL_LAYOUT_NLC, CNNL_LAYOUT_NHWC, CNNL_LAYOUT_NDHWC};

inline void ExtractNCWHD(const phi::DDim& dims,
                         const DataLayout& data_layout,
                         int* N,
                         int* C,
                         int* H,
                         int* W,
                         int* D) {
  *N = dims[0];
  if (dims.size() == 2) {
    *C = dims[1];
    *H = 1;
    *W = 1;
    *D = 1;
  } else {
    *C = data_layout == DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
    *H = data_layout == DataLayout::kNCHW ? dims[2] : dims[1];
    *W = dims.size() > 3
             ? (data_layout == DataLayout::kNCHW ? dims[3] : dims[2])
             : 1;
    *D = dims.size() > 4
             ? (data_layout == DataLayout::kNCHW ? dims[4] : dims[3])
             : 1;
  }
}

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
  const DataLayout layout = StringToDataLayout(data_layout_str);
  PADDLE_ENFORCE_EQ(use_global_stats,
                    false,
                    phi::errors::InvalidArgument(
                        "sync_batch_norm doesn't support "
                        "to set use_global_stats True. Please use batch_norm "
                        "in this case."));
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_GE(x_dims.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The Input dim size should be larger than 1."));
  PADDLE_ENFORCE_LE(x_dims.size(),
                    5,
                    phi::errors::InvalidArgument(
                        "The Input dim size should be less than 6."));
  int N, C, H, W, D;
  ExtractNCWHD(x_dims, layout, &N, &C, &H, &W, &D);

  using MPDType = typename MPTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(y);
  dev_ctx.template Alloc<MPDType>(mean_out);
  dev_ctx.template Alloc<MPDType>(variance_out);
  dev_ctx.template Alloc<MPDType>(saved_mean);
  dev_ctx.template Alloc<MPDType>(saved_variance);

  Tensor trans_x;
  Tensor trans_y;
  std::vector<int> forward_perm;
  std::vector<int> backward_perm;
  std::vector<int> trans_shape;
  const bool need_transpose =
      ((layout == DataLayout::kNCHW && x_dims.size() != 2) ||
       x_dims.size() == 5);
  if (need_transpose) {
    SetMLUTransposePerm(
        x_dims, layout, &forward_perm, &backward_perm, &trans_shape);
    trans_x.Resize(phi::make_ddim(trans_shape));
    trans_y.Resize(phi::make_ddim(trans_shape));
    dev_ctx.template Alloc<T>(&trans_x);
    dev_ctx.template Alloc<T>(&trans_y);
    MLUCnnlTensorDesc desc_x(x);
    MLUCnnlTensorDesc desc_trans_x(
        trans_shape.size(), trans_shape.data(), ToCnnlDataType(x.dtype()));
    MLUCnnl::Transpose(dev_ctx,
                       forward_perm,
                       x_dims.size(),
                       desc_x.get(),
                       GetBasePtr(&x),
                       desc_trans_x.get(),
                       GetBasePtr(&trans_x));
  } else {
    trans_x = x;
    trans_y = *y;
  }

  MLUCnnlTensorDesc desc_trans(
      trans_x,
      supported_input_layout[x_dims.size() - GET_LAYOUT_OFFSET],
      ToCnnlDataType<T>());

  bool test_mode = is_test && (!trainable_statistics);
  cnnlBatchNormMode_t mode = CNNL_BATCHNORM_SPATIAL;
  cnnlActivationMode_t act_mode = CNNL_ACTIVATION_IDENTITY;
  MLUCnnlActivationDesc activation_desc(
      act_mode, /*ceof*/ 1.0f, /*sliced_dim*/ -1);
  cnnlBatchNormOps_t bn_ops = CNNL_BATCHNORM_OPS_BN;
  MLUCnnlTensorDesc scale_desc(scale);
  if (test_mode) {  // inference
    MLUCnnlTensorDesc desc_weight_bias_mean_var(bias);
    MLUCnnl::FusedBatchNorm(dev_ctx,
                            false /*is_training*/,
                            activation_desc.get(),
                            mode,
                            bn_ops,
                            desc_trans.get(),
                            GetBasePtr(&trans_x),
                            scale_desc.get(),
                            GetBasePtr(&scale),
                            GetBasePtr(&bias),
                            GetBasePtr(&mean),
                            GetBasePtr(&variance),
                            epsilon_f,
                            momentum,
                            desc_trans.get(),
                            GetBasePtr(&trans_y),
                            GetBasePtr(mean_out),
                            GetBasePtr(variance_out),
                            nullptr,
                            nullptr);
  } else {  // training
    Tensor local_mean, local_var;
    local_mean.Resize(mean.dims());
    local_var.Resize(variance.dims());
    dev_ctx.template Alloc<MPDType>(&local_mean);
    dev_ctx.template Alloc<MPDType>(&local_var);

    MLUCnnlTensorDesc desc_mean_var(*mean_out);

    // cacl local_mean and local_var
    MLUCnnl::SyncBatchNormStats(dev_ctx,
                                desc_trans.get(),
                                GetBasePtr(&trans_x),
                                epsilon_f,
                                desc_mean_var.get(),
                                GetBasePtr(&local_mean),
                                desc_mean_var.get(),
                                GetBasePtr(&local_var));

    Tensor input_count;
    input_count.Resize(phi::make_ddim({1}));
    dev_ctx.template Alloc<MPDType>(&input_count);
    FillMLUTensorWithHostValue<MPDType>(
        dev_ctx, static_cast<MPDType>(x.numel() / C), &input_count);

    Tensor count_all;
    Tensor mean_all;
    Tensor invstd_all;

#if !defined ON_INFER
    auto comm =
        static_cast<cnclComm_t>(phi::detail::GetCCLComm(dev_ctx.GetPlace(), 0));
    auto stream = GetQueue(static_cast<C_Stream>(dev_ctx.stream()));
    if (comm) {
      int count;
      PADDLE_ENFORCE_MLU_SUCCESS(cnclGetCommCount(&count, comm));
      count_all.Resize(phi::make_ddim({count}));
      mean_all.Resize(phi::make_ddim({count, mean.numel()}));
      invstd_all.Resize(phi::make_ddim({count, variance.numel()}));
      dev_ctx.template Alloc<MPDType>(&count_all);
      dev_ctx.template Alloc<MPDType>(&mean_all);
      dev_ctx.template Alloc<MPDType>(&invstd_all);

      // sync before communication
      lastCommStream::Instance().Update(stream);

      cnclDataType_t dtype = ToCnclDataType(count_all.dtype());
      PADDLE_ENFORCE_MLU_SUCCESS(cnclAllGather(GetBasePtr(&input_count),
                                               GetBasePtr(&count_all),
                                               1,
                                               dtype,
                                               comm,
                                               stream));

      auto cncl_dtype = ToCnclDataType(mean_all.dtype());
      PADDLE_ENFORCE_MLU_SUCCESS(cnclAllGather(GetBasePtr(&local_mean),
                                               GetBasePtr(&mean_all),
                                               local_mean.numel(),
                                               cncl_dtype,
                                               comm,
                                               stream));

      PADDLE_ENFORCE_MLU_SUCCESS(cnclAllGather(GetBasePtr(&local_var),
                                               GetBasePtr(&invstd_all),
                                               local_var.numel(),
                                               cncl_dtype,
                                               comm,
                                               stream));
      // sync queue after communication processes.
      dev_ctx.Wait();
#else
    if (NO_USE_COMM) {
#endif
    } else {
      count_all = input_count;
      mean_all = local_mean;
      invstd_all = local_var;
      mean_all.Resize(phi::make_ddim({1, local_mean.numel()}));
      invstd_all.Resize(phi::make_ddim({1, local_var.numel()}));
    }

    MLUCnnlTensorDesc desc_all_mean_invstd(
        invstd_all, CNNL_LAYOUT_NC, ToCnnlDataType<MPDType>());
    MLUCnnlTensorDesc desc_moving_mean_var(*mean_out);
    MLUCnnlTensorDesc desc_saved_mean_var(*saved_mean);
    MLUCnnlTensorDesc desc_count_all(count_all);

    MLUCnnl::SyncBatchNormGatherStatsWithCounts(dev_ctx,
                                                momentum,
                                                epsilon_f,
                                                desc_all_mean_invstd.get(),
                                                GetBasePtr(&mean_all),
                                                desc_all_mean_invstd.get(),
                                                GetBasePtr(&invstd_all),
                                                desc_moving_mean_var.get(),
                                                GetBasePtr(mean_out),
                                                desc_moving_mean_var.get(),
                                                GetBasePtr(variance_out),
                                                desc_count_all.get(),
                                                GetBasePtr(&count_all),
                                                desc_saved_mean_var.get(),
                                                GetBasePtr(saved_mean),
                                                desc_saved_mean_var.get(),
                                                GetBasePtr(saved_variance));

    MLUCnnlTensorDesc desc_other_param(*saved_mean);
    MLUCnnl::SyncBatchNormElemt(dev_ctx,
                                desc_trans.get(),
                                GetBasePtr(&trans_x),
                                desc_other_param.get(),
                                GetBasePtr(saved_mean),
                                desc_other_param.get(),
                                GetBasePtr(saved_variance),
                                desc_other_param.get(),
                                GetBasePtr(&scale),
                                desc_other_param.get(),
                                GetBasePtr(&bias),
                                desc_trans.get(),
                                GetBasePtr(&trans_y));
  }
  if (need_transpose) {
    MLUCnnlTensorDesc desc_y(*y);
    MLUCnnlTensorDesc desc_trans_y(trans_y);
    MLUCnnl::Transpose(dev_ctx,
                       backward_perm,
                       trans_y.dims().size(),
                       desc_trans_y.get(),
                       GetBasePtr(&trans_y),
                       desc_y.get(),
                       GetBasePtr(y));
  }
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
  const DataLayout layout = StringToDataLayout(data_layout_str);
  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_GE(x_dims.size(),
                    2,
                    phi::errors::InvalidArgument(
                        "The Input X dim size should be larger than 1."));
  PADDLE_ENFORCE_LE(x_dims.size(),
                    5,
                    phi::errors::InvalidArgument(
                        "The Input X dim size should be less than 6."));

  int N, C, H, W, D;
  ExtractNCWHD(x_dims, layout, &N, &C, &H, &W, &D);
  PADDLE_ENFORCE_EQ(scale.dims()[0],
                    C,
                    phi::errors::InvalidArgument(
                        "Expected first dim for input parameter(scale) of "
                        "OP(sync_batch_norm) be (%d), but given (%d).",
                        C,
                        scale.dims()[0]));
  using MPDType = typename MPTypeTrait<T>::Type;
  dev_ctx.template Alloc<T>(x_grad);
  if (scale_grad && bias_grad) {
    dev_ctx.template Alloc<MPDType>(scale_grad);
    dev_ctx.template Alloc<MPDType>(bias_grad);
  }
  PADDLE_ENFORCE_EQ(scale.dims().size(),
                    1UL,
                    phi::errors::InvalidArgument(
                        "Expected rank for input parameter(scale) of "
                        "OP(sync_batch_norm) be (1), but given (%d).",
                        scale.dims().size()));
  Tensor trans_x;
  Tensor trans_dy;
  Tensor trans_dx;
  std::vector<int> forward_perm;
  std::vector<int> backward_perm;
  std::vector<int> trans_shape;
  const bool need_transpose =
      ((layout == DataLayout::kNCHW && x_dims.size() != 2) ||
       x_dims.size() == 5);
  if (need_transpose) {
    SetMLUTransposePerm(
        x_dims, layout, &forward_perm, &backward_perm, &trans_shape);
    trans_x.Resize(phi::make_ddim(trans_shape));
    trans_dy.Resize(phi::make_ddim(trans_shape));
    trans_dx.Resize(phi::make_ddim(trans_shape));
    dev_ctx.template Alloc<T>(&trans_x);
    dev_ctx.template Alloc<T>(&trans_dy);
    dev_ctx.template Alloc<T>(&trans_dx);

    MLUCnnlTensorDesc desc_x(x);
    MLUCnnlTensorDesc desc_trans_x(
        trans_shape.size(), trans_shape.data(), ToCnnlDataType(x.dtype()));
    MLUCnnl::Transpose(dev_ctx,
                       forward_perm,
                       x_dims.size(),
                       desc_x.get(),
                       GetBasePtr(&x),
                       desc_trans_x.get(),
                       GetBasePtr(&trans_x));
    MLUCnnl::Transpose(dev_ctx,
                       forward_perm,
                       x_dims.size(),
                       desc_x.get(),
                       GetBasePtr(&y_grad),
                       desc_trans_x.get(),
                       GetBasePtr(&trans_dy));
  } else {
    trans_x = x;
    trans_dy = y_grad;
    trans_dx = *x_grad;
  }
  MLUCnnlTensorDesc desc_trans(
      trans_x,
      supported_input_layout[x_dims.size() - GET_LAYOUT_OFFSET],
      ToCnnlDataType<T>());
  Tensor sum_dy, sum_dy_xmu;
  sum_dy.Resize(bias.dims());
  sum_dy_xmu.Resize(bias.dims());
  dev_ctx.template Alloc<MPDType>(&sum_dy);
  dev_ctx.template Alloc<MPDType>(&sum_dy_xmu);
  MLUCnnlTensorDesc desc_other_param(bias);
  MLUCnnl::SyncBatchnormBackwardReduce(
      dev_ctx,
      desc_trans.get(),
      GetBasePtr(&trans_dy),
      desc_trans.get(),
      GetBasePtr(&trans_x),
      desc_other_param.get(),
      GetBasePtr(&saved_mean),
      desc_other_param.get(),
      GetBasePtr(&saved_variance),
      scale_grad ? desc_other_param.get() : nullptr,
      scale_grad ? GetBasePtr(scale_grad) : nullptr,
      bias_grad ? desc_other_param.get() : nullptr,
      bias_grad ? GetBasePtr(bias_grad) : nullptr,
      desc_other_param.get(),
      GetBasePtr(&sum_dy),
      desc_other_param.get(),
      GetBasePtr(&sum_dy_xmu),
      true /*compute sum_dy, sum_dy_xmu*/,
      scale_grad ? true : false /*compute d_scale*/,
      bias_grad ? true : false /*compute d_bias*/);

  Tensor numel_count;
  numel_count.Resize(phi::make_ddim({1}));
  dev_ctx.template Alloc<int32_t>(&numel_count);
  FillMLUTensorWithHostValue<int32_t>(
      dev_ctx, static_cast<int32_t>(x.numel() / C), &numel_count);

#if !defined ON_INFER
  auto comm =
      static_cast<cnclComm_t>(phi::detail::GetCCLComm(dev_ctx.GetPlace(), 0));
  auto stream = GetQueue(static_cast<C_Stream>(dev_ctx.stream()));
  if (comm) {
    // sync before communication
    cnclDataType_t dtype = ToCnclDataType(numel_count.dtype());

    lastCommStream::Instance().Update(stream);
    PADDLE_ENFORCE_MLU_SUCCESS(cnclAllReduce(GetBasePtr(&numel_count),
                                             GetBasePtr(&numel_count),
                                             1,
                                             dtype,
                                             cnclSum,
                                             comm,
                                             stream));
    auto cncl_dtype = ToCnclDataType(sum_dy.dtype());
    PADDLE_ENFORCE_MLU_SUCCESS(cnclAllReduce(GetBasePtr(&sum_dy),
                                             GetBasePtr(&sum_dy),
                                             sum_dy.numel(),
                                             cncl_dtype,
                                             cnclSum,
                                             comm,
                                             stream));

    PADDLE_ENFORCE_MLU_SUCCESS(cnclAllReduce(GetBasePtr(&sum_dy_xmu),
                                             GetBasePtr(&sum_dy_xmu),
                                             sum_dy_xmu.numel(),
                                             cncl_dtype,
                                             cnclSum,
                                             comm,
                                             stream));
    // sync queue after communication processes.
    dev_ctx.Wait();
  }
#endif

  if (x_grad) {
    MLUCnnlTensorDesc desc_count(numel_count);
    MLUCnnl::SyncBatchNormBackwardElemt(dev_ctx,
                                        desc_trans.get(),
                                        GetBasePtr(&trans_dy),
                                        desc_trans.get(),
                                        GetBasePtr(&trans_x),
                                        desc_other_param.get(),
                                        GetBasePtr(&saved_mean),
                                        desc_other_param.get(),
                                        GetBasePtr(&saved_variance),
                                        desc_other_param.get(),
                                        GetBasePtr(&scale),
                                        desc_other_param.get(),
                                        GetBasePtr(&sum_dy),
                                        desc_other_param.get(),
                                        GetBasePtr(&sum_dy_xmu),
                                        desc_count.get(),
                                        GetBasePtr(&numel_count),
                                        desc_trans.get(),
                                        GetBasePtr(&trans_dx));
    if (need_transpose) {
      MLUCnnlTensorDesc desc_dx(*x_grad);
      MLUCnnlTensorDesc desc_trans_dx(trans_dx);
      MLUCnnl::Transpose(dev_ctx,
                         backward_perm,
                         trans_dx.dims().size(),
                         desc_trans_dx.get(),
                         GetBasePtr(&trans_dx),
                         desc_dx.get(),
                         GetBasePtr(x_grad));
    }
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sync_batch_norm,
                          mlu,
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
                          mlu,
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
