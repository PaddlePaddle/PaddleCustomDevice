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

#include <cassert>
#include <vector>

#include "kernels/funcs/sdaa_baseop.h"
#include "kernels/funcs/sdaa_funcs.h"
#include "kernels/funcs/tecodnn_conv_impl.h"
#include "paddle/extension.h"
#include "paddle/phi/backends/all_context.h"

template <typename T>
using remove_cvref = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename LeftType, typename RightType>
using same_type =
    typename std::is_same<remove_cvref<LeftType>, remove_cvref<RightType>>;

struct CreatePlanHelper {
  using ConstParamLabel = tecodnnFusedOpsConstParamLabel_t;
  using ConstParamPack = tecodnnFusedOpsConstParamPack_t;
  using FusedOpsType = tecodnnFusedOps_t;

  explicit CreatePlanHelper(const FusedOpsType fused_op_type)
      : workspace_size_(0) {
    tecodnnCreateFusedOpsPlan(&plan_, fused_op_type);
    tecodnnCreateFusedOpsConstParamPack(&const_pack_, fused_op_type);
    tecodnnCreateFusedOpsVariantParamPack(&var_pack_, fused_op_type);
  }

  ~CreatePlanHelper() {
    tecodnnDestroyFusedOpsPlan(plan_);
    tecodnnDestroyFusedOpsConstParamPack(const_pack_);
    tecodnnDestroyFusedOpsVariantParamPack(var_pack_);

    for (auto desc : filter_descs_) {
      tecodnnDestroyFilterDescriptor(desc);
    }

    for (auto desc : tensor_descs_) {
      tecodnnDestroyTensorDescriptor(desc);
    }
  }

  template <typename LabelType,
            typename ParamSetFunc,
            typename ParamType,
            typename ArgType,
            std::enable_if_t<!same_type<ArgType, paddle::Tensor>::value &&
                             !same_type<ArgType, phi::DenseTensor>::value &&
                             std::is_pointer<remove_cvref<ArgType>>::value>* =
                nullptr>
  CreatePlanHelper& BindParam(ParamSetFunc&& func,
                              ParamType&& param,
                              LabelType&& label,
                              ArgType&& arg) {
    func(param, label, std::forward<ArgType>(arg));
    return *this;
  }

  template <typename LabelType,
            typename TensorType,
            typename ParamSetFunc,
            typename ParamType,
            std::enable_if_t<same_type<TensorType, paddle::Tensor>::value ||
                             same_type<TensorType, phi::DenseTensor>::value>* =
                nullptr>
  CreatePlanHelper& BindParam(ParamSetFunc&& func,
                              ParamType&& param,
                              LabelType&& label,
                              TensorType&& tensor) {
    tecodnnTensorDescriptor_t desc;
    tecodnnFilterDescriptor_t filter_desc;
    const auto tensor_dims = phi::vectorize<int>(tensor.dims());
    switch (label) {
      case TECODNN_PARAM_WDESC:
        TECODNN_CHECK(tecodnnCreateFilterDescriptor(&filter_desc));
        TECODNN_CHECK(tecodnnSetFilter4dDescriptor(
            filter_desc,
            custom_kernel::sdaa_ops::ToTecodnnDataType(tensor.dtype()),
            TECODNN_TENSOR_CHWN,
            tensor_dims[3],
            tensor_dims[0],
            tensor_dims[1],
            tensor_dims[2]));
        filter_descs_.push_back((filter_desc));
        func(param, label, filter_desc);
        break;

      default:
        desc = custom_kernel::sdaa_ops::GetTecodnnTensorDesc(
            tensor_dims, tensor.dtype(), custom_kernel::TensorFormat::NHWC);
        tensor_descs_.push_back(desc);
        func(param, label, desc);
        break;
    }

    return *this;
  }

  template <typename LabelType, typename ArgsType>
  CreatePlanHelper& BindConstParam(LabelType&& label, ArgsType&& arg) {
    this->template BindParam<LabelType>(
        tecodnnSetFusedOpsConstParamPackAttribute,
        const_pack_,
        std::forward<LabelType>(label),
        std::forward<ArgsType>(arg));
    return *this;
  }

  template <typename LabelType, typename ArgsType>
  CreatePlanHelper& BindVarParam(LabelType&& label, ArgsType&& arg) {
    this->template BindParam<LabelType>(
        tecodnnSetFusedOpsVariantParamPackAttribute,
        var_pack_,
        std::forward<LabelType>(label),
        std::forward<ArgsType>(arg));
    return *this;
  }

  void MakeFuseOpsPlan(tecodnnHandle_t handle) {
    tecodnnMakeFusedOpsPlan(handle, plan_, const_pack_, &workspace_size_);
  }

  tecodnnFusedOpsPlan_t plan_;
  tecodnnFusedOpsConstParamPack_t const_pack_;
  tecodnnFusedOpsVariantParamPack_t var_pack_;
  std::vector<tecodnnFilterDescriptor_t> filter_descs_;
  std::vector<tecodnnTensorDescriptor_t> tensor_descs_;

  size_t workspace_size_;
};

inline std::shared_ptr<phi::DenseTensor> TransformTensor(
    const phi::CustomContext* dev_ctx,
    const paddle::Tensor& tensor,
    custom_kernel::TensorFormat dst_layout,
    custom_kernel::DataType dst_dtype) {
  int N, H, W, C, D;
  custom_kernel::sdaa_ops::ExtractNCWHD(
      tensor.dims(), phi::DataLayout::kNCHW, &N, &C, &H, &W, &D);

  phi::DenseTensor x;
  phi::DDim out_dim;
  custom_kernel::Convert_TF tf;

  if (dst_layout == custom_kernel::TensorFormat::NHWC) {
    out_dim = {N, H, W, C};
    tf = custom_kernel::Convert_TF::NCHW2NHWC;
  } else if (dst_layout == custom_kernel::TensorFormat::CHWN) {
    out_dim = {C, H, W, N};
    tf = custom_kernel::Convert_TF::NCHW2CHWN;
  } else {
    PADDLE_THROW(phi::errors::InvalidArgument("only support nhwc and chwn"));
  }

  phi::DenseTensorMeta x_meta{dst_dtype, out_dim};
  x.set_meta(x_meta);

  dev_ctx->Alloc(&x, dst_dtype);

  auto x_temp = static_cast<phi::DenseTensor*>(tensor.impl().get());
  custom_kernel::sdaa_ops::doTransformTensor(*dev_ctx, *x_temp, tf, &x);

  return std::make_shared<phi::DenseTensor>(x);
}

std::vector<std::vector<int64_t>> CustomFusedConvBNInferShape(
    const std::vector<int64_t>& conv_input,
    const std::vector<int64_t>& conv_filter,
    const std::vector<int>& strides,
    const std::vector<int>& paddings_t,
    const std::string& paddings_algorithm,
    const std::vector<int>& dilations_t,
    const int groups,
    const std::string& data_format_str,
    const std::vector<int64_t>& bn_scale,
    const std::vector<int64_t>& bn_bias,
    const std::vector<int64_t>& bn_mean,
    const std::vector<int64_t>& bn_var,
    const float bn_epsilon,
    const std::string& activation_type) {
  std::vector<int> paddings = paddings_t;
  std::vector<int> dilations = dilations_t;
  auto in_dims = phi::make_ddim(conv_input);
  auto filter_dims = phi::make_ddim(conv_filter);
  int dilation_size = dilations.size();
  for (int i = 0; i < dilation_size; ++i) {
    PADDLE_ENFORCE_GT(
        dilations[i],
        0,
        phi::errors::InvalidArgument(
            "The dilation of Op(Conv) should be larget than 0, but received "
            "dilation is %d.",
            dilations[i]));
  }
  const bool channel_last =
      (data_format_str == "NHWC" || data_format_str == "NDHWC");

  for (int i = 0; i < 2; ++i) {
    PADDLE_ENFORCE_NE(in_dims[i],
                      0,
                      phi::errors::InvalidArgument(
                          "The size of Op(Conv) inputs should not be 0."));
  }

  PADDLE_ENFORCE_EQ(
      in_dims.size() == 4 || in_dims.size() == 5,
      true,
      phi::errors::InvalidArgument(
          "The input of Op(Conv) should be a 4-D or 5-D Tensor. But "
          "received: input's dimension is %u, input's shape is [%s].",
          in_dims.size(),
          in_dims));

  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      filter_dims.size(),
      phi::errors::InvalidArgument(
          "The input's dimension and filter's dimension of "
          "Op(Conv) should be equal. But received: the input's shape is [%s], "
          "the input's dimension is %d; the filter's shape is [%s],  "
          "the filter's dimension is %d.",
          in_dims,
          in_dims.size(),
          filter_dims,
          filter_dims.size()));

  int stride_size = strides.size();
  for (int i = 0; i < stride_size; ++i) {
    PADDLE_ENFORCE_GT(
        strides[i],
        0,
        phi::errors::InvalidArgument(
            "The stride of Op(Conv) should be larget than 0, but received "
            "stride is %d.",
            strides[i]));
  }

  int in_sub_stride_size = in_dims.size() - stride_size;
  PADDLE_ENFORCE_EQ(
      in_dims.size(),
      strides.size() + 2U,
      phi::errors::InvalidArgument(
          "The difference of input's dimension and Attr(strides)'s "
          "length must be euqal to 2 for Op(Conv). "
          "But received: input's dimension is %d, input's shape is [%s]; "
          "Attr(stride)'s length is %d, Attr(stride) is [%s]; "
          "difference of input's dimention and Attr(strides)'s length = %u.",
          in_dims.size(),
          in_dims,
          strides.size(),
          phi::make_ddim(strides),
          in_sub_stride_size));

  const auto input_channels =
      channel_last ? in_dims[in_dims.size() - 1] : in_dims[1];

  PADDLE_ENFORCE_EQ(
      input_channels,
      filter_dims[1] * groups,
      phi::errors::InvalidArgument(
          "The number of input's channels should be equal to filter's channels "
          "* groups for Op(Conv). But received: the input's channels is %d, "
          "the input's shape is [%s]; the filter's channels is %d, the "
          "filter's shape is [%s]; the groups is %d, the data_format is %s. "
          "The error may come from wrong data_format setting.",
          input_channels,
          in_dims,
          filter_dims[1],
          filter_dims,
          groups,
          data_format_str));
  PADDLE_ENFORCE_EQ(
      filter_dims[0] % groups,
      0,
      phi::errors::InvalidArgument(
          "The number of output's channels (filter's first dimension) of "
          "Op(Conv) should be divided by groups. But received: "
          "the output channels is %d, the filter's shape is [%s], "
          "the groups is %d.",
          filter_dims[0],
          filter_dims,
          groups));

  phi::DDim in_data_dims;
  if (channel_last) {
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }

  phi::DDim filter_data_dims =
      phi::slice_ddim(filter_dims, 2, filter_dims.size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  custom_kernel::UpdatePaddingAndDilation(
      &paddings, &dilations, paddings_algorithm, in_data_dims, strides, ksize);

  std::vector<int64_t> output_shape({in_dims[0]});
  if (!channel_last) {
    output_shape.push_back(filter_dims[0]);
  }
  for (int i = 0; i < in_data_dims.size(); ++i) {
    if ((in_data_dims[i] <= 0 || filter_dims[i + 2] <= 0)) {
      output_shape.push_back(-1);
    } else {
      const int dkernel = dilations[i] * (filter_data_dims[i] - 1) + 1;
      int output_size =
          (in_data_dims[i] + paddings[2 * i] + paddings[2 * i + 1] - dkernel) /
              strides[i] +
          1;
      output_shape.push_back(output_size);
    }
  }
  if (channel_last) {
    output_shape.push_back(filter_dims[0]);
  }

  return {output_shape};
}

std::vector<paddle::Tensor> CustomFusedConvBN(
    const paddle::Tensor& conv_input,
    const paddle::Tensor& conv_filter,
    const std::vector<int>& strides,
    const std::vector<int>& paddings_t,
    const std::string& paddings_algorithm,
    const std::vector<int>& dilations_t,
    const int groups,
    const std::string& data_format_str,
    const paddle::Tensor& bn_scale,
    const paddle::Tensor& bn_bias,
    const paddle::Tensor& bn_mean,
    const paddle::Tensor& bn_var,
    const float bn_epsilon,
    const std::string& activation_type) {
  VLOG(4) << "Call SDAA CustomFusedConvBN Kernel";

  PADDLE_ENFORCE_EQ(
      groups,
      1,
      phi::errors::InvalidArgument("tecodnn not support group conv"
                                   "But recieved: group is %d",
                                   groups));

  PADDLE_ENFORCE_EQ(
      data_format_str == std::string("NHWC") ||
          data_format_str == std::string("NCHW"),
      true,
      phi::errors::InvalidArgument(
          "CustomConvBNFused Kernel not support layout %s.", data_format_str));

  phi::DataLayout data_layout = common::StringToDataLayout(data_format_str);
  const bool channel_last =
      (data_format_str == "NHWC" || data_format_str == "NDHWC");

  std::vector<int> paddings = paddings_t;
  std::vector<int> dilations = dilations_t;

  // get custom context and tecohandle
  auto dev_ctx = paddle::experimental::DeviceContextPool::Instance().Get(
      conv_input.place());
  auto custom_ctx = static_cast<const phi::CustomContext*>(dev_ctx);
  tecodnnHandle_t handle = custom_kernel::GetHandleFromCTX(*custom_ctx);

  // infer shape
  auto out_dims =
      CustomFusedConvBNInferShape(phi::vectorize<int64_t>(conv_input.dims()),
                                  phi::vectorize<int64_t>(conv_filter.dims()),
                                  strides,
                                  paddings,
                                  paddings_algorithm,
                                  dilations,
                                  groups,
                                  data_format_str,
                                  phi::vectorize<int64_t>(bn_scale.dims()),
                                  phi::vectorize<int64_t>(bn_bias.dims()),
                                  phi::vectorize<int64_t>(bn_mean.dims()),
                                  phi::vectorize<int64_t>(bn_var.dims()),
                                  bn_epsilon,
                                  activation_type);
  paddle::Tensor out =
      paddle::empty(out_dims.front(), conv_input.dtype(), conv_input.place());

  // update paddings and dilations
  phi::DDim in_dims = conv_input.dims();
  phi::DDim in_data_dims;
  if (channel_last) {
    in_data_dims = phi::slice_ddim(in_dims, 1, in_dims.size() - 1);
  } else {
    in_data_dims = phi::slice_ddim(in_dims, 2, in_dims.size());
  }

  phi::DDim filter_data_dims =
      phi::slice_ddim(conv_filter.dims(), 2, conv_filter.dims().size());

  std::vector<int> ksize = phi::vectorize<int>(filter_data_dims);
  custom_kernel::UpdatePaddingAndDilation(
      &paddings, &dilations, paddings_algorithm, in_data_dims, strides, ksize);

  // transform conv bn input and output if need
  paddle::Tensor conv_input_tmp, conv_filter_tmp, conv_out_tmp;
  conv_filter_tmp.set_impl(TransformTensor(custom_ctx,
                                           conv_filter,
                                           custom_kernel::TensorFormat::CHWN,
                                           custom_kernel::DataType::FLOAT16));
  if (!channel_last) {
    conv_input_tmp.set_impl(TransformTensor(custom_ctx,
                                            conv_input,
                                            custom_kernel::TensorFormat::NHWC,
                                            custom_kernel::DataType::FLOAT16));

    conv_out_tmp.set_impl(TransformTensor(
        custom_ctx, out, custom_kernel::TensorFormat::NHWC, out.dtype()));
  } else {
    conv_input_tmp.set_impl(TransformTensor(custom_ctx,
                                            conv_input,
                                            custom_kernel::TensorFormat::NCHW,
                                            custom_kernel::DataType::FLOAT16));
    conv_out_tmp = out;
  }

  // construct plan
  CreatePlanHelper plan_helper(TECODNN_FUSED_CONV_BN_SCALE_BIAS_ACTIVATION_ADD);
  tecodnnConvolutionDescriptor_t convDesc;
  // batch norm place holder
  auto bn_place_holder = TECODNN_PTR_ELEM_ALIGNED;
  // conv algo
  auto algo = TECODNN_CONVOLUTION_FWD_ALGO_0;
  // bn mode
  auto bn_mode = TECODNN_BATCHNORM_SPATIAL;

  // set conv desc
  TECODNN_CHECK(tecodnnCreateConvolutionDescriptor(&convDesc));
  TECODNN_CHECK(
      tecodnnSetConvolutionMathType(convDesc, TECODNN_TENSOR_ACC_MATH));
  TECODNN_CHECK(tecodnnSetConvolutionGroupCount(convDesc, groups));
  TECODNN_CHECK(tecodnnSetConvolution2dDescriptor(convDesc,
                                                  paddings[0],
                                                  paddings[1],
                                                  strides[0],
                                                  strides[1],
                                                  dilations[0],
                                                  dilations[1],
                                                  TECODNN_CROSS_CORRELATION,
                                                  TECODNN_DATA_FLOAT));

  plan_helper.BindConstParam(TECODNN_PARAM_XDESC, conv_input_tmp)
      .BindConstParam(TECODNN_PARAM_WDESC, conv_filter_tmp)
      .BindConstParam(TECODNN_PARAM_YDESC, conv_out_tmp)
      .BindConstParam(TECODNN_PARAM_CONV_DESC, convDesc)
      .BindConstParam(TECODNN_PARAM_CONV_ALGO, &algo)
      .BindConstParam(TECODNN_PARAM_BN_MODE, &bn_mode)
      .BindConstParam(TECODNN_PARAM_BN_SCALEBIAS_MEANVAR_DESC, bn_mean)
      .BindConstParam(TECODNN_PARAM_BN_SCALE_PLACEHOLDER, &bn_place_holder)
      .BindConstParam(TECODNN_PARAM_BN_BIAS_PLACEHOLDER, &bn_place_holder)
      .BindConstParam(TECODNN_PARAM_BN_RUNNING_MEAN_PLACEHOLDER,
                      &bn_place_holder)
      .BindConstParam(TECODNN_PARAM_BN_RUNNING_VAR_PLACEHOLDER,
                      &bn_place_holder)
      .MakeFuseOpsPlan(handle);

  // conv worksapce
  phi::DenseTensor workspace;
  if (plan_helper.workspace_size_ != 0)
    workspace.Resize({static_cast<int64_t>(plan_helper.workspace_size_)});
  custom_ctx->Alloc(&workspace, phi::DataType::INT8);

  double double_epsilon = static_cast<double>(bn_epsilon);
  plan_helper
      .BindVarParam(TECODNN_SCALE_DOUBLE_BN_EPSILON,
                    const_cast<double*>(&double_epsilon))
      .BindVarParam(TECODNN_PTR_BN_SCALE, const_cast<void*>(bn_scale.data()))
      .BindVarParam(TECODNN_PTR_BN_BIAS, const_cast<void*>(bn_bias.data()))
      .BindVarParam(TECODNN_PTR_BN_RUNNING_MEAN,
                    const_cast<void*>(bn_mean.data()))
      .BindVarParam(TECODNN_PTR_BN_RUNNING_VAR,
                    const_cast<void*>(bn_var.data()))
      .BindVarParam(TECODNN_PTR_XDATA, const_cast<void*>(conv_input_tmp.data()))
      .BindVarParam(TECODNN_PTR_WDATA,
                    const_cast<void*>(conv_filter_tmp.data()))
      .BindVarParam(TECODNN_PTR_YDATA, conv_out_tmp.data())
      .BindVarParam(TECODNN_PTR_WORKSPACE, workspace.data())
      .BindVarParam(TECODNN_SCALE_SIZE_T_WORKSPACE_SIZE_IN_BYTES,
                    &plan_helper.workspace_size_);
  tecodnnFusedOpsExecute(handle, plan_helper.plan_, plan_helper.var_pack_);

  if (!channel_last) {
    auto out_nhwc = static_cast<phi::DenseTensor*>(conv_out_tmp.impl().get());
    auto out_temp = static_cast<phi::DenseTensor*>(out.impl().get());
    custom_kernel::sdaa_ops::doTransformTensor(
        *custom_ctx, *out_nhwc, custom_kernel::Convert_TF::NHWC2NCHW, out_temp);
  } else {
    out = conv_out_tmp;
  }
  TECODNN_CHECK(tecodnnDestroyConvolutionDescriptor(convDesc));
  return {out};
}

std::vector<paddle::DataType> CustomFusedConvBNInferDtype(
    const paddle::DataType& conv_input,
    const paddle::DataType& conv_filter,
    const paddle::DataType& bn_scale,
    const paddle::DataType& bn_bias,
    const paddle::DataType& bn_mean,
    const paddle::DataType& bn_var) {
  return {conv_input};
}

PD_BUILD_OP(custom_fused_conv_bn)
    .Inputs({"Input", "Filter", "Scale", "Bias", "Mean", "Var"})
    .Attrs({"strides: std::vector<int>",
            "paddings: std::vector<int>",
            "padding_algorithm: std::string",
            "dilations: std::vector<int>",
            "groups: int",
            "data_format: std::string",
            "epsilon: float",
            "activation_type: std::string"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(CustomFusedConvBN))
    .SetInferDtypeFn(PD_INFER_DTYPE(CustomFusedConvBNInferDtype))
    .SetInferShapeFn(PD_INFER_SHAPE(CustomFusedConvBNInferShape));
