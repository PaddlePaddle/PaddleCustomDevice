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
#include "kernels/funcs/gcu_kernel_funcs.h"
#include "kernels/funcs/gcu_name_list.h"
#include "kernels/funcs/gcu_op_runner.h"
#include "paddle/phi/common/type_traits.h"

namespace custom_kernel {

template <typename T, typename Context>
void GcuConvKernel(const Context& dev_ctx,
                   const phi::DenseTensor& input,
                   const phi::DenseTensor& filter,
                   const std::vector<int>& strides,
                   const std::vector<int>& paddings,
                   const std::string& padding_algorithm,
                   int groups,
                   const std::vector<int>& dilations,
                   const std::string& data_format,
                   phi::DenseTensor* out,
                   const std::string& op_type) {
  dev_ctx.template Alloc<T>(out);

  TensorNameMap input_names;
  input_names["Input"] = {"input"};
  input_names["Filter"] = {"filter"};

  TensorValueMap inputs;
  inputs["Input"] = {const_cast<DenseTensor*>(&input)};
  inputs["Filter"] = {const_cast<DenseTensor*>(&filter)};

  TensorNameMap output_names;
  output_names["Output"] = {"out"};

  TensorValueMap outputs;
  outputs["Output"] = {out};

  GcuAttributeMap attrs;
  attrs["strides"] = strides;
  attrs["paddings"] = paddings;
  attrs["padding_algorithm"] = padding_algorithm;
  attrs["dilations"] = dilations;
  attrs["groups"] = groups;
  attrs["data_format"] = data_format;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void GcuConvGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& input,
                       const phi::DenseTensor& filter,
                       const phi::DenseTensor& out_grad,
                       const std::vector<int>& strides,
                       const std::vector<int>& paddings,
                       const std::string& padding_algorithm,
                       int groups,
                       const std::vector<int>& dilations,
                       const std::string& data_format,
                       phi::DenseTensor* input_grad,
                       phi::DenseTensor* filter_grad,
                       const std::string& op_type) {
  TensorNameMap input_names;
  input_names["Input"] = {"input"};
  input_names["Filter"] = {"filter"};
  input_names[GradVarName("Output")] = {"out_grad"};

  TensorValueMap inputs;
  inputs["Input"] = {const_cast<DenseTensor*>(&input)};
  inputs["Filter"] = {const_cast<DenseTensor*>(&filter)};
  inputs[GradVarName("Output")] = {const_cast<DenseTensor*>(&out_grad)};

  TensorNameMap output_names;
  TensorValueMap outputs;
  if (input_grad) {
    dev_ctx.template Alloc<T>(input_grad);
    output_names[GradVarName("Input")] = {"input_grad"};
    outputs[GradVarName("Input")] = {input_grad};
  }
  if (filter_grad) {
    dev_ctx.template Alloc<T>(filter_grad);
    output_names[GradVarName("Filter")] = {"filter_grad"};
    outputs[GradVarName("Filter")] = {filter_grad};
  }

  GcuAttributeMap attrs;
  attrs["strides"] = strides;
  attrs["paddings"] = paddings;
  attrs["padding_algorithm"] = padding_algorithm;
  attrs["dilations"] = dilations;
  attrs["groups"] = groups;
  attrs["data_format"] = data_format;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, op_type, dev_ctx);
}

template <typename T, typename Context>
void Conv2dKernel(const Context& dev_ctx,
                  const DenseTensor& input,
                  const DenseTensor& filter,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string& padding_algorithm,
                  const std::vector<int>& dilations,
                  int groups,
                  const std::string& data_format,
                  DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "conv2d", conv2d);

    dev_ctx.template Alloc<T>(out);

    // TODO(zhiheng.yu): only support NCHW now, NHWC need to verification
    if (data_format != "NCHW") {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Only support NCHW for now, but got : %s", data_format.c_str()));
    }

    std::vector<int64_t> padding =
        cal_aot_padding(phi::vectorize(input.dims()),
                        phi::vectorize(filter.dims()),
                        strides,
                        paddings,
                        padding_algorithm);

    // inputs
    auto input_nhwc = input;
    auto filter_nhwc = filter;
    if (data_format == "NCHW") {
      input_nhwc = ConvertNCHWToNHWC(dev_ctx, input);
      filter_nhwc = ConvertNCHWToNHWC(dev_ctx, filter);
    }

    // output
    phi::DenseTensor out_nhwc = *out;
    if (data_format == "NCHW") {
      std::vector<int64_t> out_dims = phi::vectorize(out->dims());
      std::vector<int64_t> dst_dims = reorder_vector(out_dims, {0, 2, 3, 1});

      phi::DenseTensor dst_tensor;
      phi::DenseTensorMeta meta(
          out->dtype(), phi::make_ddim(dst_dims), phi::DataLayout::NHWC);
      dst_tensor.set_meta(meta);
      dev_ctx.Alloc(&dst_tensor, dst_tensor.dtype());
      out_nhwc = dst_tensor;
    }

    auto input_gcu = GetHlirTensorV2(input_nhwc, input.dims());
    auto filter_gcu = GetHlirTensorV2(filter_nhwc, filter.dims());
    auto out_gcu = GetHlirTensorV2(out_nhwc, out->dims());
    hlir::DispatchParam params;
    params.inputs = {input_gcu, filter_gcu};
    params.outputs = {out_gcu};
    params.metadata.setValue("mode", int64_t(0));
    params.metadata.setValue("nanOpt", int64_t(0));
    params.metadata.setValue("algo", int64_t(0));
    params.metadata.setValue("stride", VectorToHlirVector(strides));
    params.metadata.setValue("padding", HlirVector(padding));
    params.metadata.setValue("dilation", VectorToHlirVector(dilations));
    params.metadata.setValue("groups", static_cast<int64_t>(groups));
    params.metadata.setValue(hlir::kAlpha, 1.0);
    params.metadata.setValue(hlir::kBelta, 0.0);
    params.stream = static_cast<topsStream_t>(dev_ctx.stream());
    AOTOPS_DEBUG(kConv2d, params);
    GCUOPS_TRACE_START(conv2d);
    auto func_ptr = GetOpFuncPtr(kConv2d, params);
    if (func_ptr) {
      auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
      PADDLE_ENFORCE(
          pass, phi::errors::InvalidArgument("dispatch %s failed!", kConv2d));
    } else {
      PADDLE_ENFORCE(
          false,
          phi::errors::InvalidArgument("not find aot func for %s", kConv2d));
    }
    FreeDispatchParam(params);
    GCUOPS_TRACE_END(conv2d);
    GcuOpStreamSync(dev_ctx);

    if (data_format == "NCHW") *out = ConvertNHWCToNCHW(dev_ctx, out_nhwc);

    PADDLE_GCU_KERNEL_END("conv2d", conv2d);
  } else {
    GcuConvKernel<T, Context>(dev_ctx,
                              input,
                              filter,
                              strides,
                              paddings,
                              padding_algorithm,
                              groups,
                              dilations,
                              data_format,
                              out,
                              "conv2d");
  }
}

template <typename T, typename Context>
void Conv2DGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& input,
                      const phi::DenseTensor& filter,
                      const phi::DenseTensor& output_grad,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const std::string& padding_algorithm,
                      const std::vector<int>& dilations,
                      int groups,
                      const std::string& data_format,
                      phi::DenseTensor* input_grad,
                      phi::DenseTensor* filter_grad) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "conv2d_grad", conv2d_grad);

    // TODO(zhiheng.yu): only support NCHW now, NHWC need to verification
    if (data_format != "NCHW") {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Only support NCHW for now, but got : %s", data_format.c_str()));
    }

    std::vector<int64_t> padding =
        cal_aot_padding(phi::vectorize(input.dims()),
                        phi::vectorize(filter.dims()),
                        strides,
                        paddings,
                        padding_algorithm);

    // inputs
    auto input_nhwc = input;
    auto filter_nhwc = filter;
    auto out_grad_nhwc = output_grad;
    if (data_format == "NCHW") {
      input_nhwc = ConvertNCHWToNHWC(dev_ctx, input);
      filter_nhwc = ConvertNCHWToNHWC(dev_ctx, filter);
      out_grad_nhwc = ConvertNCHWToNHWC(dev_ctx, output_grad);
    }
    auto filter_reverse = reverse(dev_ctx, filter_nhwc, {1, 2});

    // calculate input_grad
    if (input_grad) {
      // output
      dev_ctx.template Alloc<T>(input_grad);

      phi::DenseTensor input_grad_nhwc = *input_grad;
      if (data_format == "NCHW") {
        std::vector<int64_t> input_dims = phi::vectorize(input_grad->dims());
        std::vector<int64_t> dst_dims =
            reorder_vector(input_dims, {0, 2, 3, 1});

        phi::DenseTensor dst_tensor;
        phi::DenseTensorMeta meta(input_grad->dtype(),
                                  phi::make_ddim(dst_dims),
                                  phi::DataLayout::NHWC);
        dst_tensor.set_meta(meta);
        dev_ctx.Alloc(&dst_tensor, dst_tensor.dtype());
        input_grad_nhwc = dst_tensor;
      }

      auto out_grad_gcu = GetHlirTensorV2(out_grad_nhwc, output_grad.dims());
      auto filter_gcu = GetHlirTensorV2(filter_reverse, filter.dims());
      auto input_grad_gcu =
          GetHlirTensorV2(input_grad_nhwc, input_grad->dims());
      hlir::DispatchParam params;
      params.inputs = {out_grad_gcu, filter_gcu};
      params.outputs = {input_grad_gcu};
      params.metadata.setValue("algo", int64_t(0));
      params.metadata.setValue("stride", VectorToHlirVector(strides));
      params.metadata.setValue("padding", HlirVector(padding));
      params.metadata.setValue("dilation", VectorToHlirVector(dilations));
      params.metadata.setValue("groups", static_cast<int64_t>(groups));
      params.metadata.setValue(hlir::kAlpha, 1.0);
      params.metadata.setValue(hlir::kBelta, 0.0);
      params.stream = static_cast<topsStream_t>(dev_ctx.stream());
      AOTOPS_DEBUG(kConv2dBpi, params);
      GCUOPS_TRACE_START(conv2d_bpi);
      auto func_ptr = GetOpFuncPtr(kConv2dBpi, params);
      if (func_ptr) {
        auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
        PADDLE_ENFORCE(
            pass,
            phi::errors::InvalidArgument("dispatch %s failed!", kConv2dBpi));
      } else {
        PADDLE_ENFORCE(false,
                       phi::errors::InvalidArgument("not find aot func for %s",
                                                    kConv2dBpi));
      }
      FreeDispatchParam(params);
      GCUOPS_TRACE_END(conv2d_bpi);
      GcuOpStreamSync(dev_ctx);

      if (data_format == "NCHW")
        *input_grad = ConvertNHWCToNCHW(dev_ctx, input_grad_nhwc);
    }

    // calculate filter_grad
    if (filter_grad) {
      // output
      dev_ctx.template Alloc<T>(filter_grad);

      phi::DenseTensor filter_grad_nhwc = *filter_grad;
      if (data_format == "NCHW") {
        std::vector<int64_t> filter_dims = phi::vectorize(filter_grad->dims());
        std::vector<int64_t> dst_dims =
            reorder_vector(filter_dims, {0, 2, 3, 1});

        phi::DenseTensor dst_tensor;
        phi::DenseTensorMeta meta(filter_grad->dtype(),
                                  phi::make_ddim(dst_dims),
                                  phi::DataLayout::NHWC);
        dst_tensor.set_meta(meta);
        dev_ctx.Alloc(&dst_tensor, dst_tensor.dtype());
        filter_grad_nhwc = dst_tensor;
      }

      auto input_gcu = GetHlirTensorV2(input_nhwc, input.dims());
      auto out_grad_gcu = GetHlirTensorV2(out_grad_nhwc, output_grad.dims());
      auto filter_grad_gcu =
          GetHlirTensorV2(filter_grad_nhwc, filter_grad->dims());
      hlir::DispatchParam params;
      params.inputs = {input_gcu, out_grad_gcu};
      params.outputs = {filter_grad_gcu};
      params.metadata.setValue("algo", int64_t(0));
      params.metadata.setValue("stride", VectorToHlirVector(strides));
      params.metadata.setValue("padding", HlirVector(padding));
      params.metadata.setValue("dilation", VectorToHlirVector(dilations));
      params.metadata.setValue("groups", static_cast<int64_t>(groups));
      params.metadata.setValue(hlir::kAlpha, 1.0);
      params.metadata.setValue(hlir::kBelta, 0.0);
      params.stream = static_cast<topsStream_t>(dev_ctx.stream());
      AOTOPS_DEBUG(kConv2dBpk, params);
      GCUOPS_TRACE_START(conv2d_bpk);
      auto func_ptr = GetOpFuncPtr(kConv2dBpk, params);
      if (func_ptr) {
        auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
        PADDLE_ENFORCE(
            pass,
            phi::errors::InvalidArgument("dispatch %s failed!", kConv2dBpk));
      } else {
        PADDLE_ENFORCE(false,
                       phi::errors::InvalidArgument("not find aot func for %s",
                                                    kConv2dBpk));
      }
      FreeDispatchParam(params);
      GCUOPS_TRACE_END(conv2d_bpk);
      GcuOpStreamSync(dev_ctx);

      if (data_format == "NCHW")
        *filter_grad = ConvertNHWCToNCHW(dev_ctx, filter_grad_nhwc);
    }

    PADDLE_GCU_KERNEL_END("conv2d_grad", conv2d_grad);
  } else {
    GcuConvGradKernel<T, Context>(dev_ctx,
                                  input,
                                  filter,
                                  output_grad,
                                  strides,
                                  paddings,
                                  padding_algorithm,
                                  groups,
                                  dilations,
                                  data_format,
                                  input_grad,
                                  filter_grad,
                                  "conv2d_grad");
  }
}

template <typename T, typename Context>
void DepthwiseConv2dKernel(const Context& dev_ctx,
                           const phi::DenseTensor& input,
                           const phi::DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           phi::DenseTensor* out) {
  GcuConvKernel<T, Context>(dev_ctx,
                            input,
                            filter,
                            strides,
                            paddings,
                            padding_algorithm,
                            groups,
                            dilations,
                            data_format,
                            out,
                            "depthwise_conv2d");
}

template <typename T, typename Context>
void DepthwiseConv2dGradKernel(const Context& dev_ctx,
                               const phi::DenseTensor& input,
                               const phi::DenseTensor& filter,
                               const phi::DenseTensor& out_grad,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format,
                               phi::DenseTensor* input_grad,
                               phi::DenseTensor* filter_grad) {
  GcuConvGradKernel<T, Context>(dev_ctx,
                                input,
                                filter,
                                out_grad,
                                strides,
                                paddings,
                                padding_algorithm,
                                groups,
                                dilations,
                                data_format,
                                input_grad,
                                filter_grad,
                                "depthwise_conv2d_grad");
}

template <typename T, typename Context>
void Conv3dKernel(const Context& dev_ctx,
                  const phi::DenseTensor& input,
                  const phi::DenseTensor& filter,
                  const std::vector<int>& strides,
                  const std::vector<int>& paddings,
                  const std::string& padding_algorithm,
                  int groups,
                  const std::vector<int>& dilations,
                  const std::string& data_format,
                  phi::DenseTensor* out) {
  GcuConvKernel<T, Context>(dev_ctx,
                            input,
                            filter,
                            strides,
                            paddings,
                            padding_algorithm,
                            groups,
                            dilations,
                            data_format,
                            out,
                            "conv3d");
}

template <typename T, typename Context>
void Conv3dGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& input,
                      const phi::DenseTensor& filter,
                      const phi::DenseTensor& out_grad,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings,
                      const std::string& padding_algorithm,
                      int groups,
                      const std::vector<int>& dilations,
                      const std::string& data_format,
                      phi::DenseTensor* input_grad,
                      phi::DenseTensor* filter_grad) {
  GcuConvGradKernel<T, Context>(dev_ctx,
                                input,
                                filter,
                                out_grad,
                                strides,
                                paddings,
                                padding_algorithm,
                                groups,
                                dilations,
                                data_format,
                                input_grad,
                                filter_grad,
                                "conv3d_grad");
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(depthwise_conv2d,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::DepthwiseConv2dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(depthwise_conv2d_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::DepthwiseConv2dGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv3d,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv3dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv3d_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv3dGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv2d,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv2d_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv2DGradKernel,
                          float,
                          phi::dtype::float16) {}
