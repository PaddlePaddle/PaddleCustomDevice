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
void ConvTransposeRawKernel(const std::string& conv_type,
                            const Context& dev_ctx,
                            const DenseTensor& x,
                            const DenseTensor& filter,
                            const std::vector<int>& strides,
                            const std::vector<int>& paddings,
                            const std::vector<int>& output_padding,
                            const phi::IntArray& output_size,
                            const std::string& padding_algorithm,
                            int groups,
                            const std::vector<int>& dilations,
                            const std::string& data_format,
                            DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  TensorNameMap input_names;
  input_names["Input"] = {"input"};
  input_names["Filter"] = {"filter"};

  TensorValueMap inputs;
  inputs["Input"] = {const_cast<DenseTensor*>(&x)};
  inputs["Filter"] = {const_cast<DenseTensor*>(&filter)};

  TensorNameMap output_names;
  output_names["Out"] = {"out"};

  TensorValueMap outputs;
  outputs["Out"] = {out};

  std::vector<int> output_size_list = GetIntList(output_size.GetData());

  GcuAttributeMap attrs;
  attrs["strides"] = strides;
  attrs["paddings"] = paddings;
  attrs["output_padding"] = output_padding;
  attrs["output_size"] = output_size_list;
  attrs["padding_algorithm"] = padding_algorithm;
  attrs["groups"] = groups;
  attrs["dilations"] = dilations;
  attrs["data_format"] = data_format;

  GcuRunner(
      input_names, inputs, output_names, outputs, attrs, conv_type, dev_ctx);
}

template <typename T, typename Context>
void ConvTransposeGradRawKernel(const std::string& conv_grad_type,
                                const Context& dev_ctx,
                                const DenseTensor& x,
                                const DenseTensor& filter,
                                const DenseTensor& dout,
                                const std::vector<int>& strides,
                                const std::vector<int>& paddings,
                                const std::vector<int>& output_padding,
                                const phi::IntArray& output_size,
                                const std::string& padding_algorithm,
                                int groups,
                                const std::vector<int>& dilations,
                                const std::string& data_format,
                                DenseTensor* dx,
                                DenseTensor* dfilter) {
  dev_ctx.template Alloc<T>(dx);
  dev_ctx.template Alloc<T>(dfilter);

  TensorNameMap input_names;
  input_names["Input"] = {"input"};
  input_names["Filter"] = {"filter"};
  input_names[GradVarName("Output")] = {"dout"};

  TensorValueMap inputs;
  inputs["Input"] = {const_cast<DenseTensor*>(&x)};
  inputs["Filter"] = {const_cast<DenseTensor*>(&filter)};
  inputs[GradVarName("Output")] = {const_cast<DenseTensor*>(&dout)};

  TensorNameMap output_names;
  output_names[GradVarName("Input")] = {"dx"};
  output_names[GradVarName("Filter")] = {"dfilter"};

  TensorValueMap outputs;
  outputs[GradVarName("Input")] = {dx};
  outputs[GradVarName("Filter")] = {dfilter};

  std::vector<int> output_size_list = GetIntList(output_size.GetData());

  GcuAttributeMap attrs;
  attrs["strides"] = strides;
  attrs["paddings"] = paddings;
  attrs["output_padding"] = output_padding;
  attrs["output_size"] = output_size_list;
  attrs["padding_algorithm"] = padding_algorithm;
  attrs["groups"] = groups;
  attrs["dilations"] = dilations;
  attrs["data_format"] = data_format;

  GcuRunner(input_names,
            inputs,
            output_names,
            outputs,
            attrs,
            conv_grad_type,
            dev_ctx);
}

template <typename T, typename Context>
void Conv2dTransposeKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& output_padding,
                           const phi::IntArray& output_size,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           DenseTensor* out) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "conv2d_transpose", conv2d_transpose);

    dev_ctx.template Alloc<T>(out);

    // TODO(zhiheng.yu): only support NCHW now, NHWC need to verification
    if (data_format != "NCHW") {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Only support NCHW for now, but got : %s", data_format.c_str()));
    }
    if (!output_padding.empty()) {
      for (auto op : output_padding) {
        if (op != 0)
          PADDLE_THROW(phi::errors::Unimplemented("Unsupport output padding,"));
      }
    }

    std::vector<int64_t> padding =
        cal_aot_padding(phi::vectorize(x.dims()),
                        phi::vectorize(filter.dims()),
                        strides,
                        paddings,
                        padding_algorithm);

    // inputs
    auto input_nhwc = x;
    auto filter_nhwc = filter;
    if (data_format == "NCHW") {
      input_nhwc = ConvertNCHWToNHWC(dev_ctx, x);
      filter_nhwc = ConvertNCHWToNHWC(dev_ctx, filter);
    }
    auto filter_reverse = reverse(dev_ctx, filter_nhwc, {1, 2});

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

    auto input_gcu = GetHlirTensorV2(input_nhwc, x.dims());
    auto filter_gcu = GetHlirTensorV2(filter_reverse, filter.dims());
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
    AOTOPS_DEBUG(kConv2dBpi, params);
    GCUOPS_TRACE_START(conv2d_bpi);
    auto func_ptr = GetOpFuncPtr(kConv2dBpi, params);
    if (func_ptr) {
      auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
      PADDLE_ENFORCE(
          pass,
          phi::errors::InvalidArgument("dispatch %s failed!", kConv2dBpi));
    } else {
      PADDLE_ENFORCE(
          false,
          phi::errors::InvalidArgument("not find aot func for %s", kConv2dBpi));
    }
    FreeDispatchParam(params);
    GCUOPS_TRACE_END(conv2d_bpi);
    GcuOpStreamSync(dev_ctx);

    if (data_format == "NCHW") *out = ConvertNHWCToNCHW(dev_ctx, out_nhwc);

    PADDLE_GCU_KERNEL_END("conv2d_transpose", conv2d_transpose);
  } else {
    ConvTransposeRawKernel<T, Context>("conv2d_transpose",
                                       dev_ctx,
                                       x,
                                       filter,
                                       strides,
                                       paddings,
                                       output_padding,
                                       output_size,
                                       padding_algorithm,
                                       groups,
                                       dilations,
                                       data_format,
                                       out);
  }
}

template <typename T, typename Context>
void Conv2dTransposeGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& filter,
                               const DenseTensor& dout,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding,
                               const phi::IntArray& output_size,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format,
                               DenseTensor* dx,
                               DenseTensor* dfilter) {
  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(
        dev_ctx, "conv2d_transpose_grad", conv2d_transpose_grad);

    // TODO(zhiheng.yu): only support NCHW now, NHWC need to verification
    if (data_format != "NCHW") {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Only support NCHW for now, but got : %s", data_format.c_str()));
    }
    if (!output_padding.empty()) {
      for (auto op : output_padding) {
        if (op != 0)
          PADDLE_THROW(phi::errors::Unimplemented("Unsupport output padding,"));
      }
    }

    std::vector<int64_t> padding =
        cal_aot_padding(phi::vectorize(x.dims()),
                        phi::vectorize(filter.dims()),
                        strides,
                        paddings,
                        padding_algorithm);

    // inputs
    auto input_nhwc = x;
    auto filter_nhwc = filter;
    auto dout_nhwc = dout;
    if (data_format == "NCHW") {
      input_nhwc = ConvertNCHWToNHWC(dev_ctx, x);
      filter_nhwc = ConvertNCHWToNHWC(dev_ctx, filter);
      dout_nhwc = ConvertNCHWToNHWC(dev_ctx, dout);
    }

    // calculate dx
    if (dx) {
      // output
      dev_ctx.template Alloc<T>(dx);

      phi::DenseTensor dx_nhwc = *dx;
      if (data_format == "NCHW") {
        std::vector<int64_t> input_dims = phi::vectorize(dx->dims());
        std::vector<int64_t> dst_dims =
            reorder_vector(input_dims, {0, 2, 3, 1});

        phi::DenseTensor dst_tensor;
        phi::DenseTensorMeta meta(
            dx->dtype(), phi::make_ddim(dst_dims), phi::DataLayout::NHWC);
        dst_tensor.set_meta(meta);
        dev_ctx.Alloc(&dst_tensor, dst_tensor.dtype());
        dx_nhwc = dst_tensor;
      }

      auto dout_gcu = GetHlirTensorV2(dout_nhwc, dout.dims());
      auto filter_gcu = GetHlirTensorV2(filter_nhwc, filter.dims());
      auto dx_gcu = GetHlirTensorV2(dx_nhwc, dx->dims());
      hlir::DispatchParam params;
      params.inputs = {dout_gcu, filter_gcu};
      params.outputs = {dx_gcu};
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

      if (data_format == "NCHW") *dx = ConvertNHWCToNCHW(dev_ctx, dx_nhwc);
    }

    // calculate dfilter
    if (dfilter) {
      // output
      dev_ctx.template Alloc<T>(dfilter);

      phi::DenseTensor dfilter_nhwc = *dfilter;
      if (data_format == "NCHW") {
        std::vector<int64_t> filter_dims = phi::vectorize(dfilter->dims());
        std::vector<int64_t> dst_dims =
            reorder_vector(filter_dims, {0, 2, 3, 1});

        phi::DenseTensor dst_tensor;
        phi::DenseTensorMeta meta(
            dfilter->dtype(), phi::make_ddim(dst_dims), phi::DataLayout::NHWC);
        dst_tensor.set_meta(meta);
        dev_ctx.Alloc(&dst_tensor, dst_tensor.dtype());
        dfilter_nhwc = dst_tensor;
      }

      auto dout_gcu = GetHlirTensorV2(dout_nhwc, dout.dims());
      auto input_gcu = GetHlirTensorV2(input_nhwc, x.dims());
      auto dfilter_gcu = GetHlirTensorV2(dfilter_nhwc, dfilter->dims());
      hlir::DispatchParam params;
      params.inputs = {dout_gcu, input_gcu};
      params.outputs = {dfilter_gcu};
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
        *dfilter = ConvertNHWCToNCHW(dev_ctx, dfilter_nhwc);
    }

    PADDLE_GCU_KERNEL_END("conv2d_transpose_grad", conv2d_transpose_grad);
  } else {
    ConvTransposeGradRawKernel<T, Context>("conv2d_transpose_grad",
                                           dev_ctx,
                                           x,
                                           filter,
                                           dout,
                                           strides,
                                           paddings,
                                           output_padding,
                                           output_size,
                                           padding_algorithm,
                                           groups,
                                           dilations,
                                           data_format,
                                           dx,
                                           dfilter);
  }
}

template <typename T, typename Context>
void Conv3dTransposeKernel(const Context& dev_ctx,
                           const DenseTensor& x,
                           const DenseTensor& filter,
                           const std::vector<int>& strides,
                           const std::vector<int>& paddings,
                           const std::vector<int>& output_padding,
                           const std::vector<int>& output_size,
                           const std::string& padding_algorithm,
                           int groups,
                           const std::vector<int>& dilations,
                           const std::string& data_format,
                           DenseTensor* out) {
  ConvTransposeRawKernel<T, Context>("conv3d_transpose",
                                     dev_ctx,
                                     x,
                                     filter,
                                     strides,
                                     paddings,
                                     output_padding,
                                     output_size,
                                     padding_algorithm,
                                     groups,
                                     dilations,
                                     data_format,
                                     out);
}

template <typename T, typename Context>
void Conv3dTransposeGradKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const DenseTensor& filter,
                               const DenseTensor& dout,
                               const std::vector<int>& strides,
                               const std::vector<int>& paddings,
                               const std::vector<int>& output_padding,
                               const std::vector<int>& output_size,
                               const std::string& padding_algorithm,
                               int groups,
                               const std::vector<int>& dilations,
                               const std::string& data_format,
                               DenseTensor* dx,
                               DenseTensor* dfilter) {
  ConvTransposeGradRawKernel<T, Context>("conv3d_transpose_grad",
                                         dev_ctx,
                                         x,
                                         filter,
                                         dout,
                                         strides,
                                         paddings,
                                         output_padding,
                                         output_size,
                                         padding_algorithm,
                                         groups,
                                         dilations,
                                         data_format,
                                         dx,
                                         dfilter);
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(conv2d_transpose,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dTransposeKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv2d_transpose_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv2dTransposeGradKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv3d_transpose,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv3dTransposeKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(conv3d_transpose_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Conv3dTransposeGradKernel,
                          float,
                          phi::dtype::float16) {}
