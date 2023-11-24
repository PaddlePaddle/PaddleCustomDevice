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

namespace custom_kernel {

void pooling2d(const phi::CustomContext& dev_ctx,
               const phi::DenseTensor& input,
               const std::vector<int64_t>& ksizes,
               const std::vector<int64_t>& strides,
               const std::vector<int64_t>& paddings,
               const std::string& data_format,
               int64_t pooling_mode,
               phi::DenseTensor* out) {
  // input
  auto input_nhwc = input;
  if (data_format == "NCHW") input_nhwc = ConvertNCHWToNHWC(dev_ctx, input);

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
  auto output_gcu = GetHlirTensorV2(out_nhwc, out->dims());
  hlir::DispatchParam params;
  params.inputs = {input_gcu};
  params.outputs = {output_gcu};
  params.metadata.setValue("pooling_mode", pooling_mode);
  params.metadata.setValue("window_height", ksizes[0]);
  params.metadata.setValue("window_width", ksizes[1]);
  params.metadata.setValue("vertical_stride", strides[0]);
  params.metadata.setValue("horizontal_stride", strides[1]);
  params.metadata.setValue("vertical_padding", paddings[0]);
  params.metadata.setValue("horizontal_padding", paddings[1]);
  params.stream = static_cast<topsStream_t>(dev_ctx.stream());
  AOTOPS_DEBUG(kPooling2d, params);
  GCUOPS_TRACE_START(pooling2d);
  auto func_ptr = GetOpFuncPtr(kPooling2d, params);
  if (func_ptr) {
    auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
    PADDLE_ENFORCE(
        pass, phi::errors::InvalidArgument("dispatch %s failed!", kPooling2d));
  } else {
    PADDLE_ENFORCE(
        false,
        phi::errors::InvalidArgument("not find aot func for %s", kPooling2d));
  }
  FreeDispatchParam(params);
  GCUOPS_TRACE_END(pooling2d);
  GcuOpStreamSync(dev_ctx);

  if (data_format == "NCHW") *out = ConvertNHWCToNCHW(dev_ctx, out_nhwc);
}

template <typename T, typename Context>
void Pool2dKernel(const Context& dev_ctx,
                  const phi::DenseTensor& in_x,
                  const phi::IntArray& kernel_size,
                  const std::vector<int>& strides_t,
                  const std::vector<int>& paddings_t,
                  bool ceil_mode,
                  bool exclusive,
                  const std::string& data_format,
                  const std::string& pooling_type,
                  bool global_pooling,
                  bool adaptive,
                  const std::string& padding_algorithm,
                  phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  // aot reduce_window only support max_pool & false ceil_mode
  if (UseScatterMemory() && ceil_mode == false) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "pool2d", pool2d);

    PADDLE_ENFORCE_EQ(
        data_format,
        "NCHW",
        phi::errors::Unimplemented("Only support NCHW for now, but got : %s",
                                   data_format.c_str()));

    std::vector<int64_t> ksizes = kernel_size.GetData();

    PADDLE_ENFORCE_EQ(
        ksizes.size(),
        2,
        phi::errors::InvalidArgument("The size of kernel_size must be 2."));

    PADDLE_ENFORCE_EQ(
        strides_t.size(),
        2,
        phi::errors::InvalidArgument("The size of strides_t must be 2."));

    PADDLE_ENFORCE_EQ(
        paddings_t.size(),
        2,
        phi::errors::InvalidArgument("The size of paddings_t must be 2."));

    int64_t pooling_mode = 0;
    if (pooling_type == "max") {
      pooling_mode = /*TOPSOP_POOLING_MAX*/ 0;
    } else if (pooling_type == "avg") {
      pooling_mode = /*TOPSOP_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING*/ 2;
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("Unsupport pooling_type: %s",
                                              pooling_type.c_str()));
    }

    std::vector<int64_t> strides = vector_s64(strides_t);
    std::vector<int64_t> paddings = vector_s64(paddings_t);
    if (adaptive) {
      int64_t ih = in_x.dims().at(2);
      int64_t iw = in_x.dims().at(3);
      PADDLE_ENFORCE(ih % ksizes[0] == 0 && iw % ksizes[1] == 0,
                     phi::errors::InvalidArgument(
                         "only support: MOD(oh, ih) == 0 && MOD(ow, iw) == 0"));
      ksizes[0] = ih / ksizes[0];
      ksizes[1] = iw / ksizes[1];
      strides = {ksizes[0], ksizes[1]};
      paddings = {0, 0};
    }

    pooling2d(dev_ctx,
              in_x,
              ksizes,
              strides,
              paddings,
              data_format,
              pooling_mode,
              out);

    PADDLE_GCU_KERNEL_END("pool2d", pool2d);
  } else {
    std::vector<int> ksize(kernel_size.GetData().begin(),
                           kernel_size.GetData().end());

    TensorNameMap input_names;
    input_names["X"] = {"x"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&in_x)};

    TensorNameMap output_names;
    output_names["Out"] = {"out"};

    TensorValueMap outputs;
    outputs["Out"] = {out};

    GcuAttributeMap attrs;
    attrs["adaptive"] = adaptive;
    attrs["global_pooling"] = global_pooling;
    attrs["padding_algorithm"] = padding_algorithm;
    attrs["pooling_type"] = pooling_type;
    attrs["data_format"] = data_format;
    attrs["ksize"] = ksize;
    attrs["strides"] = strides_t;
    attrs["paddings"] = paddings_t;
    attrs["ceil_mode"] = ceil_mode;
    attrs["exclusive"] = exclusive;

    GcuRunner(
        input_names, inputs, output_names, outputs, attrs, "pool2d", dev_ctx);
  }
}

static std::vector<int64_t> gradient_padding_size(
    const phi::DDim& out_grad_shape,
    const phi::DDim& in_x_shape,
    const std::vector<int64_t>& ksizes,
    const std::vector<int64_t>& strides,
    const std::vector<int64_t>& paddings) {
  // padding for height
  int64_t pad_height_top = ksizes[0] - 1 - paddings[0];
  int64_t pad_height_interior = strides[0] - 1;
  int64_t padded_out_height = in_x_shape[2] + ksizes[0] - 1;
  int64_t pad_height_bottom = padded_out_height - pad_height_top -
                              out_grad_shape[2] -
                              (out_grad_shape[2] - 1) * pad_height_interior;

  // padding for width
  int64_t pad_width_left = ksizes[1] - 1 - paddings[1];
  int64_t pad_width_interior = strides[1] - 1;
  int64_t padded_out_width = in_x_shape[3] + ksizes[1] - 1;
  int64_t pad_width_right = padded_out_width - pad_width_left -
                            out_grad_shape[3] -
                            (out_grad_shape[3] - 1) * pad_width_interior;

  std::vector<int64_t> gradient_padding(12, 0);
  gradient_padding[2] = pad_height_top;
  gradient_padding[6] = pad_height_bottom;
  gradient_padding[10] = pad_height_interior;
  gradient_padding[3] = pad_width_left;
  gradient_padding[7] = pad_width_right;
  gradient_padding[11] = pad_width_interior;

  for (auto& padding : gradient_padding) {
    PADDLE_ENFORCE_GE(
        padding,
        0,
        phi::errors::InvalidArgument("Can not handle this paddings: %s.",
                                     VectorToString(gradient_padding).c_str()));
  }

  return gradient_padding;
}

template <typename T, typename Context>
void Pool2dGradKernel(const Context& dev_ctx,
                      const phi::DenseTensor& in_x,
                      const phi::DenseTensor& out,
                      const phi::DenseTensor& out_grad,
                      const phi::IntArray& kernel_size,
                      const std::vector<int>& strides_t,
                      const std::vector<int>& paddings_t,
                      bool ceil_mode,
                      bool exclusive,
                      const std::string& data_format,
                      const std::string& pooling_type,
                      bool global_pooling,
                      bool adaptive,
                      const std::string& padding_algorithm,
                      phi::DenseTensor* in_x_grad) {
  dev_ctx.template Alloc<T>(in_x_grad);

  if (UseScatterMemory()) {
    PADDLE_GCU_KERNEL_START(dev_ctx, "pool2d_grad", pool2d_grad);

    PADDLE_ENFORCE_EQ(
        data_format,
        "NCHW",
        phi::errors::Unimplemented("Only support NCHW for now, but got : %s",
                                   data_format.c_str()));

    std::vector<int64_t> ksizes = kernel_size.GetData();

    PADDLE_ENFORCE_EQ(
        ksizes.size(),
        2,
        phi::errors::InvalidArgument("The size of kernel_size must be 2."));

    PADDLE_ENFORCE_EQ(
        strides_t.size(),
        2,
        phi::errors::InvalidArgument("The size of strides_t must be 2."));

    PADDLE_ENFORCE_EQ(
        paddings_t.size(),
        2,
        phi::errors::InvalidArgument("The size of paddings_t must be 2."));

    std::vector<int64_t> strides = vector_s64(strides_t);
    std::vector<int64_t> in_paddings = vector_s64(paddings_t);
    if (adaptive) {
      int64_t ih = in_x.dims().at(2);
      int64_t iw = in_x.dims().at(3);
      PADDLE_ENFORCE(ih % ksizes[0] == 0 && iw % ksizes[1] == 0,
                     phi::errors::InvalidArgument(
                         "only support: MOD(oh, ih) == 0 && MOD(ow, iw) == 0"));
      ksizes[0] = ih / ksizes[0];
      ksizes[1] = iw / ksizes[1];
      strides = {ksizes[0], ksizes[1]};
      in_paddings = {0, 0};
    }

    if (pooling_type == "max") {
      std::vector<int64_t> window_dimensions{1, ksizes[0], ksizes[1], 1};
      std::vector<int64_t> window_strides{1, strides[0], strides[1], 1};
      std::vector<int64_t> paddings{0,
                                    0,
                                    in_paddings[0],
                                    in_paddings[0],
                                    in_paddings[1],
                                    in_paddings[1],
                                    0,
                                    0};

      // input
      auto input_nhwc = in_x;
      auto out_grad_nhwc = out_grad;
      if (data_format == "NCHW") {
        input_nhwc = ConvertNCHWToNHWC(dev_ctx, in_x);
        out_grad_nhwc = ConvertNCHWToNHWC(dev_ctx, out_grad);
      }

      // output
      phi::DenseTensor x_grad_nhwc = *in_x_grad;
      if (data_format == "NCHW") {
        std::vector<int64_t> out_dims = phi::vectorize(in_x_grad->dims());
        std::vector<int64_t> dst_dims = reorder_vector(out_dims, {0, 2, 3, 1});

        phi::DenseTensor dst_tensor;
        phi::DenseTensorMeta meta(in_x_grad->dtype(),
                                  phi::make_ddim(dst_dims),
                                  phi::DataLayout::NHWC);
        dst_tensor.set_meta(meta);
        dev_ctx.Alloc(&dst_tensor, dst_tensor.dtype());
        x_grad_nhwc = dst_tensor;
      }

      auto input_gcu = GetHlirTensorV2(input_nhwc, in_x.dims());
      auto out_grad_gcu = GetHlirTensorV2(out_grad_nhwc, out_grad.dims());
      auto x_grad_gcu = GetHlirTensorV2(x_grad_nhwc, in_x_grad->dims());
      hlir::DispatchParam params;
      params.inputs = {input_gcu, out_grad_gcu};
      params.outputs = {x_grad_gcu};
      params.metadata.setValue("init_value", static_cast<float>(0));
      params.metadata.setValue("select", int64_t(0));
      params.metadata.setValue("scatter", int64_t(0));
      params.metadata.setValue("window_dimensions",
                               HlirVector(window_dimensions));
      params.metadata.setValue("window_strides", HlirVector(window_strides));
      params.metadata.setValue("padding", HlirVector(paddings));
      params.stream = static_cast<topsStream_t>(dev_ctx.stream());
      AOTOPS_DEBUG(kSelectAndScatter, params);
      GCUOPS_TRACE_START(select_and_scatter);
      auto func_ptr = GetOpFuncPtr(kSelectAndScatter, params);
      if (func_ptr) {
        auto pass = hlir::HlirDispatch::dispatch(func_ptr, params);
        PADDLE_ENFORCE(pass,
                       phi::errors::InvalidArgument("dispatch %s failed!",
                                                    kSelectAndScatter));
      } else {
        PADDLE_ENFORCE(false,
                       phi::errors::InvalidArgument("not find aot func for %s",
                                                    kSelectAndScatter));
      }
      FreeDispatchParam(params);
      GCUOPS_TRACE_END(select_and_scatter);
      GcuOpStreamSync(dev_ctx);

      if (data_format == "NCHW")
        *in_x_grad = ConvertNHWCToNCHW(dev_ctx, x_grad_nhwc);
    } else if (pooling_type == "avg") {
      int64_t pooling_mode = /*TOPSOP_POOLING_AVERAGE_COUNT_INCLUDE_PADDING*/ 1;

      // calc pooling divide factor
      auto ones = ones_like(dev_ctx, in_x);
      auto pool_factor = EmptyTensor(dev_ctx, out.meta());
      pooling2d(dev_ctx,
                ones,
                ksizes,
                strides,
                in_paddings,
                data_format,
                pooling_mode,
                &pool_factor);

      // out_grad div by kernel factor
      auto div_out_grad = div_compute(dev_ctx, out_grad, pool_factor);

      // padding div_out_grad
      auto pads = gradient_padding_size(
          div_out_grad.dims(), in_x.dims(), ksizes, strides, in_paddings);
      auto padded_out_grad =
          pad(dev_ctx, div_out_grad, pads, /*constant pad*/ 0, 0.f);

      // pooling backward
      pooling2d(dev_ctx,
                padded_out_grad,
                ksizes,
                {1, 1},
                {0, 0},
                data_format,
                pooling_mode,
                in_x_grad);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented("Unsupport pooling_type: %s",
                                              pooling_type.c_str()));
    }

    PADDLE_GCU_KERNEL_END("pool2d_grad", pool2d_grad);
  } else {
    std::vector<int> ksize(kernel_size.GetData().begin(),
                           kernel_size.GetData().end());

    TensorNameMap input_names;
    input_names["X"] = {"x"};
    input_names["Y"] = {"y"};
    input_names[GradVarName("Out")] = {"out_grad"};

    TensorValueMap inputs;
    inputs["X"] = {const_cast<DenseTensor*>(&in_x)};
    inputs["Y"] = {const_cast<DenseTensor*>(&out)};
    inputs[GradVarName("Out")] = {const_cast<DenseTensor*>(&out_grad)};

    TensorNameMap output_names;
    output_names[GradVarName("X")] = {"in_x_grad"};

    TensorValueMap outputs;
    outputs[GradVarName("X")] = {in_x_grad};

    GcuAttributeMap attrs;
    attrs["adaptive"] = adaptive;
    attrs["global_pooling"] = global_pooling;
    attrs["padding_algorithm"] = padding_algorithm;
    attrs["pooling_type"] = pooling_type;
    attrs["data_format"] = data_format;
    attrs["ksize"] = ksize;
    attrs["strides"] = strides_t;
    attrs["paddings"] = paddings_t;
    attrs["ceil_mode"] = ceil_mode;
    attrs["exclusive"] = exclusive;

    GcuRunner(input_names,
              inputs,
              output_names,
              outputs,
              attrs,
              "pool2d_grad",
              dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(pool2d,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Pool2dKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(pool2d_grad,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::Pool2dGradKernel,
                          float,
                          phi::dtype::float16) {}
