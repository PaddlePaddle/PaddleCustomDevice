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

template <typename T, typename Context>
void NearestInterpKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& out_size,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& size_tensor,
    const paddle::optional<phi::DenseTensor>& scale_tensor,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    phi::DenseTensor* out) {
  auto stream = dev_ctx.stream();

  auto input = x;
  auto input_dims = input.dims();
  PADDLE_ENFORCE_EQ(
      input_dims.size(),
      4UL,
      phi::errors::External("NPU Interpolate Kernel only support 4-D Tensor."));

  phi::DataLayout data_layout = StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  ExtractNCDWH(input_dims, data_layout, &n, &c, &in_d, &in_h, &in_w);

  // To-do(qili93): need to support align_corners = true case, try ReSizeD
  PADDLE_ENFORCE_EQ(
      align_corners,
      false,
      phi::errors::InvalidArgument(
          "NPU Interpolate Kernel has diff when align_corners is true."));

  float scale_h = -1;
  float scale_w = -1;

  // Priority: SizeTensor > OutSize > Scale > scale > out_h & out_w
  if (size_tensor && size_tensor->size() > 0) {
    auto list_new_shape_tensor = size_tensor.get();
    std::vector<int32_t> output_h(1);
    std::vector<int32_t> output_w(1);
    TensorToVector(dev_ctx, *(list_new_shape_tensor[0]), dev_ctx, &output_h);
    TensorToVector(dev_ctx, *(list_new_shape_tensor[1]), dev_ctx, &output_w);
    out_h = output_h[0];
    out_w = output_w[0];
  } else if (out_size) {
    auto out_size_data =
        get_new_data_from_tensor<int>(dev_ctx, out_size.get_ptr());
    out_h = out_size_data[0];
    out_w = out_size_data[1];
  } else {
    if (scale_tensor) {
      auto scale_data =
          get_new_data_from_tensor<float>(dev_ctx, scale_tensor.get_ptr());
      if (scale_data.size() > 1) {
        scale_h = scale_data[0];
        scale_w = scale_data[1];
      } else {
        scale_h = scale_data[0];
        scale_w = scale_data[0];
      }
      PADDLE_ENFORCE_EQ(
          scale_w > 0,
          true,
          phi::errors::InvalidArgument(
              "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      PADDLE_ENFORCE_EQ(
          scale_h > 0,
          true,
          phi::errors::InvalidArgument(
              "The scale_h in input 'Scale' Tensor of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_h));
    } else {
      if (scale.size() > 1) {
        scale_h = scale[0];
        scale_w = scale[1];

        PADDLE_ENFORCE_EQ(
            scale_w > 0,
            true,
            phi::errors::InvalidArgument(
                "The scale_w in Attr(scale) of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_w));
        PADDLE_ENFORCE_EQ(
            scale_h > 0,
            true,
            phi::errors::InvalidArgument(
                "The scale_h in Attr(scale) of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_h));
      }
    }
    if (scale_h > 0. && scale_w > 0.) {
      out_h = static_cast<int>(in_h * scale_h);
      out_w = static_cast<int>(in_w * scale_w);
    }
  }
  PADDLE_ENFORCE_GT(out_h,
                    0,
                    phi::errors::InvalidArgument(
                        "out_h in Attr(out_shape) of Op(interpolate) "
                        "should be greater than 0."));
  PADDLE_ENFORCE_GT(out_w,
                    0,
                    phi::errors::InvalidArgument(
                        "out_w in Attr(out_shape) of Op(interpolate) "
                        "should be greater than 0."));
  phi::DDim dim_out;
  if (data_layout == phi::DataLayout::kNCHW) {
    dim_out = {n, c, out_h, out_w};
  } else {
    dim_out = {n, out_h, out_w, c};
  }

  phi::DenseTensorMeta out_meta = {out->dtype(), dim_out};
  out->set_meta(out_meta);
  dev_ctx.template Alloc<T>(out);

  if (in_h == out_h && in_w == out_w) {
    TensorCopy(dev_ctx, input, false, out);
    return;
  }

  // To-do(qili93): need to support bilineare, try ResizeD
  // Add bilineare by zhulei
  if ("nearest" == interp_method) {
    NpuOpRunner runner;
    runner.SetType("ResizeNearestNeighborV2")
        .AddInput(input)
        .AddInput(dev_ctx, std::vector<int32_t>{out_h, out_w})
        .AddOutput(*out)
        .AddAttr("align_corners", align_corners)
        .AddAttr("half_pixel_centers", false);
    runner.Run(stream);
  } else if ("bilinear" == interp_method) {
    PADDLE_THROW(
        phi::errors::Unimplemented(" %s is not supported.", interp_method));
  }
}

template <typename T, typename Context>
void NearestInterpGradKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& out_size,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& size_tensor,
    const paddle::optional<phi::DenseTensor>& scale_tensor,
    const phi::DenseTensor& out_grad,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    phi::DenseTensor* dx) {
  auto input = x;
  auto input_grad = dx;
  auto output_grad = out_grad;

  auto stream = dev_ctx.stream();

  phi::DataLayout data_layout = StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  // To-do(qili93): need to support align_corners = true case, try ReSizeD
  PADDLE_ENFORCE_EQ(
      align_corners,
      false,
      phi::errors::InvalidArgument(
          "NPU Interpolate Kernel has diff when align_corners is true."));

  float scale_h = -1;
  float scale_w = -1;

  // Priority: SizeTensor > OutSize > Scale > scale > out_h & out_w
  if (size_tensor && size_tensor->size() > 0) {
    auto list_new_size_tensor = size_tensor.get();
    std::vector<int32_t> output_h(1);
    std::vector<int32_t> output_w(1);
    TensorToVector(dev_ctx, *(list_new_size_tensor[0]), dev_ctx, &output_h);
    TensorToVector(dev_ctx, *(list_new_size_tensor[1]), dev_ctx, &output_w);
    out_h = output_h[0];
    out_w = output_w[0];
  } else if (out_size) {
    auto out_size_data =
        get_new_data_from_tensor<int>(dev_ctx, out_size.get_ptr());
    out_h = out_size_data[0];
    out_w = out_size_data[1];
  } else {
    if (scale_tensor) {
      auto scale_data =
          get_new_data_from_tensor<float>(dev_ctx, scale_tensor.get_ptr());
      if (scale_data.size() > 1) {
        scale_h = scale_data[0];
        scale_w = scale_data[1];
      } else {
        scale_w = scale_data[0];
        scale_h = scale_data[0];
      }
      PADDLE_ENFORCE_EQ(
          scale_w > 0,
          true,
          phi::errors::InvalidArgument(
              "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
      PADDLE_ENFORCE_EQ(
          scale_h > 0,
          true,
          phi::errors::InvalidArgument(
              "The scale_h in input 'Scale' Tensor of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_h));
    } else {
      if (scale.size() > 1) {
        scale_h = scale[0];
        scale_w = scale[1];
        PADDLE_ENFORCE_EQ(
            scale_w > 0,
            true,
            phi::errors::InvalidArgument(
                "The scale_w in Attr(scale) of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_w));
        PADDLE_ENFORCE_EQ(
            scale_h > 0,
            true,
            phi::errors::InvalidArgument(
                "The scale_h in Attr(scale) of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_h));
      }
    }
    if (scale_h > 0. && scale_w > 0.) {
      out_h = static_cast<int>(in_h * scale_h);
      out_w = static_cast<int>(in_w * scale_w);
    }
  }

  phi::DDim dim_grad;
  if (data_layout == phi::DataLayout::kNCHW) {
    dim_grad = {n, c, in_h, in_w};
  } else {
    dim_grad = {n, in_h, in_w, c};
  }

  phi::DenseTensorMeta input_grad_meta = {input.dtype(), dim_grad};
  input_grad->set_meta(input_grad_meta);
  dev_ctx.template Alloc<T>(input_grad);

  if (in_h == out_h && in_w == out_w) {
    TensorCopy(dev_ctx, output_grad, false, input_grad);
    return;
  }

  // To-do(qili93): need to support bilineare, try ResizeGradD
  if ("nearest" == interp_method) {
    NpuOpRunner runner;
    runner.SetType("ResizeNearestNeighborV2Grad")
        .AddInput(output_grad)
        .AddInput(dev_ctx, std::vector<int32_t>{in_h, in_w})
        .AddOutput(*input_grad)
        .AddAttr("align_corners", align_corners)
        .AddAttr("half_pixel_centers", false);
    runner.Run(stream);
  } else if ("bilinear" == interp_method) {
    PADDLE_THROW(
        phi::errors::Unimplemented(" %s is not supported.", interp_method));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(nearest_interp_v2,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::NearestInterpKernel,
                          float,
                          phi::dtype::float16,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(nearest_interp_v2_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::NearestInterpGradKernel,
                          float,
                          phi::dtype::float16,
                          double) {}
