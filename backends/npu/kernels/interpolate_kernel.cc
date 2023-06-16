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

#include "kernels/interpolate_kernel.h"

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "kernels/funcs/slice_utils.h"
namespace custom_kernel {

template <typename T, typename Context>
static void Interpolate1DKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& out_size,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& size_tensor,
    const paddle::optional<phi::DenseTensor>& scale_tensor,
    const std::string& data_layout_str,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    phi::DenseTensor* output) {
  auto input = x;
  auto stream = dev_ctx.stream();

  phi::DataLayout data_layout = StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  float scale_w = -1;
  // Priority: SizeTensor > OutSize > Scale > scale > out_h & out_w
  if (size_tensor && size_tensor->size() > 0) {
    auto new_size = get_new_shape(dev_ctx, size_tensor.get());
    out_w = new_size[0];
  } else {
    if (scale_tensor) {
      auto scale_data =
          get_new_data_from_tensor<float>(dev_ctx, scale_tensor.get_ptr());
      scale_w = scale_data[0];
      PADDLE_ENFORCE_EQ(
          scale_w > 0,
          true,
          phi::errors::InvalidArgument(
              "The scale_w in input 'Scale' Tensor of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_w));
    } else {
      if (scale.size() > 0) {
        scale_w = scale[0];

        PADDLE_ENFORCE_EQ(
            scale_w > 0,
            true,
            phi::errors::InvalidArgument(
                "The scale_w in Attr(scale) of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_w));
      }
    }
    if (scale_w > 0.) {
      out_w = static_cast<int>(in_w * scale_w);
    }
    if (out_size) {
      auto out_size_data =
          get_new_data_from_tensor<int>(dev_ctx, out_size.get_ptr());
      out_w = out_size_data[0];
    }
  }
  PADDLE_ENFORCE_GT(out_w,
                    0,
                    phi::errors::InvalidArgument(
                        "out_w in Attr(out_shape) of Op(interpolate) "
                        "should be greater than 0."));
  phi::DDim dim_out;
  if (data_layout == phi::DataLayout::kNCHW) {
    dim_out = {n, c, out_w};
  } else {
    dim_out = {n, out_w, c};
  }
  output->Resize(dim_out);
  dev_ctx.template Alloc<T>(output);

  if (in_w == out_w) {
    TensorCopy(dev_ctx, x, false, output);
    return;
  }
  float ratio_w = 0.f;
  if (out_w > 1) {
    float new_scale_w = 0.f;
    new_scale_w = (scale_w > 0) ? static_cast<float>(1. / scale_w)
                                : static_cast<float>(in_w) / out_w;
    ratio_w = (align_corners) ? static_cast<float>(in_w - 1) / (out_w - 1)
                              : static_cast<float>(new_scale_w);
  }
  if ("linear" == interp_method) {
    std::vector<float> scales = {scale_w};
    std::vector<int> output_size;
    if (data_layout == phi::DataLayout::kNCHW) {
      output_size = {n, c, 1, out_w};
      input.Resize({n, c, 1, in_w});
    } else {
      output_size = {n, 1, out_w, c};
      input.Resize({n, 1, in_w, c});
    }
    output->Resize(phi::make_ddim(output_size));

    std::string coordinate_transformation_mode =
        align_corners ? "align_corners" : "half_pixel";
    NpuOpRunner linear_runner;
    linear_runner.SetType("ResizeD")
        .AddInput(input)
        .AddAttr("sizes", output_size)
        .AddAttr("coordinate_transformation_mode",
                 coordinate_transformation_mode)
        .AddAttr("mode", interp_method)
        .AddAttr("scales", scales)
        .AddOutput(*output)
        .Run(dev_ctx.stream());
    output->Resize(dim_out);
  }
}

template <typename T, typename Context>
static void Interpolate2DKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& out_size,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& size_tensor,
    const paddle::optional<phi::DenseTensor>& scale_tensor,
    const std::string& data_layout_str,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    phi::DenseTensor* output) {
  auto stream = dev_ctx.stream();

  phi::DataLayout data_layout = StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  ExtractNCDWH(x.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  float scale_w = -1;
  float scale_h = -1;

  // Priority: SizeTensor > OutSize > Scale > scale > out_h & out_w
  if (size_tensor && size_tensor->size() > 0) {
    auto new_size = get_new_shape(dev_ctx, size_tensor.get());
    out_h = new_size[0];
    out_w = new_size[1];
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
      if (scale.size() > 0) {
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
    if (scale_w > 0. && scale_h > 0.) {
      out_h = static_cast<int>(in_h * scale_h);
      out_w = static_cast<int>(in_w * scale_w);
    }
    if (out_size) {
      auto out_size_data =
          get_new_data_from_tensor<int>(dev_ctx, out_size.get_ptr());
      out_h = out_size_data[0];
      out_w = out_size_data[1];
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
  output->Resize(dim_out);
  dev_ctx.template Alloc<T>(output);

  if (in_h == out_h && in_w == out_w) {
    TensorCopy(dev_ctx, x, false, output);
    return;
  }
  if ("nearest" == interp_method) {
    NpuOpRunner runner;
    runner.SetType("ResizeNearestNeighborV2")
        .AddInput(x)
        .AddInput(dev_ctx, std::vector<int32_t>{out_h, out_w})
        .AddOutput(*output)
        .AddAttr("align_corners", align_corners)
        .AddAttr("half_pixel_centers", false);
    runner.Run(stream);
  } else if ("bilinear" == interp_method) {
    if (align_corners == true) {
      // ResizeBilinearV2 only support fp32 input.
      if (x.dtype() == phi::DataType::FLOAT16) {
        phi::DenseTensor tmp_x, tmp_out;
        tmp_x.Resize(x.dims());
        dev_ctx.template Alloc<float>(&tmp_x);
        tmp_out.Resize(output->dims());
        dev_ctx.template Alloc<float>(&tmp_out);
        const auto& cast_runner1 =
            NpuOpRunner("Cast", {x}, {tmp_x}, {{"dst_type", ACL_FLOAT}});
        cast_runner1.Run(stream);

        NpuOpRunner runner;
        runner.SetType("ResizeBilinearV2")
            .AddInput(tmp_x)
            .AddInput(dev_ctx, std::vector<int32_t>{out_h, out_w})
            .AddOutput(tmp_out)
            .AddAttr("align_corners", align_corners)
            .AddAttr("half_pixel_centers", !align_corners);
        runner.Run(stream);

        const auto& cast_runner2 = NpuOpRunner(
            "Cast", {tmp_out}, {*output}, {{"dst_type", ACL_FLOAT16}});
        cast_runner2.Run(stream);
      } else {
        NpuOpRunner runner;
        runner.SetType("ResizeBilinearV2")
            .AddInput(x)
            .AddInput(dev_ctx, std::vector<int32_t>{out_h, out_w})
            .AddOutput(*output)
            .AddAttr("align_corners", align_corners)
            .AddAttr("half_pixel_centers", !align_corners);
        runner.Run(stream);
      }
    } else {
      BilinearFwdNpu<T, Context>(dev_ctx,
                                 &x,
                                 output,
                                 scale_h,
                                 scale_w,
                                 align_corners,
                                 align_mode,
                                 data_layout);
    }
  } else if ("bicubic" == interp_method) {
    std::string coordinate_transformation_mode = "half_pixel";
    if (align_corners == true) {
      coordinate_transformation_mode = "align_corners";
    }
    NpuOpRunner runner;
    runner.SetType("ResizeD")
        .AddInput(x)
        .AddOutput(*output)
        .AddAttr("sizes", std::vector<int32_t>{n, c, out_h, out_w})
        .AddAttr("scales", std::vector<float>{scale_h, scale_w})
        .AddAttr("roi", std::vector<float>{})
        .AddAttr("coordinate_transformation_mode",
                 coordinate_transformation_mode)
        .AddAttr("cubic_coeff_a", static_cast<float>(-0.75))
        .AddAttr("exclude_outside", static_cast<int64_t>(0))
        .AddAttr("extrapolation_value", static_cast<float>(0.0))
        .AddAttr("mode", (std::string) "cubic")
        .AddAttr("nearest_mode", (std::string) "round_prefer_floor");
    runner.Run(stream);
  }
}

template <typename T, typename Context>
static void Interpolate3DKernel(
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
    phi::DenseTensor* output) {
  auto stream = dev_ctx.stream();

  phi::DataLayout data_layout = StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  ExtractNCDWH(x.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  float scale_d = -1;
  float scale_w = -1;
  float scale_h = -1;

  // Priority: SizeTensor > OutSize > Scale > scale > out_h & out_w
  if (size_tensor && size_tensor->size() > 0) {
    auto new_size = get_new_shape(dev_ctx, size_tensor.get());
    out_d = new_size[0];
    out_h = new_size[1];
    out_w = new_size[2];
  } else {
    if (scale_tensor) {
      auto scale_data =
          get_new_data_from_tensor<float>(dev_ctx, scale_tensor.get_ptr());
      if (scale_data.size() > 1) {
        scale_d = scale_data[0];
        scale_h = scale_data[1];
        scale_w = scale_data[2];
      } else {
        scale_d = scale_data[0];
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
      PADDLE_ENFORCE_EQ(
          scale_d > 0,
          true,
          phi::errors::InvalidArgument(
              "The scale_d in input 'Scale' Tensor of Operator(interpolate) "
              "should be greater than 0, but received value is %d.",
              scale_d));
    } else {
      if (scale.size() > 0) {
        scale_d = scale[0];
        scale_h = scale[1];
        scale_w = scale[2];

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
        PADDLE_ENFORCE_EQ(
            scale_d > 0,
            true,
            phi::errors::InvalidArgument(
                "The scale_d in Attr(scale) of Operator(interpolate) "
                "should be greater than 0, but received value is %d.",
                scale_d));
      }
    }
    if (scale_w > 0. && scale_h > 0. && scale_d > 0.) {
      out_d = static_cast<int>(in_d * scale_d);
      out_h = static_cast<int>(in_h * scale_h);
      out_w = static_cast<int>(in_w * scale_w);
    }
    if (out_size) {
      auto out_size_data =
          get_new_data_from_tensor<int>(dev_ctx, out_size.get_ptr());
      out_d = out_size_data[0];
      out_h = out_size_data[1];
      out_w = out_size_data[2];
    }
  }
  PADDLE_ENFORCE_GT(out_d,
                    0,
                    phi::errors::InvalidArgument(
                        "out_d in Attr(out_shape) of Op(interpolate) "
                        "should be greater than 0."));
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
    dim_out = {n, c, out_d, out_h, out_w};
  } else {
    dim_out = {n, out_d, out_h, out_w, c};
  }
  output->Resize(dim_out);
  dev_ctx.template Alloc<T>(output);

  if (in_d == out_d && in_h == out_h && in_w == out_w) {
    TensorCopy(dev_ctx, x, false, output);
    return;
  }
  if ("trilinear" == interp_method) {
    NpuOpRunner runner;
    runner.SetType("UpsampleTrilinear3d")
        .AddInput(x)
        .AddOutput(*output)
        .AddAttr("align_corners", align_corners)
        .AddAttr("output_size", phi::vectorize(dim_out));
    runner.Run(stream);
  } else if ("nearest" == interp_method) {
    if (align_corners == false) {
      std::vector<int32_t> output_size = {out_d, out_h, out_w};
      NpuOpRunner runner;
      runner.SetType("UpsampleNearest3d")
          .AddInput(x)
          .AddOutput(*output)
          .AddAttr("output_size", output_size);
      runner.Run(stream);
    } else {
      PADDLE_THROW(phi::errors::Unimplemented(
          "Nearest 3D interpolate with align_corners == true is not "
          "supported."));
    }
  }
}
template <typename T, typename Context>
void InterpolateKernel(
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
  auto input_dims = x.dims();
  if (input_dims.size() == 3) {  // 1D interpolation
    Interpolate1DKernel<T, Context>(dev_ctx,
                                    x,
                                    out_size,
                                    size_tensor,
                                    scale_tensor,
                                    data_layout_str,
                                    out_w,
                                    scale,
                                    interp_method,
                                    align_corners,
                                    align_mode,
                                    out);
  } else if (input_dims.size() == 4) {  // 2D interpolation
    Interpolate2DKernel<T, Context>(dev_ctx,
                                    x,
                                    out_size,
                                    size_tensor,
                                    scale_tensor,
                                    data_layout_str,
                                    out_h,
                                    out_w,
                                    scale,
                                    interp_method,
                                    align_corners,
                                    align_mode,
                                    out);
  } else if (input_dims.size() == 5) {  // 3D interpolation
    Interpolate3DKernel<T, Context>(dev_ctx,
                                    x,
                                    out_size,
                                    size_tensor,
                                    scale_tensor,
                                    data_layout_str,
                                    out_d,
                                    out_h,
                                    out_w,
                                    scale,
                                    interp_method,
                                    align_corners,
                                    align_mode,
                                    out);
  }
}

template <typename T, typename Context>
void BilinearInterpKernel(
    const Context& ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& out_size,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& size_tensor,
    const paddle::optional<phi::DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    phi::DenseTensor* output) {
  InterpolateKernel<T, Context>(ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale,
                                interp_method,
                                align_corners,
                                align_mode,
                                output);
}

template <typename T, typename Context>
void NearestInterpKernel(
    const Context& ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& out_size,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& size_tensor,
    const paddle::optional<phi::DenseTensor>& scale_tensor,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    phi::DenseTensor* output) {
  InterpolateKernel<T, Context>(ctx,
                                x,
                                out_size,
                                size_tensor,
                                scale_tensor,
                                data_layout,
                                out_d,
                                out_h,
                                out_w,
                                scale,
                                interp_method,
                                align_corners,
                                align_mode,
                                output);
}

template <typename T, typename Context>
static void Interpolate2DBwdKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& out_size,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& size_tensor,
    const paddle::optional<phi::DenseTensor>& scale_tensor,
    const phi::DenseTensor& out_grad,
    const std::string& data_layout_str,
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

  float scale_h = -1;
  float scale_w = -1;

  // Priority: SizeTensor > OutSize > Scale > scale > out_h & out_w
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
  if (out_size) {
    auto out_size_data =
        get_new_data_from_tensor<int>(dev_ctx, out_size.get_ptr());
    out_h = out_size_data[0];
    out_w = out_size_data[1];
  }
  if (size_tensor && size_tensor->size() > 0) {
    // have size tensor
    auto new_size = get_new_shape(dev_ctx, size_tensor.get());
    out_h = new_size[0];
    out_w = new_size[1];
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
    if (align_corners == true) {
      // ResizeBilinearV2 only support fp32 input.
      if (x.dtype() == phi::DataType::FLOAT16) {
        phi::DenseTensor tmp_x, tmp_out;
        tmp_x.Resize(output_grad.dims());
        dev_ctx.template Alloc<float>(&tmp_x);
        tmp_out.Resize(input_grad->dims());
        dev_ctx.template Alloc<float>(&tmp_out);
        const auto& cast_runner1 = NpuOpRunner(
            "Cast", {output_grad}, {tmp_x}, {{"dst_type", ACL_FLOAT}});
        cast_runner1.Run(stream);

        NpuOpRunner runner;
        runner.SetType("ResizeBilinearV2Grad")
            .AddInput(tmp_x)
            .AddInput(dev_ctx, std::vector<int32_t>{in_h, in_w})
            .AddOutput(tmp_out)
            .AddAttr("align_corners", align_corners)
            .AddAttr("half_pixel_centers", !align_corners);
        runner.Run(stream);

        const auto& cast_runner2 = NpuOpRunner(
            "Cast", {tmp_out}, {*input_grad}, {{"dst_type", ACL_FLOAT16}});
        cast_runner2.Run(stream);
      } else {
        NpuOpRunner runner;
        runner.SetType("ResizeBilinearV2Grad")
            .AddInput(output_grad)
            .AddInput(dev_ctx, std::vector<int32_t>{in_h, in_w})
            .AddOutput(*input_grad)
            .AddAttr("align_corners", align_corners)
            .AddAttr("half_pixel_centers", !align_corners);
        runner.Run(stream);
      }
    } else {
      BilinearBwdNpu<T, Context>(dev_ctx,
                                 &output_grad,
                                 input_grad,
                                 scale_h,
                                 scale_w,
                                 align_corners,
                                 align_mode,
                                 data_layout);
    }
  }
}

template <typename T, typename Context>
void InterpolateGradKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& out_size,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& size_tensor,
    const paddle::optional<phi::DenseTensor>& scale_tensor,
    const phi::DenseTensor& output_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    phi::DenseTensor* x_grad) {
  auto output_grad_dims = output_grad.dims();
  if (output_grad_dims.size() == 3) {  // 1D interpolation grad
    // Interpolate1DBwdKernel<T, Context>(dev_ctx,
    //                                 x,
    //                                 out_size,
    //                                 size_tensor,
    //                                 scale_tensor,
    //                                 output_grad,
    //                                 data_layout,
    //                                 out_w,
    //                                 scale,
    //                                 interp_method,
    //                                 align_corners,
    //                                 align_mode,
    //                                 x_grad);
  } else if (output_grad_dims.size() == 4) {  // 2D interpolation grad
    Interpolate2DBwdKernel<T, Context>(dev_ctx,
                                       x,
                                       out_size,
                                       size_tensor,
                                       scale_tensor,
                                       output_grad,
                                       data_layout,
                                       out_h,
                                       out_w,
                                       scale,
                                       interp_method,
                                       align_corners,
                                       align_mode,
                                       x_grad);

  } else if (output_grad_dims.size() == 5) {  // 3D interpolation grad
    // Interpolate3DBwdKernel<T, Context>(dev_ctx,
    //                                 x,
    //                                 out_size,
    //                                 size_tensor,
    //                                 scale_tensor,
    //                                 output_grad,
    //                                 data_layout,
    //                                 out_d,
    //                                 out_h,
    //                                 out_w,
    //                                 scale,
    //                                 interp_method,
    //                                 align_corners,
    //                                 align_mode,
    //                                 x_grad);
  }
}

template <typename T, typename Context>
void BilinearInterpGradKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& out_size,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& size_tensor,
    const paddle::optional<phi::DenseTensor>& scale_tensor,
    const phi::DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    phi::DenseTensor* x_grad) {
  InterpolateGradKernel<T, Context>(dev_ctx,
                                    x,
                                    out_size,
                                    size_tensor,
                                    scale_tensor,
                                    out_grad,
                                    data_layout,
                                    out_d,
                                    out_h,
                                    out_w,
                                    scale,
                                    interp_method,
                                    align_corners,
                                    align_mode,
                                    x_grad);
}

template <typename T, typename Context>
void NearestInterpGradKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const paddle::optional<phi::DenseTensor>& out_size,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& size_tensor,
    const paddle::optional<phi::DenseTensor>& scale_tensor,
    const phi::DenseTensor& out_grad,
    const std::string& data_layout,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    phi::DenseTensor* x_grad) {
  InterpolateGradKernel<T, Context>(dev_ctx,
                                    x,
                                    out_size,
                                    size_tensor,
                                    scale_tensor,
                                    out_grad,
                                    data_layout,
                                    out_d,
                                    out_h,
                                    out_w,
                                    scale,
                                    interp_method,
                                    align_corners,
                                    align_mode,
                                    x_grad);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(nearest_interp,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::NearestInterpKernel,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(nearest_interp_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::NearestInterpGradKernel,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(bilinear_interp,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::BilinearInterpKernel,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(bilinear_interp_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::BilinearInterpGradKernel,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
