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

#include <iostream>

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/extension.h"

namespace custom_kernel {

namespace funcs {

inline void ExtractNCDWH(const phi::DDim& dims,
                         const DataLayout& data_layout,
                         int* N,
                         int* C,
                         int* D,
                         int* H,
                         int* W) {
  *N = dims[0];

  if (dims.size() == 3) {
    *C = data_layout == DataLayout::kNCHW ? dims[1] : dims[2];
    *D = 1;
    *H = 1;
    *W = data_layout == DataLayout::kNCHW ? dims[2] : dims[1];
  } else if (dims.size() == 4) {
    *C = data_layout == DataLayout::kNCHW ? dims[1] : dims[3];
    *D = 1;
    *H = data_layout == DataLayout::kNCHW ? dims[2] : dims[1];
    *W = data_layout == DataLayout::kNCHW ? dims[3] : dims[2];
  } else {
    *C = data_layout == DataLayout::kNCHW ? dims[1] : dims[4];
    *D = data_layout == DataLayout::kNCHW ? dims[2] : dims[1];
    *H = data_layout == DataLayout::kNCHW ? dims[3] : dims[2];
    *W = data_layout == DataLayout::kNCHW ? dims[4] : dims[3];
  }
}

inline std::vector<int> GetNewShape(
    const Context& dev_ctx,
    const std::vector<const phi::DenseTensor*>& list_new_shape_tensor) {
  // get tensor from
  std::vector<int> vec_new_shape;
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    const auto& src_place = tensor->place();
    PADDLE_ENFORCE_EQ(tensor->dims(),
                      phi::make_ddim({1}),
                      phi::errors::InvalidArgument(
                          "The shape of dimension tensor should be [1],"
                          "but received d%.",
                          tensor->dims()));
    if (src_place.GetType() == phi::AllocationType::CUSTOM) {
      phi::DenseTensor temp;
      TensorCopy(dev_ctx, *tensor, true, &temp, phi::CPUPlace());
      vec_new_shape.push_back(static_cast<int32_t>(*temp.data<int32_t>()));
    } else {
      vec_new_shape.push_back(static_cast<int32_t>(*tensor->data<int32_t>()));
    }
  }

  return std::move(vec_new_shape);
}

template <typename T>
inline std::vector<T> GetNewDataFromTensor(
    const Context& dev_ctx, const phi::DenseTensor* new_data_tensor) {
  std::vector<T> vec_new_data;
  auto* new_data = new_data_tensor->data<T>();
  phi::DenseTensor cpu_starts_tensor;
  const auto& src_place = new_data_tensor->place();
  if (src_place.GetType() == phi::AllocationType::CUSTOM) {
    TensorCopy(
        dev_ctx, *new_data_tensor, true, &cpu_starts_tensor, phi::CPUPlace());
    new_data = cpu_starts_tensor.data<T>();
  }
  vec_new_data = std::vector<T>(new_data, new_data + new_data_tensor->numel());

  return std::move(vec_new_data);
}

}  // namespace funcs

template <typename T, typename Context>
void NearestInterpKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& input,
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
  VLOG(4) << "Call SDAA NearestInterpKernel";

  auto* input_data = input.data<T>();

  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  funcs::ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  // Priority: SizeTensor > OutSize > Scale > scale > out_h & out_w
  float scale_w = -1;
  float scale_h = -1;
  if (size_tensor && size_tensor->size() > 0) {
    // have size tensor
    auto new_size = funcs::GetNewShape(dev_ctx, size_tensor.get());
    out_h = new_size[0];
    out_w = new_size[1];
  } else if (out_size) {
    auto out_size_data =
        funcs::GetNewDataFromTensor<int>(dev_ctx, out_size.get_ptr());
    out_h = out_size_data[0];
    out_w = out_size_data[1];
  } else {
    if (scale_tensor) {
      auto scale_data =
          funcs::GetNewDataFromTensor<float>(dev_ctx, scale_tensor.get_ptr());
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
        scale_w = scale[1];
        scale_h = scale[0];

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
  if (data_layout == DataLayout::kNCHW) {
    dim_out = {n, c, out_h, out_w};
  } else {
    dim_out = {n, out_h, out_w, c};
  }
  output->Resize(dim_out);

  if (in_h == out_h && in_w == out_w) {
    // input and output are the same -> copy
    // TensorCopy() will alloc,
    // so don't use dev_ctx.templat Alloc<T> to alloc memory
    TensorCopy(dev_ctx, input, true, output);
    return;
  }

  dev_ctx.template Alloc<T>(output);

  float ratio_h = static_cast<float>(out_h) / in_h;
  float ratio_w = static_cast<float>(out_w) / in_w;

  if (data_layout == DataLayout::kNCHW) {
    // transform data layout
    phi::DDim in_x_NHWC_dims =
        sdaa_ops::doDimPermute(input, Convert_TF::NCHW2NHWC);
    phi::DDim out_NHWC_dims =
        sdaa_ops::doDimPermute(*output, Convert_TF::NCHW2NHWC);

    phi::DenseTensor in_x_NHWC, out_NHWC;
    phi::DenseTensorMeta in_x_NHWC_meta = {input.dtype(), in_x_NHWC_dims};
    phi::DenseTensorMeta out_NHWC_meta = {output->dtype(), out_NHWC_dims};
    in_x_NHWC.set_meta(in_x_NHWC_meta);
    out_NHWC.set_meta(out_NHWC_meta);

    dev_ctx.template Alloc<T>(&in_x_NHWC);
    dev_ctx.template Alloc<T>(&out_NHWC);

    // NCHW -> NHWC
    sdaa_ops::doTransformTensor(
        dev_ctx, input, Convert_TF::NCHW2NHWC, &in_x_NHWC);
    sdaa_ops::doTransformTensor(
        dev_ctx, *output, Convert_TF::NCHW2NHWC, &out_NHWC);

    sdaa_ops::doNearestInterpolateForward(
        dev_ctx, in_x_NHWC, ratio_w, ratio_h, 0.f, align_corners, &out_NHWC);

    // NHWC -> NCHW
    sdaa_ops::doTransformTensor(
        dev_ctx, out_NHWC, Convert_TF::NHWC2NCHW, output);
  } else {
    // NHWC
    sdaa_ops::doNearestInterpolateForward(
        dev_ctx, input, ratio_w, ratio_h, 0.f, align_corners, output);
  }
}

template <typename T, typename Context>
void NearestInterpGradKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& input,
    const paddle::optional<phi::DenseTensor>& out_size,
    const paddle::optional<std::vector<const phi::DenseTensor*>>& size_tensor,
    const paddle::optional<phi::DenseTensor>& scale_tensor,
    const phi::DenseTensor& output_grad,
    const std::string& data_layout_str,
    int out_d,
    int out_h,
    int out_w,
    const std::vector<float>& scale,
    const std::string& interp_method,
    bool align_corners,
    int align_mode,
    phi::DenseTensor* input_grad) {
  VLOG(4) << "Call SDAA NearestInterpGradKernel";

  const DataLayout data_layout = common::StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  funcs::ExtractNCDWH(input.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);

  float scale_h = -1;
  float scale_w = -1;
  // Priority: SizeTensor > OutSize > Scale > scale > out_h & out_w
  if (size_tensor && size_tensor->size() > 0) {
    // have size tensor
    auto new_size = funcs::GetNewShape(dev_ctx, size_tensor.get());
    out_h = new_size[0];
    out_w = new_size[1];
  } else if (out_size) {
    auto out_size_data =
        funcs::GetNewDataFromTensor<int>(dev_ctx, out_size.get_ptr());
    out_h = out_size_data[0];
    out_w = out_size_data[1];
  } else {
    if (scale_tensor) {
      auto scale_data =
          funcs::GetNewDataFromTensor<float>(dev_ctx, scale_tensor.get_ptr());
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
        scale_w = scale[1];
        scale_h = scale[0];

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
      if (scale_w > 0. && scale_h > 0.) {
        out_h = static_cast<int>(in_h * scale_h);
        out_w = static_cast<int>(in_w * scale_w);
      }
    }
  }

  auto* output_grad_data = output_grad.data<T>();
  phi::DDim dim_grad;
  if (data_layout == DataLayout::kNCHW) {
    dim_grad = {n, c, in_h, in_w};
  } else {
    dim_grad = {n, in_h, in_w, c};
  }
  input_grad->Resize(dim_grad);

  if (in_h == out_h && in_w == out_w) {
    // input and output are the same -> copy
    // TensorCopy() will alloc,
    // so don't use dev_ctx.templat Alloc<T> to alloc memory
    TensorCopy(dev_ctx, output_grad, true, input_grad);
    return;
  }

  dev_ctx.template Alloc<T>(input_grad);

  float ratio_h = static_cast<float>(out_h) / in_h;
  float ratio_w = static_cast<float>(out_w) / in_w;
  if (data_layout == DataLayout::kNCHW) {
    // transform data layout
    phi::DDim in_x_NHWC_dims =
        sdaa_ops::doDimPermute(output_grad, Convert_TF::NCHW2NHWC);
    phi::DDim out_NHWC_dims =
        sdaa_ops::doDimPermute(*input_grad, Convert_TF::NCHW2NHWC);

    phi::DenseTensor in_x_NHWC, out_NHWC;
    phi::DenseTensorMeta in_x_NHWC_meta = {output_grad.dtype(), in_x_NHWC_dims};
    phi::DenseTensorMeta out_NHWC_meta = {input_grad->dtype(), out_NHWC_dims};
    in_x_NHWC.set_meta(in_x_NHWC_meta);
    out_NHWC.set_meta(out_NHWC_meta);

    dev_ctx.template Alloc<T>(&in_x_NHWC);
    dev_ctx.template Alloc<T>(&out_NHWC);

    // NCHW -> NHWC
    sdaa_ops::doTransformTensor(
        dev_ctx, output_grad, Convert_TF::NCHW2NHWC, &in_x_NHWC);
    sdaa_ops::doTransformTensor(
        dev_ctx, *input_grad, Convert_TF::NCHW2NHWC, &out_NHWC);

    sdaa_ops::doNearestInterpolateBackward(
        dev_ctx, in_x_NHWC, ratio_w, ratio_h, 0.f, align_corners, &out_NHWC);

    // NHWC -> NCHW
    sdaa_ops::doTransformTensor(
        dev_ctx, out_NHWC, Convert_TF::NHWC2NCHW, input_grad);
  } else {
    // NHWC
    sdaa_ops::doNearestInterpolateBackward(
        dev_ctx, output_grad, ratio_w, ratio_h, 0.f, align_corners, input_grad);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(nearest_interp,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::NearestInterpKernel,
                          float) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(nearest_interp_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::NearestInterpGradKernel,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
