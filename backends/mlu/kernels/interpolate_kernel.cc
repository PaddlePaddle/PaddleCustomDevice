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

#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

inline std::vector<int> get_new_shape_mlu(
    const phi::CustomContext& dev_ctx,
    const std::vector<const Tensor*>& list_new_shape_tensor) {
  // get tensor from
  std::vector<int> vec_new_shape;
  for (size_t i = 0; i < list_new_shape_tensor.size(); ++i) {
    auto tensor = list_new_shape_tensor[i];
    PADDLE_ENFORCE_EQ(
        tensor->dims(),
        phi::make_ddim({1}),
        phi::errors::InvalidArgument("shape of dim tensor should be [1]"));
    std::vector<int32_t> temp_vec(1);
    dev_ctx.Wait();
    TensorToVector(dev_ctx, *tensor, dev_ctx, &temp_vec);
    vec_new_shape.push_back(temp_vec[0]);
  }

  return vec_new_shape;
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
  PADDLE_ENFORCE_GE(input_dims.size(),
                    4,
                    phi::errors::External("MLU Interpolate kernel supports x "
                                          "range greater or equal than 4."));
  PADDLE_ENFORCE_LE(input_dims.size(),
                    5,
                    phi::errors::External("MLU Interpolate kernel supports x "
                                          "range less or equal than 5. "));

  phi::DataLayout data_layout = StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  ExtractNCDWH(input_dims, data_layout, &n, &c, &in_d, &in_h, &in_w);
  int align_center = align_corners ? 0 : (align_mode == 1 ? 0 : 1);

  float scale_d = -1;
  float scale_h = -1;
  float scale_w = -1;

  // Priority: SizeTensor > OutSize > Scale > scale > out_h & out_w
  if (size_tensor && size_tensor->size() > 0) {
    // have SizeTensor
    VLOG(5) << "[Interp] get out_w and out_w from SizeTensor";
    auto list_new_shape_tensor = size_tensor.get();

    if (list_new_shape_tensor.size() <= 2) {
      auto output_h =
          get_new_data_from_tensor<int>(dev_ctx, list_new_shape_tensor[0]);
      auto output_w =
          get_new_data_from_tensor<int>(dev_ctx, list_new_shape_tensor[1]);
      out_h = output_h[0];
      out_w = output_w[0];
    } else {
      auto output_d =
          get_new_data_from_tensor<int>(dev_ctx, list_new_shape_tensor[0]);
      auto output_h =
          get_new_data_from_tensor<int>(dev_ctx, list_new_shape_tensor[1]);
      auto output_w =
          get_new_data_from_tensor<int>(dev_ctx, list_new_shape_tensor[2]);
      out_h = output_h[0];
      out_w = output_w[0];
      out_d = output_d[0];
    }
  } else if (out_size) {
    VLOG(5) << "[Interp] get out_w and out_w from OutSize";
    auto out_size_data =
        get_new_data_from_tensor<int>(dev_ctx, out_size.get_ptr());
    out_h = out_size_data[0];
    out_w = out_size_data[1];
  } else {
    if (scale_tensor) {
      VLOG(5) << "[Interp] get out_w and out_w from ScaleTensor";
      std::vector<float> scale_data;
      scale_data =
          get_new_data_from_tensor<float>(dev_ctx, scale_tensor.get_ptr());
      if (scale_data.size() == 1) {
        scale_h = scale_data[0];
        scale_w = scale_data[0];
      } else if (scale_data.size() == 2) {
        scale_h = scale_data[0];
        scale_w = scale_data[1];
      } else {
        scale_d = scale_data[0];
        scale_h = scale_data[1];
        scale_w = scale_data[2];
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
      if (scale.size() > 1 && scale.size() <= 2) {
        scale_h = scale[0];
        scale_w = scale[1];

        PADDLE_ENFORCE_EQ(
            scale_w > 0 && scale_h > 0,
            true,
            phi::errors::InvalidArgument("scale  of Op(interpolate) "
                                         "should be greater than 0."));
      } else if (scale.size() > 2) {
        scale_d = scale[0];
        scale_h = scale[1];
        scale_w = scale[2];
        PADDLE_ENFORCE_EQ(
            scale_d > 0 && scale_w > 0 && scale_h > 0,
            true,
            phi::errors::InvalidArgument("scale  of Op(interpolate) "
                                         "should be greater than 0."));
      }
    }
    if (scale_h > 0. && scale_w > 0.) {
      VLOG(5) << "[Interp] get out_w and out_w from scale";
      out_h = static_cast<int>(in_h * scale_h);
      out_w = static_cast<int>(in_w * scale_w);
    }

    if (scale_d > 0.) {
      out_d = static_cast<int>(in_d * scale_d);
    }
  }

  if (out_h == 1 && out_w == 1) {
    align_center = 0;
  }

  VLOG(5) << "[Interp] n: " << n << " in_d: " << in_d << " in_h: " << in_h
          << " in_w: " << in_w << " out_d: " << out_d << " out_h: " << out_h
          << " out_w: " << out_w << " c: " << c;
  PADDLE_ENFORCE_GT(out_h,
                    0,
                    phi::errors::InvalidArgument("out_h in Attr(out_shape) of "
                                                 "Op(interpolate) "
                                                 "should be greater than 0."));
  PADDLE_ENFORCE_GT(out_w,
                    0,
                    phi::errors::InvalidArgument("out_w in Attr(out_shape) of "
                                                 "Op(interpolate) "
                                                 "should be greater than 0."));

  // do transpose according to cnnl's constraints
  // cnnlInterp_v2 only accepts NHWC when mode is CNNL_INTERP_BILINEAR and
  // CNNL_INTERP_NEAREST,
  phi::DDim dim_in, dim_in_trans, dim_out, dim_out_trans;
  Tensor transformed_input, transformed_output;
  bool need_transpose = input_dims.size() != 2;
  if (input_dims.size() == 4) {
    // need to do transpose if layout is kNCHW
    need_transpose &= data_layout == DataLayout::kNCHW;
    if (need_transpose) {
      // if need_transpose, do the following
      // 1. transpose x NCHW -> NHWC
      // 2. interpolation in(NHWC) -> out(NHWC)
      // 3. transpose out NHWC -> HCHW
      // dim_in = {n, c, in_h, in_w};
      dim_in_trans = {n, in_h, in_w, c};
      dim_out = {n, c, out_h, out_w};
      dim_out_trans = {n, out_h, out_w, c};
      out->Resize(dim_out);
      dev_ctx.template Alloc<T>(out);

      if (in_h == out_h && in_w == out_w) {
        TensorCopy(dev_ctx, x, false, out);
        return;
      }
      // do transpose on x tensor, then do interpolation
      MLUCnnlTensorDesc input_desc(
          x, CNNL_LAYOUT_NCHW, ToCnnlDataType(x.dtype()));

      transformed_input.Resize(dim_in_trans);
      dev_ctx.template Alloc<T>(&transformed_input);
      transformed_output.Resize(dim_out_trans);
      dev_ctx.template Alloc<T>(&transformed_output);

      MLUCnnlTensorDesc input_reshaped_desc(
          transformed_input,
          CNNL_LAYOUT_NHWC,
          ToCnnlDataType(transformed_input.dtype()));
      const std::vector<int> perm = {0, 2, 3, 1};
      MLUCnnl::Transpose(dev_ctx,
                         perm,
                         input_dims.size(),
                         input_desc.get(),
                         GetBasePtr(&x),
                         input_reshaped_desc.get(),
                         GetBasePtr(&transformed_input));
    } else {
      // if no need_transpose, do the following
      // 1. interpolation in(NHWC) -> out(NHWC)
      // dim_in = {n, in_h, in_w, c};
      dim_out = {n, out_h, out_w, c};
      out->Resize(dim_out);
      dev_ctx.template Alloc<T>(out);

      if (in_h == out_h && in_w == out_w) {
        TensorCopy(dev_ctx, x, false, out);
        return;
      }
      transformed_input = x;
      transformed_output = *out;
    }

    MLUCnnlTensorDesc input_desc(transformed_input,
                                 CNNL_LAYOUT_NHWC,
                                 ToCnnlDataType(transformed_input.dtype()));
    MLUCnnlTensorDesc output_desc(transformed_output,
                                  CNNL_LAYOUT_NHWC,
                                  ToCnnlDataType(transformed_output.dtype()));
    MLUCnnl::Interp(dev_ctx,
                    GetMLUCnnlInterpMode(interp_method),
                    align_corners,
                    align_center,
                    input_desc.get(),
                    GetBasePtr(&transformed_input),
                    output_desc.get(),
                    GetBasePtr(&transformed_output));

    if (need_transpose) {
      // if need_transpose, reshape out back to NCHW
      const std::vector<int> perm = {0, 3, 1, 2};
      MLUCnnlTensorDesc output_reshape_desc(
          *out, CNNL_LAYOUT_NCHW, ToCnnlDataType(out->dtype()));
      MLUCnnl::Transpose(dev_ctx,
                         perm,
                         dim_out_trans.size(),
                         output_desc.get(),
                         GetBasePtr(&transformed_output),
                         output_reshape_desc.get(),
                         GetBasePtr(out));
    }
  } else {
    PADDLE_ENFORCE_EQ(
        interp_method,
        "trilinear",
        phi::errors::External("MLU Interpolate kernel only supports 5D "
                              "data in trilinear mode."));

    // need to do transpose if layout is kNCDHW
    need_transpose &= data_layout == DataLayout::kNCHW;
    if (need_transpose) {
      // if need_transpose, do the following
      // 1. transpose x NCDHW -> NDHWC
      // 2. interpolation in(NDHWC) -> out(NDHWC)
      // 3. transpose out NDHWC -> HCDHW
      // dim_in = {n, c, in_d, in_h, in_w};
      dim_in_trans = {n, in_d, in_h, in_w, c};
      dim_out = {n, c, out_d, out_h, out_w};
      dim_out_trans = {n, out_d, out_h, out_w, c};
      out->Resize(dim_out);
      dev_ctx.template Alloc<T>(out);

      if (in_h == out_h && in_w == out_w && in_d == out_d) {
        TensorCopy(dev_ctx, x, false, out);
        return;
      }
      // do transpose on x tensor (HCDHW -> NDHWC), then do interpolation
      MLUCnnlTensorDesc input_desc(
          x, CNNL_LAYOUT_NCDHW, ToCnnlDataType(x.dtype()));

      transformed_input.Resize(dim_in_trans);
      dev_ctx.template Alloc<T>(&transformed_input);
      transformed_output.Resize(dim_out_trans);
      dev_ctx.template Alloc<T>(&transformed_output);

      MLUCnnlTensorDesc input_reshaped_desc(
          transformed_input,
          CNNL_LAYOUT_NDHWC,
          ToCnnlDataType(transformed_input.dtype()));
      const std::vector<int> perm = {0, 2, 3, 4, 1};
      MLUCnnl::Transpose(dev_ctx,
                         perm,
                         input_dims.size(),
                         input_desc.get(),
                         GetBasePtr(&x),
                         input_reshaped_desc.get(),
                         GetBasePtr(&transformed_input));
    } else {
      // if no need_transpose, do the following
      // 1. interpolation in(NDHWC) -> out(NDHWC)
      // dim_in = {n, in_d, in_h, in_w, c};
      dim_out = {n, out_d, out_h, out_w, c};
      out->Resize(dim_out);
      dev_ctx.template Alloc<T>(out);

      if (in_h == out_h && in_w == out_w && in_d == out_d) {
        TensorCopy(dev_ctx, x, false, out);
        return;
      }
      transformed_input = x;
      transformed_output = *out;
    }

    MLUCnnlTensorDesc input_desc(transformed_input,
                                 CNNL_LAYOUT_NDHWC,
                                 ToCnnlDataType(transformed_input.dtype()));
    MLUCnnlTensorDesc output_desc(transformed_output,
                                  CNNL_LAYOUT_NDHWC,
                                  ToCnnlDataType(transformed_output.dtype()));
    // use trilinear mode in HCDHW layout
    MLUCnnl::Interp(dev_ctx,
                    GetMLUCnnlInterpMode(interp_method),
                    align_corners,
                    align_center,
                    input_desc.get(),
                    GetBasePtr(&transformed_input),
                    output_desc.get(),
                    GetBasePtr(&transformed_output));

    if (need_transpose) {
      // if need_transpose, reshape out back (NDHWC -> NCDHW)
      const std::vector<int> perm = {0, 4, 1, 2, 3};
      MLUCnnlTensorDesc output_reshape_desc(
          *out, CNNL_LAYOUT_NCDHW, ToCnnlDataType(out->dtype()));
      MLUCnnl::Transpose(dev_ctx,
                         perm,
                         dim_out_trans.size(),
                         output_desc.get(),
                         GetBasePtr(&transformed_output),
                         output_reshape_desc.get(),
                         GetBasePtr(out));
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
  auto output_grad_dims = out_grad.dims();
  PADDLE_ENFORCE_EQ(
      output_grad_dims.size(),
      4,
      phi::errors::External("XPU Interpolategrad kernel only support 2d"));

  auto input_dims = x.dims();
  const DataLayout data_layout = StringToDataLayout(data_layout_str);
  int n, c, in_d, in_h, in_w;
  ExtractNCDWH(x.dims(), data_layout, &n, &c, &in_d, &in_h, &in_w);
  int align_center = align_corners ? 0 : (align_mode == 1 ? 0 : 1);

  float scale_h = -1;
  float scale_w = -1;

  if (size_tensor && size_tensor->size() > 0) {
    // have size tensor
    auto new_size = get_new_shape_mlu(dev_ctx, size_tensor.get());
    out_h = new_size[0];
    out_w = new_size[1];
  } else {
    if (scale_tensor) {
      std::vector<float> scale_data;
      scale_data =
          get_new_data_from_tensor<float>(dev_ctx, scale_tensor.get_ptr());
      if (scale_data.size() > 1) {
        scale_h = scale_data[0];
        scale_w = scale_data[1];
      } else {
        scale_h = scale_data[0];
        scale_w = scale_data[0];
      }
      PADDLE_ENFORCE_EQ(
          scale_w > 0 && scale_h > 0,
          true,
          phi::errors::InvalidArgument("scale  of Op(interpolate) "
                                       "should be greater than 0."));
    } else {
      if (scale.size() > 1) {
        scale_h = scale[0];
        scale_w = scale[1];

        PADDLE_ENFORCE_EQ(
            scale_w > 0 && scale_h > 0,
            true,
            phi::errors::InvalidArgument("scale  of Op(interpolate) "
                                         "should be greater than 0."));
      }
    }
    if (scale_h > 0. && scale_w > 0.) {
      out_h = static_cast<int>(in_h * scale_h);
      out_w = static_cast<int>(in_w * scale_w);
    }
    if (out_size) {
      std::vector<int32_t> out_size_data;
      out_size_data =
          get_new_data_from_tensor<int>(dev_ctx, out_size.get_ptr());
      out_h = out_size_data[0];
      out_w = out_size_data[1];
    }
  }

  if (out_h == 1 && out_w == 1) {
    align_center = 0;
  }

  phi::DDim dim_grad;
  phi::DDim dim_out_grad, dim_out_trans_grad, dim_in_grad, dim_in_trans_grad;
  Tensor transformed_output_grad, transformed_input_grad;
  bool need_transpose =
      input_dims.size() != 2 && data_layout == DataLayout::kNCHW;

  if (need_transpose) {
    // if need_transpose, do the following
    // 1. transpose out_grad NCHW -> NHWC
    // 2. InterpBackward out_grad(NHWC) -> dx(NHWC)
    // 3. transpose dx NHWC -> HCHW
    // dim_out_grad = {n, c, out_h, out_w};
    dim_out_trans_grad = {n, out_h, out_w, c};
    dim_in_grad = {n, c, in_h, in_w};
    dim_in_trans_grad = {n, in_h, in_w, c};
    dx->Resize(dim_in_grad);
    dev_ctx.template Alloc<T>(dx);

    if (in_h == out_h && in_w == out_w) {
      TensorCopy(dev_ctx, out_grad, false, dx);
      return;
    }
    // do transpose on x tensor, then do interpolation
    MLUCnnlTensorDesc input_desc(
        out_grad, CNNL_LAYOUT_NCHW, ToCnnlDataType(out_grad.dtype()));

    transformed_output_grad.Resize(dim_out_trans_grad);
    dev_ctx.template Alloc<T>(&transformed_output_grad);
    transformed_input_grad.Resize(dim_in_trans_grad);
    dev_ctx.template Alloc<T>(&transformed_input_grad);

    MLUCnnlTensorDesc input_reshaped_desc(
        transformed_output_grad,
        CNNL_LAYOUT_NHWC,
        ToCnnlDataType(transformed_output_grad.dtype()));
    const std::vector<int> perm = {0, 2, 3, 1};
    MLUCnnl::Transpose(dev_ctx,
                       perm,
                       input_dims.size(),
                       input_desc.get(),
                       GetBasePtr(&out_grad),
                       input_reshaped_desc.get(),
                       GetBasePtr(&transformed_output_grad));
  } else {
    // if no need_transpose, do the following
    // 1. InterpBackward out_grad(NHWC) -> dx(NHWC)
    dim_in_grad = {n, in_h, in_w, c};
    dx->Resize(dim_in_grad);
    dev_ctx.template Alloc<T>(dx);

    if (in_h == out_h && in_w == out_w) {
      TensorCopy(dev_ctx, out_grad, false, dx);
      return;
    }
    transformed_output_grad = out_grad;
    transformed_input_grad = *dx;
  }

  MLUCnnlTensorDesc input_desc(transformed_output_grad,
                               CNNL_LAYOUT_NHWC,
                               ToCnnlDataType(transformed_output_grad.dtype()));
  MLUCnnlTensorDesc output_desc(transformed_input_grad,
                                CNNL_LAYOUT_NHWC,
                                ToCnnlDataType(transformed_input_grad.dtype()));
  MLUCnnl::InterpBackward(dev_ctx,
                          GetMLUCnnlInterpBackwardMode(interp_method),
                          align_corners,
                          align_center,
                          input_desc.get(),
                          GetBasePtr(&transformed_output_grad),
                          output_desc.get(),
                          GetBasePtr(&transformed_input_grad));

  if (need_transpose) {
    const std::vector<int> perm = {0, 3, 1, 2};
    MLUCnnlTensorDesc output_reshape_desc(
        *dx, CNNL_LAYOUT_NCHW, ToCnnlDataType(dx->dtype()));
    MLUCnnl::Transpose(dev_ctx,
                       perm,
                       dim_in_trans_grad.size(),
                       output_desc.get(),
                       GetBasePtr(&transformed_input_grad),
                       output_reshape_desc.get(),
                       GetBasePtr(dx));
  }
}

template <typename T, typename Context>
void BilinearInterpKernel(
    const Context& dev_ctx,
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
    phi::DenseTensor* out) {
  InterpolateKernel<T, Context>(dev_ctx,
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
                                out);
}

template <typename T, typename Context>
void NearestInterpKernel(
    const Context& dev_ctx,
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
    phi::DenseTensor* out) {
  InterpolateKernel<T, Context>(dev_ctx,
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
                                out);
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
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::NearestInterpKernel,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(nearest_interp_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::NearestInterpGradKernel,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(bilinear_interp,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::BilinearInterpKernel,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}

PD_REGISTER_PLUGIN_KERNEL(bilinear_interp_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::BilinearInterpGradKernel,
                          float,
                          phi::dtype::float16) {
  kernel->InputAt(1).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(2).SetBackend(phi::Backend::ALL_BACKEND);
  kernel->InputAt(3).SetBackend(phi::Backend::ALL_BACKEND);
}
