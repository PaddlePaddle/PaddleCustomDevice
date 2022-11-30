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
#include "kernels/funcs/op_command.h"

namespace custom_kernel {

template <typename T = int>
inline void UpdatePadding(std::vector<T>* paddings,
                          const bool global_pooling,
                          const bool adaptive,
                          const std::string padding_algorithm,
                          const phi::DDim data_dims,
                          const std::vector<T>& strides,
                          const std::vector<T>& kernel_size) {
  // set padding size == data_dims.size() * 2
  auto data_shape = phi::vectorize<T>(data_dims);
  if (static_cast<int>(paddings->size()) == data_dims.size()) {
    for (int i = 0; i < data_dims.size(); ++i) {
      T copy_pad = *(paddings->begin() + 2 * i);
      paddings->insert(paddings->begin() + 2 * i + 1, copy_pad);
    }
  } else {
    PADDLE_ENFORCE_EQ(data_dims.size() * 2,
                      paddings->size(),
                      phi::errors::InvalidArgument(
                          "Paddings size %d should be the same or twice as the "
                          "pooling size %d.",
                          paddings->size(),
                          data_dims.size() * 2));
  }

  // when padding_algorithm is "VALID" or "SAME"
  if (padding_algorithm == "SAME") {
    for (int i = 0; i < data_dims.size(); ++i) {
      T out_size = (data_dims[i] + strides[i] - 1) / strides[i];
      T pad_sum =
          std::max((out_size - 1) * strides[i] + kernel_size[i] - data_shape[i],
                   static_cast<T>(0));
      T pad_0 = pad_sum / 2;
      T pad_1 = pad_sum - pad_0;
      *(paddings->begin() + i * 2) = pad_0;
      *(paddings->begin() + i * 2 + 1) = pad_1;
    }
  } else if (padding_algorithm == "VALID") {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }

  // if global_pooling == true or adaptive == true, padding will be ignore
  if (global_pooling || adaptive) {
    for (auto it = paddings->begin(); it != paddings->end(); it++) {
      *it = 0;
    }
  }
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
  PADDLE_ENFORCE_EQ(exclusive,
                    true,
                    phi::errors::InvalidArgument(
                        "Pool only support exclusive=true, but got false."));
  PADDLE_ENFORCE_EQ(
      adaptive && data_format == "NHWC",
      false,
      phi::errors::InvalidArgument("AdaptivePool only support channel first."));
  dev_ctx.template Alloc<T>(out);

  std::vector<int> ksize(kernel_size.GetData().begin(),
                         kernel_size.GetData().end());
  auto strides = strides_t;
  auto paddings = paddings_t;
  const bool channel_last = data_format == "NHWC";

  auto in_x_dims = in_x.dims();
  auto out_dims = out->dims();
  phi::DDim data_dims;
  phi::DDim out_data_dims;

  std::vector<int> ksize_vec(4, 1);
  std::vector<int> strides_vec(4, 1);

  if (channel_last) {
    data_dims = phi::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
    out_data_dims = phi::slice_ddim(out_dims, 1, out_dims.size() - 1);
    ksize_vec[1] = ksize[0];
    ksize_vec[2] = ksize[1];
    strides_vec[1] = strides[0];
    strides_vec[2] = strides[1];
  } else {
    data_dims = phi::slice_ddim(in_x_dims, 2, in_x_dims.size());
    out_data_dims = phi::slice_ddim(out_dims, 2, out_dims.size());
    ksize_vec[2] = ksize[0];
    ksize_vec[3] = ksize[1];
    strides_vec[2] = strides[0];
    strides_vec[3] = strides[1];
  }
  UpdatePadding(&paddings,
                global_pooling,
                adaptive,
                padding_algorithm,
                data_dims,
                strides,
                ksize);
  PADDLE_ENFORCE_LT(
      std::max(paddings[0], paddings[1]),
      ksize[0],
      phi::errors::InvalidArgument(
          "Paddings should be less than %d, but max(pads[0], pads[1]) is %d.",
          ksize[0],
          std::max(paddings[0], paddings[1])));
  PADDLE_ENFORCE_LT(
      std::max(paddings[2], paddings[3]),
      ksize[1],
      phi::errors::InvalidArgument(
          "Paddings should be less than %d, but max(pads[2], pads[3]) is %d.",
          ksize[1],
          std::max(paddings[2], paddings[3])));

  if (adaptive) {
    std::string pooling_mode = "AdaptiveAvgPool2d";
    if (pooling_type == "max") {
      pooling_mode = "AdaptiveMaxPool2d";
    }
    experimental::OpCommand(pooling_mode)
        .Input(
            in_x,
            experimental::TensorDescMaker("x").FromTensor(in_x).SetDataLayout(
                channel_last ? phi::DataLayout::NHWC : phi::DataLayout::NCHW))
        .Output(
            *out,
            experimental::TensorDescMaker("y").FromTensor(*out).SetDataLayout(
                channel_last ? phi::DataLayout::NHWC : phi::DataLayout::NCHW))
        .Attr("output_size", phi::vectorize<int>(out_data_dims))
        .Run(dev_ctx);
  } else {
    std::string pooling_mode = "AvgPoolV2";
    if (pooling_type == "max") {
      pooling_mode = "MaxPoolV3";
    }
    experimental::OpCommand(pooling_mode)
        .Input(
            in_x,
            experimental::TensorDescMaker("x").FromTensor(in_x).SetDataLayout(
                channel_last ? phi::DataLayout::NHWC : phi::DataLayout::NCHW))
        .Output(
            *out,
            experimental::TensorDescMaker("y").FromTensor(*out).SetDataLayout(
                channel_last ? phi::DataLayout::NHWC : phi::DataLayout::NCHW))
        .Attr("ksize", ksize_vec)
        .Attr("strides", strides_vec)
        .Attr("padding_mode", std::string("CALCULATED"))
        .Attr("pads", paddings)
        .Attr("data_format", data_format)
        .Attr("global_pooling", global_pooling)
        .Attr("ceil_mode", ceil_mode)
        .Attr("exclusive", exclusive)
        .Run(dev_ctx);
  }
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

  std::vector<int> ksize(kernel_size.GetData().begin(),
                         kernel_size.GetData().end());
  auto strides = strides_t;
  auto paddings = paddings_t;

  const bool channel_last = data_format == "NHWC";

  // update paddings
  auto in_x_dims = in_x.dims();
  auto out_dims = out.dims();
  phi::DDim data_dims;
  phi::DDim out_data_dims;
  std::vector<int> ksize_vec(4, 1);
  std::vector<int> strides_vec(4, 1);

  if (channel_last) {
    data_dims = phi::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
    out_data_dims = phi::slice_ddim(out_dims, 1, out_dims.size() - 1);
    ksize_vec[1] = ksize[0];
    ksize_vec[2] = ksize[1];
    strides_vec[1] = strides[0];
    strides_vec[2] = strides[1];
  } else {
    data_dims = phi::slice_ddim(in_x_dims, 2, in_x_dims.size());
    out_data_dims = phi::slice_ddim(out_dims, 2, out_dims.size());
    ksize_vec[2] = ksize[0];
    ksize_vec[3] = ksize[1];
    strides_vec[2] = strides[0];
    strides_vec[3] = strides[1];
  }
  UpdatePadding(&paddings,
                global_pooling,
                adaptive,
                padding_algorithm,
                data_dims,
                strides,
                ksize);

  PADDLE_ENFORCE_LT(
      std::max(paddings[0], paddings[1]),
      ksize[0],
      phi::errors::InvalidArgument(
          "Paddings should be less than %d, but max(pads[0], pads[1]) is %d.",
          ksize[0],
          std::max(paddings[0], paddings[1])));
  PADDLE_ENFORCE_LT(
      std::max(paddings[2], paddings[3]),
      ksize[1],
      phi::errors::InvalidArgument(
          "Paddings should be less than %d, but max(pads[2], pads[3]) is %d.",
          ksize[1],
          std::max(paddings[2], paddings[3])));

  if (adaptive || (global_pooling && pooling_type == "max")) {
    PADDLE_ENFORCE_EQ(data_dims[0] % out_data_dims[0],
                      0,
                      phi::errors::InvalidArgument(
                          "When adaptive = True, H and W must be divisible, "
                          "but input dims is %s, output dims is %s",
                          data_dims,
                          out_data_dims));
    PADDLE_ENFORCE_EQ(data_dims[1] % out_data_dims[1],
                      0,
                      phi::errors::InvalidArgument(
                          "When adaptive = True, H and W must be divisible, "
                          "but input dims is %s, output dims is %s",
                          data_dims,
                          out_data_dims));
    if (channel_last) {
      strides_vec[1] = data_dims[0] / out_data_dims[0];
      strides_vec[2] = data_dims[1] / out_data_dims[1];
      ksize_vec[1] = strides_vec[1];
      ksize_vec[2] = strides_vec[2];
    } else {
      strides_vec[2] = data_dims[0] / out_data_dims[0];
      strides_vec[3] = data_dims[1] / out_data_dims[1];
      ksize_vec[2] = strides_vec[2];
      ksize_vec[3] = strides_vec[3];
    }
  }

  if (pooling_type == "max") {
    PADDLE_ENFORCE(
        !global_pooling,
        phi::errors::Unavailable("Computing gradients of global pooling is not "
                                 "supported, which means ksize < x1"));

    experimental::OpCommand("MaxPoolV3Grad")
        .Input(in_x,
               experimental::TensorDescMaker("orig_input")
                   .FromTensor(in_x)
                   .SetDataLayout(channel_last ? phi::DataLayout::NHWC
                                               : phi::DataLayout::NCHW))
        .Input(out,
               experimental::TensorDescMaker("orig_output")
                   .FromTensor(out)
                   .SetDataLayout(channel_last ? phi::DataLayout::NHWC
                                               : phi::DataLayout::NCHW))
        .Input(out_grad,
               experimental::TensorDescMaker("grad")
                   .FromTensor(out_grad)
                   .SetDataLayout(channel_last ? phi::DataLayout::NHWC
                                               : phi::DataLayout::NCHW))
        .Output(*in_x_grad,
                experimental::TensorDescMaker("out_grad")
                    .FromTensor(*in_x_grad)
                    .SetDataLayout(channel_last ? phi::DataLayout::NHWC
                                                : phi::DataLayout::NCHW))
        .Attr("ksize", ksize_vec)
        .Attr("strides", strides_vec)
        .Attr("padding_mode", std::string("CALCULATED"))
        .Attr("pads", paddings)
        .Attr("data_format", data_format)
        .Attr("global_pooling", global_pooling)
        .Attr("ceil_mode", ceil_mode)  // 0: floor, 1: ceil
        .Attr("exclusive", exclusive)
        .Run(dev_ctx);
  } else if (pooling_type == "avg") {
    PADDLE_ENFORCE(strides[0] == strides[1],
                   phi::errors::InvalidArgument(
                       "AvgPoolGrad dose not support Asymmetric strides. but "
                       "strides = (%d, %d)",
                       strides[0],
                       strides[1]));
    phi::DenseTensor in_x_dims;
    TensorFromVector(
        dev_ctx, phi::vectorize(in_x.dims()), phi::CPUContext(), &in_x_dims);

    experimental::OpCommand("AvgPoolV2Grad")
        .Input(in_x_dims,
               experimental::TensorDescMaker("orig_input_shape")
                   .FromTensor(in_x_dims)
                   .SetDataLayout(phi::DataLayout::NCHW))
        .Input(out_grad,
               experimental::TensorDescMaker("input_grad")
                   .FromTensor(out_grad)
                   .SetDataLayout(channel_last ? phi::DataLayout::NHWC
                                               : phi::DataLayout::NCHW))
        .Output(*in_x_grad,
                experimental::TensorDescMaker("out_grad")
                    .FromTensor(*in_x_grad)
                    .SetDataLayout(channel_last ? phi::DataLayout::NHWC
                                                : phi::DataLayout::NCHW))
        .Attr("ksize", ksize_vec)
        .Attr("strides", strides_vec)
        .Attr("padding_mode", std::string("CALCULATED"))
        .Attr("pads", paddings)
        .Attr("data_format", data_format)
        .Attr("global_pooling", global_pooling)
        .Attr("ceil_mode", ceil_mode)  // 0: floor, 1: ceil
        .Attr("exclusive", exclusive)
        .Run(dev_ctx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(pool2d,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::Pool2dKernel,
                          float,
                          phi::dtype::float16) {}
PD_REGISTER_PLUGIN_KERNEL(pool2d_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::Pool2dGradKernel,
                          float,
                          phi::dtype::float16) {}
