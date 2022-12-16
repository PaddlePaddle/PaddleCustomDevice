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

  phi::DenseTensor in_x_tensor(in_x), out_tensor(*out);
  std::vector<int> ksize_vec(4, 1);
  std::vector<int> strides_vec(4, 1);

  if (channel_last) {
    data_dims = phi::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
    out_data_dims = phi::slice_ddim(out_dims, 1, out_dims.size() - 1);
    ksize_vec[1] = ksize[0];
    ksize_vec[2] = ksize[1];
    strides_vec[1] = strides[0];
    strides_vec[2] = strides[1];
    phi::DenseTensorMeta in_x_meta = {
        in_x_tensor.dtype(), in_x_tensor.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta out_meta = {
        out_tensor.dtype(), out_tensor.dims(), phi::DataLayout::kNHWC};
    in_x_tensor.set_meta(in_x_meta);
    out_tensor.set_meta(out_meta);
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

    // AdaptiveAvgPool2d only support NCHW
    phi::DenseTensor transformed_input, transformed_output;
    if (pooling_type == "avg" && channel_last) {
      transformed_input.Resize(phi::make_dim(
          in_x_dims[0], in_x_dims[3], in_x_dims[1], in_x_dims[2]));
      dev_ctx.template Alloc<T>(&transformed_input);
      transformed_output.Resize(
          phi::make_dim(out_dims[0], out_dims[3], out_dims[1], out_dims[2]));
      dev_ctx.template Alloc<T>(&transformed_output);

      const auto& trans_runner =
          NpuOpRunner("TransData",
                      {in_x_tensor},
                      {transformed_input},
                      {{"src_format", std::string("NHWC")},
                       {"dst_format", std::string("NCHW")}});
      trans_runner.Run(dev_ctx.stream());
    } else {
      transformed_input = in_x_tensor;
      transformed_output = out_tensor;
    }

    const auto& runner =
        NpuOpRunner(pooling_mode,
                    {transformed_input},
                    {transformed_output},
                    {{"output_size", phi::vectorize<int>(out_data_dims)}});
    runner.Run(dev_ctx.stream());

    if (pooling_type == "avg" && channel_last) {
      const auto& trans_runner =
          NpuOpRunner("TransData",
                      {transformed_output},
                      {out_tensor},
                      {{"src_format", std::string("NCHW")},
                       {"dst_format", std::string("NHWC")}});
      trans_runner.Run(dev_ctx.stream());
    }
  } else {
    std::string pooling_mode = "AvgPoolV2";
    if (pooling_type == "max") {
      PADDLE_ENFORCE_EQ(
          exclusive,
          true,
          phi::errors::InvalidArgument(
              "MaxPool only support exclusive=false, but got true"));
      pooling_mode = "MaxPoolV3";
    }

    if (strides_vec[2] >= 64) {
      NPUAttributeMap attrs = {{"ksize", ksize_vec},
                               {"strides", strides_vec},
                               {"padding_mode", std::string("CALCULATED")},
                               {"pads", paddings},
                               {"data_format", data_format},
                               {"global_pooling", global_pooling},
                               {"ceil_mode", ceil_mode},
                               {"exclusive", exclusive}};
      phi::DenseTensor out_tensor_tmp;
      out_tensor_tmp.Resize(phi::make_dim(
          in_x_dims[0], in_x_dims[1], in_x_dims[2] / 2, in_x_dims[3] / 2));
      dev_ctx.template Alloc<T>(&out_tensor_tmp);
      if (channel_last) {
        attrs["strides"] = std::vector<int32_t>({1, 2, 2, 1});
        strides_vec[1] /= 2;
        strides_vec[2] /= 2;
      } else {
        attrs["strides"] = std::vector<int32_t>({1, 1, 2, 2});
        strides_vec[2] /= 2;
        strides_vec[3] /= 2;
      }
      attrs["ksize"] = std::vector<int32_t>({1, 1, 1, 1});
      NpuOpRunner runner0;
      runner0.SetType(pooling_mode);
      runner0.AddInput(in_x_tensor);
      runner0.AddOutput(out_tensor_tmp);
      runner0.AddAttrs(attrs);
      runner0.Run(dev_ctx.stream());
      // twice
      attrs["ksize"] = ksize_vec;
      attrs["strides"] = strides_vec;
      NpuOpRunner runner1;
      runner1.SetType(pooling_mode);
      runner1.AddInput(out_tensor_tmp);
      runner1.AddOutput(out_tensor);
      runner1.AddAttrs(attrs);
      runner1.Run(dev_ctx.stream());
      return;
    }

    const auto& runner =
        NpuOpRunner(pooling_mode,
                    {in_x_tensor},
                    {out_tensor},
                    {{"ksize", ksize_vec},
                     {"strides", strides_vec},
                     {"padding_mode", std::string("CALCULATED")},
                     {"pads", paddings},
                     {"data_format", data_format},
                     {"global_pooling", global_pooling},
                     {"ceil_mode", ceil_mode},
                     {"exclusive", exclusive}});
    runner.Run(dev_ctx.stream());
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

  phi::DenseTensor in_x_tensor(in_x), out_tensor(out),
      out_grad_tensor(out_grad), in_x_grad_tensor(*in_x_grad);
  if (channel_last) {
    data_dims = phi::slice_ddim(in_x_dims, 1, in_x_dims.size() - 1);
    out_data_dims = phi::slice_ddim(out_dims, 1, out_dims.size() - 1);
    ksize_vec[1] = ksize[0];
    ksize_vec[2] = ksize[1];
    strides_vec[1] = strides[0];
    strides_vec[2] = strides[1];
    phi::DenseTensorMeta in_x_meta = {
        in_x_tensor.dtype(), in_x_tensor.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta out_meta = {
        out_tensor.dtype(), out_tensor.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta out_grad_meta = {out_grad_tensor.dtype(),
                                          out_grad_tensor.dims(),
                                          phi::DataLayout::kNHWC};
    phi::DenseTensorMeta in_x_grad_meta = {in_x_grad_tensor.dtype(),
                                           in_x_grad_tensor.dims(),
                                           phi::DataLayout::kNHWC};
    in_x_tensor.set_meta(in_x_meta);
    out_tensor.set_meta(out_meta);
    out_grad_tensor.set_meta(out_grad_meta);
    in_x_grad_tensor.set_meta(in_x_grad_meta);
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
    // PADDLE_ENFORCE_EQ(data_dims[0] % out_data_dims[0],
    //                   0,
    //                   phi::errors::InvalidArgument(
    //                       "When adaptive = True, H and W must be divisible, "
    //                       "but input dims is %s, output dims is %s",
    //                       data_dims,
    //                       out_data_dims));
    // PADDLE_ENFORCE_EQ(data_dims[1] % out_data_dims[1],
    //                   0,
    //                   phi::errors::InvalidArgument(
    //                       "When adaptive = True, H and W must be divisible, "
    //                       "but input dims is %s, output dims is %s",
    //                       data_dims,
    //                       out_data_dims));
    // if (channel_last) {
    //   strides_vec[1] = data_dims[0] / out_data_dims[0];
    //   strides_vec[2] = data_dims[1] / out_data_dims[1];
    //   ksize_vec[1] = strides_vec[1];
    //   ksize_vec[2] = strides_vec[2];
    // } else {
    //   strides_vec[2] = data_dims[0] / out_data_dims[0];
    //   strides_vec[3] = data_dims[1] / out_data_dims[1];
    //   ksize_vec[2] = strides_vec[2];
    //   ksize_vec[3] = strides_vec[3];
    // }
    if (channel_last) {
      strides_vec[1] = std::floor(data_dims[0] / out_data_dims[0]);
      strides_vec[2] = std::floor(data_dims[1] / out_data_dims[1]);
      ksize_vec[1] = data_dims[0] - ((out_data_dims[0] - 1) * strides_vec[1]);
      ksize_vec[2] = data_dims[1] - ((out_data_dims[1] - 1) * strides_vec[2]);
    } else {
      strides_vec[2] = std::floor(data_dims[0] / out_data_dims[0]);
      strides_vec[3] = std::floor(data_dims[1] / out_data_dims[1]);
      ksize_vec[2] = data_dims[0] - ((out_data_dims[0] - 1) * strides_vec[2]);
      ksize_vec[3] = data_dims[1] - ((out_data_dims[1] - 1) * strides_vec[3]);
    }
    for (auto& pad : paddings) {
      pad = 0;
    }
  }
  for (auto strides : strides_vec) {
    std::cout << "[DEBUG] strides value: " << strides << std::endl;
  }
  for (auto ksize : ksize_vec) {
    std::cout << "[DEBUG] ksize value: " << ksize << std::endl;
  }
  for (auto out_data_dim : phi::vectorize<int>(out_data_dims)) {
    std::cout << "[DEBUG] out_data_dims value: " << out_data_dim << std::endl;
  }
  NPUAttributeMap attrs = {{"ksize", ksize_vec},
                           {"strides", strides_vec},
                           {"padding_mode", std::string("CALCULATED")},
                           {"pads", paddings},
                           {"data_format", data_format},
                           {"global_pooling", global_pooling},
                           {"ceil_mode", ceil_mode},
                           {"exclusive", exclusive}};

  if (pooling_type == "max") {
    if (global_pooling) {
      for (auto& s : strides_vec) {
        s = 1;
      }
      PADDLE_ENFORCE_LT(std::max(data_dims[0], data_dims[1]),
                        255,
                        phi::errors::InvalidArgument(
                            "MaxPoolGrad H, W must be less than 255 when "
                            "global_pooling = True, but got %s",
                            data_dims));
      attrs["global_pooling"] = false;
    }

    const auto& runner = NpuOpRunner("MaxPoolV3Grad",
                                     {in_x_tensor, out_tensor, out_grad_tensor},
                                     {in_x_grad_tensor},
                                     attrs);  // 0: floor, 1: ceil
    runner.Run(dev_ctx.stream());
  } else if (pooling_type == "avg") {
    PADDLE_ENFORCE(strides[0] == strides[1],
                   phi::errors::InvalidArgument(
                       "AvgPoolGrad dose not support Asymmetric strides. but "
                       "strides = (%d, %d)",
                       strides[0],
                       strides[1]));
    if (strides_vec[2] >= 64) {
      phi::DenseTensor in_x_grad_tensor_tmp;
      auto in_x_grad_tensor_dims = in_x_grad_tensor.dims();
      std::cout << "[DEBUG] in_x_grad_tensor_dims value: "
                << in_x_grad_tensor_dims.to_str() << std::endl;
      std::cout << "[DEBUG] out_grad_tensor_dims value: "
                << out_grad_tensor.dims().to_str() << std::endl;
      std::cout << "[DEBUG] in_x_dims value: " << in_x_dims.to_str()
                << std::endl;
      in_x_grad_tensor_tmp.Resize(phi::make_dim(in_x_grad_tensor_dims[0],
                                                in_x_grad_tensor_dims[1],
                                                in_x_grad_tensor_dims[2] / 2,
                                                in_x_grad_tensor_dims[3] / 2));
      dev_ctx.template Alloc<T>(&in_x_grad_tensor_tmp);
      std::cout << "[DEBUG] 222222222222 " << std::endl;
      if (channel_last) {
        strides_vec[1] /= 4;
        strides_vec[2] /= 4;
      } else {
        strides_vec[2] /= 4;
        strides_vec[3] /= 4;
      }
      attrs["strides"] = strides_vec;
      // attrs["strides"] = std::vector<int32_t>({1,1,2,2});
      // attrs["ksize"] = std::vector<int32_t>({1,1,1,1});
      std::cout << "[DEBUG] 33333333333 " << std::endl;
      NpuOpRunner runner0;
      runner0.SetType("AvgPoolV2Grad");
      runner0.AddInput(dev_ctx,
                       phi::vectorize<int>(in_x_grad_tensor_tmp.dims()));
      runner0.AddInput(out_grad_tensor);
      runner0.AddOutput(in_x_grad_tensor_tmp);
      runner0.AddAttrs(attrs);
      runner0.Run(dev_ctx.stream());
      std::cout << "[DEBUG] 4444444444 " << std::endl;
      // twice
      for (auto strides : strides_vec) {
        std::cout << "[DEBUG] strides value: " << strides << std::endl;
      }
      for (auto ksize : ksize_vec) {
        std::cout << "[DEBUG] ksize value: " << ksize << std::endl;
      }
      if (channel_last) {
        attrs["strides"] = std::vector<int32_t>({1, 2, 2, 1});
      } else {
        attrs["strides"] = std::vector<int32_t>({1, 1, 2, 2});
      }
      attrs["ksize"] = std::vector<int32_t>({1, 1, 1, 1});
      NpuOpRunner runner1;
      runner1.SetType("AvgPoolV2Grad");
      runner1.AddInput(dev_ctx, phi::vectorize<int>(in_x_dims));
      runner1.AddInput(in_x_grad_tensor_tmp);
      runner1.AddOutput(in_x_grad_tensor);
      runner1.AddAttrs(attrs);
      runner1.Run(dev_ctx.stream());
      return;
    }

    NpuOpRunner runner;
    runner.SetType("AvgPoolV2Grad");
    runner.AddInput(dev_ctx, phi::vectorize<int>(in_x.dims()));
    runner.AddInput(out_grad_tensor);
    runner.AddOutput(in_x_grad_tensor);
    runner.AddAttrs(attrs);
    runner.Run(dev_ctx.stream());
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
