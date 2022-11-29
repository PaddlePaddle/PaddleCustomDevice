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

template <typename T, typename Context>
void BatchNormKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& running_mean,
                     const phi::DenseTensor& running_var,
                     const phi::DenseTensor& scale,
                     const phi::DenseTensor& bias,
                     bool is_test,
                     float momentum,
                     float epsilon,
                     const std::string& data_layout_str,
                     bool use_global_stats,
                     bool trainable_stats,
                     phi::DenseTensor* y,
                     phi::DenseTensor* mean_out,
                     phi::DenseTensor* variance_out,
                     phi::DenseTensor* saved_mean,
                     phi::DenseTensor* saved_variance,
                     phi::DenseTensor* reserve_space) {
  bool test_mode = is_test && (!trainable_stats);
  bool training = !test_mode && !use_global_stats;

  phi::DataLayout data_layout = phi::StringToDataLayout(data_layout_str);

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());

  PADDLE_ENFORCE_EQ(
      x_dims.size() >= 2 && x_dims.size() <= 5,
      true,
      phi::errors::InvalidArgument(
          "The size of input's dimensions should be between 2 and 5"
          "But received: the size of input's dimensions is [%d]",
          x_dims.size()));

  if (x_dims.size() == 2 && data_layout == phi::DataLayout::kNHWC) {
    data_layout = phi::DataLayout::kNCHW;
  } else if (x_dims.size() ==
             3) {  // transform 3d tensor to 4d tensor to satisfy the format
    if (data_layout == phi::DataLayout::kNCHW) {
      x_dims.push_back(1);  // expand NCL -> NCL1
    } else {
      x_dims.insert(x_dims.begin() + 2, 1);  // expand NLC -> NL1C
    }
  }

  dev_ctx.template Alloc<T>(y);

  if (!training) {
    experimental::OpCommand("BNInfer")
        .Input(x,
               experimental::TensorDescMaker("x", x)
                   .SetDataLayout(data_layout)
                   .SetDims(phi::make_ddim(x_dims)))
        .Input(scale,
               experimental::TensorDescMaker("scale", scale)
                   .SetDataLayout(phi::DataLayout::ANY))
        .Input(bias,
               experimental::TensorDescMaker("offset", bias)
                   .SetDataLayout(phi::DataLayout::ANY))
        .Input(running_mean,
               experimental::TensorDescMaker("mean", running_mean)
                   .SetDataLayout(phi::DataLayout::ANY))
        .Input(running_var,
               experimental::TensorDescMaker("variance", running_var)
                   .SetDataLayout(phi::DataLayout::ANY))
        .Output(*y,
                experimental::TensorDescMaker("y", *y)
                    .SetDataLayout(data_layout)
                    .SetDims(phi::make_ddim(x_dims)))
        .Attr("epsilon", epsilon)
        .Run(dev_ctx);
  } else {
    dev_ctx.template Alloc<float>(mean_out);
    dev_ctx.template Alloc<float>(variance_out);
    dev_ctx.template Alloc<float>(saved_mean);
    dev_ctx.template Alloc<float>(saved_variance);

    phi::DenseTensor sum, square_sum;
    sum.Resize(running_mean.dims());
    square_sum.Resize(running_mean.dims());
    dev_ctx.template Alloc<float>(&sum);
    dev_ctx.template Alloc<float>(&square_sum);

    experimental::OpCommand("BNTrainingReduce")
        .Input(x,
               experimental::TensorDescMaker("x", x)
                   .SetDataLayout(data_layout)
                   .SetDims(phi::make_ddim(x_dims)))
        .Output(sum,
                experimental::TensorDescMaker("sum", sum)
                    .SetDataLayout(phi::DataLayout::ANY))
        .Output(square_sum,
                experimental::TensorDescMaker("square_sum", square_sum)
                    .SetDataLayout(phi::DataLayout::ANY))
        .Run(dev_ctx);

    phi::DenseTensor tmp_mean, tmp_variance;
    tmp_mean.Resize(mean_out->dims());
    dev_ctx.template Alloc<T>(&tmp_mean);
    tmp_variance.Resize(variance_out->dims());
    dev_ctx.template Alloc<T>(&tmp_variance);
    // NOTE(wangran16): ref input must be variable
    experimental::OpCommandHelper::MarkAsParameter(mean_out);
    experimental::OpCommandHelper::MarkAsParameter(variance_out);

    experimental::OpCommand("BNTrainingUpdate")
        .Input(x,
               experimental::TensorDescMaker("x", x)
                   .SetDataLayout(data_layout)
                   .SetDims(phi::make_ddim(x_dims)))
        .Input(sum,
               experimental::TensorDescMaker("sum", sum)
                   .SetDataLayout(phi::DataLayout::ANY))
        .Input(square_sum,
               experimental::TensorDescMaker("square_sum", square_sum)
                   .SetDataLayout(phi::DataLayout::ANY))
        .Input(scale,
               experimental::TensorDescMaker("scale", scale)
                   .SetDataLayout(phi::DataLayout::ANY))
        .Input(bias,
               experimental::TensorDescMaker("offset", bias)
                   .SetDataLayout(phi::DataLayout::ANY))
        .Input(running_mean,
               experimental::TensorDescMaker("mean", running_mean)
                   .SetDataLayout(phi::DataLayout::ANY))
        .Input(running_var,
               experimental::TensorDescMaker("variance", running_var)
                   .SetDataLayout(phi::DataLayout::ANY))
        .Output(
            *y,
            experimental::TensorDescMaker("y", *y).SetDataLayout(data_layout))
        .Output(tmp_mean,
                experimental::TensorDescMaker("mean", tmp_mean)
                    .SetDataLayout(phi::DataLayout::ANY))
        .Output(tmp_variance,
                experimental::TensorDescMaker("variance", tmp_variance)
                    .SetDataLayout(phi::DataLayout::ANY))
        .Output(*saved_mean,
                experimental::TensorDescMaker("batch_mean", *saved_mean)
                    .SetDataLayout(phi::DataLayout::ANY))
        .Output(*saved_variance,
                experimental::TensorDescMaker("batch_variance", *saved_variance)
                    .SetDataLayout(phi::DataLayout::ANY))
        .Attr("epsilon", epsilon)
        .Attr("factor", momentum)
        .Run(dev_ctx);
  }
}

template <typename T, typename Context>
void BatchNormGradKernel(
    const Context& dev_ctx,
    const phi::DenseTensor& x,
    const phi::DenseTensor& scale,
    const phi::DenseTensor& bias,
    const paddle::optional<phi::DenseTensor>& mean,
    const paddle::optional<phi::DenseTensor>& variance,
    const phi::DenseTensor& saved_mean,
    const phi::DenseTensor& saved_inv_variance,
    const paddle::optional<phi::DenseTensor>& reserve_space,
    const phi::DenseTensor& d_y,
    float momentum,
    float epsilon,
    const std::string& data_layout_str,
    bool is_test,
    bool use_global_stats,
    bool trainable_statistics,
    phi::DenseTensor* d_x,
    phi::DenseTensor* d_scale,
    phi::DenseTensor* d_bias) {
  phi::DataLayout data_layout = phi::StringToDataLayout(data_layout_str);

  use_global_stats = is_test || use_global_stats;

  phi::DenseTensor scale_grad_tmp, bias_grad_tmp;
  scale_grad_tmp.Resize(scale.dims());
  bias_grad_tmp.Resize(bias.dims());
  dev_ctx.template Alloc<float>(&scale_grad_tmp);
  dev_ctx.template Alloc<float>(&bias_grad_tmp);

  if (d_scale == nullptr) {
    d_scale = &scale_grad_tmp;
  }
  if (d_bias == nullptr) {
    d_bias = &bias_grad_tmp;
  }

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());

  PADDLE_ENFORCE_EQ(
      x_dims.size() >= 2 && x_dims.size() <= 5,
      true,
      phi::errors::InvalidArgument(
          "The size of input's dimensions should be between 2 and 5"
          "But received: the size of input's dimensions is [%d]",
          x_dims.size()));

  if (x_dims.size() == 2 && data_layout == phi::DataLayout::kNHWC) {
    data_layout = phi::DataLayout::kNCHW;
  } else if (x_dims.size() ==
             3) {  // transform 3d tensor to 4d tensor to satisfy the format
    if (data_layout == phi::DataLayout::kNCHW) {
      x_dims.push_back(1);  // expand NCL -> NCL1
    } else {
      x_dims.insert(x_dims.begin() + 2, 1);  // expand NLC -> NL1C
    }
  }

  if (d_scale && d_bias) {
    dev_ctx.template Alloc<float>(d_scale);
    dev_ctx.template Alloc<float>(d_bias);

    if (use_global_stats) {
      const auto* running_mean = mean.get_ptr();
      const auto* running_variance = variance.get_ptr();

      experimental::OpCommand("BNTrainingUpdateGrad")
          .Input(d_y,
                 experimental::TensorDescMaker("grads", d_y)
                     .SetDataLayout(data_layout)
                     .SetDims(phi::make_ddim(x_dims)))
          .Input(x,
                 experimental::TensorDescMaker("x", x)
                     .SetDataLayout(data_layout)
                     .SetDims(phi::make_ddim(x_dims)))
          .Input(*running_mean,
                 experimental::TensorDescMaker("batch_mean", *running_mean)
                     .SetDataLayout(phi::DataLayout::ANY))
          .Input(
              *running_variance,
              experimental::TensorDescMaker("batch_variance", *running_variance)
                  .SetDataLayout(phi::DataLayout::ANY))
          .Output(*d_scale,
                  experimental::TensorDescMaker("diff_scale", *d_scale)
                      .SetDataLayout(phi::DataLayout::ANY))
          .Output(*d_bias,
                  experimental::TensorDescMaker("diff_offset", *d_bias)
                      .SetDataLayout(phi::DataLayout::ANY))
          .Attr("epsilon", epsilon)
          .Run(dev_ctx);
    } else {
      experimental::OpCommand("BNTrainingUpdateGrad")
          .Input(d_y,
                 experimental::TensorDescMaker("grads", d_y)
                     .SetDataLayout(data_layout)
                     .SetDims(phi::make_ddim(x_dims)))
          .Input(x,
                 experimental::TensorDescMaker("x", x)
                     .SetDataLayout(data_layout)
                     .SetDims(phi::make_ddim(x_dims)))
          .Input(saved_mean,
                 experimental::TensorDescMaker("batch_mean", saved_mean)
                     .SetDataLayout(phi::DataLayout::ANY))
          .Input(saved_inv_variance,
                 experimental::TensorDescMaker("batch_variance",
                                               saved_inv_variance)
                     .SetDataLayout(phi::DataLayout::ANY))
          .Output(*d_scale,
                  experimental::TensorDescMaker("diff_scale", *d_scale)
                      .SetDataLayout(phi::DataLayout::ANY))
          .Output(*d_bias,
                  experimental::TensorDescMaker("diff_offset", *d_bias)
                      .SetDataLayout(phi::DataLayout::ANY))
          .Attr("epsilon", epsilon)
          .Run(dev_ctx);
    }
  }
  if (d_x) {
    dev_ctx.template Alloc<T>(d_x);
    if (use_global_stats) {
      const auto* running_variance = variance.get_ptr();
      experimental::OpCommand("BNInferGrad")
          .Input(d_y,
                 experimental::TensorDescMaker("grads", d_y)
                     .SetDataLayout(data_layout)
                     .SetDims(phi::make_ddim(x_dims)))
          .Input(scale,
                 experimental::TensorDescMaker("scale", scale)
                     .SetDataLayout(phi::DataLayout::ANY))
          .Input(
              *running_variance,
              experimental::TensorDescMaker("batch_variance", *running_variance)
                  .SetDataLayout(phi::DataLayout::ANY))
          .Output(*d_x,
                  experimental::TensorDescMaker("x_backprop", *d_x)
                      .SetDataLayout(phi::DataLayout::ANY)
                      .SetDims(phi::make_ddim(x_dims)))
          .Attr("epsilon", epsilon)
          .Run(dev_ctx);
    } else {
      experimental::OpCommand("BNTrainingReduceGrad")
          .Input(d_y,
                 experimental::TensorDescMaker("grads", d_y)
                     .SetDataLayout(data_layout))
          .Input(
              x,
              experimental::TensorDescMaker("x", x).SetDataLayout(data_layout))
          .Input(*d_scale,
                 experimental::TensorDescMaker("diff_scale", *d_scale)
                     .SetDataLayout(phi::DataLayout::ANY))
          .Input(*d_bias,
                 experimental::TensorDescMaker("diff_offset", *d_bias)
                     .SetDataLayout(phi::DataLayout::ANY))
          .Input(scale,
                 experimental::TensorDescMaker("scale", scale)
                     .SetDataLayout(phi::DataLayout::ANY))
          .Input(saved_mean,
                 experimental::TensorDescMaker("batch_mean", saved_mean)
                     .SetDataLayout(phi::DataLayout::ANY))
          .Input(saved_inv_variance,
                 experimental::TensorDescMaker("batch_variance",
                                               saved_inv_variance)
                     .SetDataLayout(phi::DataLayout::ANY))
          .Output(*d_x,
                  experimental::TensorDescMaker("y", *d_x).SetDataLayout(
                      phi::DataLayout::ANY))
          .Attr("epsilon", epsilon)
          .Run(dev_ctx);
    }
  }
}

template <typename T, typename Context>
void BatchNormInferKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& mean,
                          const phi::DenseTensor& variance,
                          const phi::DenseTensor& scale,
                          const phi::DenseTensor& bias,
                          float momentum,
                          float epsilon,
                          const std::string& data_layout_str,
                          phi::DenseTensor* y,
                          phi::DenseTensor* mean_out,
                          phi::DenseTensor* variance_out) {
  phi::DataLayout data_layout = StringToDataLayout(data_layout_str);

  std::vector<int> x_dims = phi::vectorize<int>(x.dims());

  PADDLE_ENFORCE_EQ(
      x_dims.size() >= 2 && x_dims.size() <= 5,
      true,
      phi::errors::InvalidArgument(
          "The size of input's dimensions should be between 2 and 5"
          "But received: the size of input's dimensions is [%d]",
          x_dims.size()));

  if (x_dims.size() == 2 && data_layout == phi::DataLayout::kNHWC) {
    data_layout = phi::DataLayout::kNCHW;
  } else if (x_dims.size() ==
             3) {  // transform 3d tensor to 4d tensor to satisfy the format
    if (data_layout == phi::DataLayout::kNCHW) {
      x_dims.push_back(1);  // expand NCL -> NCL1
    } else {
      x_dims.insert(x_dims.begin() + 2, 1);  // expand NLC -> NL1C
    }
  }

  dev_ctx.template Alloc<T>(y);
  experimental::OpCommand("BNInfer")
      .Input(x,
             experimental::TensorDescMaker("x", x)
                 .SetDataLayout(data_layout)
                 .SetDims(phi::make_ddim(x_dims)))
      .Input(scale,
             experimental::TensorDescMaker("scale", scale)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Input(bias,
             experimental::TensorDescMaker("offset", bias)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Input(mean,
             experimental::TensorDescMaker("mean", mean)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Input(variance,
             experimental::TensorDescMaker("variance", variance)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Output(*y,
              experimental::TensorDescMaker("y", *y)
                  .SetDataLayout(data_layout)
                  .SetDims(phi::make_ddim(x_dims)))
      .Attr("epsilon", epsilon)
      .Run(dev_ctx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(batch_norm,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormKernel,
                          phi::dtype::float16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(batch_norm_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormGradKernel,
                          phi::dtype::float16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(batch_norm_infer,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormInferKernel,
                          phi::dtype::float16,
                          float,
                          double) {}
