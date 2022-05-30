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
void BatchNormKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& scale,
                     const phi::DenseTensor& bias,
                     const phi::DenseTensor& running_mean,
                     const phi::DenseTensor& running_var,
                     float momentum,
                     float epsilon,
                     const std::string& data_layout_str,
                     bool is_test,
                     bool use_global_stats,
                     bool trainable_stats,
                     bool fuse_with_relu,
                     phi::DenseTensor* y,
                     phi::DenseTensor* mean_out,
                     phi::DenseTensor* variance_out,
                     phi::DenseTensor* saved_mean,
                     phi::DenseTensor* saved_variance,
                     phi::DenseTensor* reserve_space) {
  bool test_mode = is_test && (!trainable_stats);
  bool training = !test_mode && !use_global_stats;

  phi::DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_layout_str);

  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_EQ((x_dims.size() == 4UL || x_dims.size() == 3UL),
                    true,
                    phi::errors::InvalidArgument(
                        "The input tensor X's dimension must equal to 3 or 4. "
                        " But got X's shape = [%s], X's dimension = [%d].",
                        x_dims.to_str(),
                        x_dims.size()));

  dev_ctx.template Alloc<T>(y);

  phi::DenseTensor x_tensor(x), y_tesnor(*y);
  if (data_layout == phi::DataLayout::kNHWC) {
    phi::DenseTensorMeta x_meta = {x.dtype(), x.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta y_meta = {
        y->dtype(), y->dims(), phi::DataLayout::kNHWC};
    x_tensor.set_meta(x_meta);
    y_tesnor.set_meta(y_meta);
  }

  auto stream = dev_ctx.stream();
  if (!training) {
    const auto& runner_infer =
        NpuOpRunner("BNInfer",
                    {x_tensor, scale, bias, running_mean, running_var},
                    {y_tesnor},
                    {{"epsilon", epsilon}});
    runner_infer.Run(stream);
  } else {
    dev_ctx.template Alloc<T>(mean_out);
    dev_ctx.template Alloc<T>(variance_out);
    dev_ctx.template Alloc<T>(saved_mean);
    dev_ctx.template Alloc<T>(saved_variance);

    phi::DenseTensor sum, square_sum;
    sum.Resize(running_mean.dims());
    square_sum.Resize(running_mean.dims());
    dev_ctx.template Alloc<T>(&sum);
    dev_ctx.template Alloc<T>(&square_sum);
    // BNTrainingReduce ONLY support rank = 4
    if (x.dims().size() == 3) {
      auto x_shape_vec = phi::vectorize(x.dims());
      if (data_layout == phi::DataLayout::kNCHW) {
        x_shape_vec.push_back(1);  // expand NCL -> NCL1
      } else {
        x_shape_vec.insert(x_shape_vec.begin() + 2, 1);  // expand NLC -> NL1C
      }
      auto x_new_shape = phi::make_ddim(x_shape_vec);
      x_tensor.Resize(x_new_shape);
      x_tensor.Resize(x_new_shape);
    }
    const auto& runner_reduce = NpuOpRunner("BNTrainingReduce",
                                            {x_tensor},
                                            {sum, square_sum},
                                            {{"epsilon", epsilon}});
    runner_reduce.Run(stream);

    const auto& runner_update = NpuOpRunner(
        "BNTrainingUpdate",
        {x_tensor, sum, square_sum, scale, bias, running_mean, running_var},
        {y_tesnor, *mean_out, *variance_out, *saved_mean, *saved_variance},
        {{"factor", momentum}, {"epsilon", epsilon}});
    runner_update.Run(stream);
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
    bool fuse_with_relu,
    phi::DenseTensor* d_x,
    phi::DenseTensor* d_scale,
    phi::DenseTensor* d_bias) {
  phi::DataLayout data_layout =
      paddle::framework::StringToDataLayout(data_layout_str);

  use_global_stats = is_test || use_global_stats;

  phi::DenseTensor x_tensor(x), dy_tensor(d_y);
  if (data_layout == phi::DataLayout::kNHWC) {
    phi::DenseTensorMeta x_meta = {x.dtype(), x.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta dy_meta = {
        d_y.dtype(), d_y.dims(), phi::DataLayout::kNHWC};
    x_tensor.set_meta(x_meta);
    dy_tensor.set_meta(dy_meta);
  }

  phi::DenseTensor scale_grad_tmp, bias_grad_tmp;
  scale_grad_tmp.Resize(scale.dims());
  bias_grad_tmp.Resize(bias.dims());
  dev_ctx.template Alloc<T>(&scale_grad_tmp);
  dev_ctx.template Alloc<T>(&bias_grad_tmp);

  if (d_scale == nullptr) {
    d_scale = &scale_grad_tmp;
  }
  if (d_bias == nullptr) {
    d_bias = &bias_grad_tmp;
  }

  auto stream = dev_ctx.stream();
  if (d_scale && d_bias) {
    dev_ctx.template Alloc<T>(d_scale);
    dev_ctx.template Alloc<T>(d_bias);

    if (use_global_stats) {
      const auto* running_mean = mean.get_ptr();
      const auto* running_variance = variance.get_ptr();
      const auto& runner_update =
          NpuOpRunner("BNTrainingUpdateGrad",
                      {dy_tensor, x_tensor, *running_mean, *running_variance},
                      {*d_scale, *d_bias},
                      {{"epsilon", epsilon}});
      runner_update.Run(stream);
    } else {
      const auto& runner_update =
          NpuOpRunner("BNTrainingUpdateGrad",
                      {dy_tensor, x_tensor, saved_mean, saved_inv_variance},
                      {*d_scale, *d_bias},
                      {{"epsilon", epsilon}});
      runner_update.Run(stream);
    }
  }
  if (d_x) {
    dev_ctx.template Alloc<T>(d_x);
    phi::DenseTensor dx_tensor(*d_x);
    if (data_layout == phi::DataLayout::kNHWC) {
      phi::DenseTensorMeta dx_meta = {
          d_x->dtype(), d_x->dims(), phi::DataLayout::kNHWC};
      dx_tensor.set_meta(dx_meta);
    }
    if (use_global_stats) {
      if (x.dims().size() == 3) {
        // BNInferGrad only support x rank = 4,
        auto x_shape_vec = phi::vectorize(d_x->dims());
        if (data_layout == phi::DataLayout::kNCHW) {
          x_shape_vec.push_back(1);  // expand NCL -> NCL1
        } else {
          x_shape_vec.insert(x_shape_vec.begin() + 2, 1);  // expand NLC -> NL1C
        }
        auto x_new_shape = phi::make_ddim(x_shape_vec);
        dx_tensor.Resize(x_new_shape);
        dy_tensor.Resize(x_new_shape);
      }
      const auto* running_variance = variance.get_ptr();
      const auto& runner_infer =
          NpuOpRunner("BNInferGrad",
                      {dy_tensor, scale, *running_variance},
                      {dx_tensor},
                      {{"epsilon", epsilon}});
      runner_infer.Run(stream);
    } else {
      const auto& runner_reduce = NpuOpRunner("BNTrainingReduceGrad",
                                              {dy_tensor,
                                               x_tensor,
                                               *d_scale,
                                               *d_bias,
                                               scale,
                                               saved_mean,
                                               saved_inv_variance},
                                              {dx_tensor},
                                              {{"epsilon", epsilon}});
      runner_reduce.Run(stream);
    }
  }
}

template <typename T, typename Context>
void BatchNormInferKernel(const Context& dev_ctx,
                          const phi::DenseTensor& x,
                          const phi::DenseTensor& scale,
                          const phi::DenseTensor& bias,
                          const phi::DenseTensor& mean,
                          const phi::DenseTensor& variance,
                          float momentum,
                          float epsilon,
                          const std::string& data_layout_str,
                          phi::DenseTensor* y,
                          phi::DenseTensor* mean_out,
                          phi::DenseTensor* variance_out) {
  phi::DataLayout data_layout = StringToDataLayout(data_layout_str);

  const auto& x_dims = x.dims();
  PADDLE_ENFORCE_EQ((x_dims.size() == 4UL || x_dims.size() == 3UL),
                    true,
                    phi::errors::InvalidArgument(
                        "The input tensor X's dimension must equal to 3 or 4. "
                        " But got X's shape = [%s], X's dimension = [%d].",
                        x_dims.to_str(),
                        x_dims.size()));

  dev_ctx.template Alloc<T>(y);

  phi::DenseTensor x_tensor(x);
  phi::DenseTensor y_tesnor(*y);
  if (data_layout == phi::DataLayout::kNHWC) {
    phi::DenseTensorMeta x_meta = {x.dtype(), x.dims(), phi::DataLayout::kNHWC};
    phi::DenseTensorMeta y_meta = {
        y->dtype(), y->dims(), phi::DataLayout::kNHWC};
    x_tensor.set_meta(x_meta);
    y_tesnor.set_meta(y_meta);
  }

  auto stream = dev_ctx.stream();
  const auto& runner_infer =
      NpuOpRunner("BNInfer",
                  {x_tensor, scale, bias, mean, variance},
                  {y_tesnor},
                  {{"epsilon", epsilon}});
  runner_infer.Run(stream);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(batch_norm,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormKernel,
                          phi::dtype::float16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(batch_norm_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormGradKernel,
                          phi::dtype::float16,
                          float,
                          double) {}

PD_REGISTER_PLUGIN_KERNEL(batch_norm_infer,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::BatchNormInferKernel,
                          phi::dtype::float16,
                          float,
                          double) {}
