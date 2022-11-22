/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

namespace custom_kernel {

template <typename T, typename Context>
void nll_loss_npu_impl(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& labels,
                       const paddle::optional<phi::DenseTensor>& weight,
                       int64_t ignore_index,
                       const std::string& reduction,
                       phi::DenseTensor* out,
                       phi::DenseTensor* total_weight) {
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(total_weight);
  auto stream = dev_ctx.stream();

  phi::DenseTensor weight_tensor;
  if (weight.get_ptr() != nullptr) {
    weight_tensor = *(weight.get_ptr());
  } else {
    weight_tensor.Resize(phi::make_ddim({x.dims()[1]}));
    dev_ctx.template Alloc<T>(&weight_tensor);
    FillNpuTensorWithConstant<T>(&weight_tensor, dev_ctx, static_cast<T>(1));
  }

  if (ignore_index >= 0 && ignore_index < x.dims()[1]) {
    phi::DenseTensor zero;
    zero.Resize(phi::make_ddim({1}));
    auto zero_ptr = dev_ctx.template Alloc<T>(&zero);
    FillNpuTensorWithConstant<T>(&zero, dev_ctx, static_cast<T>(0));

    auto size = paddle::experimental::SizeOf(weight_tensor.dtype());
    auto dst_ptr = weight_tensor.data();
    dst_ptr += ignore_index * size;
    aclrtMemcpyAsync(
        dst_ptr, size, zero_ptr, size, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
  }

  const auto& runner_nll_loss =
      NpuOpRunner("NLLLoss",
                  {x, labels, weight_tensor},
                  {*out, *total_weight},
                  {{"reduction", reduction}, {"ignore_index", ignore_index}});
  runner_nll_loss.Run(stream);
}

template <typename T, typename Context>
void NLLLossKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& labels,
                   const paddle::optional<phi::DenseTensor>& weight,
                   int64_t ignore_index,
                   const std::string& reduction,
                   phi::DenseTensor* out,
                   phi::DenseTensor* total_weight) {
  auto x_dims = x.dims();
  auto labels_dims = labels.dims();
  const auto batch_size = x_dims[0];

  if (x_dims.size() == 2) {
    PADDLE_ENFORCE_EQ(labels_dims.size(),
                      1,
                      phi::errors::InvalidArgument(
                          "The size of labels should be 1 when size of x is 2, "
                          "but got size of labels is %d.",
                          labels.dims().size()));
    PADDLE_ENFORCE_EQ(
        batch_size,
        labels_dims[0],
        phi::errors::InvalidArgument(
            "The first dimension of labels should be equal to that of x, "
            "but got first dimension of labels is %d and x is %d.",
            labels_dims[0],
            batch_size));
    nll_loss_npu_impl<T, Context>(
        dev_ctx, x, labels, weight, ignore_index, reduction, out, total_weight);
  } else if (x_dims.size() == 4) {
    PADDLE_ENFORCE(
        false,
        phi::errors::Unimplemented("The size of x only supports 2, size 4 is "
                                   "not implemented in temporary."));
  }
}

template <typename T, typename Context>
void nll_loss_grad_npu_impl(const Context& dev_ctx,
                            const phi::DenseTensor& x,
                            const phi::DenseTensor& labels,
                            const paddle::optional<phi::DenseTensor>& weight,
                            const phi::DenseTensor& total_weight,
                            const phi::DenseTensor& d_out,
                            int64_t ignore_index,
                            const std::string& reduction,
                            phi::DenseTensor* dx) {
  dev_ctx.template Alloc<T>(dx);
  auto stream = dev_ctx.stream();

  phi::DenseTensor weight_tensor;
  if (weight.get_ptr() != nullptr) {
    weight_tensor = *(weight.get_ptr());
  } else {
    weight_tensor.Resize(phi::make_ddim({x.dims()[1]}));
    dev_ctx.template Alloc<T>(&weight_tensor);
    FillNpuTensorWithConstant<T>(&weight_tensor, dev_ctx, static_cast<T>(1));
  }

  if (ignore_index >= 0 && ignore_index < x.dims()[1]) {
    phi::DenseTensor zero;
    zero.Resize(phi::make_ddim({1}));
    auto zero_ptr = dev_ctx.template Alloc<T>(&zero);
    FillNpuTensorWithConstant<T>(&zero, dev_ctx, static_cast<T>(0));

    auto size = paddle::experimental::SizeOf(weight_tensor.dtype());
    auto dst_ptr = weight_tensor.data();
    dst_ptr += ignore_index * size;
    aclrtMemcpyAsync(
        dst_ptr, size, zero_ptr, size, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
  }

  const auto& runner_nll_loss =
      NpuOpRunner("NLLLossGrad",
                  {x, d_out, labels, weight_tensor, total_weight},
                  {*dx},
                  {{"reduction", reduction}, {"ignore_index", ignore_index}});
  runner_nll_loss.Run(stream);
}

template <typename T, typename Context>
void NLLLossGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& labels,
                       const paddle::optional<phi::DenseTensor>& weight,
                       const phi::DenseTensor& total_weight,
                       const phi::DenseTensor& d_out,
                       int64_t ignore_index,
                       const std::string& reduction,
                       phi::DenseTensor* dx) {
  auto x_dims = x.dims();
  auto labels_dims = labels.dims();
  const auto batch_size = x_dims[0];

  if (x_dims.size() == 2) {
    nll_loss_grad_npu_impl<T, Context>(dev_ctx,
                                       x,
                                       labels,
                                       weight,
                                       total_weight,
                                       d_out,
                                       ignore_index,
                                       reduction,
                                       dx);
  } else if (x_dims.size() == 4) {
    PADDLE_ENFORCE(
        false,
        phi::errors::Unimplemented("The size of x only supports 2, size 4 is "
                                   "not implemented in temporary."));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    nll_loss, npu, ALL_LAYOUT, custom_kernel::NLLLossKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(
    nll_loss_grad, npu, ALL_LAYOUT, custom_kernel::NLLLossGradKernel, float) {}
