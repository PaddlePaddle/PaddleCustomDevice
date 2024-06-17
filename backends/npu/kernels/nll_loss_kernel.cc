// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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
#include "kernels/funcs/slice_utils.h"
#include "paddle/phi/kernels/funcs/tensor_formatter.h"

namespace custom_kernel {
template <typename T, typename Context>
void NllLossRawKernel(const Context& dev_ctx,
                      const phi::DenseTensor& x,
                      const phi::DenseTensor& labels,
                      const paddle::optional<phi::DenseTensor>& weight,
                      int64_t ignore_index,
                      const std::string& reduction,
                      phi::DenseTensor* out,
                      phi::DenseTensor* total_weight) {
  auto x_dims = x.dims();
  phi::Scalar weight_default = 1.0;
  int64_t reduction_int = 1;
  if (reduction == "none") {
    reduction_int = 0;
  } else if (reduction == "sum") {
    reduction_int = 2;
  }

  phi::DenseTensor weight_tensor;
  auto weight_size = phi::make_ddim({x.dims()[1]});
  if (weight.get_ptr() == nullptr) {
    weight_tensor.ResizeAndAllocate(weight_size);
    dev_ctx.template Alloc<float>(&weight_tensor);
    EXEC_NPU_CMD(
        aclnnInplaceFillScalar, dev_ctx, weight_tensor, weight_default);
  } else {
    weight_tensor = *weight.get_ptr();
  }

  if (x_dims.size() == 4 && total_weight->dims().size() == 0) {
    total_weight->Resize(phi::make_ddim({1}));
  }

  // std::cout << "x_dims.size()" << x_dims.size() << std::endl;
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(total_weight);
  if (x_dims.size() == 2) {
    EXEC_NPU_CMD(aclnnNLLLoss,
                 dev_ctx,
                 x,
                 labels,
                 weight_tensor,
                 reduction_int,
                 ignore_index,
                 *out,
                 *total_weight);
  } else if (x_dims.size() == 4) {
    EXEC_NPU_CMD(aclnnNLLLoss2d,
                 dev_ctx,
                 x,
                 labels,
                 weight_tensor,
                 reduction_int,
                 ignore_index,
                 *out,
                 *total_weight);
  }
}

template <typename T, typename Context>
void NllLossGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& labels,
                       const paddle::optional<phi::DenseTensor>& weight,
                       const phi::DenseTensor& total_weight,
                       const phi::DenseTensor& d_out,
                       int64_t ignore_index,
                       const std::string& reduction,
                       phi::DenseTensor* dx) {
  auto x_dims = x.dims();
  phi::Scalar weight_default = 1.0;
  int64_t reduction_int = 1;
  if (reduction == "none") {
    reduction_int = 0;
  } else if (reduction == "sum") {
    reduction_int = 2;
  }

  phi::DenseTensor weight_tensor;
  auto weight_size = phi::make_ddim({x.dims()[1]});
  if (weight.get_ptr() == nullptr) {
    weight_tensor.ResizeAndAllocate(weight_size);
    dev_ctx.template Alloc<float>(&weight_tensor);
    EXEC_NPU_CMD(
        aclnnInplaceFillScalar, dev_ctx, weight_tensor, weight_default);
  } else {
    weight_tensor = *weight.get_ptr();
  }

  dev_ctx.template Alloc<T>(dx);
  if (x_dims.size() == 2) {
    EXEC_NPU_CMD(aclnnNLLLossBackward,
                 dev_ctx,
                 d_out,
                 x,
                 labels,
                 weight_tensor,
                 reduction_int,
                 ignore_index,
                 total_weight,
                 *dx);
  } else if (x_dims.size() == 4) {
    EXEC_NPU_CMD(aclnnNLLLoss2dBackward,
                 dev_ctx,
                 d_out,
                 x,
                 labels,
                 weight_tensor,
                 reduction_int,
                 ignore_index,
                 total_weight,
                 *dx);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(nll_loss,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::NllLossRawKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}

PD_REGISTER_PLUGIN_KERNEL(nll_loss_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::NllLossGradKernel,
                          float,
                          phi::dtype::float16,
                          phi::dtype::bfloat16) {}
