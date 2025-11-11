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

#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void HuberLossKernel(const Context& dev_ctx,
                     const phi::DenseTensor& input,
                     const phi::DenseTensor& label,
                     float delta,
                     phi::DenseTensor* out,
                     phi::DenseTensor* residual) {
  // compute y-x
  cnnlDataType_t data_type = ToCnnlDataType<T>();
  dev_ctx.template Alloc<T>(residual);
  MLUCnnlTensorDesc input_disc(input);
  MLUCnnlOpTensorDesc sub_op_desc(
      CNNL_OP_TENSOR_SUB, data_type, CNNL_NOT_PROPAGATE_NAN);
  MLUCnnl::OpTensor(dev_ctx,
                    sub_op_desc.get(),
                    input_disc.get(),
                    GetBasePtr(&label),
                    input_disc.get(),
                    GetBasePtr(&input),
                    input_disc.get(),
                    GetBasePtr(residual),
                    data_type);

  // compute smoothl1loss
  dev_ctx.template Alloc<T>(out);
  cnnlSmoothL1LossAlgorithm_t smoothl1_algo =
      CNNL_SMOOTHL1LOSS_REDUCTION_NONE;  // defines whether to do reduction
                                         // here
  MLUCnnl::SmoothL1LossForward(
      dev_ctx,
      input_disc.get(),
      GetBasePtr(&input),
      input_disc.get(), /* target has same shape as x */
      GetBasePtr(&label),
      delta,
      smoothl1_algo,
      input_disc.get(), /* out has same shape as x */
      GetBasePtr(out));

  // compute multiply by delta
  Tensor scale_tensor, bias_tensor;
  scale_tensor.Resize(phi::make_ddim({1}));
  dev_ctx.template Alloc<T>(&scale_tensor);
  bias_tensor.Resize(phi::make_ddim({1}));
  dev_ctx.template Alloc<T>(&bias_tensor);
  FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(delta), &scale_tensor);
  FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0.f), &bias_tensor);
  const int axis = std::max(out->dims().size() - 1, 0);
  MLUCnnlTensorDesc scale_desc(scale_tensor);
  MLUCnnlTensorDesc bias_desc(bias_tensor);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::Scale(dev_ctx,
                 axis,
                 out_desc.get(),
                 GetBasePtr(out),
                 scale_desc.get(),
                 GetBasePtr(&scale_tensor),
                 bias_desc.get(),
                 GetBasePtr(&bias_tensor),
                 out_desc.get(),
                 GetBasePtr(out));
}

template <typename T, typename Context>
void HuberLossGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& residual,
                         const phi::DenseTensor& dout,
                         float delta,
                         phi::DenseTensor* dx,
                         phi::DenseTensor* dy) {
  Tensor t_grad_rd;
  t_grad_rd.Resize(residual.dims());
  dev_ctx.template Alloc<T>(&t_grad_rd);
  MLUCnnlTensorDesc t_grad_rd_desc(t_grad_rd);
  if (dx || dy) {
    Tensor t_zero;
    t_zero.Resize(residual.dims());
    dev_ctx.template Alloc<T>(&t_zero);
    FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0.f), &t_zero);
    MLUCnnlTensorDesc residual_desc(residual);
    MLUCnnlTensorDesc dout_desc(dout);

    cnnlSmoothL1LossAlgorithm_t smoothl1_algo =
        CNNL_SMOOTHL1LOSS_REDUCTION_NONE;  // defines whether to do reduction
                                           // here
    MLUCnnl::SmoothL1LossBackward(dev_ctx,
                                  residual_desc.get(),
                                  GetBasePtr(&residual),
                                  residual_desc.get(),
                                  GetBasePtr(&t_zero),
                                  dout_desc.get(),
                                  GetBasePtr(&dout),
                                  delta,
                                  smoothl1_algo,
                                  t_grad_rd_desc.get(),
                                  GetBasePtr(&t_grad_rd));
  }

  // compute multiply by delta
  Tensor scale_tensor, bias_tensor;
  scale_tensor.Resize(phi::make_ddim({1}));
  dev_ctx.template Alloc<T>(&scale_tensor);
  bias_tensor.Resize(phi::make_ddim({1}));
  dev_ctx.template Alloc<T>(&bias_tensor);
  FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0.f), &bias_tensor);
  const int axis = std::max(t_grad_rd.dims().size() - 1, 0);

  MLUCnnlTensorDesc scale_desc(scale_tensor);
  MLUCnnlTensorDesc bias_desc(bias_tensor);

  if (dx) {
    dev_ctx.template Alloc<T>(dx);
    FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(-delta), &scale_tensor);
    MLUCnnlTensorDesc out_desc(*dx);
    MLUCnnl::Scale(dev_ctx,
                   axis,
                   t_grad_rd_desc.get(),
                   GetBasePtr(&t_grad_rd),
                   scale_desc.get(),
                   GetBasePtr(&scale_tensor),
                   bias_desc.get(),
                   GetBasePtr(&bias_tensor),
                   out_desc.get(),
                   GetBasePtr(dx));
  }
  if (dy) {
    dev_ctx.template Alloc<T>(dy);
    FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(delta), &scale_tensor);
    MLUCnnlTensorDesc out_desc(*dy);
    MLUCnnl::Scale(dev_ctx,
                   axis,
                   t_grad_rd_desc.get(),
                   GetBasePtr(&t_grad_rd),
                   scale_desc.get(),
                   GetBasePtr(&scale_tensor),
                   bias_desc.get(),
                   GetBasePtr(&bias_tensor),
                   out_desc.get(),
                   GetBasePtr(dy));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(huber_loss,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::HuberLossKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(huber_loss_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::HuberLossGradKernel,
                          float,
                          phi::dtype::float16) {}
