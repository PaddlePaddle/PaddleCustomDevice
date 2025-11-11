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

#include "kernels/funcs/elementwise_utils.h"
#include "kernels/funcs/logic_op.h"
#include "kernels/funcs/reduce_op.h"

namespace custom_kernel {

template <typename T, typename Context>
void KLDivLossKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& label,
                     const std::string& reduction,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);
  phi::DenseTensor out_tmp;
  out_tmp.Resize(x.dims());
  dev_ctx.template Alloc<T>(&out_tmp);
  // formula: label * (log(label) - x)
  // 0. mark label >=0
  phi::DenseTensor tensor_zeros;
  tensor_zeros.Resize(label.dims());
  dev_ctx.template Alloc<T>(&tensor_zeros);
  MLUCnnlTensorDesc tensor_zeros_desc(tensor_zeros);

  auto value = static_cast<T>(0);
  MLUCnnl::Fill(dev_ctx,
                CNNL_POINTER_MODE_HOST,
                &value,
                tensor_zeros_desc.get(),
                GetBasePtr(&tensor_zeros));

  phi::DenseTensor condiction_out;
  condiction_out.Resize(label.dims());
  dev_ctx.template Alloc<T>(&condiction_out);
  MLULogicOp(dev_ctx, label, tensor_zeros, "greater_equal", &condiction_out);

  MLUCnnlTensorDesc label_desc(label);
  // 1. log(label) ->log_label
  phi::DenseTensor log_label;
  log_label.Resize(label.dims());
  dev_ctx.template Alloc<T>(&log_label);

  MLUCnnlTensorDesc loglabel_desc(log_label);

  MLUCnnl::Log(dev_ctx,
               CNNL_COMPUTATION_HIGH_PRECISION,
               CNNL_LOG_E,
               label_desc.get(),
               GetBasePtr(&label),
               loglabel_desc.get(),
               GetBasePtr(&log_label));
  // 2. optensor --sub( log(label) - x)->sub_out
  phi::DenseTensor sub_out;
  sub_out.Resize(x.dims());
  dev_ctx.template Alloc<T>(&sub_out);
  MLUOpTensorKernel<T>(dev_ctx, log_label, x, -1, CNNL_OP_TENSOR_SUB, &sub_out);
  // 3. optensor-- mul(label,sub_out) -->out_tmp
  MLUOpTensorKernel<T>(
      dev_ctx, label, sub_out, -1, CNNL_OP_TENSOR_MUL, &out_tmp);

  MLUCnnlTensorDesc out_tmp_desc(out_tmp);
  MLUCnnlTensorDesc condition_desc(condiction_out);
  MLUCnnl::Select(dev_ctx,
                  condition_desc.get(),
                  GetBasePtr(&condiction_out),
                  out_tmp_desc.get(),
                  GetBasePtr(&out_tmp),
                  tensor_zeros_desc.get(),
                  GetBasePtr(&tensor_zeros),
                  out_tmp_desc.get(),
                  GetBasePtr(&out_tmp));

  std::vector<int64_t> axes = {-1};
  // 4.reduction
  if ("none" == reduction) {
    *out = out_tmp;
  } else if ("batchmean" == reduction) {
    const int batch = x.dims()[0];
    float alpha = 1.0 / batch;
    float beta = 0;
    MLUCnnlTensorDesc out_desc(*out);
    MLUReduceOp<T>(dev_ctx, out_tmp, axes, false, true, "reduce_sum", out);
    MLUCnnl::Transform(dev_ctx,
                       &alpha,
                       &beta,
                       out_desc.get(),
                       GetBasePtr(out),
                       out_desc.get(),
                       GetBasePtr(out));
  } else if ("mean" == reduction) {
    MLUReduceOp<T>(dev_ctx, out_tmp, axes, false, true, "reduce_mean", out);
  } else if ("sum" == reduction) {
    MLUReduceOp<T>(dev_ctx, out_tmp, axes, false, true, "reduce_sum", out);
  }
}

template <typename T, typename Context>
void KLDivLossGradKernel(const Context& dev_ctx,
                         const phi::DenseTensor& x,
                         const phi::DenseTensor& label,
                         const phi::DenseTensor& d_out,
                         const std::string& reduction,
                         phi::DenseTensor* d_x) {
  dev_ctx.template Alloc<T>(d_x);
  // formula: dx = -1 * label * d_out
  // Relu(label) make label >=0
  phi::DenseTensor label_clip;
  label_clip.Resize(label.dims());
  dev_ctx.template Alloc<T>(&label_clip);
  MLUCnnlActivationDesc act_desc(CNNL_ACTIVATION_RELU, 1.0);
  MLUCnnlTensorDesc label_desc(label);
  MLUCnnlTensorDesc label_clip_desc(label_clip);

  MLUCnnl::Active(dev_ctx,
                  act_desc.get(),
                  label_desc.get(),
                  GetBasePtr(&label),
                  label_clip_desc.get(),
                  GetBasePtr(&label_clip));

  // label * d_out
  phi::DenseTensor out_tmp;
  out_tmp.Resize(x.dims());
  dev_ctx.template Alloc<T>(&out_tmp);
  MLUOpTensorKernel<T>(
      dev_ctx, label_clip, d_out, -1, CNNL_OP_TENSOR_MUL, &out_tmp);

  MLUCnnlTensorDesc out_tmp_desc(out_tmp);

  float alpha;
  float beta = 0;
  if ("mean" == reduction) {
    alpha = static_cast<float>(-1.0 / d_x->numel());
  } else if ("batchmean" == reduction) {
    alpha = static_cast<float>(-1.0 / d_x->dims()[0]);
  } else {
    alpha = -1.0;
  }

  MLUCnnlTensorDesc dx_desc(*d_x);
  MLUCnnl::Transform(dev_ctx,
                     &alpha,
                     &beta,
                     out_tmp_desc.get(),
                     GetBasePtr(&out_tmp),
                     dx_desc.get(),
                     GetBasePtr(d_x));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(kldiv_loss,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::KLDivLossKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(kldiv_loss_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::KLDivLossGradKernel,
                          float,
                          phi::dtype::float16) {}
