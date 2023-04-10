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
#include "kernels/funcs/reduce_op.h"

namespace custom_kernel {

template <typename T, typename Context>
void MaxRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::IntArray& axes,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DenseTensor* out) {
  MLUReduceOp<T>(
      dev_ctx, x, axes.GetData(), keep_dim, reduce_all, "reduce_max", out);
}

template <typename T, typename Context>
void MaxKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               bool keep_dim,
               phi::DenseTensor* out) {
  bool reduce_all = false;
  if (dims.size() == 0) {
    reduce_all = true;
  }
  custom_kernel::MaxRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T, typename Context>
void MaxGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& out,
                   const phi::DenseTensor& out_grad,
                   const phi::IntArray& reduce_dims_in,
                   bool keep_dim,
                   bool reduce_all,
                   phi::DenseTensor* x_grad) {
  auto reduce_dims = reduce_dims_in.GetData();
  auto stream = dev_ctx.stream();
  dev_ctx.template Alloc<T>(x_grad);
  if (x.dims().size() == 0) {
    TensorCopy(dev_ctx, out_grad, true, x_grad);
    return;
  }
  // broadcast
  auto x_dims_vec = phi::vectorize(x.dims());
  if (reduce_all) {
    reduce_dims.clear();
    for (size_t d = 0; d < x_dims_vec.size(); ++d) {
      reduce_dims.push_back(static_cast<int>(d));
    }
  }

  phi::DenseTensor tmp_out(out), tmp_out_grad(out_grad);
  auto tmp_out_dims_vec = x_dims_vec;
  for (auto d : reduce_dims) {
    if (d < 0) {
      d += x_dims_vec.size();
    }
    tmp_out_dims_vec[d] = 1;
  }

  tmp_out.Resize(phi::make_ddim(tmp_out_dims_vec));
  tmp_out_grad.Resize(phi::make_ddim(tmp_out_dims_vec));

  phi::DenseTensor transformed_out;
  phi::DenseTensorMeta meta = {x.dtype(), phi::make_ddim(x_dims_vec)};
  transformed_out.set_meta(meta);
  dev_ctx.template Alloc<T>(&transformed_out);
  MLUCnnlTensorDesc tmp_out_desc(tmp_out);
  MLUCnnlTensorDesc transformed_out_desc(transformed_out);
  MLUCnnl::BroadcastTo(dev_ctx,
                       tmp_out_desc.get(),
                       GetBasePtr(&tmp_out),
                       transformed_out_desc.get(),
                       GetBasePtr(&transformed_out));

  phi::DenseTensor transformed_out_grad;
  phi::DenseTensorMeta grad_meta = {x.dtype(), phi::make_ddim(x_dims_vec)};
  transformed_out_grad.set_meta(grad_meta);
  dev_ctx.template Alloc<T>(&transformed_out_grad);
  MLUCnnlTensorDesc tmp_out_grad_desc(tmp_out_grad);
  MLUCnnlTensorDesc transformed_out_grad_desc(transformed_out_grad);
  MLUCnnl::BroadcastTo(dev_ctx,
                       tmp_out_grad_desc.get(),
                       GetBasePtr(&tmp_out_grad),
                       transformed_out_grad_desc.get(),
                       GetBasePtr(&transformed_out_grad));
  // compare
  phi::DenseTensor equal_cond;
  equal_cond.Resize(x_grad->dims());
  dev_ctx.template Alloc<bool>(&equal_cond);
  MLUCnnlTensorDesc x_desc(x);
  MLUCnnlTensorDesc equal_cond_desc(equal_cond);
  MLUCnnl::Logic(dev_ctx,
                 CNNL_LOGIC_OP_EQ,
                 x_desc.get(),
                 GetBasePtr(&x),
                 transformed_out_desc.get(),
                 GetBasePtr(&transformed_out),
                 equal_cond_desc.get(),
                 GetBasePtr(&equal_cond));

  // select
  phi::DenseTensor t_zero;
  t_zero.Resize(x_grad->dims());
  dev_ctx.template Alloc<T>(&t_zero);
  MLUCnnlTensorDesc t_zero_desc(t_zero);
  auto value = static_cast<T>(0);
  MLUCnnl::Fill(dev_ctx,
                CNNL_POINTER_MODE_HOST,
                &value,
                t_zero_desc.get(),
                GetBasePtr(&t_zero));
  t_zero.Resize(x_grad->dims());

  MLUCnnlTensorDesc x_grad_desc(*x_grad);
  MLUCnnl::Select(dev_ctx,
                  equal_cond_desc.get(),
                  GetBasePtr(&equal_cond),
                  transformed_out_grad_desc.get(),
                  GetBasePtr(&transformed_out_grad),
                  t_zero_desc.get(),
                  GetBasePtr(&t_zero),
                  x_grad_desc.get(),
                  GetBasePtr(x_grad));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(max_raw,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::MaxRawKernel,
                          int32_t,
                          phi::dtype::float16,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(max,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::MaxKernel,
                          int32_t,
                          phi::dtype::float16,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(max_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::MaxGradKernel,
                          bool,
                          int32_t,
                          phi::dtype::float16,
                          float) {}
