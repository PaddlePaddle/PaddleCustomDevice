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

#include <limits.h>

#include "kernels/funcs/elementwise_utils.h"
#include "kernels/funcs/logic_op.h"
#include "kernels/funcs/mlu_baseop.h"
#include "kernels/funcs/mlu_funcs.h"

namespace custom_kernel {

template <typename T, typename Context>
void PnormKernel(const Context& dev_ctx,
                 const phi::DenseTensor& x,
                 float porder,
                 int axis,
                 float epsilon,
                 bool keepdim,
                 bool asvector,
                 phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto xdim = x.dims();
  axis = axis < 0 ? xdim.size() + axis : axis;

  NormalizeDesc normalize_desc(&axis,
                               1, /* axis_num */
                               CNNL_NOT_PROPAGATE_NAN,
                               epsilon,
                               porder,
                               0, /* channel_shared */
                               0 /* across_spatial */);
  Tensor p_norm_out;
  auto ori_out_dims = out->dims();
  auto out_dims_vec = phi::vectorize(x.dims());
  out_dims_vec[axis] = 1;
  auto norm_dims = phi::make_ddim(out_dims_vec);
  out->Resize(norm_dims);
  p_norm_out.Resize(x.dims());
  dev_ctx.template Alloc<T>(&p_norm_out);
  MLUCnnlTensorDesc in_desc(x);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnlTensorDesc p_norm_desc(p_norm_out);
  MLUCnnl::Normalize(dev_ctx,
                     normalize_desc.get(),
                     in_desc.get(),
                     GetBasePtr(&x),
                     in_desc.get(),
                     nullptr, /* scale_tensor */
                     p_norm_desc.get(),
                     out_desc.get(),
                     GetBasePtr(&p_norm_out),
                     GetBasePtr(out));
  out->Resize(ori_out_dims);
}

template <typename T, typename Context>
void PnormGradKernel(const Context& dev_ctx,
                     const phi::DenseTensor& x,
                     const phi::DenseTensor& y,
                     const phi::DenseTensor& dy,
                     float porder,
                     int axis,
                     float epsilon,
                     bool keepdim,
                     bool asvector,
                     phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto xdim = x.dims();
  axis = axis < 0 ? xdim.size() + axis : axis;

  phi::DenseTensor y_share(y);
  phi::DenseTensor dy_share(dy);
  auto ydim = xdim;
  if (!keepdim) {
    ydim[axis] = 1;
  } else {
    ydim = y.dims();
  }
  y_share.Resize(ydim);
  dy_share.Resize(ydim);
  if (porder == 0) {
    FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0), out);
    out->Resize(xdim);
  } else if (porder == INFINITY || porder == -INFINITY) {
    phi::DenseTensor x_abs;
    x_abs.Resize(xdim);
    dev_ctx.template Alloc<T>(&x_abs);
    MLUCnnlTensorDesc abs_in_desc(x);
    MLUCnnlTensorDesc abs_out_desc(x_abs);
    MLUCnnl::Abs(dev_ctx,
                 abs_in_desc.get(),
                 GetBasePtr(&x),
                 abs_out_desc.get(),
                 GetBasePtr(&x_abs));

    phi::DenseTensor t_cond;
    t_cond.Resize(xdim);
    MLULogicOp(dev_ctx, x_abs, y_share, "equal", out);

    phi::DenseTensor t_zero;
    t_zero.Resize({1});
    dev_ctx.template Alloc<T>(&t_zero);
    FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0), &t_zero);

    phi::DenseTensor x_sign;
    x_sign.Resize(xdim);
    dev_ctx.template Alloc<T>(&x_sign);
    MLUCnnlTensorDesc sign_out_desc(x_sign);
    MLUCnnl::Sign(dev_ctx,
                  abs_in_desc.get(),
                  GetBasePtr(&x),
                  sign_out_desc.get(),
                  GetBasePtr(&x_sign));

    MLUOpTensorKernel<T>(
        dev_ctx, x_sign, dy_share, -1, CNNL_OP_TENSOR_MUL, out);

    MLUCnnlTensorDesc cond_desc(t_cond);
    MLUCnnlTensorDesc zero_desc(t_zero);
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnl::Select(dev_ctx,
                    cond_desc.get(),
                    GetBasePtr(&t_cond),
                    out_desc.get(),
                    GetBasePtr(out),
                    zero_desc.get(),
                    GetBasePtr(&t_zero),
                    out_desc.get(),
                    GetBasePtr(out));
  } else {
    phi::DenseTensor x_abs;
    x_abs.Resize(xdim);
    dev_ctx.template Alloc<T>(&x_abs);
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnlTensorDesc abs_in_desc(x);
    MLUCnnlTensorDesc abs_out_desc(x_abs);
    MLUCnnl::Abs(dev_ctx,
                 abs_in_desc.get(),
                 GetBasePtr(&x),
                 abs_out_desc.get(),
                 GetBasePtr(&x_abs));

    phi::DenseTensor x_sign;
    x_sign.Resize(xdim);
    dev_ctx.template Alloc<T>(&x_sign);
    MLUCnnlTensorDesc sign_out_desc(x_sign);
    MLUCnnl::Sign(dev_ctx,
                  abs_in_desc.get(),
                  GetBasePtr(&x),
                  sign_out_desc.get(),
                  GetBasePtr(&x_sign));

    phi::DenseTensor y_pow;
    y_pow.Resize(ydim);
    dev_ctx.template Alloc<T>(&y_pow);
    if (porder >= 1) {
      phi::DenseTensor t_exp;
      t_exp.Resize({1});
      dev_ctx.template Alloc<float>(&t_exp);
      FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(porder - 1), &t_exp);
      MLUBinaryOp<POW, T>(dev_ctx, x_abs, t_exp, -1, &x_abs);
      MLUBinaryOp<POW, T>(dev_ctx, y_share, t_exp, -1, &y_pow);

      MLUCnnlTensorDesc y_pow_desc(y_pow);
      MLUCnnl::Div(dev_ctx,
                   CNNL_COMPUTATION_HIGH_PRECISION,
                   abs_out_desc.get(),
                   GetBasePtr(&x_abs),
                   y_pow_desc.get(),
                   GetBasePtr(&y_pow),
                   out_desc.get(),
                   GetBasePtr(out));
    } else {
      phi::DenseTensor t_exp;
      t_exp.Resize({1});
      dev_ctx.template Alloc<float>(&t_exp);
      FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(1 - porder), &t_exp);
      MLUBinaryOp<POW, T>(dev_ctx, x_abs, t_exp, -1, &x_abs);
      MLUBinaryOp<POW, T>(dev_ctx, y_share, t_exp, -1, &y_pow);

      MLUCnnlTensorDesc y_pow_desc(y_pow);
      MLUCnnl::Div(dev_ctx,
                   CNNL_COMPUTATION_HIGH_PRECISION,
                   y_pow_desc.get(),
                   GetBasePtr(&y_pow),
                   abs_out_desc.get(),
                   GetBasePtr(&x_abs),
                   out_desc.get(),
                   GetBasePtr(out));
    }

    MLUOpTensorKernel<T>(dev_ctx, *out, x_sign, -1, CNNL_OP_TENSOR_MUL, out);
    MLUOpTensorKernel<T>(dev_ctx, *out, dy_share, -1, CNNL_OP_TENSOR_MUL, out);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(p_norm,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::PnormKernel,
                          float,
                          phi::dtype::float16) {}

PD_REGISTER_PLUGIN_KERNEL(p_norm_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::PnormGradKernel,
                          float,
                          phi::dtype::float16) {}
