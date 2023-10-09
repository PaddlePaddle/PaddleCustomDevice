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
  Tensor in_t, out_t;
  auto need_cast_flag =
      x.dtype() == phi::DataType::INT64 || x.dtype() == phi::DataType::BOOL
          ? true
          : false;
  if (need_cast_flag) {
    in_t.Resize(x.dims());
    out_t.Resize(out->dims());
    dev_ctx.template Alloc<int>(&in_t);
    dev_ctx.template Alloc<T>(out);
    MLUCnnlTensorDesc in_desc(x);
    MLUCnnlTensorDesc casted_in_desc(in_t);
    cnnlCastDataType_t cast_type = GetCastDataType(x.dtype(), DataType::INT32);
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  in_desc.get(),
                  GetBasePtr(&x),
                  casted_in_desc.get(),
                  GetBasePtr(&in_t));
    MLUReduceOp<int32_t>(dev_ctx,
                         in_t,
                         axes.GetData(),
                         keep_dim,
                         reduce_all,
                         "reduce_max",
                         &out_t);

    // cast back to int64 or bool
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnlTensorDesc casted_out_desc(out_t);
    cnnlCastDataType_t cast_back_type =
        GetCastDataType(DataType::INT32, x.dtype());
    MLUCnnl::Cast(dev_ctx,
                  cast_back_type,
                  casted_out_desc.get(),
                  GetBasePtr(&out_t),
                  out_desc.get(),
                  GetBasePtr(out));
  } else {
    in_t = x;
    MLUReduceOp<T>(
        dev_ctx, in_t, axes.GetData(), keep_dim, reduce_all, "reduce_max", out);
  }
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
  auto need_cast_for_int64 = x.dtype() == phi::DataType::INT64 ? true : false;
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

  Tensor tmp_x, tmp_out, tmp_out_grad, tmp_x_grad;
  if (need_cast_for_int64) {
    tmp_x.Resize(x.dims());
    tmp_out.Resize(out.dims());
    tmp_out_grad.Resize(out_grad.dims());
    tmp_x_grad.Resize(x_grad->dims());
    dev_ctx.template Alloc<int>(&tmp_x);
    dev_ctx.template Alloc<int>(&tmp_out);
    dev_ctx.template Alloc<int>(&tmp_out_grad);
    dev_ctx.template Alloc<int>(&tmp_x_grad);
    MLUCnnlTensorDesc in_desc(x);
    MLUCnnlTensorDesc casted_in_desc(tmp_x);
    MLUCnnlTensorDesc out_desc(out);
    MLUCnnlTensorDesc casted_out_desc(tmp_out);
    MLUCnnlTensorDesc out_grad_desc(out_grad);
    MLUCnnlTensorDesc casted_out_grad_desc(tmp_out_grad);
    cnnlCastDataType_t cast_type =
        GetCastDataType(DataType::INT64, DataType::INT32);
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  in_desc.get(),
                  GetBasePtr(&x),
                  casted_in_desc.get(),
                  GetBasePtr(&tmp_x));
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  out_desc.get(),
                  GetBasePtr(&out),
                  casted_out_desc.get(),
                  GetBasePtr(&tmp_out));
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  out_grad_desc.get(),
                  GetBasePtr(&out_grad),
                  casted_out_grad_desc.get(),
                  GetBasePtr(&tmp_out_grad));
  } else {
    tmp_x = x;
    tmp_out = out;
    tmp_out_grad = out_grad;
    tmp_x_grad = *x_grad;
  }
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
  if (need_cast_for_int64) {
    phi::DenseTensorMeta meta = {phi::DataType::INT32,
                                 phi::make_ddim(x_dims_vec)};
    transformed_out.set_meta(meta);
    dev_ctx.template Alloc<int>(&transformed_out);
  } else {
    phi::DenseTensorMeta meta = {x.dtype(), phi::make_ddim(x_dims_vec)};
    transformed_out.set_meta(meta);
    dev_ctx.template Alloc<T>(&transformed_out);
  }

  MLUCnnlTensorDesc tmp_out_desc(tmp_out);
  MLUCnnlTensorDesc transformed_out_desc(transformed_out);
  MLUCnnl::BroadcastTo(dev_ctx,
                       tmp_out_desc.get(),
                       GetBasePtr(&tmp_out),
                       transformed_out_desc.get(),
                       GetBasePtr(&transformed_out));

  phi::DenseTensor transformed_out_grad;
  if (need_cast_for_int64) {
    phi::DenseTensorMeta meta = {phi::DataType::INT32,
                                 phi::make_ddim(x_dims_vec)};
    transformed_out_grad.set_meta(meta);
    dev_ctx.template Alloc<int>(&transformed_out_grad);
  } else {
    phi::DenseTensorMeta meta = {x.dtype(), phi::make_ddim(x_dims_vec)};
    transformed_out_grad.set_meta(meta);
    dev_ctx.template Alloc<T>(&transformed_out_grad);
  }

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
  MLUCnnlTensorDesc x_desc(tmp_x);
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
  if (need_cast_for_int64) {
    dev_ctx.template Alloc<int>(&t_zero);
    FillMLUTensorWithHostValue(dev_ctx, static_cast<int>(0), &t_zero);
  } else {
    dev_ctx.template Alloc<T>(&t_zero);
    FillMLUTensorWithHostValue(dev_ctx, static_cast<T>(0), &t_zero);
  }

  MLUCnnlTensorDesc t_zero_desc(t_zero);
  MLUCnnlTensorDesc x_grad_desc(tmp_x_grad);
  MLUCnnl::Select(dev_ctx,
                  equal_cond_desc.get(),
                  GetBasePtr(&equal_cond),
                  transformed_out_grad_desc.get(),
                  GetBasePtr(&transformed_out_grad),
                  t_zero_desc.get(),
                  GetBasePtr(&t_zero),
                  x_grad_desc.get(),
                  GetBasePtr(&tmp_x_grad));

  if (need_cast_for_int64) {
    MLUCnnlTensorDesc casted_x_grad_desc(*x_grad);
    cnnlCastDataType_t cast_type =
        GetCastDataType(DataType::INT32, DataType::INT64);
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  x_grad_desc.get(),
                  GetBasePtr(&tmp_x_grad),
                  casted_x_grad_desc.get(),
                  GetBasePtr(x_grad));
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(max_raw,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::MaxRawKernel,
                          bool,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(max,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::MaxKernel,
                          bool,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {}

PD_REGISTER_PLUGIN_KERNEL(max_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::MaxGradKernel,
                          bool,
                          int32_t,
                          int64_t,
                          phi::dtype::float16,
                          float) {}
