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

#include <set>

#include "kernels/funcs/reduce_op.h"

namespace custom_kernel {

template <typename T, typename Context>
void SumRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::IntArray& axes,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DataType out_dtype,
                  phi::DenseTensor* out) {
  Tensor in_t, out_t;
  auto need_cast_for_int64 =
      x.dtype() == phi::DataType::INT64 || x.dtype() == phi::DataType::BOOL
          ? true
          : false;
  if (need_cast_for_int64) {
    in_t.Resize(x.dims());
    out_t.Resize(out->dims());
    dev_ctx.template Alloc<int>(&in_t);
    dev_ctx.template Alloc<int64_t>(
        out);  // the output of bool and int64 are int64
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
                         "reduce_sum",
                         &out_t);

    // cast back to int64
    MLUCnnlTensorDesc out_desc(*out);
    MLUCnnlTensorDesc casted_out_desc(out_t);
    cnnlCastDataType_t cast_back_type =
        GetCastDataType(DataType::INT32, DataType::INT64);
    MLUCnnl::Cast(dev_ctx,
                  cast_back_type,
                  casted_out_desc.get(),
                  GetBasePtr(&out_t),
                  out_desc.get(),
                  GetBasePtr(out));
  } else {
    in_t = x;
    MLUReduceOp<T>(
        dev_ctx, in_t, axes.GetData(), keep_dim, reduce_all, "reduce_sum", out);
  }
}

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               phi::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out) {
  bool reduce_all = false;
  if (dims.size() == 0) {
    reduce_all = true;
  }
  custom_kernel::SumRawKernel<T>(
      dev_ctx, x, dims, keep_dim, reduce_all, out_dtype, out);
}

template <typename T, typename Context>
void SumGradKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::DenseTensor& out_grad,
                   const phi::IntArray& dims_array,
                   bool keep_dim,
                   bool reduce_all,
                   phi::DenseTensor* x_grad) {
  auto reduce_dims = dims_array.GetData();
  dev_ctx.template Alloc<T>(x_grad);

  // The reduce_dims has full dim, set the reduce_all is True
  const auto& input_dim_size = x.dims().size();
  std::set<int> dims_set(reduce_dims.begin(), reduce_dims.end());
  bool full_dim = true;
  for (auto i = 0; i < input_dim_size; i++) {
    if (dims_set.find(i) == dims_set.end()) {
      full_dim = false;
      break;
    }
  }
  reduce_all = (reduce_all || reduce_dims.size() == 0 || full_dim);

  auto in_dims = phi::vectorize(x.dims());
  if (reduce_all) {
    reduce_dims.clear();
    for (size_t d = 0; d < in_dims.size(); ++d) {
      reduce_dims.push_back(static_cast<int>(d));
    }
  }
  for (auto& d : reduce_dims) {
    if (d < 0) {
      d = d + in_dims.size();
    }
  }

  Tensor tmp_out;
  if (x_grad->dtype() == out_grad.dtype()) {
    tmp_out = out_grad;
  } else {
    phi::DenseTensorMeta meta = {x_grad->dtype(), out_grad.dims()};
    tmp_out.set_meta(meta);
    dev_ctx.template Alloc<T>(&tmp_out);

    MLUCnnlTensorDesc out_grad_desc(out_grad);
    MLUCnnlTensorDesc casted_out_grad(*x_grad);
    cnnlCastDataType_t cast_type =
        GetCastDataType(out_grad.dtype(), x_grad->dtype());
    MLUCnnl::Cast(dev_ctx,
                  cast_type,
                  out_grad_desc.get(),
                  GetBasePtr(&out_grad),
                  casted_out_grad.get(),
                  GetBasePtr(&tmp_out));
  }
  auto tmp_output_dims = in_dims;
  for (auto d : reduce_dims) {
    tmp_output_dims[d] = 1;
  }
  tmp_out.Resize(phi::make_ddim(tmp_output_dims));

  MLUCnnlTensorDesc out_desc(tmp_out);
  MLUCnnlTensorDesc in_grad_desc(*x_grad);

  MLUCnnl::BroadcastTo(dev_ctx,
                       out_desc.get(),
                       GetBasePtr(&tmp_out),
                       in_grad_desc.get(),
                       GetBasePtr(x_grad));
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sum_raw,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SumRawKernel,
                          int32_t,
                          phi::dtype::float16,
                          float) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(sum,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SumKernel,
                          int32_t,
                          bool,
                          int64_t,
                          phi::dtype::float16,
                          float) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(sum_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::SumGradKernel,
                          phi::dtype::float16,
                          int32_t,
                          bool,
                          int64_t,
                          float) {}
