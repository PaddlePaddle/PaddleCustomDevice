// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"
namespace custom_kernel {

bool CheckDtype(const phi::DataType& dt) {
  static std::vector<phi::DataType> tecodnn_support_dtype = {
      phi::DataType::FLOAT32,
      phi::DataType::FLOAT16,
      phi::DataType::INT32,
      phi::DataType::INT64,
      phi::DataType::FLOAT64};
  auto it =
      std::find(tecodnn_support_dtype.begin(), tecodnn_support_dtype.end(), dt);
  if (it != tecodnn_support_dtype.end()) {
    return true;
  } else {
    return false;
  }
}

template <typename T, typename Context>
void SumRawKernel(const Context& dev_ctx,
                  const phi::DenseTensor& x,
                  const phi::IntArray& axes,
                  bool keep_dim,
                  bool reduce_all,
                  phi::DataType out_dtype,
                  phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA SumRawKernel";

  auto tecodnn_support = CheckDtype(out->dtype());
  PADDLE_ENFORCE_EQ(
      tecodnn_support,
      true,
      phi::errors::InvalidArgument("The output dtype %s tecodnn don't support",
                                   out->dtype()));

  dev_ctx.Alloc(out, out->dtype());
  auto dims = axes.GetData();

  std::vector<int64_t> reduce_dims;
  int nDims = x.dims().size();
  if (reduce_all) {
    for (size_t i = 0; i < nDims; i++) {
      reduce_dims.push_back(i);
    }
  } else {
    for (size_t i = 0; i < dims.size(); ++i) {
      if (dims[i] < 0) {
        reduce_dims.push_back(dims[i] + nDims);
      } else {
        reduce_dims.push_back(dims[i]);
      }
    }
  }

  // no need to cast dtype
  if (out->dtype() == x.dtype()) {
    sdaa_ops::doSumTensor(dev_ctx, x, reduce_dims, out);
  } else {
    // cast x tensor to out dtype
    phi::DenseTensor in_t;
    in_t.Resize(x.dims());
    dev_ctx.Alloc(&in_t, out->dtype());
    sdaa_ops::doCastTensor(dev_ctx, x, &in_t);
    sdaa_ops::doSumTensor(dev_ctx, in_t, reduce_dims, out);
  }
}

template <typename T, typename Context>
void SumKernel(const Context& dev_ctx,
               const phi::DenseTensor& x,
               const phi::IntArray& dims,
               phi::DataType out_dtype,
               bool keep_dim,
               phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA SumKernel";
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
                   const phi::IntArray& dims,
                   bool keep_dim,
                   bool reduce_all,
                   phi::DenseTensor* x_grad) {
  VLOG(4) << "Call SDAA SumGradKernel";
  dev_ctx.template Alloc<T>(x_grad);
  phi::DenseTensor out_grad_temp(out_grad);
  if (reduce_all || dims.size() == 0) {
    std::vector<int64_t> out_dims(x.dims().size(), 1);
    out_grad_temp.Resize(phi::make_ddim(out_dims));
  } else if (keep_dim) {
    out_grad_temp.Resize(out_grad.dims());
  } else {
    auto reduce_dims = dims.GetData();
    auto out_dims_vec = phi::vectorize(x.dims());
    for (size_t i = 0; i < reduce_dims.size(); i++) {
      if (reduce_dims[i] < 0) {
        reduce_dims[i] += x.dims().size();
      }
      out_dims_vec[reduce_dims[i]] = 1;
    }
    out_grad_temp.Resize(phi::make_ddim(out_dims_vec));
  }
  sdaa_ops::doExpandTensor(dev_ctx, out_grad_temp, x_grad);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(sum_raw,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SumRawKernel,
                          bool,
                          float,
                          double,
                          int32_t,
                          int64_t,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(sum,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SumKernel,
                          bool,
                          float,
                          double,
                          int32_t,
                          int64_t,
                          phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::UNDEFINED);
}

PD_REGISTER_PLUGIN_KERNEL(sum_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::SumGradKernel,
                          bool,
                          float,
                          double,
                          int32_t,
                          int64_t,
                          phi::dtype::float16) {}
