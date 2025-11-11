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

#include "kernels/funcs/sdaa_baseop.h"
#include "paddle/phi/extension.h"
namespace custom_kernel {

template <typename T, typename Context>
void MeanRawKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::IntArray& axes,
                   bool keep_dim,
                   bool reduce_all,
                   phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA MeanRawKernel";
  auto dims = axes.GetData();
  std::vector<int64_t> reduce_dims;
  int nDims = x.dims().size();
  auto out_dims = out->dims();
  dev_ctx.template Alloc<T>(out);

  if (nDims == 0) {
    TensorCopy(dev_ctx, x, false, out);
    out->Resize(out_dims);  // copy will reset the dims.
    return;
  }

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

  sdaa_ops::doMeanTensor(dev_ctx, x, reduce_dims, out);
}

template <typename T, typename Context>
void MeanKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& dims,
                bool keep_dim,
                phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA MeanKernel";
  bool reduce_all = false;
  if (dims.size() == 0) {
    reduce_all = true;
  }
  custom_kernel::MeanRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T, typename Context>
void MeanAllKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   phi::DenseTensor* out) {
  VLOG(4) << "Call SDAA MeanAllKernel";

  custom_kernel::MeanRawKernel<T>(
      dev_ctx, x, phi::vectorize<int64_t>(x.dims()), true, true, out);
}

template <typename T, typename Context>
void MeanAllGradKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& grad,
                       phi::DenseTensor* x_grad) {
  PADDLE_ENFORCE_EQ(grad.numel(),
                    1,
                    phi::errors::InvalidArgument(
                        "Mean Gradient Input Tensor len should be 1. But "
                        "received Out@Grad's elements num is %d.",
                        grad.numel()));
  dev_ctx.template Alloc<T>(x_grad);
  const float alpha = 1.0f / static_cast<float>(x_grad->numel());
  sdaa_ops::doExpandTensor(dev_ctx, grad, x_grad);
  sdaa_ops::doScaleTensor(dev_ctx, *x_grad, alpha, 0, true, false, x_grad);
}

template <typename T, typename Context>
void MeanGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& out_grad,
                    const phi::IntArray& dims,
                    bool keep_dim,
                    bool reduce_all,
                    phi::DenseTensor* x_grad) {
  VLOG(4) << "call sdaa mean grad kernel";
  dev_ctx.template Alloc<T>(x_grad);
  phi::DenseTensor out_grad_temp(out_grad);
  float constant = 1;
  if (reduce_all || dims.size() == 0) {
    std::vector<int64_t> out_dims(x.dims().size(), 1);
    out_grad_temp.Resize(phi::make_ddim(out_dims));
    constant = 1.f / static_cast<float>(x.numel());
  } else {
    auto reduce_dims = dims.GetData();
    auto out_dims_vec = phi::vectorize(x.dims());

    for (size_t i = 0; i < reduce_dims.size(); i++) {
      if (reduce_dims[i] < 0) {
        reduce_dims[i] += x.dims().size();
      }
      constant *= out_dims_vec[reduce_dims[i]];
      out_dims_vec[reduce_dims[i]] = 1;
    }
    constant = 1 / constant;
    out_grad_temp.Resize(phi::make_ddim(out_dims_vec));
  }
  sdaa_ops::doExpandTensor(dev_ctx, out_grad_temp, x_grad);
  sdaa_ops::doScaleTensor(dev_ctx, *x_grad, constant, 0, true, false, x_grad);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(mean_raw,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MeanRawKernel,
                          float,
                          phi::dtype::float16) {}
PD_REGISTER_PLUGIN_KERNEL(mean,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MeanKernel,
                          float,
                          phi::dtype::float16) {}
PD_REGISTER_PLUGIN_KERNEL(mean_all,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MeanAllKernel,
                          float,
                          phi::dtype::float16) {}
PD_REGISTER_PLUGIN_KERNEL(mean_all_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MeanAllGradKernel,
                          float,
                          phi::dtype::float16) {}
PD_REGISTER_PLUGIN_KERNEL(mean_grad,
                          sdaa,
                          ALL_LAYOUT,
                          custom_kernel::MeanGradKernel,
                          float,
                          phi::dtype::float16) {}
