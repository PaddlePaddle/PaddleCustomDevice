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

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "kernels/funcs/op_command.h"

namespace custom_kernel {

template <typename T, typename Context>
void FullKernel(const Context& dev_ctx,
                const phi::IntArray& shape,
                const phi::Scalar& val,
                phi::DataType dtype,
                phi::DenseTensor* out);

template <typename T, typename Context>
void MeanRawKernel(const Context& dev_ctx,
                   const phi::DenseTensor& x,
                   const phi::IntArray& axes,
                   bool keep_dim,
                   bool reduce_all,
                   phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  auto axes_data = axes.GetData();

  std::vector<int32_t> dims;
  if (reduce_all) {
    for (auto i = 0; i < x.dims().size(); ++i) {
      dims.push_back(i);
    }
  } else {
    for (auto i = 0; i < axes_data.size(); ++i) {
      dims.push_back(axes_data[i] < 0 ? x.dims().size() + axes_data[i]
                                      : axes_data[i]);
    }
  }

  phi::DenseTensor reduce_axes;
  experimental::OpCommandHelper::VectorToHostTensor(
      dev_ctx, dims, &reduce_axes);
  experimental::OpCommand("ReduceMean")
      .Input(x,
             experimental::TensorDescMaker("x", x).SetDataLayout(
                 phi::DataLayout::ANY))
      .Input(reduce_axes,
             experimental::TensorDescMaker("axes", reduce_axes)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Output(*out,
              experimental::TensorDescMaker("y", *out).SetDataLayout(
                  phi::DataLayout::ANY))
      .Attr("keep_dims", keep_dim)
      .Run(dev_ctx);
}

template <typename T, typename Context>
void MeanKernel(const Context& dev_ctx,
                const phi::DenseTensor& x,
                const phi::IntArray& dims,
                bool keep_dim,
                phi::DenseTensor* out) {
  bool reduce_all = false;
  custom_kernel::MeanRawKernel<T>(dev_ctx, x, dims, keep_dim, reduce_all, out);
}

template <typename T, typename Context>
void MeanGradKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& out_grad,
                    const phi::IntArray& axes,
                    bool keep_dim,
                    bool reduce_all,
                    phi::DenseTensor* x_grad) {
  auto reduce_axes = axes.GetData();

  int reduce_numel = 1;

  if (reduce_all) {
    reduce_axes.clear();
    for (int d = 0; d < x.dims().size(); ++d) {
      reduce_axes.push_back(static_cast<int>(d));
    }
  }
  for (auto& d : reduce_axes) {
    if (d < 0) {
      d += x.dims().size();
    }
    reduce_numel *= x.dims()[d];
  }

  auto out_grad_dims = x.dims();
  for (auto d : reduce_axes) {
    out_grad_dims[d] = 1;
  }

  phi::DenseTensor reciprocal_reduce_numel;
  phi::DenseTensor value;
  experimental::OpCommandHelper::ScalarToHostTensor(
      dev_ctx, static_cast<T>(1.0 / reduce_numel), &value);
  custom_kernel::FullKernel<T, Context>(dev_ctx,
                                        phi::vectorize(x.dims()),
                                        value,
                                        x.dtype(),
                                        &reciprocal_reduce_numel);

  dev_ctx.template Alloc<T>(x_grad);
  experimental::OpCommand("Mul")
      .Input(out_grad,
             experimental::TensorDescMaker("x1", out_grad)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Input(reciprocal_reduce_numel,
             experimental::TensorDescMaker("x2", reciprocal_reduce_numel)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Output(*x_grad,
              experimental::TensorDescMaker("y", *x_grad)
                  .SetDataLayout(phi::DataLayout::ANY))
      .Run(dev_ctx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(
    mean_raw, npu, ALL_LAYOUT, custom_kernel::MeanRawKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(
    mean, npu, ALL_LAYOUT, custom_kernel::MeanKernel, float) {}

PD_REGISTER_PLUGIN_KERNEL(
    mean_grad, npu, ALL_LAYOUT, custom_kernel::MeanGradKernel, float) {}
