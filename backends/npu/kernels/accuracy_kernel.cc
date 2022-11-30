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
void FullLikeKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::Scalar& val,
                    phi::DataType dtype,
                    phi::DenseTensor* out);

template <typename T, typename Context>
void AccuracyRawKernel(const Context& dev_ctx,
                       const phi::DenseTensor& inference,
                       const phi::DenseTensor& indices, /* int64 */
                       const phi::DenseTensor& label,   /* int64 */
                       phi::DenseTensor* accuracy,      /* float */
                       phi::DenseTensor* correct,       /* int */
                       phi::DenseTensor* total /* int */) {
  dev_ctx.template Alloc<float>(accuracy);
  dev_ctx.template Alloc<int>(correct);
  // dev_ctx.template Alloc<int>(total);

  size_t num_samples = inference.dims()[0];
  size_t class_dim = inference.dims()[1];

  if (num_samples == 0) {
    return;
  }

  phi::DenseTensor equal;
  equal.Resize({num_samples, class_dim});
  dev_ctx.template Alloc<bool>(&equal);

  experimental::OpCommand("Equal")
      .Input(indices,
             experimental::TensorDescMaker("x1")
                 .FromTensor(indices)
                 .SetDataLayout(phi::DataLayout::ANY)
                 .SetDims({num_samples, class_dim}))
      .Input(label,
             experimental::TensorDescMaker("x2")
                 .FromTensor(label)
                 .SetDataLayout(phi::DataLayout::ANY)
                 .SetDims({num_samples, 1}))
      .Output(equal,
              experimental::TensorDescMaker("y")
                  .FromTensor(equal)
                  .SetDataLayout(phi::DataLayout::ANY)
                  .SetDims({num_samples, class_dim}))
      .Run(dev_ctx);

  phi::DenseTensor equal_int;
  equal_int.Resize({num_samples, class_dim});
  dev_ctx.template Alloc<int>(&equal_int);
  experimental::OpCommand("Cast")
      .Input(equal,
             experimental::TensorDescMaker("x")
                 .FromTensor(equal)
                 .SetDataLayout(phi::DataLayout::ANY)
                 .SetDims({num_samples, class_dim}))
      .Output(equal_int,
              experimental::TensorDescMaker("y")
                  .FromTensor(equal_int)
                  .SetDataLayout(phi::DataLayout::ANY))
      .Attr(
          "dst_type",
          static_cast<int>(experimental::ConvertToNpuDtype(equal_int.dtype())))
      .Run(dev_ctx);

  phi::DenseTensor reduce_max;
  reduce_max.Resize({num_samples});
  dev_ctx.template Alloc<int>(&reduce_max);

  phi::DenseTensor reduce_max_axes;
  experimental::OpCommandHelper::VectorToHostTensor(
      dev_ctx, std::vector<int>({1}), &reduce_max_axes);
  experimental::OpCommand("ReduceMax")
      .Input(equal_int,
             experimental::TensorDescMaker("x")
                 .FromTensor(equal_int)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Input(reduce_max_axes,
             experimental::TensorDescMaker("axes")
                 .FromTensor(reduce_max_axes)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Output(reduce_max,
              experimental::TensorDescMaker("y")
                  .FromTensor(reduce_max)
                  .SetDataLayout(phi::DataLayout::ANY))
      .Attr("keep_dims", false)
      .Run(dev_ctx);

  phi::DenseTensor reduce_sum_axes;
  experimental::OpCommandHelper::VectorToHostTensor(
      dev_ctx, std::vector<int>({0}), &reduce_sum_axes);
  experimental::OpCommand("ReduceSum")
      .Input(reduce_max,
             experimental::TensorDescMaker("x")
                 .FromTensor(reduce_max)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Input(reduce_sum_axes,
             experimental::TensorDescMaker("axes")
                 .FromTensor(reduce_sum_axes)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Output(
          *correct,
          experimental::TensorDescMaker("y").FromTensor(*correct).SetDataLayout(
              phi::DataLayout::ANY))
      .Attr("keep_dims", false)
      .Run(dev_ctx);

  phi::DenseTensor total_value;
  experimental::OpCommandHelper::ScalarToHostTensor(
      dev_ctx, static_cast<int>(num_samples), &total_value);
  custom_kernel::FullLikeKernel<int, Context>(
      dev_ctx, *total, total_value, total->dtype(), total);

  phi::DenseTensor correct_float, total_float;
  correct_float.Resize({1});
  total_float.Resize({1});
  dev_ctx.template Alloc<float>(&correct_float);
  dev_ctx.template Alloc<float>(&total_float);

  experimental::OpCommand("Cast")
      .Input(
          *correct,
          experimental::TensorDescMaker("x").FromTensor(*correct).SetDataLayout(
              phi::DataLayout::ANY))
      .Output(correct_float,
              experimental::TensorDescMaker("y")
                  .FromTensor(correct_float)
                  .SetDataLayout(phi::DataLayout::ANY))
      .Attr("dst_type",
            static_cast<int>(
                experimental::ConvertToNpuDtype(correct_float.dtype())))
      .Run(dev_ctx);

  experimental::OpCommand("Cast")
      .Input(
          *total,
          experimental::TensorDescMaker("x").FromTensor(*total).SetDataLayout(
              phi::DataLayout::ANY))
      .Output(total_float,
              experimental::TensorDescMaker("y")
                  .FromTensor(total_float)
                  .SetDataLayout(phi::DataLayout::ANY))
      .Attr("dst_type",
            static_cast<int>(
                experimental::ConvertToNpuDtype(total_float.dtype())))
      .Run(dev_ctx);

  experimental::OpCommand("Div")
      .Input(correct_float,
             experimental::TensorDescMaker("x1")
                 .FromTensor(correct_float)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Input(total_float,
             experimental::TensorDescMaker("x2")
                 .FromTensor(total_float)
                 .SetDataLayout(phi::DataLayout::ANY))
      .Output(*accuracy,
              experimental::TensorDescMaker("y")
                  .FromTensor(*accuracy)
                  .SetDataLayout(phi::DataLayout::ANY))
      .Run(dev_ctx);
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(accuracy,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::AccuracyRawKernel,
                          float,
                          phi::dtype::float16,
                          int,
                          int64_t) {
  kernel->InputAt(1).SetDataType(phi::DataType::INT64);
  kernel->InputAt(2).SetDataType(phi::DataType::INT64);
}
