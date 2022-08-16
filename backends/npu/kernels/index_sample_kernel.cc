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

namespace custom_kernel {

template <typename T, typename Context>
void IndexSampleGather(const Context& dev_ctx,
                       const phi::DenseTensor* index,
                       const phi::DenseTensor* input,
                       phi::DenseTensor* out) {
  auto index_dims = index->dims();
  auto input_dims = input->dims();
  auto batch_size = input_dims[0];
  auto index_length = index_dims[1];

  std::vector<T> gather_index_vec;
  std::vector<T> index_vec;
  TensorToVector(dev_ctx, *index, dev_ctx, &index_vec);
  for (auto i = 0; i < batch_size; ++i) {
    for (auto j = 0; j < index_length; j++) {
      gather_index_vec.push_back(i);
      gather_index_vec.push_back(index_vec[i * index_length + j]);
    }
  }
  phi::DenseTensor gather_index;
  TensorFromVector(dev_ctx, gather_index_vec, dev_ctx, &gather_index);
  gather_index.Resize({batch_size, index_length, 2});

  NpuOpRunner runner;
  runner.SetType("GatherNd")
      .AddInput(*input)
      .AddInput(gather_index)
      .AddOutput(*out);
  runner.Run(dev_ctx.stream());
}

template <typename T, typename Context>
void IndexSampleKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& index,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  const auto& index_type = index.dtype();
  if (index_type == phi::DenseTensorMeta::DataType::INT32) {
    IndexSampleGather<int32_t, Context>(dev_ctx, &index, &x, out);
  } else {
    IndexSampleGather<int64_t, Context>(dev_ctx, &index, &x, out);
  }
}

template <typename T, typename Context>
void IndexSampleGradScatter(const Context& dev_ctx,
                            const phi::DenseTensor* index,
                            const phi::DenseTensor* out_grad,
                            phi::DenseTensor* x_grad) {
  auto index_dims = index->dims();
  auto input_dims = x_grad->dims();
  auto batch_size = input_dims[0];
  auto index_length = index_dims[1];

  std::vector<T> scatter_index_vec;
  std::vector<T> index_vec;
  TensorToVector(dev_ctx, *index, dev_ctx, &index_vec);
  for (auto i = 0; i < batch_size; ++i) {
    for (auto j = 0; j < index_length; j++) {
      scatter_index_vec.push_back(i);
      scatter_index_vec.push_back(index_vec[i * index_length + j]);
    }
  }
  phi::DenseTensor scatter_index;
  TensorFromVector(dev_ctx, scatter_index_vec, dev_ctx, &scatter_index);
  scatter_index.Resize({batch_size, index_length, 2});

  NpuOpRunner runner;
  runner.SetType("ScatterNd")
      .AddInput(scatter_index)
      .AddInput(*out_grad)
      .AddInput(dev_ctx, phi::vectorize<T>(x_grad->dims()))
      .AddOutput(*x_grad);
  runner.Run(dev_ctx.stream());
}

template <typename T, typename Context>
void IndexSampleGradKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const phi::DenseTensor& index,
                           const phi::DenseTensor& out_grad,
                           phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  const auto& index_type = index.dtype();
  if (index_type == phi::DenseTensorMeta::DataType::INT32) {
    IndexSampleGradScatter<int32_t, Context>(
        dev_ctx, &index, &out_grad, x_grad);
  } else {
    IndexSampleGradScatter<int64_t, Context>(
        dev_ctx, &index, &out_grad, x_grad);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(index_sample,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::IndexSampleKernel,
                          phi::dtype::float16,
                          float,
                          int32_t,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(index_sample_grad,
                          ascend,
                          ALL_LAYOUT,
                          custom_kernel::IndexSampleGradKernel,
                          phi::dtype::float16,
                          float,
                          int32_t,
                          int64_t) {}
