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

  // CPU implementation for index
  std::vector<T> gather_index_vec;
  std::vector<T> index_vec;
  // make sure device value is ready to copy
  dev_ctx.Wait();
  TensorToVector(dev_ctx, *index, dev_ctx, &index_vec);

  for (auto i = 0; i < batch_size; ++i) {
    for (auto j = 0; j < index_length; j++) {
      gather_index_vec.push_back(i);
      PADDLE_ENFORCE_GE(
          index_vec[i * index_length + j],
          0,
          phi::errors::InvalidArgument(
              "Variable value (index) of OP(index_sample) "
              "expected >= 0 and < %ld, but got %ld. Please check input "
              "value.",
              input_dims[1],
              index_vec[i * index_length + j]));
      PADDLE_ENFORCE_LT(
          index_vec[i * index_length + j],
          input_dims[1],
          phi::errors::InvalidArgument(
              "Variable value (index) of OP(index_sample) "
              "expected >= 0 and < %ld, but got %ld. Please check input "
              "value.",
              input_dims[1],
              index_vec[i * index_length + j]));
      gather_index_vec.push_back(index_vec[i * index_length + j]);
    }
  }

  phi::DenseTensor gather_index;
  TensorFromVector(dev_ctx, gather_index_vec, dev_ctx, &gather_index);
  dev_ctx.Wait();
  gather_index.Resize({batch_size, index_length, 2});

  MLUCnnlTensorDesc x_desc(*input);
  MLUCnnlTensorDesc index_desc(gather_index);
  MLUCnnlTensorDesc out_desc(*out);
  MLUCnnl::GatherNd(dev_ctx,
                    x_desc.get(),
                    GetBasePtr(input),
                    index_desc.get(),
                    GetBasePtr(&gather_index),
                    out_desc.get(),
                    GetBasePtr(out));
}

template <typename T, typename Context>
void IndexSampleKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& index,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  const auto& index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(
      index_type_match,
      true,
      phi::errors::InvalidArgument("Index holds the wrong type, it holds [%s],"
                                   "but desires to be [%s] or [%s]",
                                   index_type,
                                   phi::DataType::INT32,
                                   phi::DataType::INT64));
  if (index_type == phi::DataType::INT32) {
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
  // CPU implementation for index
  std::vector<T> scatter_index_vec;
  std::vector<T> index_vec;
  dev_ctx.Wait();
  TensorToVector(dev_ctx, *index, dev_ctx, &index_vec);
  for (auto i = 0; i < batch_size; ++i) {
    for (auto j = 0; j < index_length; j++) {
      scatter_index_vec.push_back(i);
      scatter_index_vec.push_back(index_vec[i * index_length + j]);
    }
  }
  phi::DenseTensor scatter_index;
  TensorFromVector(dev_ctx, scatter_index_vec, dev_ctx, &scatter_index);
  dev_ctx.Wait();
  scatter_index.Resize({batch_size, index_length, 2});
  cnnlScatterNdMode_t mode = CNNL_SCATTERND_ADD;

  MLUCnnlTensorDesc index_desc(scatter_index);
  MLUCnnlTensorDesc updates_desc(*out_grad);
  MLUCnnlTensorDesc out_desc(*x_grad);

  auto value = static_cast<T>(0);
  MLUCnnl::Fill(dev_ctx,
                CNNL_POINTER_MODE_HOST,
                &value,
                out_desc.get(),
                GetBasePtr(x_grad));

  MLUCnnl::ScatterNd(dev_ctx,
                     mode,
                     index_desc.get(),
                     GetBasePtr(&scatter_index),
                     updates_desc.get(),
                     GetBasePtr(out_grad),
                     out_desc.get(),
                     GetBasePtr(x_grad),
                     out_desc.get(),
                     GetBasePtr(x_grad));
}

template <typename T, typename Context>
void IndexSampleGradKernel(const Context& dev_ctx,
                           const phi::DenseTensor& x,
                           const phi::DenseTensor& index,
                           const phi::DenseTensor& out_grad,
                           phi::DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);

  const auto& index_type = index.dtype();
  bool index_type_match =
      index_type == phi::DataType::INT32 || index_type == phi::DataType::INT64;
  PADDLE_ENFORCE_EQ(
      index_type_match,
      true,
      phi::errors::InvalidArgument("Index holds the wrong type, it holds [%s],"
                                   "but desires to be [%s] or [%s]",
                                   index_type,
                                   phi::DataType::INT32,
                                   phi::DataType::INT64));
  if (index_type == phi::DataType::INT32) {
    IndexSampleGradScatter<int32_t, Context>(
        dev_ctx, &index, &out_grad, x_grad);
  } else {
    IndexSampleGradScatter<int64_t, Context>(
        dev_ctx, &index, &out_grad, x_grad);
  }
}

}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(index_sample,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::IndexSampleKernel,
                          phi::dtype::float16,
                          float,
                          int32_t,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(index_sample_grad,
                          mlu,
                          ALL_LAYOUT,
                          custom_kernel::IndexSampleGradKernel,
                          phi::dtype::float16,
                          float,
                          int32_t,
                          int64_t) {}
