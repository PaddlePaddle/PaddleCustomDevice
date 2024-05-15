// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
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

#include "kernels/funcs/gcu_kernel_funcs.h"

namespace custom_kernel {
template <typename T, typename Context>
extern void GatherNdKernel(const Context &dev_ctx,
                           const phi::DenseTensor &x,
                           const phi::DenseTensor &index,
                           phi::DenseTensor *out);

template <typename T, typename Context, typename IndexT = int>
void IndexSampleGather(const Context &dev_ctx,
                       const phi::DenseTensor &input,
                       const phi::DenseTensor &index,
                       phi::DenseTensor *output) {
  auto index_dims = index.dims();
  auto input_dims = input.dims();
  auto batch_size = input_dims[0];
  auto index_length = index_dims[1];

  // CPU implementation for index
  std::vector<IndexT, PinnedAllocatorForSTL<IndexT>> gather_index_vec;
  std::vector<IndexT> index_vec;
  // make sure device value is ready to copy
  dev_ctx.Wait();
  TensorToVector(dev_ctx, index, dev_ctx, &index_vec);

  for (auto i = 0; i < batch_size; ++i) {
    for (auto j = 0; j < index_length; ++j) {
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
  //   dev_ctx.Wait();
  gather_index.Resize({batch_size, index_length, 2});
  custom_kernel::GatherNdKernel<T, Context>(
      dev_ctx, input, gather_index, output);
}

template <typename T, typename Context>
void IndexSampleKernel(const Context &dev_ctx,
                       const phi::DenseTensor &x,
                       const phi::DenseTensor &index,
                       phi::DenseTensor *out) {
  PADDLE_GCU_KERNEL_TRACE("index_sample");
  dev_ctx.template Alloc<T>(out);
  if (LaunchAOTKernel()) {
    phi::DenseTensor input_index = MaybeCreateOrTrans64To32bits(dev_ctx, index);
    bool sparse_grad = false;
    int64_t axis = (x.dims().size() > 1) ? (x.dims().size() - 1) : 0;
    LAUNCH_TOPSATENOP(
        topsatenGather, dev_ctx, *out, x, axis, input_index, sparse_grad);
    // auto index_type = index.dtype();
    // bool index_type_match = index_type == phi::DataType::INT32 ||
    //                         index_type == phi::DataType::INT64;
    // PADDLE_ENFORCE_EQ(index_type_match, true,
    //                   phi::errors::InvalidArgument(
    //                       "Input(Index) holds the wrong type, it holds %s,
    //                       but " "desires to be %s or %s",
    //                       phi::DataTypeToString(index_type).c_str(),
    //                       phi::DataTypeToString(phi::DataType::INT32).c_str(),
    //                       phi::DataTypeToString(phi::DataType::INT64).c_str()));
    // if (index_type == phi::DataType::INT32) {
    //   IndexSampleGather<T, Context, int>(dev_ctx, x, index, out);
    // } else if (index_type == phi::DataType::INT64) {
    //   IndexSampleGather<T, Context, int64_t>(dev_ctx, x, index, out);
    // }

  } else {  // kernel impl base on JIT
    THROW_JIT_UNIMPLEMENTED();
  }
}
}  // namespace custom_kernel

PD_REGISTER_PLUGIN_KERNEL(index_sample,
                          gcu,
                          ALL_LAYOUT,
                          custom_kernel::IndexSampleKernel,
                          phi::dtype::float16,
                          float,
                          int32_t) {}
