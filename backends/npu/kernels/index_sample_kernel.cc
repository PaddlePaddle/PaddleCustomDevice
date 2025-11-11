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
void ConcatKernel(const Context& dev_ctx,
                  const std::vector<const phi::DenseTensor*>& ins,
                  const phi::Scalar& axis_scalar,
                  phi::DenseTensor* out);

template <typename T, typename Context>
void GatherNdKernel(const Context& dev_ctx,
                    const phi::DenseTensor& x,
                    const phi::DenseTensor& index,
                    phi::DenseTensor* out);

template <typename T, typename Context>
void IndexSampleGather(const Context& dev_ctx,
                       const phi::DenseTensor* index,
                       const phi::DenseTensor* input,
                       phi::DenseTensor* out) {
  auto index_dims = index->dims();
  auto input_dims = input->dims();
  auto batch_size = input_dims[0];
  auto index_length = index_dims[1];
  auto stream = dev_ctx.stream();

  // NPU implementation depends on spliting input in the first dimension.
  // When the input shape is (100,1), this implementation costs lots of time
  // to Slice and Gather. It makes more sense when input shape is (1,100).
  if (input_dims[1] > input_dims[0] * 100) {
    phi::DenseTensor tmp_input;
    phi::DenseTensorMeta input_meta = {input->dtype(),
                                       phi::make_ddim({input->numel()})};
    tmp_input.set_meta(input_meta);
    dev_ctx.template Alloc<T>(&tmp_input);
    TensorCopy(dev_ctx, *input, false, &tmp_input);
    tmp_input.Resize({input->numel()});
    std::vector<phi::DenseTensor> tmp_output;
    for (auto i = 0; i < batch_size; ++i) {
      // adds
      phi::DenseTensor tmp_index, tmp_slice_t, trans_index;
      tmp_index.Resize(index->dims());
      tmp_slice_t.Resize(phi::make_ddim({index_length}));
      trans_index.Resize(index->dims());
      dev_ctx.template Alloc<int32_t>(&tmp_index);
      dev_ctx.template Alloc<int32_t>(&tmp_slice_t);
      dev_ctx.template Alloc<int32_t>(&trans_index);
      if (index->dtype() == phi::DataType::INT64) {
        const auto& cast_runner = NpuOpRunner(
            "Cast", {*index}, {trans_index}, {{"dst_type", ACL_INT32}});
        cast_runner.Run(stream);
        const auto& adds_runner =
            NpuOpRunner("Adds",
                        {trans_index},
                        {tmp_index},
                        {{"value", static_cast<float>(i * input_dims[1])}});
        adds_runner.Run(stream);
      } else {
        const auto& adds_runner =
            NpuOpRunner("Adds",
                        {*index},
                        {tmp_index},
                        {{"value", static_cast<float>(i * input_dims[1])}});
        adds_runner.Run(stream);
      }
      tmp_index.Resize({index->numel()});
      // slice
      NpuOpRunner runner1;
      runner1.SetType("Slice")
          .AddInput(tmp_index)
          .AddInput(dev_ctx, std::vector<int32_t>{i * index_length})
          .AddInput(dev_ctx, std::vector<int32_t>{index_length})
          .AddOutput(tmp_slice_t)
          .Run(stream);
      // gather
      phi::DenseTensor tmp_out_t;
      phi::DenseTensorMeta meta = {tmp_input.dtype(),
                                   phi::make_ddim({index_length})};
      tmp_out_t.set_meta(meta);
      if (tmp_input.dtype() == phi::DataType::FLOAT32) {
        dev_ctx.template Alloc<float>(&tmp_out_t);
      } else if (tmp_input.dtype() == phi::DataType::INT32) {
        dev_ctx.template Alloc<int32_t>(&tmp_out_t);
      } else if (tmp_input.dtype() == phi::DataType::INT64) {
        dev_ctx.template Alloc<int64_t>(&tmp_out_t);
      } else if (tmp_input.dtype() == phi::DataType::FLOAT16) {
        dev_ctx.template Alloc<phi::dtype::float16>(&tmp_out_t);
      }

      NpuOpRunner gather_runner;
      gather_runner.SetType("GatherV2")
          .AddInput(tmp_input)
          .AddInput(tmp_slice_t)
          .AddInput(dev_ctx, std::vector<int32_t>{0})
          .AddOutput(tmp_out_t)
          .Run(stream);
      tmp_output.push_back(tmp_out_t);
    }

    int count = 0;
    std::vector<const phi::DenseTensor*> concat_input;
    for (int i = 0; i < tmp_output.size(); ++i) {
      count += tmp_output[i].numel();
      concat_input.push_back(&tmp_output[i]);
    }

    auto dtype = concat_input[0]->dtype();
    auto out_dim = out->dims();
    phi::DenseTensorMeta meta = {dtype, phi::make_ddim({count})};
    out->set_meta(meta);

    phi::Scalar axis = 0;
    if (dtype == phi::DataType::FLOAT32) {
      custom_kernel::ConcatKernel<float, Context>(
          dev_ctx, concat_input, axis, out);
    } else if (dtype == phi::DataType::INT32) {
      custom_kernel::ConcatKernel<int32_t, Context>(
          dev_ctx, concat_input, axis, out);
    } else if (dtype == phi::DataType::INT64) {
      custom_kernel::ConcatKernel<int64_t, Context>(
          dev_ctx, concat_input, axis, out);
    } else if (dtype == phi::DataType::FLOAT16) {
      custom_kernel::ConcatKernel<phi::dtype::float16, Context>(
          dev_ctx, concat_input, axis, out);
    }
    out->Resize(out_dim);

  } else {
    // CPU implementation for index
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

    auto dtype = input->dtype();
    if (dtype == phi::DataType::FLOAT32) {
      custom_kernel::GatherNdKernel<float, Context>(
          dev_ctx, *input, gather_index, out);
    } else if (dtype == phi::DataType::INT32) {
      custom_kernel::GatherNdKernel<int32_t, Context>(
          dev_ctx, *input, gather_index, out);
    } else if (dtype == phi::DataType::INT64) {
      custom_kernel::GatherNdKernel<int64_t, Context>(
          dev_ctx, *input, gather_index, out);
    } else if (dtype == phi::DataType::FLOAT16) {
      custom_kernel::GatherNdKernel<phi::dtype::float16, Context>(
          dev_ctx, *input, gather_index, out);
    }
  }
}

template <typename T, typename Context>
void IndexSampleKernel(const Context& dev_ctx,
                       const phi::DenseTensor& x,
                       const phi::DenseTensor& index,
                       phi::DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  const auto& index_type = index.dtype();
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
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::IndexSampleKernel,
                          phi::dtype::float16,
                          float,
                          int32_t,
                          int64_t) {}

PD_REGISTER_PLUGIN_KERNEL(index_sample_grad,
                          npu,
                          ALL_LAYOUT,
                          custom_kernel::IndexSampleGradKernel,
                          phi::dtype::float16,
                          float,
                          int32_t,
                          int64_t) {}
