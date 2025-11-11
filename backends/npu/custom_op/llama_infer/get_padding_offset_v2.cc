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

#include <iostream>
#include <vector>

#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"
#include "paddle/extension.h"

std::vector<paddle::Tensor> GetPaddingOffsetV2(
    const paddle::Tensor& input_ids,
    const paddle::Tensor& cum_offsets,
    const paddle::Tensor& token_num,
    const paddle::Tensor& seq_len) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          input_ids.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  std::vector<int64_t> input_ids_shape = input_ids.shape();
  const int bsz = input_ids.shape()[0];
  const int seq_length = input_ids_shape[1];
  auto cpu_token_num = token_num.copy_to(paddle::CPUPlace(), true);

  const int token_num_data = cpu_token_num.data<int64_t>()[0];

  auto input_ids_tensor =
      static_cast<const phi::DenseTensor*>(input_ids.impl().get());
  auto cum_offsets_tensor =
      static_cast<const phi::DenseTensor*>(cum_offsets.impl().get());
  auto token_num_tensor =
      static_cast<phi::DenseTensor*>(token_num.impl().get());
  auto seq_len_tensor =
      static_cast<const phi::DenseTensor*>(seq_len.impl().get());
  token_num_tensor->Resize(phi::make_ddim({1, 1}));

  std::shared_ptr<phi::DenseTensor> x_remove_padding_out_tensor =
      std::make_shared<phi::DenseTensor>();
  x_remove_padding_out_tensor->Resize(phi::make_ddim({bsz * seq_length}));
  dev_ctx->Alloc(x_remove_padding_out_tensor.get(), paddle::DataType::INT64);

  std::shared_ptr<phi::DenseTensor> cum_offsets_out_tensor =
      std::make_shared<phi::DenseTensor>();
  cum_offsets_out_tensor->Resize(cum_offsets_tensor->dims());
  dev_ctx->Alloc(cum_offsets_out_tensor.get(), cum_offsets_tensor->dtype());

  std::shared_ptr<phi::DenseTensor> padding_offset_tensor =
      std::make_shared<phi::DenseTensor>();
  padding_offset_tensor->Resize(phi::make_ddim({bsz * seq_length}));
  dev_ctx->Alloc(padding_offset_tensor.get(), paddle::DataType::INT32);

  std::shared_ptr<phi::DenseTensor> cu_seqlens_q_tensor =
      std::make_shared<phi::DenseTensor>();
  cu_seqlens_q_tensor->Resize(phi::make_ddim({bsz + 1}));
  dev_ctx->Alloc(cu_seqlens_q_tensor.get(), paddle::DataType::INT32);

  std::shared_ptr<phi::DenseTensor> cu_seqlens_k_tensor =
      std::make_shared<phi::DenseTensor>();
  cu_seqlens_k_tensor->Resize(phi::make_ddim({bsz + 1}));
  dev_ctx->Alloc(cu_seqlens_k_tensor.get(), paddle::DataType::INT32);

  std::vector<phi::DenseTensor> inputs = {
      *input_ids_tensor,
      *cum_offsets_tensor,
      *token_num_tensor,
      *seq_len_tensor,
  };

  std::vector<phi::DenseTensor> outputs = {*x_remove_padding_out_tensor,
                                           *cum_offsets_out_tensor,
                                           *padding_offset_tensor,
                                           *cu_seqlens_q_tensor,
                                           *cu_seqlens_k_tensor};

  const auto& runner = NpuOpRunner("GetPaddingOffset", inputs, outputs);
  runner.Run(stream);

  x_remove_padding_out_tensor->Resize(phi::make_ddim({token_num_data}));

  return {paddle::Tensor(x_remove_padding_out_tensor),
          paddle::Tensor(cum_offsets_out_tensor),
          paddle::Tensor(padding_offset_tensor),
          paddle::Tensor(cu_seqlens_q_tensor),
          paddle::Tensor(cu_seqlens_k_tensor)};
}

std::vector<std::vector<int64_t>> GetPaddingOffsetV2InferShape(
    const std::vector<int64_t>& input_ids_shape,
    const std::vector<int64_t>& cum_offsets_shape,
    const std::vector<int64_t>& token_num_shape,
    const std::vector<int64_t>& seq_len_shape) {
  int64_t bsz = seq_len_shape[0];
  int64_t seq_len = input_ids_shape[1];
  return {{-1}, {bsz}, {-1}, {bsz + 1}, {bsz + 1}};
}

std::vector<paddle::DataType> GetPaddingOffsetV2InferDtype(
    const paddle::DataType& input_ids_dtype,
    const paddle::DataType& cum_offsets_dtype,
    const paddle::DataType& token_num_dtype,
    const paddle::DataType& seq_len_dtype) {
  return {input_ids_dtype,
          seq_len_dtype,
          seq_len_dtype,
          seq_len_dtype,
          seq_len_dtype};
}

PD_BUILD_OP(get_padding_offset_v2)
    .Inputs({"input_ids", "cum_offsets", "token_num", "seq_len"})
    .Outputs({"x_remove_padding",
              "cum_offsets_out",
              "padding_offset",
              "cu_seqlens_q",
              "cu_seqlens_k"})
    .SetKernelFn(PD_KERNEL(GetPaddingOffsetV2))
    .SetInferShapeFn(PD_INFER_SHAPE(GetPaddingOffsetV2InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetPaddingOffsetV2InferDtype));
