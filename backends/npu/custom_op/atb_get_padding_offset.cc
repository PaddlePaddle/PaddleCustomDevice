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
#ifdef PADDLE_WITH_ASCEND_TRANSFORMER_ACC
#include <acl/acl.h>
#include <atb/atb_infer.h>

#include "paddle/extension.h"
#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

std::shared_ptr<PpAscendAtbOpBase> g_atbUnpadOp;

std::vector<paddle::Tensor> AtbUnpadOp(const paddle::Tensor& input_ids,
                                       const paddle::Tensor& cum_offsets,
                                       const paddle::Tensor& token_num,
                                       const paddle::Tensor& seq_len) {

  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(input_ids.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  if (!g_atbUnpadOp) {
    std::cout << "Run In AtbUnpadOp: " << std::endl;
    g_atbUnpadOp.reset(new PpAscendAtbOpBase("AtbUnpadOp"));

    atb::Operation *op = nullptr;
    atb::infer::UnpadParam unpadParam;

    atb::CreateOperation(unpadParam, &op);
    g_atbUnpadOp->operation_.reset(op);
  }
  auto token_num_cpu = token_num.copy_to(paddle::CPUPlace(), true);
  auto token_num_tensor =  static_cast<phi::DenseTensor *>(token_num.impl().get());
  token_num_tensor->Resize(phi::make_ddim({1, 1})); // token_num传过来没有shape信息，手动给一下

  auto cum_offsets_tensor =  static_cast<phi::DenseTensor *>(cum_offsets.impl().get());
  cum_offsets_tensor->Resize(phi::make_ddim({cum_offsets.shape()[0], 1})); // 加速库需要的shape[xx, 1]

  std::vector<const phi::DenseTensor *> inputs;
  inputs.push_back(static_cast<const phi::DenseTensor *>(input_ids.impl().get()));
  inputs.push_back(static_cast<const phi::DenseTensor *>(cum_offsets.impl().get()));
  inputs.push_back(static_cast<const phi::DenseTensor *>(token_num.impl().get()));
  inputs.push_back(static_cast<const phi::DenseTensor *>(seq_len.impl().get()));

  int64_t bsz = input_ids.shape()[0];
  int64_t max_seq_len_in_batch = input_ids.shape()[1];
  std::shared_ptr<phi::DenseTensor> ids_remove_tensor = std::make_shared<phi::DenseTensor>();
  ids_remove_tensor->Resize(phi::make_ddim({1, bsz * max_seq_len_in_batch})); // 加速库按最大空间去infershape
  dev_ctx->Alloc(ids_remove_tensor.get(),
    static_cast<const phi::DenseTensor *>(input_ids.impl().get())->dtype());

  std::shared_ptr<phi::DenseTensor> cum_offsets_out_tensor = std::make_shared<phi::DenseTensor>();
  cum_offsets_out_tensor->Resize(phi::make_ddim({bsz, 1}));
  dev_ctx->Alloc(cum_offsets_out_tensor.get(),
    static_cast<const phi::DenseTensor *>(cum_offsets.impl().get())->dtype());

  std::shared_ptr<phi::DenseTensor> padding_offset_tensor = std::make_shared<phi::DenseTensor>();
  padding_offset_tensor->Resize(phi::make_ddim({1, bsz * max_seq_len_in_batch}));
  dev_ctx->Alloc(padding_offset_tensor.get(),
    static_cast<const phi::DenseTensor *>(cum_offsets.impl().get())->dtype());

  std::vector<const phi::DenseTensor *> outputs;
  outputs.push_back(ids_remove_tensor.get());
  outputs.push_back(cum_offsets_out_tensor.get());
  outputs.push_back(padding_offset_tensor.get());

  g_atbUnpadOp->Execute(stream, inputs, outputs, dev_ctx);

  ids_remove_tensor->Resize(phi::make_ddim({1, token_num_cpu.data<int64_t>()[0]})); // 修改ids的shape为实际token数量
  return {paddle::Tensor(ids_remove_tensor), paddle::Tensor(cum_offsets_out_tensor), paddle::Tensor(padding_offset_tensor)};
}

std::vector<std::vector<int64_t>> GetPaddingOffsetInferShape(const std::vector<int64_t>& input_ids_shape,
                                                             const std::vector<int64_t>& cum_offsets_shape,
                                                             const std::vector<int64_t>& token_num_shape,
                                                             const std::vector<int64_t>& seq_len_shape) {
  int64_t bsz = input_ids_shape[0];
  int64_t seq_len = input_ids_shape[1];
  return {{-1}, {bsz}, {-1}};
}

std::vector<paddle::DataType> GetPaddingOffsetInferDtype(const paddle::DataType& input_ids_dtype,
                                                         const paddle::DataType& cum_offsets_dtype,
                                                         const paddle::DataType& token_num_dtype,
                                                         const paddle::DataType& seq_len_dtype) {
  return {input_ids_dtype, seq_len_dtype, seq_len_dtype};
}

PD_BUILD_OP(get_padding_offset)
    .Inputs({"input_ids", "cum_offsets", "token_num", "seq_len"})
    .Outputs({"x_remove_padding", "cum_offsets_out", "padding_offset"})
    .SetKernelFn(PD_KERNEL(AtbUnpadOp))
    .SetInferShapeFn(PD_INFER_SHAPE(GetPaddingOffsetInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetPaddingOffsetInferDtype));
#endif
