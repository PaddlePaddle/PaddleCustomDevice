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

std::shared_ptr<PpAscendAtbOpBase> g_atbPadOp;

std::vector<paddle::Tensor> AtbPadOp(const paddle::Tensor& tmp_out,
                                     const paddle::Tensor& padding_offset,
                                     const paddle::Tensor& seq_lens,
                                     const paddle::Tensor& input_ids) {

  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(tmp_out.place()));
  auto stream = static_cast<aclrtStream>(dev_ctx->stream());

  if (!g_atbPadOp) {
    std::cout << "Run In AtbPadOp: " << std::endl;
    g_atbPadOp.reset(new PpAscendAtbOpBase("AtbPadOp"));

    atb::Operation *op = nullptr;
    atb::infer::PadParam padParam;

    atb::CreateOperation(padParam, &op);
    g_atbPadOp->operation_.reset(op);
  }

  auto tmp_out_tensor =  static_cast<phi::DenseTensor *>(tmp_out.impl().get());
  tmp_out_tensor->Resize(phi::make_ddim({tmp_out.shape()[0] * tmp_out.shape()[1], tmp_out.shape()[2]})); // [token_num,hidden_dim]

  std::vector<const phi::DenseTensor *> inputs;
  inputs.push_back(tmp_out_tensor);
  inputs.push_back(static_cast<const phi::DenseTensor *>(padding_offset.impl().get()));
  inputs.push_back(static_cast<const phi::DenseTensor *>(seq_lens.impl().get()));
  inputs.push_back(static_cast<const phi::DenseTensor *>(input_ids.impl().get()));

  int64_t bsz = seq_lens.shape()[0];
  int64_t dim_embed = tmp_out.shape()[1];
  std::shared_ptr<phi::DenseTensor> out_tensor = std::make_shared<phi::DenseTensor>();
  out_tensor->Resize(phi::make_ddim({bsz, dim_embed}));
  dev_ctx->Alloc(out_tensor.get(),
    static_cast<const phi::DenseTensor *>(tmp_out.impl().get())->dtype());

  std::vector<const phi::DenseTensor *> outputs;
  outputs.push_back(out_tensor.get());

  g_atbPadOp->Execute(stream, inputs, outputs);

  return {paddle::Tensor(out_tensor)};
}


std::vector<std::vector<int64_t>> RebuildPaddingInferShape(const std::vector<int64_t>& tmp_out_shape,
                                                           const std::vector<int64_t>& padding_offset_shape,
                                                           const std::vector<int64_t>& seq_lens_shape,
                                                           const std::vector<int64_t>& input_ids_shape) {
  int64_t bsz = seq_lens_shape[0];
  int64_t dim_embed = tmp_out_shape[1];
  return {{bsz, dim_embed}};
}

std::vector<paddle::DataType> RebuildPaddingInferDtype(const paddle::DataType& tmp_out_dtype,
                                                       const paddle::DataType& padding_offset_dtype,
                                                       const paddle::DataType& seq_lens_dtype,
                                                       const paddle::DataType& input_ids_dtype) {
  return {tmp_out_dtype};
}

PD_BUILD_OP(rebuild_padding)
    .Inputs({"tmp_out", "padding_offset", "seq_lens", "input_ids"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(AtbPadOp))
    .SetInferShapeFn(PD_INFER_SHAPE(RebuildPaddingInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(RebuildPaddingInferDtype));
#endif
