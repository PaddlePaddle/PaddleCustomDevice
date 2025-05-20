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
#include "habanalabs/perf_lib_layer_params.h"
#include "kernels/funcs.h"
#include "kernels/hpu_funcs.h"
#include "kernels/hpu_operator.h"
#include "paddle/extension.h"
#include "utils/utils.h"

namespace custom_kernel {

enum TENSOR_IDS_IN { INPUT_IDS = 0, FIRST_TOKEN_IDS, PRE_IDS, NEXT_TOKENS };

enum TENSOR_IDS_OUT { INPUT_IDS_OUT = 0 };

class RecoverBlockOp : public HpuFusedOperator {
 public:
  RecoverBlockOp() : HpuFusedOperator("RecoverBlock_", false) {}

  template <typename T>
  void AddNode(ConvertTensors* ct,
               int recover_id,
               int ori_seq_len_encoder,
               int step_idx_now) {
    std::vector<DIMS> inputs_dims = ct->GetDims();
    synSliceParamsV2 params;
    // slice first_token_ids
    VLOG(6) << "RecoverBlockOp: slice first_token_ids";
    synTensor first_token_ids = createTensorFromCT(ct, FIRST_TOKEN_IDS);
    synTensor first_token_ids_out =
        createTensorNoPresist("first_token_ids_out", syn_type_int64, {1, 1});
    for (size_t i = 0; i < inputs_dims[FIRST_TOKEN_IDS].size(); i++) {
      params.axes[i] = i;
      params.steps[i] = 1;
      params.starts[i] = 0;
      params.ends[i] = inputs_dims[FIRST_TOKEN_IDS]
                                  [inputs_dims[FIRST_TOKEN_IDS].size() - 1 - i];
    }
    params.starts[0] = 0;
    params.ends[0] = 1;
    params.starts[1] = recover_id;
    params.ends[1] = recover_id + 1;
    std::vector<synTensor> slice_first_token_ids_in = {first_token_ids};
    std::vector<synTensor> slice_first_token_ids_out = {first_token_ids_out};
    AddNode_IOP<synSliceParamsV2>(slice_first_token_ids_in,
                                  slice_first_token_ids_out,
                                  params,
                                  "slice",
                                  guid_ + "slice_first_token_ids");

    // slice pre_ids
    VLOG(6) << "RecoverBlockOp: slice pre_ids";
    synTensor pre_ids = createTensorFromCT(ct, PRE_IDS);
    synTensor pre_ids_out = createTensorNoPresist(
        "pre_ids_out", syn_type_int64, {1, step_idx_now - 1});
    for (size_t i = 0; i < inputs_dims[PRE_IDS].size(); i++) {
      params.axes[i] = i;
      params.steps[i] = 1;
      params.starts[i] = 0;
      params.ends[i] =
          inputs_dims[PRE_IDS][inputs_dims[PRE_IDS].size() - 1 - i];
    }
    params.starts[0] = ori_seq_len_encoder;
    params.ends[0] = ori_seq_len_encoder + step_idx_now - 1;
    params.starts[1] = recover_id;
    params.ends[1] = recover_id + 1;
    std::vector<synTensor> slice_pre_ids_in = {pre_ids};
    std::vector<synTensor> slice_pre_ids_out = {pre_ids_out};
    AddNode_IOP<synSliceParamsV2>(slice_pre_ids_in,
                                  slice_pre_ids_out,
                                  params,
                                  "slice",
                                  guid_ + "slice_pre_ids");

    // slice next_tokens
    VLOG(6) << "RecoverBlockOp: slice next_tokens";
    synTensor next_tokens = createTensorFromCT(ct, NEXT_TOKENS);
    synTensor next_tokens_out =
        createTensorNoPresist("next_tokens_out", syn_type_int64, {1, 1});
    for (size_t i = 0; i < inputs_dims[NEXT_TOKENS].size(); i++) {
      params.axes[i] = i;
      params.steps[i] = 1;
      params.starts[i] = 0;
      params.ends[i] =
          inputs_dims[NEXT_TOKENS][inputs_dims[NEXT_TOKENS].size() - 1 - i];
    }
    params.starts[0] = 0;
    params.ends[0] = 1;
    params.starts[1] = recover_id;
    params.ends[1] = recover_id + 1;
    std::vector<synTensor> slice_next_tokens_in = {next_tokens};
    std::vector<synTensor> slice_next_tokens_out = {next_tokens_out};
    AddNode_IOP<synSliceParamsV2>(slice_next_tokens_in,
                                  slice_next_tokens_out,
                                  params,
                                  "slice",
                                  guid_ + "slice_next_tokens");

    // slice_insert input_ids
    VLOG(6) << "RecoverBlockOp: slice_insert input_ids";
    synSectionHandle section_input_ids = createSection();
    synTensor input_ids =
        createTensorFromCT(ct, INPUT_IDS, true, section_input_ids);
    synTensor input_ids_out1 = createTensorNoPresist(
        "input_ids_out1", syn_type_int64, inputs_dims[INPUT_IDS]);
    for (size_t i = 0; i < inputs_dims[INPUT_IDS].size(); i++) {
      params.axes[i] = i;
      params.steps[i] = 1;
      params.starts[i] = 0;
      params.ends[i] =
          inputs_dims[INPUT_IDS][inputs_dims[INPUT_IDS].size() - 1 - i];
    }
    params.starts[0] = 0;
    params.ends[0] = 1;
    params.starts[1] = recover_id;
    params.ends[1] = recover_id + 1;
    std::vector<synTensor> slice_insert_input_ids_in = {input_ids,
                                                        first_token_ids_out};
    std::vector<synTensor> slice_insert_input_ids_out = {input_ids_out1};
    AddNode_IOP<synSliceParamsV2>(slice_insert_input_ids_in,
                                  slice_insert_input_ids_out,
                                  params,
                                  "slice_insert",
                                  guid_ + "slice_insert_first_token_ids");

    synTensor input_ids_out2 = createTensorNoPresist(
        "input_ids_out2", syn_type_int64, inputs_dims[INPUT_IDS]);
    params.starts[0] = ori_seq_len_encoder;
    params.ends[0] = ori_seq_len_encoder + step_idx_now - 1;
    params.starts[1] = recover_id;
    params.ends[1] = recover_id + 1;
    slice_insert_input_ids_in = {input_ids_out1, pre_ids_out};
    slice_insert_input_ids_out = {input_ids_out2};
    AddNode_IOP<synSliceParamsV2>(slice_insert_input_ids_in,
                                  slice_insert_input_ids_out,
                                  params,
                                  "slice_insert",
                                  guid_ + "slice_insert_pre_ids");

    synTensor input_ids_out =
        createTensorFromCT(ct, INPUT_IDS, false, section_input_ids);
    params.starts[0] = ori_seq_len_encoder + step_idx_now - 1;
    params.ends[0] = ori_seq_len_encoder + step_idx_now;
    params.starts[1] = recover_id;
    params.ends[1] = recover_id + 1;
    slice_insert_input_ids_in = {input_ids_out2, next_tokens_out};
    slice_insert_input_ids_out = {input_ids_out};
    AddNode_IOP<synSliceParamsV2>(slice_insert_input_ids_in,
                                  slice_insert_input_ids_out,
                                  params,
                                  "slice_insert",
                                  guid_ + "slice_insert_next_tokens");
  }
};

#define INSERT_TENSOR_TO_CT(tnsr, ct, rank, id, is_input)                     \
  {                                                                           \
    auto dims = tnsr.dims();                                                  \
    int dim_size = dims.size();                                               \
                                                                              \
    PD_CHECK(dim_size == rank,                                                \
             #tnsr "'s dimensions should be " #rank " but it is      ",       \
             dim_size);                                                       \
                                                                              \
    PD_CHECK(ct.GetDims(is_input).size() == id,                               \
             #tnsr "'s offset is not correct");                               \
                                                                              \
    auto tnsr##_dt = static_cast<const phi::DenseTensor*>(tnsr.impl().get()); \
    ct.Add(*tnsr##_dt, is_input);                                             \
  }

template <typename T>
void recover_block(const paddle::Tensor& input_ids,
                   const paddle::Tensor& first_token_ids,
                   const paddle::Tensor& pre_ids,
                   const paddle::Tensor& next_tokens,
                   int recover_id,
                   int ori_seq_len_encoder,
                   int step_idx_now) {
  custom_kernel::ConvertTensors ct;

  INSERT_TENSOR_TO_CT(input_ids, ct, 2, INPUT_IDS, true);
  INSERT_TENSOR_TO_CT(first_token_ids, ct, 2, FIRST_TOKEN_IDS, true);
  INSERT_TENSOR_TO_CT(pre_ids, ct, 2, PRE_IDS, true);
  INSERT_TENSOR_TO_CT(next_tokens, ct, 2, NEXT_TOKENS, true);

  INSERT_TENSOR_TO_CT(input_ids, ct, 2, INPUT_IDS_OUT, false);

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(pre_ids.place()));

  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, nullptr_t>("RecoverBlock", inputs_dims, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    RecoverBlockOp op;

    op.AddNode<T>(&ct, recover_id, ori_seq_len_encoder, step_idx_now);

    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx->stream()), tensors);
}

}  // namespace custom_kernel

void RecoverBlock(const paddle::Tensor& input_ids,
                  const paddle::Tensor& first_token_ids,
                  const paddle::Tensor& pre_ids,
                  const paddle::Tensor& next_tokens,
                  int recover_id,
                  int ori_seq_len_encoder,
                  int step_idx_now) {
  return custom_kernel::recover_block<int64_t>(input_ids,
                                               first_token_ids,
                                               pre_ids,
                                               next_tokens,
                                               recover_id,
                                               ori_seq_len_encoder,
                                               step_idx_now);
}

PD_BUILD_OP(recover_block)
    .Inputs({"input_ids", "first_token_ids", "pre_ids", "next_tokens"})
    .Attrs({"recover_id: int", "ori_seq_len_encoder: int", "step_idx_now: int"})
    .Outputs({"input_ids_out"})
    .SetInplaceMap({{"input_ids", "input_ids_out"}})
    .SetKernelFn(PD_KERNEL(RecoverBlock));
