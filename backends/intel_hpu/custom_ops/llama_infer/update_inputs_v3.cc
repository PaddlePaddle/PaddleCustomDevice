// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

enum TENSOR_IDS_IN {
  STOP_FLAGS = 0,
  STEP_IDX,
  SEQ_LENS_THIS_TIME,
  SEQ_LENS_ENCODER,
  SEQ_LENS_DECODER,
  MAX_DEC_LEN,
  INPUT_IDS,
  STOP_NUMS,
  NEXT_TOKENS,
  IS_BLOCK_STEP,
  END_IDS,
  KWARGS_NEXT_TOKENS,
  NEXT_TOKENS_I32,
  KWARGS_NEXT_TOKENS_I32,
};

enum TENSOR_IDS_OUT {
  NOT_NEED_STOP_OUT = 0,
  INPUT_IDS_OUT,
  NEXT_TOKENS_OUT,
  KWARGS_NEXT_TOKENS_OUT,
  NEXT_TOKENS_OUT_I32,
  KWARGS_NEXT_TOKENS_OUT_I32,
};

class UpdateInputsV3Op : public HpuFusedOperator {
 public:
  UpdateInputsV3Op() : HpuFusedOperator("UpdateInputsV3_", false) {}

  void AddNode(ConvertTensors* ct) {
    std::vector<DIMS> inputs_dims = ct->GetDims();

    synTensor stop_flags = createTensorFromCT(ct, STOP_FLAGS, true);
    synTensor step_idx = createTensorFromCT(ct, STEP_IDX, true);
    synTensor seq_lens_this_time =
        createTensorFromCT(ct, SEQ_LENS_THIS_TIME, true);
    synTensor seq_lens_encoder = createTensorFromCT(ct, SEQ_LENS_ENCODER, true);
    synTensor seq_lens_decoder = createTensorFromCT(ct, SEQ_LENS_DECODER, true);
    synTensor max_dec_len = createTensorFromCT(ct, MAX_DEC_LEN, true);
    synTensor stop_nums = createTensorFromCT(ct, STOP_NUMS, true);
    synTensor is_block_step = createTensorFromCT(ct, IS_BLOCK_STEP, true);
    synTensor end_ids = createTensorFromCT(ct, END_IDS, true);

    synTensor not_need_stop_out =
        createTensorFromCT(ct, NOT_NEED_STOP_OUT, false);

    synSectionHandle section_input_ids = createSection();
    synTensor input_ids =
        createTensorFromCT(ct, INPUT_IDS, true, section_input_ids);
    synTensor input_ids_out =
        createTensorFromCT(ct, INPUT_IDS_OUT, false, section_input_ids);

    synSectionHandle section_next_tokens = createSection();
    synTensor next_tokens =
        createTensorFromCT(ct, NEXT_TOKENS, true, section_next_tokens);
    synTensor next_tokens_out =
        createTensorFromCT(ct, NEXT_TOKENS_OUT, false, section_next_tokens);

    synSectionHandle section_i32_next_tokens = createSection();
    synTensor next_tokens_i32 =
        createTensorFromCT(ct, NEXT_TOKENS_I32, true, section_i32_next_tokens);
    synTensor next_tokens_i32_out = createTensorFromCT(
        ct, NEXT_TOKENS_OUT_I32, false, section_i32_next_tokens);

    std::vector<synTensor> cast_next_tokens_in = {next_tokens};
    std::vector<synTensor> cast_next_tokens_out = {next_tokens_i32};
    AddNodeCast(cast_next_tokens_in,
                cast_next_tokens_out,
                "cast_i64_to_i32",
                guid_ + "cast_next_tokens_out");

    synSectionHandle section_kwargs_next_tokens = createSection();
    synTensor kwargs_next_tokens = createTensorFromCT(
        ct, KWARGS_NEXT_TOKENS, true, section_kwargs_next_tokens);
    synTensor kwargs_next_tokens_out = createTensorFromCT(
        ct, KWARGS_NEXT_TOKENS_OUT, false, section_kwargs_next_tokens);

    synSectionHandle section_i32_kwargs_next_tokens = createSection();
    synTensor kwargs_next_tokens_i32 = createTensorFromCT(
        ct, KWARGS_NEXT_TOKENS_I32, true, section_i32_kwargs_next_tokens);
    synTensor kwargs_next_tokens_i32_out = createTensorFromCT(
        ct, KWARGS_NEXT_TOKENS_OUT_I32, false, section_i32_kwargs_next_tokens);

    std::vector<synTensor> cast_kwargs_next_tokens_in = {kwargs_next_tokens};
    std::vector<synTensor> cast_kwargs_next_tokens_out = {
        kwargs_next_tokens_i32};
    AddNodeCast(cast_kwargs_next_tokens_in,
                cast_kwargs_next_tokens_out,
                "cast_i64_to_i32",
                guid_ + "cast_kwargs_next_tokens_out");

    synTensor stop_flag_now_int_out = createTensorNoPresist(
        "stop_flag_now_int_out", syn_type_int32, inputs_dims[IS_BLOCK_STEP]);

    synTensor stop_sum = createTensorNoPresist(
        "stop_sum", syn_type_int64, inputs_dims[STOP_NUMS]);

    std::vector<synTensor> ins = {stop_flags,
                                  step_idx,
                                  seq_lens_this_time,
                                  seq_lens_encoder,
                                  seq_lens_decoder,
                                  max_dec_len,
                                  next_tokens_i32,
                                  is_block_step,
                                  end_ids,
                                  kwargs_next_tokens_i32};
    std::vector<synTensor> outs = {
        stop_flag_now_int_out, next_tokens_i32_out, kwargs_next_tokens_i32_out};
    std::string node = "custom_update_inputs_v3";
    AddNode_IO(ins, outs, node, guid_ + node);

    // convert next_tokens output back
    std::vector<synTensor> cast_next_tokens_back_in = {next_tokens_i32_out};
    std::vector<synTensor> cast_next_tokens_back_out = {next_tokens_out};
    AddNodeCast(cast_next_tokens_back_in,
                cast_next_tokens_back_out,
                "cast_i32_to_i64",
                guid_ + "cast_next_tokens_back");

    // convert kwargs_next_tokens output back
    std::vector<synTensor> cast_kwargs_next_tokens_back_in = {
        kwargs_next_tokens_i32_out};
    std::vector<synTensor> cast_kwargs_next_tokens_back_out = {
        kwargs_next_tokens_out};
    AddNodeCast(cast_kwargs_next_tokens_back_in,
                cast_kwargs_next_tokens_back_out,
                "cast_i32_to_i64",
                guid_ + "cast_kwargs_next_tokens_back");

    synSliceParams params = {{0}};
    for (size_t i = 0; i < inputs_dims[INPUT_IDS].size(); i++) {
      params.axes[i] = i;
      params.steps[i] = 1;
      params.starts[i] = 0;
      params.ends[i] =
          inputs_dims[INPUT_IDS][inputs_dims[INPUT_IDS].size() - 1 - i];
    }
    params.starts[inputs_dims[INPUT_IDS].size() - 1 - 1] = 0;
    params.ends[inputs_dims[INPUT_IDS].size() - 1 - 1] = 1;

    std::vector<synTensor> set_value_in = {input_ids, next_tokens_out};
    std::vector<synTensor> set_value_out = {input_ids_out};

    AddNode_IOP<synSliceParams>(set_value_in,
                                set_value_out,
                                params,
                                "slice_insert",
                                guid_ + "slice_insert");

    ns_Reduction::Params reduce_params = {0};
    std::vector<synTensor> reduce_in = {stop_flag_now_int_out};
    std::vector<synTensor> reduce_out = {stop_sum};
    AddNodeReduceSum<int64_t>(
        reduce_in, reduce_out, reduce_params, guid_ + "reduceSum");

    std::vector<synTensor> less_in = {stop_sum, stop_nums};
    std::vector<synTensor> less_out = {not_need_stop_out};
    AddNode_IO(less_in, less_out, "less_fwd_i64", guid_ + "less_fwd_i64");
  }
};

#define INSERT_TENSOR_TO_CT(tnsr, ct, id, is_input)                           \
  {                                                                           \
    PD_CHECK(ct.GetDims(is_input).size() == id,                               \
             #tnsr "'s offset is not correct");                               \
                                                                              \
    auto tnsr##_dt = static_cast<const phi::DenseTensor*>(tnsr.impl().get()); \
    ct.Add(*tnsr##_dt, is_input);                                             \
  }

template <typename T>
void update_inputs_v3(const paddle::Tensor& stop_flags,
                      const paddle::Tensor& step_idx,
                      const paddle::Tensor& not_need_stop,
                      const paddle::Tensor& seq_lens_this_time,
                      const paddle::Tensor& seq_lens_encoder,
                      const paddle::Tensor& seq_lens_decoder,
                      const paddle::Tensor& max_dec_len,
                      const paddle::Tensor& input_ids,
                      const paddle::Tensor& stop_nums,
                      const paddle::Tensor& next_tokens,
                      const paddle::Tensor& is_block_step,
                      const paddle::Tensor& end_ids,
                      const paddle::Tensor& kwargs_next_tokens,
                      const paddle::Tensor& next_tokens_i32,
                      const paddle::Tensor& kwargs_next_tokens_i32) {
  custom_kernel::ConvertTensors ct;

  INSERT_TENSOR_TO_CT(stop_flags, ct, STOP_FLAGS, true);
  INSERT_TENSOR_TO_CT(step_idx, ct, STEP_IDX, true);
  INSERT_TENSOR_TO_CT(seq_lens_this_time, ct, SEQ_LENS_THIS_TIME, true);
  INSERT_TENSOR_TO_CT(seq_lens_encoder, ct, SEQ_LENS_ENCODER, true);
  INSERT_TENSOR_TO_CT(seq_lens_decoder, ct, SEQ_LENS_DECODER, true);
  INSERT_TENSOR_TO_CT(max_dec_len, ct, MAX_DEC_LEN, true);
  INSERT_TENSOR_TO_CT(input_ids, ct, INPUT_IDS, true);
  INSERT_TENSOR_TO_CT(stop_nums, ct, STOP_NUMS, true);
  INSERT_TENSOR_TO_CT(next_tokens, ct, NEXT_TOKENS, true);
  INSERT_TENSOR_TO_CT(is_block_step, ct, IS_BLOCK_STEP, true);
  INSERT_TENSOR_TO_CT(end_ids, ct, END_IDS, true);
  INSERT_TENSOR_TO_CT(kwargs_next_tokens, ct, KWARGS_NEXT_TOKENS, true);

  INSERT_TENSOR_TO_CT(not_need_stop, ct, NOT_NEED_STOP_OUT, false);
  INSERT_TENSOR_TO_CT(input_ids, ct, INPUT_IDS_OUT, false);
  INSERT_TENSOR_TO_CT(next_tokens, ct, NEXT_TOKENS_OUT, false);
  INSERT_TENSOR_TO_CT(kwargs_next_tokens, ct, KWARGS_NEXT_TOKENS_OUT, false);

  INSERT_TENSOR_TO_CT(next_tokens_i32, ct, NEXT_TOKENS_I32, true);
  INSERT_TENSOR_TO_CT(kwargs_next_tokens_i32, ct, KWARGS_NEXT_TOKENS_I32, true);

  INSERT_TENSOR_TO_CT(next_tokens_i32, ct, NEXT_TOKENS_OUT_I32, false);
  INSERT_TENSOR_TO_CT(
      kwargs_next_tokens_i32, ct, KWARGS_NEXT_TOKENS_OUT_I32, false);

  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          stop_flags.place()));

  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<T, nullptr_t>("UpdateInputsV3", inputs_dims, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    UpdateInputsV3Op op;

    op.AddNode(&ct);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx->stream()), tensors);
}

}  // namespace custom_kernel

void UpdateInputsV3(const paddle::Tensor& stop_flags,
                    const paddle::Tensor& step_idx,
                    const paddle::Tensor& not_need_stop,
                    const paddle::Tensor& seq_lens_this_time,
                    const paddle::Tensor& seq_lens_encoder,
                    const paddle::Tensor& seq_lens_decoder,
                    const paddle::Tensor& max_dec_len,
                    const paddle::Tensor& input_ids,
                    const paddle::Tensor& stop_nums,
                    const paddle::Tensor& next_tokens,
                    const paddle::Tensor& is_block_step,
                    const paddle::Tensor& end_ids,
                    const paddle::Tensor& kwargs_next_tokens) {
  paddle::Tensor kwargs_next_tokens_i32 =
      paddle::full(kwargs_next_tokens.shape(),
                   0,
                   phi::DataType::INT32,
                   kwargs_next_tokens.place());

  paddle::Tensor next_tokens_i32 = paddle::full(
      next_tokens.shape(), 0, phi::DataType::INT32, next_tokens.place());

  return custom_kernel::update_inputs_v3<float>(stop_flags,
                                                step_idx,
                                                not_need_stop,
                                                seq_lens_this_time,
                                                seq_lens_encoder,
                                                seq_lens_decoder,
                                                max_dec_len,
                                                input_ids,
                                                stop_nums,
                                                next_tokens,
                                                is_block_step,
                                                end_ids,
                                                kwargs_next_tokens,
                                                next_tokens_i32,
                                                kwargs_next_tokens_i32);
}

std::vector<std::vector<int64_t>> UpdateInputsV3InferShape(
    const std::vector<int64_t>& stop_flags_shape,
    const std::vector<int64_t>& step_idx_shape,
    const std::vector<int64_t>& not_need_stop_shape,
    const std::vector<int64_t>& seq_lens_this_time_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& max_dec_len_shape,
    const std::vector<int64_t>& input_ids_shape,
    const std::vector<int64_t>& stop_nums_shape,
    const std::vector<int64_t>& next_tokens_shape,
    const std::vector<int64_t>& is_block_step_shape,
    const std::vector<int64_t>& end_ids_shape,
    const std::vector<int64_t>& kwargs_next_tokens_shape) {
  return {stop_flags_shape};
}

std::vector<paddle::DataType> UpdateInputsV3InferDtype(
    const paddle::DataType& stop_flags_dtype,
    const paddle::DataType& step_idx_dtype,
    const paddle::DataType& not_need_stop_dtype,
    const paddle::DataType& seq_lens_this_time_dtype,
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& max_dec_len_dtype,
    const paddle::DataType& input_ids_dtype,
    const paddle::DataType& stop_nums_dtype,
    const paddle::DataType& next_tokens_dtype,
    const paddle::DataType& is_block_step_dtype,
    const paddle::DataType& end_ids_dtype,
    const paddle::DataType& kwargs_next_tokens_dtype) {
  return {stop_flags_dtype};
}

PD_BUILD_OP(update_inputs_v3)
    .Inputs({"stop_flags",
             "step_idx",
             "not_need_stop",
             "seq_lens_this_time",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "max_dec_len",
             "input_ids",
             "stop_nums",
             "next_tokens",
             "is_block_step",
             "end_ids",
             "kwargs_next_tokens"})
    .Outputs({"not_need_stop_out", "next_tokens_out", "kwargs_next_tokens_out"})
    .SetInplaceMap({{"not_need_stop", "not_need_stop_out"},
                    {"next_tokens", "next_tokens_out"},
                    {"kwargs_next_tokens", "kwargs_next_tokens_out"}})
    .SetKernelFn(PD_KERNEL(UpdateInputsV3))
    .SetInferShapeFn(PD_INFER_SHAPE(UpdateInputsV3InferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(UpdateInputsV3InferDtype));
