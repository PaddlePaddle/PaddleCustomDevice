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
#include <dlfcn.h>
#include <omp.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/time.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "habanalabs/perf_lib_layer_params.h"
#include "habanalabs/synapse_api.h"
#include "habanalabs/synapse_common_types.h"
#include "kernels/funcs.h"
#include "kernels/hpu_operator.h"
#include "paddle/extension.h"
#include "utils/utils.h"

namespace custom_kernel {
class PostProcessOp : public HpuOperator {
 public:
  explicit PostProcessOp(const std::string& guid) : HpuOperator(guid) {}
  synTensor createTensorFromCT(ConvertTensors& ct,
                               int idx,
                               std::string name,
                               bool is_input = true,
                               synSectionHandle section = nullptr) {
    if (is_input) {
      auto inputs = ct.GetTensors();
      synTensor t = createTensor(inputs[idx].dims.size(),
                                 inputs[idx].type,
                                 inputs[idx].dims,
                                 true,
                                 name.c_str(),
                                 section);
      return t;
    }

    auto outputs = ct.GetTensors(false);
    synTensor t = createTensor(outputs[idx].dims.size(),
                               outputs[idx].type,
                               outputs[idx].dims,
                               true,
                               name.c_str(),
                               section);
    return t;
  }

  synTensor createTensorNoPresist(std::string name,
                                  synDataType dtype,
                                  std::vector<int64_t> dims,
                                  synSectionHandle section = nullptr) {
    synTensor t =
        createTensor(dims.size(), dtype, dims, false, name.c_str(), section);
    return t;
  }

  void AddCastNode(synTensor x, synTensor y, std::string node_guid) {
    synTensor syn_inputs[1] = {x};
    synTensor syn_outputs[1] = {y};
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     1,
                                     1,
                                     nullptr,
                                     0,
                                     node_guid.c_str(),
                                     node_guid.c_str(),
                                     nullptr,
                                     nullptr);
    std::string failure_info = "[RUNTIME]";
    failure_info += guid_.c_str();
    failure_info += "::";
    failure_info += node_guid.c_str();
    failure_info += "  synNodeCreate() failed = ";
    PD_CHECK(status == synSuccess, failure_info.c_str(), status);
  }
};
}  // namespace custom_kernel

namespace custom_kernel {
class GetStopFlagsMultiOp : public PostProcessOp {
 public:
  GetStopFlagsMultiOp() : PostProcessOp("set_value_by_flags") {}

  void AddEqualNode(synTensor x, synTensor y, synTensor out) {
    synTensor syn_inputs[2] = {x, y};
    synTensor syn_outputs[1] = {out};

    std::string node_guid = "equal_fwd_i64";
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     2,
                                     1,
                                     nullptr,
                                     0,
                                     node_guid.c_str(),
                                     node_guid.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] GetStopFlagsMultiOp::Where synNodeCreate () failed = ",
             status);
  }

  void AddOrNode(synTensor x, synTensor y, synTensor out) {
    synTensor syn_inputs[2] = {x, y};
    synTensor syn_outputs[1] = {out};

    std::string node_guid = "or_fwd_i8";
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     2,
                                     1,
                                     nullptr,
                                     0,
                                     node_guid.c_str(),
                                     node_guid.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] GetStopFlagsMultiOp::Where synNodeCreate () failed = ",
             status);
  }

  void AddWhereNode(synTensor condition,
                    synTensor x,
                    synTensor y,
                    synTensor out) {
    synTensor syn_inputs[3] = {condition, x, y};
    synTensor syn_outputs[1] = {out};

    std::string node_guid = "where_fwd_i64";
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     3,
                                     1,
                                     nullptr,
                                     0,
                                     node_guid.c_str(),
                                     node_guid.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] GetStopFlagsMultiOp::Where synNodeCreate () failed = ",
             status);
  }
};
}  // namespace custom_kernel

std::vector<paddle::Tensor> GetStopFlagsMulti(const paddle::Tensor& topk_ids,
                                              const paddle::Tensor& stop_flags,
                                              const paddle::Tensor& end_ids,
                                              int64_t mode) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          topk_ids.place()));

  PD_CHECK(topk_ids.dtype() == paddle::DataType::INT64);
  PD_CHECK(stop_flags.dtype() == paddle::DataType::BOOL);
  PD_CHECK(end_ids.dtype() == paddle::DataType::INT64);

  auto end_ids_dt = static_cast<const phi::DenseTensor*>(end_ids.impl().get());
  auto topk_ids_out = topk_ids.copy_to(topk_ids.place(), true);
  auto topk_ids_out_dt =
      static_cast<const phi::DenseTensor*>(topk_ids_out.impl().get());
  auto stop_flags_out = stop_flags.copy_to(stop_flags.place(), true);
  auto stop_flags_out_dt =
      static_cast<const phi::DenseTensor*>(stop_flags_out.impl().get());

  custom_kernel::ConvertTensors ct;
  ct.Add(*topk_ids_out_dt);
  ct.Add(*stop_flags_out_dt);
  ct.Add(*end_ids_dt);

  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<int32_t, nullptr_t>(
      "TokenPenaltyMultiScores", inputs_dims, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    custom_kernel::GetStopFlagsMultiOp op;

    synSectionHandle section_shared0 = op.createSection();
    synSectionHandle section_shared1 = op.createSection();
    synSectionHandle section_shared2 = op.createSection();

    synTensor topk_in =
        op.createTensorFromCT(ct, 0, "topk_in", true, section_shared0);
    synTensor end_in =
        op.createTensorFromCT(ct, 2, "end_in", true, section_shared2);
    synTensor flag_in =
        op.createTensorFromCT(ct, 1, "flag_in", true, section_shared1);
    synTensor topk_out =
        op.createTensorFromCT(ct, 0, "topk_out", true, section_shared0);
    op.AddWhereNode(flag_in, end_in, topk_in, topk_out);

    synTensor equal =
        op.createTensorNoPresist("equal", syn_type_int8, inputs_dims[1]);
    op.AddEqualNode(topk_out, end_in, equal);

    synTensor flag_out =
        op.createTensorFromCT(ct, 1, "flag_out", true, section_shared1);
    op.AddOrNode(equal, flag_in, flag_out);

    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors;
  tensors["topk_in"] =
      reinterpret_cast<uint64_t>(topk_ids_out_dt->data<int64_t>());
  tensors["topk_out"] =
      reinterpret_cast<uint64_t>(topk_ids_out_dt->data<int64_t>());
  tensors["flag_in"] =
      reinterpret_cast<uint64_t>(stop_flags_out_dt->data<bool>());
  tensors["flag_out"] =
      reinterpret_cast<uint64_t>(stop_flags_out_dt->data<bool>());
  tensors["end_in"] = reinterpret_cast<uint64_t>(end_ids_dt->data<int64_t>());

  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx->stream()), tensors);

  return {paddle::Tensor(topk_ids_out), paddle::Tensor(stop_flags_out)};
}

std::vector<std::vector<int64_t>> GetStopFlagsMultiInferShape(
    const std::vector<int64_t>& topk_ids_shape,
    const std::vector<int64_t>& stop_flags_shape,
    const std::vector<int64_t>& end_ids_shape) {
  return {topk_ids_shape, stop_flags_shape};
}

std::vector<paddle::DataType> GetStopFlagsMultiInferDtype(
    const paddle::DataType& topk_ids_dtype,
    const paddle::DataType& stop_flags_dtype,
    const paddle::DataType& end_ids_dtype) {
  return {topk_ids_dtype, stop_flags_dtype};
}

PD_BUILD_OP(set_stop_value_multi_ends)
    .Inputs({"topk_ids", "stop_flags", "end_ids"})
    .Attrs({"mode: int64_t"})
    .Outputs({"topk_ids_out", "stop_flags_out"})
    .SetKernelFn(PD_KERNEL(GetStopFlagsMulti))
    .SetInferShapeFn(PD_INFER_SHAPE(GetStopFlagsMultiInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(GetStopFlagsMultiInferDtype));

namespace custom_kernel {
class SetValueByFlagAndIdxOp : public PostProcessOp {
 public:
  SetValueByFlagAndIdxOp() : PostProcessOp("custom_set_value_bfai") {}

  void AddSetValueBFAINode(synTensor* inputs, synTensor* outputs) {
    synStatus status = synNodeCreate(graphHandle_,
                                     inputs,
                                     outputs,
                                     4,
                                     1,
                                     nullptr,
                                     0,
                                     guid_.c_str(),
                                     guid_.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] SetValue::AddSetValue synNodeCreate() failed = ",
             status);
  }
};

}  // namespace custom_kernel

std::vector<paddle::Tensor> SetValueByFlagsAndIdx(
    const paddle::Tensor& pre_ids_all,
    const paddle::Tensor& pre_ids_now,
    const paddle::Tensor& step_idx,
    const paddle::Tensor& stop_flags) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          pre_ids_all.place()));

  auto pre_ids_now_dt =
      static_cast<const phi::DenseTensor*>(pre_ids_now.impl().get());
  auto step_idx_dt =
      static_cast<const phi::DenseTensor*>(step_idx.impl().get());
  auto stop_flags_dt =
      static_cast<const phi::DenseTensor*>(stop_flags.impl().get());
  auto pre_ids_all_dt =
      static_cast<const phi::DenseTensor*>(pre_ids_all.impl().get());

  phi::DenseTensor out(*pre_ids_all_dt);

  custom_kernel::ConvertTensors ct;
  ct.Add(*pre_ids_all_dt);
  ct.Add(*pre_ids_now_dt);
  ct.Add(*step_idx_dt);
  ct.Add(*stop_flags_dt);
  ct.Add(out, false);

  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<int32_t, nullptr_t>(
      "SetValueByFlagsAndIdx", inputs_dims, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    custom_kernel::SetValueByFlagAndIdxOp op;

    synSectionHandle section_shared = op.createSection();
    synTensor pre_ids_all_in =
        op.createTensorFromCT(ct, 0, "pre_ids_all_in", true, section_shared);
    synTensor pre_ids_all_i32 = op.createTensorNoPresist(
        "pre_ids_all_i32", syn_type_int32, inputs_dims[0]);
    op.AddCastNode(pre_ids_all_in, pre_ids_all_i32, "cast_i64_to_i32");

    synTensor pre_ids_now_in = op.createTensorFromCT(ct, 1, "pre_ids_now_in");
    synTensor pre_ids_now_i32 = op.createTensorNoPresist(
        "pre_ids_now_i32", syn_type_int32, inputs_dims[1]);
    op.AddCastNode(pre_ids_now_in, pre_ids_now_i32, "cast_i64_to_i32");

    synTensor step_idx_in = op.createTensorFromCT(ct, 2, "step_idx_in");
    synTensor step_idx_i32 = op.createTensorNoPresist(
        "step_idx_i32", syn_type_int32, inputs_dims[2]);
    op.AddCastNode(step_idx_in, step_idx_i32, "cast_i64_to_i32");

    synTensor stop_flags_in = op.createTensorFromCT(ct, 3, "stop_flags_in");
    synTensor out_i32 =
        op.createTensorNoPresist("out_i32", syn_type_int32, inputs_dims[0]);
    synTensor inputs[] = {
        pre_ids_all_i32, pre_ids_now_i32, step_idx_i32, stop_flags_in};
    synTensor outputs[] = {out_i32};
    op.AddSetValueBFAINode(inputs, outputs);

    synTensor out_i64 =
        op.createTensorFromCT(ct, 0, "pre_ids_all_out", false, section_shared);
    op.AddCastNode(out_i32, out_i64, "cast_i32_to_i64");

    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors;
  tensors["pre_ids_all_in"] =
      reinterpret_cast<uint64_t>(pre_ids_all_dt->data<int64_t>());
  tensors["pre_ids_now_in"] =
      reinterpret_cast<uint64_t>(pre_ids_now_dt->data<int64_t>());
  tensors["step_idx_in"] =
      reinterpret_cast<uint64_t>(step_idx_dt->data<int64_t>());
  tensors["stop_flags_in"] =
      reinterpret_cast<uint64_t>(stop_flags_dt->data<bool>());
  tensors["pre_ids_all_out"] = reinterpret_cast<uint64_t>(out.data<int64_t>());

  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx->stream()), tensors);

  return {stop_flags};
}

std::vector<std::vector<int64_t>> SetValueByFlagsAndIdxInferShape(
    const std::vector<int64_t>& pre_ids_all_shape,
    const std::vector<int64_t>& pre_ids_now_shape,
    const std::vector<int64_t>& step_idx_shape,
    const std::vector<int64_t>& stop_flags_shape) {
  return {stop_flags_shape};
}

std::vector<paddle::DataType> SetValueByFlagsAndIdxInferDtype(
    const paddle::DataType& pre_ids_all_dtype,
    const paddle::DataType& pre_ids_now_dtype,
    const paddle::DataType& step_idx_dtype,
    const paddle::DataType& stop_flags_dtype) {
  return {stop_flags_dtype};
}

PD_BUILD_OP(set_value_by_flags_and_idx)
    .Inputs({"pre_ids_all", "pre_ids_now", "step_idx", "stop_flags"})
    .Outputs({"stop_flags_out"})
    .SetKernelFn(PD_KERNEL(SetValueByFlagsAndIdx))
    .SetInferShapeFn(PD_INFER_SHAPE(SetValueByFlagsAndIdxInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SetValueByFlagsAndIdxInferDtype));

namespace custom_kernel {
class TokenPenaltyMultiScoresOp : public PostProcessOp {
 public:
  TokenPenaltyMultiScoresOp() : PostProcessOp("custom_logits_by_repeats") {}

  void AddFullNode(synTensor t, float value) {
    ns_ConstantKernel::Params params;
    params.constant.f = value;
    std::string node_guid = "constant_f32";
    synTensor syn_outputs[1] = {t};
    synStatus status = synNodeCreate(graphHandle_,
                                     nullptr,
                                     syn_outputs,
                                     0,
                                     1,
                                     &params,
                                     sizeof(params),
                                     node_guid.c_str(),
                                     node_guid.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess,
        "[RUNTIME] :TokenPenaltyMultiScores:Full synNodeCreate() failed =  ",
        status);
  }

  void AddLogitsByRepeatsNode(synTensor repeats,
                              synTensor logits,
                              synTensor penalty,
                              synTensor frequency,
                              synTensor presence,
                              synTensor logits_out) {
    synTensor syn_inputs[] = {repeats, logits, penalty, frequency, presence};
    synTensor syn_outputs[] = {logits_out};
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     5,
                                     1,
                                     nullptr,
                                     0,
                                     guid_.c_str(),
                                     guid_.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] TokenPenaltyMultiScores::LogitsByRepeats "
             "synNodeCreate() failed = ",
             status);
  }

  void AddMinLengthLogitsNode(synTensor logits,
                              synTensor cur_lens,
                              synTensor min_lens,
                              synTensor eos,
                              synTensor logits_no_eos) {
    synTensor syn_inputs[] = {logits, cur_lens, min_lens, eos};
    synTensor syn_outputs[] = {logits_no_eos};
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     4,
                                     1,
                                     nullptr,
                                     0,
                                     "custom_logits_wo_eos",
                                     "custom_logits_wo_eos",
                                     nullptr,
                                     nullptr);
    PD_CHECK(status == synSuccess,
             "[RUNTIME] TokenPenaltyMultiScores::MinLengthLogits "
             "synNodeCreate() failed = ",
             status);
  }

  void AddScatterAddNode(synTensor x,
                         synTensor index,
                         synTensor update,
                         synTensor output) {
    synTensor syn_inputs[3] = {x, index, update};
    synTensor syn_outputs[1] = {output};

    ns_ScatterKernel::Params params;
    params.axis = 0;

    std::string node_guid = "unsorted_scatter_add_fwd_f32";
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     3,
                                     1,
                                     &params,
                                     sizeof(params),
                                     node_guid.c_str(),
                                     node_guid.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess,
        "[RUNTIME] TokenPenaltyMultiScores::reshape synNodeCreate () failed = ",
        status);
  }

  void AddWrapNode(synTensor x, synTensor y) {
    synTensor syn_inputs[1] = {x};
    synTensor syn_outputs[1] = {y};

    std::string node_guid = "custom_wrap_update";
    synStatus status = synNodeCreate(graphHandle_,
                                     syn_inputs,
                                     syn_outputs,
                                     1,
                                     1,
                                     nullptr,
                                     0,
                                     node_guid.c_str(),
                                     node_guid.c_str(),
                                     nullptr,
                                     nullptr);
    PD_CHECK(
        status == synSuccess,
        "[RUNTIME] TokenPenaltyMultiScores::Wrap synNodeCreate() failed = ",
        status);
  }
};

}  // namespace custom_kernel

std::vector<paddle::Tensor> TokenPenaltyMultiScores(
    const paddle::Tensor& pre_ids,
    const paddle::Tensor& logits,
    const paddle::Tensor& penalty_scores,
    const paddle::Tensor& frequency_scores,
    const paddle::Tensor& presence_scores,
    const paddle::Tensor& cur_len,
    const paddle::Tensor& min_len,
    const paddle::Tensor& eos_token_id) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(pre_ids.place()));

  auto pre_ids_dt = static_cast<const phi::DenseTensor*>(pre_ids.impl().get());
  auto penalty_dt =
      static_cast<const phi::DenseTensor*>(penalty_scores.impl().get());
  auto frequency_dt =
      static_cast<const phi::DenseTensor*>(frequency_scores.impl().get());
  auto presence_dt =
      static_cast<const phi::DenseTensor*>(presence_scores.impl().get());
  auto cur_len_dt = static_cast<const phi::DenseTensor*>(cur_len.impl().get());
  auto min_len_dt = static_cast<const phi::DenseTensor*>(min_len.impl().get());
  auto eos_token_id_dt =
      static_cast<const phi::DenseTensor*>(eos_token_id.impl().get());

  auto output = logits.copy_to(logits.place(), true);
  auto logits_out_dt =
      static_cast<const phi::DenseTensor*>(output.impl().get());

  enum TENSOR_IDS {
    PRE_IDS = 0,
    LOGITS,
    PENALTY,
    FREQUENCY,
    PRESENCE,
    CUR_LEN,
    MIN_LEN,
    EOS_TOKEN
  };

  custom_kernel::ConvertTensors ct;
  ct.Add(*pre_ids_dt);
  ct.Add(*logits_out_dt);
  ct.Add(*penalty_dt);
  ct.Add(*frequency_dt);
  ct.Add(*presence_dt);
  ct.Add(*cur_len_dt);
  ct.Add(*min_len_dt);
  ct.Add(*eos_token_id_dt);

  std::vector<DIMS> inputs_dims = ct.GetDims();
  OpCacheOperator op_info;
  op_info.prepareOpInfo<int32_t, nullptr_t>(
      "TokenPenaltyMultiScores", inputs_dims, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    custom_kernel::TokenPenaltyMultiScoresOp op;

    synSectionHandle section_shared = op.createSection();
    synTensor logits_in =
        op.createTensorFromCT(ct, LOGITS, "logits_in", true, section_shared);

    synTensor cur_len_in =
        op.createTensorFromCT(ct, CUR_LEN, "cur_len_in", true);
    synTensor cur_len_i32 = op.createTensorNoPresist(
        "cur_len_i32", syn_type_int32, inputs_dims[CUR_LEN]);
    op.AddCastNode(cur_len_in, cur_len_i32, "cast_i64_to_i32");

    synTensor min_len_in =
        op.createTensorFromCT(ct, MIN_LEN, "min_len_in", true);
    synTensor min_len_i32 = op.createTensorNoPresist(
        "min_len_i32", syn_type_int32, inputs_dims[MIN_LEN]);
    op.AddCastNode(min_len_in, min_len_i32, "cast_i64_to_i32");

    synTensor eos_in = op.createTensorFromCT(ct, EOS_TOKEN, "eos_in", true);
    synTensor eos_i32 = op.createTensorNoPresist(
        "eos_i32", syn_type_int32, inputs_dims[EOS_TOKEN]);
    op.AddCastNode(eos_in, eos_i32, "cast_i64_to_i32");

    synTensor logits_mark_eos = op.createTensorNoPresist(
        "logits_mark_eos", syn_type_float, inputs_dims[LOGITS]);
    op.AddMinLengthLogitsNode(
        logits_in, cur_len_i32, min_len_i32, eos_i32, logits_mark_eos);
    synTensor pre_ids_in =
        op.createTensorFromCT(ct, PRE_IDS, "pre_ids_in", true);
    synTensor pre_ids_i32 = op.createTensorNoPresist(
        "pre_ids_i32", syn_type_int32, inputs_dims[PRE_IDS]);
    op.AddCastNode(pre_ids_in, pre_ids_i32, "cast_i64_to_i32");

    synTensor update = op.createTensorNoPresist(
        "update", syn_type_float, inputs_dims[PRE_IDS]);
    op.AddWrapNode(pre_ids_i32, update);

    synTensor zeros =
        op.createTensorNoPresist("zeros", syn_type_float, inputs_dims[LOGITS]);
    op.AddFullNode(zeros, 0.0f);

    synTensor pre_ids_times = op.createTensorNoPresist(
        "pre_ids_times", syn_type_float, inputs_dims[LOGITS]);
    op.AddScatterAddNode(zeros, pre_ids_i32, update, pre_ids_times);
    synTensor penalty_in =
        op.createTensorFromCT(ct, PENALTY, "penalty_in", true);
    synTensor frequency_in =
        op.createTensorFromCT(ct, FREQUENCY, "frequency_in", true);
    synTensor presence_in =
        op.createTensorFromCT(ct, PRESENCE, "presence_in", true);

    synTensor logits_out =
        op.createTensorFromCT(ct, LOGITS, "logits_out", true, section_shared);

    op.AddLogitsByRepeatsNode(pre_ids_times,
                              logits_mark_eos,
                              penalty_in,
                              frequency_in,
                              presence_in,
                              logits_out);

    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors;
  tensors["pre_ids_in"] =
      reinterpret_cast<uint64_t>(pre_ids_dt->data<int64_t>());
  tensors["logits_in"] =
      reinterpret_cast<uint64_t>(logits_out_dt->data<float>());
  tensors["penalty_in"] = reinterpret_cast<uint64_t>(penalty_dt->data<float>());
  tensors["frequency_in"] =
      reinterpret_cast<uint64_t>(frequency_dt->data<float>());
  tensors["presence_in"] =
      reinterpret_cast<uint64_t>(presence_dt->data<float>());
  tensors["cur_len_in"] =
      reinterpret_cast<uint64_t>(cur_len_dt->data<int64_t>());
  tensors["min_len_in"] =
      reinterpret_cast<uint64_t>(min_len_dt->data<int64_t>());
  tensors["eos_in"] =
      reinterpret_cast<uint64_t>(eos_token_id_dt->data<int64_t>());
  tensors["logits_out"] =
      reinterpret_cast<uint64_t>(logits_out_dt->data<float>());

  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx->stream()), tensors);

  return {paddle::Tensor(output)};
}

std::vector<std::vector<int64_t>> TokenPenaltyMultiScoresInferShape(
    const std::vector<int64_t>& pre_ids_shape,
    const std::vector<int64_t>& logits_shape,
    const std::vector<int64_t>& penalty_scores_shape,
    const std::vector<int64_t>& frequency_scores_shape,
    const std::vector<int64_t>& presence_scores_shape,
    const std::vector<int64_t>& cur_len_shape,
    const std::vector<int64_t>& min_len_shape,
    const std::vector<int64_t>& eos_token_id_shape) {
  return {logits_shape};
}

std::vector<paddle::DataType> TokenPenaltyMultiScoresInferDtype(
    const paddle::DataType& pre_ids_dtype,
    const paddle::DataType& logits_dtype,
    const paddle::DataType& penalty_scores_dtype,
    const paddle::DataType& frequency_scores_dtype,
    const paddle::DataType& presence_scores_dtype,
    const paddle::DataType& cur_len_dtype,
    const paddle::DataType& min_len_dtype,
    const paddle::DataType& eos_token_id_dtype) {
  return {logits_dtype};
}
PD_BUILD_OP(get_token_penalty_multi_scores)
    .Inputs({"pre_ids",
             "logits",
             "penalty_scores",
             "frequency_scores",
             "presence_scores",
             "cur_len",
             "min_len",
             "eos_token_id"})
    .Outputs({"logits_out"})
    .SetKernelFn(PD_KERNEL(TokenPenaltyMultiScores))
    .SetInferShapeFn(PD_INFER_SHAPE(TokenPenaltyMultiScoresInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(TokenPenaltyMultiScoresInferDtype));

constexpr char kSEP = '/';

std::string DirName(const std::string& filepath) {
  auto pos = filepath.rfind(kSEP);
  if (pos == std::string::npos) {
    return "";
  }
  return filepath.substr(0, pos);
}

bool FileExists(const std::string& filepath) {
  struct stat buffer;
  return (stat(filepath.c_str(), &buffer) == 0);
}

void MkDir(const char* path) {
  std::string path_error(path);
  path_error += " mkdir failed!";
  if (mkdir(path, 0755)) {
    if (errno != EEXIST) {
      throw std::runtime_error(path_error);
    }
  }
}

void MkDirRecursively(const char* fullpath) {
  if (*fullpath == '\0') return;  // empty string
  if (FileExists(fullpath)) return;
  MkDirRecursively(DirName(fullpath).c_str());
  MkDir(fullpath);
}

template <typename data_t>
void saveToFile(std::ostream& os,
                const void* x_data,
                std::vector<int64_t> shape,
                int64_t x_numel,
                const char type_id) {
  // 1.type
  os.write(reinterpret_cast<const char*>(&type_id), sizeof(type_id));
  // 2.data
  uint64_t size = x_numel * sizeof(data_t);
  os.write(static_cast<const char*>(x_data),
           static_cast<std::streamsize>(size));
}

template <typename data_t>
void save_with_output_kernel(const paddle::Tensor& x,
                             const paddle::Tensor& batch_idx,
                             const paddle::Tensor& step_idx,
                             std::string file_path,
                             int64_t rank_id,
                             char type_id) {
  std::vector<int64_t> x_shape = x.shape();

  if (rank_id >= 0) {
    file_path += "_rank_" + std::to_string(rank_id);
  }

  int batch_idx_data = -1, step_idx_data = -1;

  if (batch_idx.is_custom_device()) {
    paddle::Tensor batch_idx_cpu =
        batch_idx.copy_to<int32_t>(paddle::CPUPlace());
    batch_idx_data = batch_idx_cpu.data<int32_t>()[0];
  } else {
    batch_idx_data = batch_idx.data<int32_t>()[0];
  }
  if (step_idx.is_custom_device()) {
    paddle::Tensor step_idx_cpu = step_idx.copy_to<int64_t>(paddle::CPUPlace());
    step_idx_data = step_idx_cpu.data<int64_t>()[0];
  } else {
    step_idx_data = step_idx.data<int64_t>()[0];
  }
  auto x_data = x.data<data_t>();

  if (batch_idx_data >= 0) {
    file_path += "_batch_" + std::to_string(batch_idx_data);
  }
  if (step_idx_data >= 0) {
    file_path += "_step_" + std::to_string(step_idx_data);
  }
  MkDirRecursively(DirName(file_path).c_str());
  std::ofstream fout(file_path, std::ios::binary);
  fout.write("0", 1);
  saveToFile<data_t>(fout, x_data, x_shape, x.numel(), type_id);
  fout.seekp(std::ios::beg);
  fout.write("1", 1);
  fout.close();
}

std::vector<paddle::Tensor> SaveWithOutputForward(
    const paddle::Tensor& x,
    const paddle::Tensor& batch_idx,
    const paddle::Tensor& step_idx,
    std::string file_path,
    int64_t rank_id) {
  auto out = x.copy_to(paddle::CPUPlace(), false);
  switch (x.type()) {
    case paddle::DataType::FLOAT32:
      save_with_output_kernel<float>(
          out, batch_idx, step_idx, file_path, rank_id, '0');
      break;
    case paddle::DataType::INT64:
      save_with_output_kernel<int64_t>(
          out, batch_idx, step_idx, file_path, rank_id, '1');
      break;
    case paddle::DataType::INT32:
      save_with_output_kernel<int32_t>(
          out, batch_idx, step_idx, file_path, rank_id, '2');
      break;
    default:
      PD_THROW(
          "function SaveWithOutputForward is not implemented for data type");
  }
  return {out};
}

std::vector<std::vector<int64_t>> SaveWithOutputInferShape(
    const std::vector<int64_t>& x_shape,
    const std::vector<int64_t>& batch_idx_shape,
    const std::vector<int64_t>& step_idx_shape) {
  return {x_shape};
}

std::vector<paddle::DataType> SaveWithOutputInferDtype(
    const paddle::DataType& x_dtype,
    const paddle::DataType& batch_idx_dtype,
    const paddle::DataType& step_idx_dtype) {
  return {x_dtype};
}

PD_BUILD_OP(save_with_output)
    .Inputs({"x", "batch_idx", "step_idx"})
    .Attrs({"file_path: std::string", "rank_id: int64_t"})
    .Outputs({"out"})
    .SetKernelFn(PD_KERNEL(SaveWithOutputForward))
    .SetInferShapeFn(PD_INFER_SHAPE(SaveWithOutputInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SaveWithOutputInferDtype));
