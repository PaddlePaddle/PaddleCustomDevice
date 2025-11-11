// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may
// not use this file except in compliance with the License.
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

class FusedIndexSelectKernel : public HpuFusedOperator {
 public:
  FusedIndexSelectKernel() : HpuFusedOperator("fused_index_select_fwd") {}

  void AddNode(ConvertTensors& ct, int batch_size) {
    auto inputs = ct.GetTensors();
    auto outputs = ct.GetTensors(false);
    auto num = outputs.size();

    auto index = createTensorFromCT(&ct, num);
    auto index_length = inputs[num].dims[0];

    for (decltype(num) i = 0; i < num; i++) {
      auto x = createTensorFromCT(&ct, i);
      synTensor out = nullptr;
      bool need_padding = (batch_size != index_length);
      std::string i_str = std::to_string(i);

      // output padding is needed by index_select when index length < bsz
      if (need_padding) {
        std::string out_name = "tmp_out" + i_str;
        std::vector<int64_t> out_dims = {index_length, outputs[i].dims[1]};
        out = createTensorNoPresist(out_name, outputs[i].type, out_dims);
      } else {
        out = createTensorFromCT(&ct, i, false);
      }

      ns_GatherKernel::Params params;
      params.axis = static_cast<int32_t>(inputs[i].dims.size()) - 1;

      std::vector<synTensor> ins = {x, index};
      std::vector<synTensor> outs = {out};
      std::string node_name = "group_index_select_" + i_str;

      switch (inputs[i].type) {
        case syn_type_fixed:
          AddNodeIndexSelect<bool>(ins, outs, params, node_name);
          break;
        case syn_type_single:
          AddNodeIndexSelect<float>(ins, outs, params, node_name);
          break;
        case syn_type_int32:
          AddNodeIndexSelect<int32_t>(ins, outs, params, node_name);
          break;
        case syn_type_int64:
          AddNodeIndexSelect<int64_t>(ins, outs, params, node_name);
          break;
        default:
          PD_CHECK(false,
                   "[RUNTIME] unexpected x type encountered in "
                   "FusedIndexSelect for AddNodeIndexSelect");
          break;
      }

      if (need_padding) {
        std::string pad_name = "pad_" + i_str;
        std::vector<int64_t> pad_dims = {batch_size - inputs[num].dims[0],
                                         inputs[i].dims[1]};
        auto pad = createTensorNoPresist(pad_name, inputs[i].type, pad_dims);
        ns_ConstantKernel::Params const_params;
        const_params.constant.i = 0;
        std::vector<synTensor> full_out = {pad};
        std::string full_node_name = "full_zeros_" + i_str;

        switch (inputs[i].type) {
          case syn_type_fixed:
            AddNodeFull<bool>(full_out, const_params, full_node_name);
            break;
          case syn_type_single:
            AddNodeFull<float>(full_out, const_params, full_node_name);
            break;
          case syn_type_int32:
            AddNodeFull<int32_t>(full_out, const_params, full_node_name);
            break;
          case syn_type_int64:
            AddNodeFull<int64_t>(full_out, const_params, full_node_name);
            break;
          default:
            PD_CHECK(false,
                     "[RUNTIME] unexpected x type encountered in "
                     "FusedIndexSelect for AddNodeFull");
            break;
        }

        std::vector<synTensor> concat_ins = {out, pad};
        auto padding_out = createTensorFromCT(&ct, i, false);
        std::vector<synTensor> concat_outs = {padding_out};
        synConcatenateParams concatParams;
        concatParams.axis = 1;
        std::string concat_node_name = "concat_" + i_str;
        AddNodeConcat(concat_ins, concat_outs, concatParams, concat_node_name);
      }
    }

    PD_CHECK(inputs.size() == (outputs.size() + 1),
             "[RUNTIME] in and out tensor numbers don't match");
  }
};

}  // namespace custom_kernel

#define DEF_DENSE_TENSOR_IN_AND_OUT(name)                                  \
  auto name##_t = static_cast<const phi::DenseTensor*>(name.impl().get()); \
  ct.Add(name##_t);                                                        \
  std::shared_ptr<phi::DenseTensor> name##_o_t =                           \
      std::make_shared<phi::DenseTensor>();                                \
  name##_o_t->Resize(phi::make_ddim({batch_size, name.dims()[1]}));        \
  dev_ctx->Alloc(name##_o_t.get(), name.dtype());                          \
  ct.Add(name##_o_t.get(), false);                                         \
  results.push_back(paddle::Tensor(name##_o_t));

std::vector<paddle::Tensor> FusedIndexSelectForward(
    const paddle::Tensor& temperature,
    const paddle::Tensor& top_p,
    const paddle::Tensor& step_index,
    const paddle::Tensor& prompt_token_idx,
    const paddle::Tensor& pre_token_ids,
    const paddle::Tensor& stop_flags,
    const paddle::Tensor& seq_lens_encoder,
    const paddle::Tensor& seq_lens_decoder,
    const paddle::Tensor& frequency_penalties,
    const paddle::Tensor& presence_penalties,
    const paddle::Tensor& repetition_penalties,
    const paddle::Tensor& min_dec_lens,
    const paddle::Tensor& sampled_ids,
    const int batch_size) {
  auto dev_ctx = static_cast<const phi::CustomContext*>(
      paddle::experimental::DeviceContextPool::Instance().Get(
          temperature.place()));

  custom_kernel::ConvertTensors ct;
  std::vector<paddle::Tensor> results;

  DEF_DENSE_TENSOR_IN_AND_OUT(temperature);
  DEF_DENSE_TENSOR_IN_AND_OUT(top_p);
  DEF_DENSE_TENSOR_IN_AND_OUT(step_index);
  DEF_DENSE_TENSOR_IN_AND_OUT(prompt_token_idx);
  DEF_DENSE_TENSOR_IN_AND_OUT(pre_token_ids);
  DEF_DENSE_TENSOR_IN_AND_OUT(stop_flags);
  DEF_DENSE_TENSOR_IN_AND_OUT(seq_lens_encoder);
  DEF_DENSE_TENSOR_IN_AND_OUT(seq_lens_decoder);
  DEF_DENSE_TENSOR_IN_AND_OUT(frequency_penalties);
  DEF_DENSE_TENSOR_IN_AND_OUT(presence_penalties);
  DEF_DENSE_TENSOR_IN_AND_OUT(repetition_penalties);
  DEF_DENSE_TENSOR_IN_AND_OUT(min_dec_lens);

  auto sampled_ids_t =
      static_cast<const phi::DenseTensor*>(sampled_ids.impl().get());
  ct.Add(sampled_ids_t);

  std::vector<DIMS> inputs_dims = ct.GetDims();
  std::vector<DIMS> outputs_dims = ct.GetDims(false);
  inputs_dims.insert(
      inputs_dims.end(), outputs_dims.begin(), outputs_dims.end());
  OpCacheOperator op_info;
  op_info.prepareOpInfo<float, nullptr_t>(
      "FusedIndexSelectKernel", inputs_dims, nullptr);
  auto recipe = op_info.GetRecipe();

  if (recipe == nullptr) {
    custom_kernel::FusedIndexSelectKernel op;
    op.AddNode(ct, batch_size);
    op.Compile();
    op_info.setOp(op);
    recipe = op_info.GetRecipe();
  }

  std::map<std::string, uint64_t> tensors = ct.GetDeviceAddr();
  RecipeRunner runner(recipe);
  runner.Run(reinterpret_cast<C_Stream>(dev_ctx->stream()), tensors);

  return results;
}

std::vector<std::vector<int64_t>> FusedIndexSelectInferShape(
    const std::vector<int64_t>& temperature_shape,
    const std::vector<int64_t>& top_p_shape,
    const std::vector<int64_t>& step_index_shape,
    const std::vector<int64_t>& prompt_token_idx_shape,
    const std::vector<int64_t>& pre_token_ids_shape,
    const std::vector<int64_t>& stop_flags_shape,
    const std::vector<int64_t>& seq_lens_encoder_shape,
    const std::vector<int64_t>& seq_lens_decoder_shape,
    const std::vector<int64_t>& frequency_penalties_shape,
    const std::vector<int64_t>& presence_penalties_shape,
    const std::vector<int64_t>& repetition_penalties_shape,
    const std::vector<int64_t>& min_dec_lens_shape,
    const std::vector<int64_t>& sampled_ids_shape) {
  return {temperature_shape};
}

std::vector<paddle::DataType> FusedIndexSelectInferDtype(
    const paddle::DataType& temperature_dtype,
    const paddle::DataType& top_p_dtype,
    const paddle::DataType& step_index_dtype,
    const paddle::DataType& prompt_token_idx_dtype,
    const paddle::DataType& pre_token_ids_dtype,
    const paddle::DataType& stop_flags_dtype,
    const paddle::DataType& seq_lens_encoder_dtype,
    const paddle::DataType& seq_lens_decoder_dtype,
    const paddle::DataType& frequency_penalties_dtype,
    const paddle::DataType& presence_penalties_dtype,
    const paddle::DataType& repetition_penalties_dtype,
    const paddle::DataType& min_dec_lens_dtype,
    const paddle::DataType& sampled_ids_dtype) {
  return {temperature_dtype};
}

PD_BUILD_OP(fused_index_select)
    .Inputs({"temperature",
             "top_p",
             "step_index",
             "prompt_token_idx",
             "pre_token_ids",
             "stop_flags",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "frequency_penalties",
             "presence_penalties",
             "repetition_penalties",
             "min_dec_lens",
             "sampled_ids"})
    .Outputs({"temperature_ext",
              "top_p_ext",
              "step_index_ext",
              "prompt_token_idx_ext",
              "pre_token_ids_ext",
              "stop_flags_ext",
              "seq_lens_encoder_ext",
              "seq_lens_decoder_ext",
              "frequency_penalties_ext",
              "presence_penalties_ext",
              "repetition_penalties_ext",
              "min_dec_lens_ext"})
    .Attrs({"batch_size: int"})
    .SetKernelFn(PD_KERNEL(FusedIndexSelectForward))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedIndexSelectInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(FusedIndexSelectInferDtype));
