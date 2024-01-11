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
#include <hccl/hccl.h>
#include <hccl/hccl_types.h>
#include "llama_layer_parallel_op.h"
#include "llama_layer/llama_blockattn_smooth_parallel_operation.h"
#include "paddle/extension.h"
#include "kernels/funcs/format_utils.h"
#include "kernels/funcs/npu_funcs.h"
#include "kernels/funcs/npu_op_runner.h"

static int32_t layer_num = 80; /* TODO:65B，写死8卡 */
std::shared_ptr<PpAtbLlamaBlockAttnLayerParallelOp> g_llamaBlockAttnSmoothEncoderOp;
std::shared_ptr<PpAtbLlamaBlockAttnLayerParallelOp> g_llamaBlockAttnSmoothDecoderOp;
static uint64_t executeCount = 0;
static bool g_isEncoder = true;

static bool first_run = true;
static paddle::Tensor norm_blank_bias; 
static paddle::Tensor self_out_norm_blank_bias;
static paddle::Tensor empty_offset;

void PerpareLlamaBlockAttnSmoothEncoderInputs(
    const paddle::Tensor &hidden,
    const paddle::Tensor &norm_weight,
    const paddle::Tensor &norm_blank_bias,
    const paddle::Tensor &qkv_mix_weight,
    const paddle::Tensor &qkv_deq_scale,
    const paddle::Tensor &self_out_linear_weight,
    const paddle::Tensor &self_out_linear_shift,
    const paddle::Tensor &self_out_linear_smooth,
    const paddle::Tensor &self_out_linear_deq_scale,
    const paddle::Tensor &self_out_norm_weight,
    const paddle::Tensor &self_out_norm_blank_bias,
    const paddle::Tensor &mlp_gate_up_weight,
    const paddle::Tensor &mlp_deq_scale,
    const paddle::Tensor &mlp_down_weight,
    const paddle::Tensor &mlp_down_shift,
    const paddle::Tensor &mlp_down_smooth,
    const paddle::Tensor &mlp_down_deq_scale,
    const paddle::Tensor &cos_table,
    const paddle::Tensor &sin_table,
    const paddle::Tensor &attention_mask,
    const paddle::Tensor &cache_key,
    const paddle::Tensor &cache_value,
    const paddle::Tensor &seq_len,
    const paddle::Tensor &block_tables,
    const phi::DenseTensor &slot_mapping_tensor,
    std::vector<const phi::DenseTensor *> &inputs) {

  auto hidden_tensor = static_cast<const phi::DenseTensor *>(hidden.impl().get());
  auto norm_weight_tensor = static_cast<const phi::DenseTensor *>(norm_weight.impl().get());
  auto norm_blank_bias_tensor = static_cast<const phi::DenseTensor *>(norm_blank_bias.impl().get());
  auto qkv_mix_weight_tensor = static_cast<const phi::DenseTensor *>(qkv_mix_weight.impl().get());
  auto qkv_deq_scale_tensor = static_cast<const phi::DenseTensor *>(qkv_deq_scale.impl().get());
  auto self_out_linear_weight_tensor = static_cast<phi::DenseTensor *>(self_out_linear_weight.impl().get());
  auto self_out_linear_shift_tensor = static_cast<phi::DenseTensor *>(self_out_linear_shift.impl().get());
  auto self_out_linear_smooth_tensor = static_cast<phi::DenseTensor *>(self_out_linear_smooth.impl().get());
  auto self_out_linear_deq_scale_tensor = static_cast<const phi::DenseTensor *>(self_out_linear_deq_scale.impl().get());
  auto self_out_norm_weight_tensor = static_cast<const phi::DenseTensor *>(self_out_norm_weight.impl().get());
  auto self_out_norm_blank_bias_tensor = static_cast<const phi::DenseTensor *>(self_out_norm_blank_bias.impl().get());
  auto mlp_gate_up_weight_tensor = static_cast<phi::DenseTensor *>(mlp_gate_up_weight.impl().get());
  auto mlp_deq_scale_tensor = static_cast<phi::DenseTensor *>(mlp_deq_scale.impl().get());
  auto mlp_down_weight_tensor = static_cast<phi::DenseTensor *>(mlp_down_weight.impl().get());
  auto mlp_down_shift_tensor = static_cast<phi::DenseTensor *>(mlp_down_shift.impl().get());
  auto mlp_down_smooth_tensor = static_cast<phi::DenseTensor *>(mlp_down_smooth.impl().get());
  auto mlp_down_deq_scale_tensor = static_cast<phi::DenseTensor *>(mlp_down_deq_scale.impl().get());
  auto cos_table_tensor = static_cast<const phi::DenseTensor *>(cos_table.impl().get());
  auto sin_table_tensor = static_cast<const phi::DenseTensor *>(sin_table.impl().get());
  auto attention_mask_tensor = static_cast<const phi::DenseTensor *>(attention_mask.impl().get());
  auto cache_key_tensor = static_cast<const phi::DenseTensor *>(cache_key.impl().get());
  auto cache_value_tensor = static_cast<const phi::DenseTensor *>(cache_value.impl().get());
  auto seq_len_tensor = static_cast<const phi::DenseTensor *>(seq_len.impl().get());
  auto block_tables_tensor = static_cast<const phi::DenseTensor *>(block_tables.impl().get());

  inputs.push_back(hidden_tensor);
  inputs.push_back(norm_weight_tensor);
  inputs.push_back(norm_blank_bias_tensor);
  inputs.push_back(qkv_mix_weight_tensor);
  inputs.push_back(qkv_deq_scale_tensor);
  inputs.push_back(self_out_linear_weight_tensor);
  inputs.push_back(self_out_linear_shift_tensor);
  inputs.push_back(self_out_linear_smooth_tensor);
  inputs.push_back(self_out_linear_deq_scale_tensor);
  inputs.push_back(self_out_norm_weight_tensor);
  inputs.push_back(self_out_norm_blank_bias_tensor);
  inputs.push_back(mlp_gate_up_weight_tensor);
  inputs.push_back(mlp_deq_scale_tensor);
  inputs.push_back(mlp_down_weight_tensor);
  inputs.push_back(mlp_down_shift_tensor);
  inputs.push_back(mlp_down_smooth_tensor);
  inputs.push_back(mlp_down_deq_scale_tensor);
  inputs.push_back(cos_table_tensor);
  inputs.push_back(sin_table_tensor);
  inputs.push_back(attention_mask_tensor);
  inputs.push_back(cache_key_tensor);
  inputs.push_back(cache_value_tensor);
  inputs.push_back(seq_len_tensor);
  inputs.push_back(block_tables_tensor);
  inputs.push_back(&slot_mapping_tensor);
}

void PerpareLlamaBlockAttnSmoothDecoderInputs(
    const paddle::Tensor &hidden,
    const paddle::Tensor &norm_weight,
    const paddle::Tensor &norm_blank_bias,
    const paddle::Tensor &qkv_mix_weight,
    const paddle::Tensor &qkv_deq_scale,
    const paddle::Tensor &self_out_linear_weight,
    const paddle::Tensor &self_out_linear_shift,
    const paddle::Tensor &self_out_linear_smooth,
    const paddle::Tensor &self_out_linear_deq_scale,
    const paddle::Tensor &self_out_norm_weight,
    const paddle::Tensor &self_out_norm_blank_bias,
    const paddle::Tensor &mlp_gate_up_weight,
    const paddle::Tensor &mlp_deq_scale,
    const paddle::Tensor &mlp_down_weight,
    const paddle::Tensor &mlp_down_shift,
    const paddle::Tensor &mlp_down_smooth,
    const paddle::Tensor &mlp_down_deq_scale,
    const paddle::Tensor &cos_table,
    const paddle::Tensor &sin_table,
    const paddle::Tensor &attention_mask,
    const paddle::Tensor &cache_key,
    const paddle::Tensor &cache_value,
    const phi::DenseTensor &seq_len_tensor,
    const paddle::Tensor &block_tables,
    const phi::DenseTensor &slot_mapping_tensor,
    std::vector<const phi::DenseTensor *> &inputs) {

  auto hidden_tensor = static_cast<const phi::DenseTensor *>(hidden.impl().get());
  auto norm_weight_tensor = static_cast<const phi::DenseTensor *>(norm_weight.impl().get());
  auto norm_blank_bias_tensor = static_cast<const phi::DenseTensor *>(norm_blank_bias.impl().get());
  auto qkv_mix_weight_tensor = static_cast<const phi::DenseTensor *>(qkv_mix_weight.impl().get());
  auto qkv_deq_scale_tensor = static_cast<const phi::DenseTensor *>(qkv_deq_scale.impl().get());
  auto self_out_linear_weight_tensor = static_cast<phi::DenseTensor *>(self_out_linear_weight.impl().get());
  auto self_out_linear_shift_tensor = static_cast<phi::DenseTensor *>(self_out_linear_shift.impl().get());
  auto self_out_linear_smooth_tensor = static_cast<phi::DenseTensor *>(self_out_linear_smooth.impl().get());
  auto self_out_linear_deq_scale_tensor = static_cast<const phi::DenseTensor *>(self_out_linear_deq_scale.impl().get());
  auto self_out_norm_weight_tensor = static_cast<const phi::DenseTensor *>(self_out_norm_weight.impl().get());
  auto self_out_norm_blank_bias_tensor = static_cast<const phi::DenseTensor *>(self_out_norm_blank_bias.impl().get());
  auto mlp_gate_up_weight_tensor = static_cast<phi::DenseTensor *>(mlp_gate_up_weight.impl().get());
  auto mlp_deq_scale_tensor = static_cast<phi::DenseTensor *>(mlp_deq_scale.impl().get());
  auto mlp_down_weight_tensor = static_cast<phi::DenseTensor *>(mlp_down_weight.impl().get());
  auto mlp_down_shift_tensor = static_cast<phi::DenseTensor *>(mlp_down_shift.impl().get());
  auto mlp_down_smooth_tensor = static_cast<phi::DenseTensor *>(mlp_down_smooth.impl().get());
  auto mlp_down_deq_scale_tensor = static_cast<phi::DenseTensor *>(mlp_down_deq_scale.impl().get());
  auto cos_table_tensor = static_cast<const phi::DenseTensor *>(cos_table.impl().get());
  auto sin_table_tensor = static_cast<const phi::DenseTensor *>(sin_table.impl().get());
  auto attention_mask_tensor = static_cast<const phi::DenseTensor *>(attention_mask.impl().get());
  auto cache_key_tensor = static_cast<const phi::DenseTensor *>(cache_key.impl().get());
  auto cache_value_tensor = static_cast<const phi::DenseTensor *>(cache_value.impl().get());
  auto block_tables_tensor = static_cast<const phi::DenseTensor *>(block_tables.impl().get());

  inputs.push_back(hidden_tensor);
  inputs.push_back(norm_weight_tensor);
  inputs.push_back(norm_blank_bias_tensor);
  inputs.push_back(qkv_mix_weight_tensor);
  inputs.push_back(qkv_deq_scale_tensor);
  inputs.push_back(self_out_linear_weight_tensor);
  inputs.push_back(self_out_linear_shift_tensor);
  inputs.push_back(self_out_linear_smooth_tensor);
  inputs.push_back(self_out_linear_deq_scale_tensor);
  inputs.push_back(self_out_norm_weight_tensor);
  inputs.push_back(self_out_norm_blank_bias_tensor);
  inputs.push_back(mlp_gate_up_weight_tensor);
  inputs.push_back(mlp_deq_scale_tensor);
  inputs.push_back(mlp_down_weight_tensor);
  inputs.push_back(mlp_down_shift_tensor);
  inputs.push_back(mlp_down_smooth_tensor);
  inputs.push_back(mlp_down_deq_scale_tensor);
  inputs.push_back(cos_table_tensor);
  inputs.push_back(sin_table_tensor);
  inputs.push_back(attention_mask_tensor);
  inputs.push_back(cache_key_tensor);
  inputs.push_back(cache_value_tensor);
  inputs.push_back(&seq_len_tensor);
  inputs.push_back(block_tables_tensor);
  inputs.push_back(&slot_mapping_tensor);
}


static bool isEncoderToken(const paddle::Tensor &encoder_seq_len)
{
  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(encoder_seq_len.place()));
  std::vector<int32_t> seq_len_vec;
  auto seq_len_tensor = const_cast<phi::DenseTensor *>(static_cast<const phi::DenseTensor *>(encoder_seq_len.impl().get()));
  custom_kernel::TensorToVector(*dev_ctx, *seq_len_tensor, *dev_ctx, &seq_len_vec);

  for(auto array: seq_len_vec) {
    if (array > 0) {
      return true; // 只要encoder非零，即认为是prefill阶段
    }
  }

  return false;
}

void InitAtbLlamaBlockAttnSmoothLayerOp(std::shared_ptr<PpAtbLlamaBlockAttnLayerParallelOp> &block_op,
                                  float rmsNormEps, int32_t head_num, int32_t head_dim, HcclComm comm,
                                  float inputRmsNormScale, float selfRmsNormScale, float selfQuantScale, float mlpQuantScale, int layerid)
{
    if(!block_op){
      std::cout << "Run In Block Attention Smooth Parallel isPrefill:" << g_isEncoder <<
      " head_num: " << head_num << " head_dim: " << head_dim << std::endl;
      block_op.reset(new PpAtbLlamaBlockAttnLayerParallelOp("LlamaBlockAttnSmoothLayerParallelOp", layer_num, g_isEncoder, false));
    }

    std::string device_id_str = getenv("FLAGS_selected_npus");
    int device_id = stoi(device_id_str);
    int nranks = 8;

    atb::Operation *op = nullptr; 
    LlamaBlockAttnSmoothParallelParam param = {rmsNormEps,
                                         head_num,
                                         head_dim,
                                         device_id,
                                         nranks,
                                         1.0 / std::sqrt(head_dim), // qkScale
                                         2, // rotaryCoeff
                                         true,
                                         comm,
                                         g_isEncoder, // isPrefill
                                         selfRmsNormScale,
                                         0,
                                         selfQuantScale,
                                         0,
                                         inputRmsNormScale,
                                         0,
                                         mlpQuantScale,
                                         0}; 
    LlamaBlockAttnSmoothParallelOperation(param, &op);
    block_op->operations_.at(layerid).reset(op);
}

std::vector<paddle::Tensor> LlamaBlockAttnSmoothLayerParallelOp(
    const paddle::Tensor &hidden,
    const paddle::Tensor &norm_weight,
    const paddle::Tensor &qkv_mix_weight,
    const paddle::Tensor &qkv_deq_scale,
    const paddle::Tensor &self_out_linear_weight,
    const paddle::Tensor &self_out_linear_shift,
    const paddle::Tensor &self_out_linear_smooth,
    const paddle::Tensor &self_out_linear_deq_scale,
    const paddle::Tensor &self_out_norm_weight,
    const paddle::Tensor &mlp_gate_up_weight,
    const paddle::Tensor &mlp_deq_scale,
    const paddle::Tensor &mlp_down_weight,
    const paddle::Tensor &mlp_down_shift,
    const paddle::Tensor &mlp_down_smooth,
    const paddle::Tensor &mlp_down_deq_scale,
    const paddle::Tensor &cos_table,
    const paddle::Tensor &sin_table,
    const paddle::Tensor &attention_mask,
    const paddle::Tensor &cache_key,
    const paddle::Tensor &cache_value,
    const paddle::Tensor &decoder_seq_len,
    const paddle::Tensor &encoder_seq_len,
    const paddle::Tensor &block_tables,
    int32_t block_size,
    float rmsNormEps,
    float inputRmsNormScale,
    float selfRmsNormScale,
    float selfQuantScale,
    float mlpQuantScale) {

  int32_t batch_size = hidden.shape().at(0);
  int32_t head_num = cache_key.shape().at(1);
  int32_t head_dim = cache_key.shape().at(3);
  int32_t max_batch_size = attention_mask.shape().at(0);

  auto dev_ctx = static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(hidden.place()));

  auto stream = static_cast<aclrtStream>(dev_ctx->stream());
  auto comm = reinterpret_cast<HcclComm>(phi::detail::GetCCLComm(hidden.place(), 0));

  int layer_id = executeCount % layer_num;
  if (layer_id == 0) {
    g_isEncoder = isEncoderToken(encoder_seq_len);
  }

  inputRmsNormScale *= 127;
  selfRmsNormScale *= 127;
  selfQuantScale *= 127;
  mlpQuantScale *= 127;
  if (g_isEncoder && (!g_llamaBlockAttnSmoothEncoderOp || !g_llamaBlockAttnSmoothEncoderOp->operations_.at(layer_id))) {
    InitAtbLlamaBlockAttnSmoothLayerOp(g_llamaBlockAttnSmoothEncoderOp, rmsNormEps, head_num, head_dim, comm, inputRmsNormScale, selfRmsNormScale, selfQuantScale, mlpQuantScale, layer_id);
  } else if (!g_isEncoder && (!g_llamaBlockAttnSmoothDecoderOp || !g_llamaBlockAttnSmoothDecoderOp->operations_.at(layer_id))) {
    InitAtbLlamaBlockAttnSmoothLayerOp(g_llamaBlockAttnSmoothDecoderOp, rmsNormEps, head_num, head_dim, comm, inputRmsNormScale, selfRmsNormScale, selfQuantScale, mlpQuantScale, layer_id);
  }

  if (layer_id == 0) {
    if (g_isEncoder) {
      g_llamaBlockAttnSmoothEncoderOp->output_->Resize(phi::make_ddim(hidden.shape()));
      g_llamaBlockAttnSmoothEncoderOp->UpdateInputTensorAndParam(block_tables, encoder_seq_len, block_size);
    } else {
      g_llamaBlockAttnSmoothDecoderOp->output_->Resize(phi::make_ddim(hidden.shape()));
      g_llamaBlockAttnSmoothDecoderOp->UpdateInputTensorAndParam(block_tables, decoder_seq_len, block_size);
    }
    executeCount = 0;
  }
  std::vector<const phi::DenseTensor *> inputs;

  executeCount++;
  if (first_run) {
      norm_blank_bias = paddle::full(norm_weight.shape(), 0, paddle::DataType::FLOAT16, hidden.place()); 
      self_out_norm_blank_bias = paddle::full(self_out_norm_weight.shape(), 0, paddle::DataType::FLOAT16, hidden.place()); 
      empty_offset = paddle::full({}, 0, paddle::DataType::INT8, hidden.place());
      first_run = false;
  }

  if (g_isEncoder) {
    PerpareLlamaBlockAttnSmoothEncoderInputs(hidden,
                                       norm_weight,
                                       norm_blank_bias,
                                       qkv_mix_weight,
                                       qkv_deq_scale,
                                       self_out_linear_weight,
                                       self_out_linear_shift,
                                       self_out_linear_smooth,
                                       self_out_linear_deq_scale,
                                       self_out_norm_weight,
                                       self_out_norm_blank_bias,
                                       mlp_gate_up_weight,
                                       mlp_deq_scale,
                                       mlp_down_weight,
                                       mlp_down_shift,
                                       mlp_down_smooth,
                                       mlp_down_deq_scale,
                                       cos_table,
                                       sin_table,
                                       attention_mask,
                                       cache_key,
                                       cache_value,
                                       encoder_seq_len,
                                       block_tables,
                                       g_llamaBlockAttnSmoothEncoderOp->slot_mapping_tensor_,
                                       inputs);
    std::vector<const phi::DenseTensor *> outputs = {g_llamaBlockAttnSmoothEncoderOp->output_.get()};
    g_llamaBlockAttnSmoothEncoderOp->Execute(stream, inputs, outputs, dev_ctx, layer_id);
    return {paddle::Tensor(g_llamaBlockAttnSmoothEncoderOp->output_)};
  }
  PerpareLlamaBlockAttnSmoothDecoderInputs(hidden,
                                     norm_weight,
                                     norm_blank_bias,
                                     qkv_mix_weight,
                                     qkv_deq_scale,
                                     self_out_linear_weight,
                                     self_out_linear_shift,
                                     self_out_linear_smooth,
                                     self_out_linear_deq_scale,
                                     self_out_norm_weight,
                                     self_out_norm_blank_bias,
                                     mlp_gate_up_weight,
                                     mlp_deq_scale,
                                     mlp_down_weight,
                                     mlp_down_shift,
                                     mlp_down_smooth,
                                     mlp_down_deq_scale,
                                     cos_table,
                                     sin_table,
                                     attention_mask,
                                     cache_key,
                                     cache_value,
                                     g_llamaBlockAttnSmoothDecoderOp->token_offset_tensor_,
                                     block_tables,
                                     g_llamaBlockAttnSmoothDecoderOp->slot_mapping_tensor_,
                                     inputs);
  std::vector<const phi::DenseTensor *> outputs = {g_llamaBlockAttnSmoothDecoderOp->output_.get()};
  g_llamaBlockAttnSmoothDecoderOp->Execute(stream, inputs, outputs, dev_ctx, layer_id);
  return {paddle::Tensor(g_llamaBlockAttnSmoothDecoderOp->output_)};
}

std::vector<std::vector<int64_t>> LlamaBlockAttnSmoothLayerOpInferShape(
    const std::vector<int64_t> &hidden_shape,
    const std::vector<int64_t> &norm_weight_shape,
    const std::vector<int64_t> &qkv_mix_weight_shape,
    const std::vector<int64_t> &qkv_deq_Scale_shape,
    const std::vector<int64_t> &self_out_linear_weight_shape,
    const std::vector<int64_t> &self_out_linear_shift_shape,
    const std::vector<int64_t> &self_out_linear_smooth_shape,
    const std::vector<int64_t> &self_out_linear_deq_scale_shape,
    const std::vector<int64_t> &self_out_norm_weight_shape,
    const std::vector<int64_t> &mlp_gate_up_weight_shape,
    const std::vector<int64_t> &mlp_deq_scale_shape,
    const std::vector<int64_t> &mlp_down_weight_shape,
    const std::vector<int64_t> &mlp_down_shift_shape,
    const std::vector<int64_t> &mlp_down_smooth_shape,
    const std::vector<int64_t> &mlp_down_deq_scale_shape,
    const std::vector<int64_t> &cos_table_shape,
    const std::vector<int64_t> &sin_table_shape,
    const std::vector<int64_t> &attention_mask_shape,
    const std::vector<int64_t> &cacheK_shape,
    const std::vector<int64_t> &cacheV_shape,
    const std::vector<int64_t> &decoder_seq_len_shape,
    const std::vector<int64_t> &encoder_seq_len_shape,
    const std::vector<int64_t> &block_tables_shape,
    int32_t block_size,
    float rmsNormEps,
    float inputRmsNormScale,
    float selfRmsNormScale,
    float selfQuantScale,
    float mlpQuantScale) {

  return {hidden_shape};
}


PD_BUILD_OP(llama_blockattn_smooth_layer_parallel)
    .Inputs({"Hidden",
             "NormWeight",
             "QKVMixWeight",
             "QKVDeqScale",
             "SelfOutLinearWeight",
             "SelfOutLinearShift",
             "SelfOutLinearSmooth",
             "SelfOutLinearDeqScale",
             "SelfOutNormWeight",
             "MlpGateUpWeight",
             "MlpDeqScale",
             "MlpDownWeight",
             "MlpDownShift",
             "MlpDownSmooth",
             "MlpDownDeqScale",
             "CosTable",
             "SinTable",
             "AttentionMask",
             "Cache_K",
             "Cache_V",
             "DecoderSeqLength",
             "EncoderSeqLength",
             "BlockTables"})
    .Outputs({"Out"})
    .Attrs({"block_size: int", "rmsNormEps: float", "inputRmsNormScale: float", "selfRmsNormScale: float", "selfQuantScale: float", "mlpQuantScale: float"})
    .SetKernelFn(PD_KERNEL(LlamaBlockAttnSmoothLayerParallelOp))
    .SetInferShapeFn(PD_INFER_SHAPE(
        LlamaBlockAttnSmoothLayerOpInferShape)); // neccessary if the op has muti_inputs

#endif
