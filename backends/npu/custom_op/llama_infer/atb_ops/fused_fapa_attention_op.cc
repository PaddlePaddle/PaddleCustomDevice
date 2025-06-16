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

#ifdef PADDLE_WITH_ATB
#include "fused_fapa_attn_op_utils.h"


constexpr int32_t kFusedFaPaAttnLayerBegin = 1;
constexpr int32_t kFusedFaPaAttnLayerEnd = 2;

static bool first_or_second_flag = false;
static int layer_id = 0;

void FusedFaPaAttnLayerOpPrefillStage(
  const phi::CustomContext &dev_ctx,
  const paddle::Tensor &qkv_weight,
  const paddle::optional<paddle::Tensor> &qkv_bias,
  const paddle::optional<paddle::Tensor> &qkv_deq_scale,
  const paddle::Tensor &cache_k,
  const paddle::Tensor &cache_v,
  float qkv_quant_scale,
  int64_t max_seq_len,
  int64_t head_num,
  int64_t kv_head_num,
  int64_t head_dim,
  int64_t emb_dim,
  int64_t ntokens,
  bool trans_qkv,
  bool use_neox_style,
  bool use_alibi) {

bool use_matmul_int8 = qkv_deq_scale.is_initialized();

int64_t batch_size =
    FusedFaPaGlobalVar::Instance().get_seqlens_encoder()->size;
void *slots_data = FusedFaPaGlobalVar::Instance().get_slots_encoder();
void *seqlens_dev =
    FusedFaPaGlobalVar::Instance().get_seqlens_encoder()->dev_ptr;
void *seqlens_host =
    FusedFaPaGlobalVar::Instance().get_seqlens_encoder()->host_ptr;

void *first_out_data =
    FusedFaPaGlobalVar::Instance().get_out_encoder()->first->data();
void *second_out_data =
    FusedFaPaGlobalVar::Instance().get_out_encoder()->second->data();

auto &runner = *FusedFaPaGlobalVar::Instance().get_encoder_runner(layer_id);

if (runner.is_initialized()) {
  runner.reset_variant_pack();
} else {
  atb_layers::FaPaAttentionParam param;
  param.use_alibi = use_alibi;
  param.trans_qkv_weight = trans_qkv;
  param.rope_neox = use_neox_style;
  param.has_qkv_bias = qkv_bias.is_initialized();
  param.head_num = head_num;
  param.kv_head_num = kv_head_num;
  param.head_dim = head_dim;
  param.is_prefill = true;
  param.qkv_quant_scale = qkv_quant_scale * 127.0f;
  param.use_matmul_int8 = use_matmul_int8;
  runner.create(param);
}

if (first_or_second_flag) {
  runner.bind_input(
      second_out_data, phi::DataType::FLOAT16, {ntokens, emb_dim});
} else {
  runner.bind_input(
      first_out_data, phi::DataType::FLOAT16, {ntokens, emb_dim});
}
runner.bind_input(qkv_weight);
if (qkv_bias.is_initialized()) {
  runner.bind_input(qkv_bias.get());
}
if (qkv_deq_scale.is_initialized()) {
  runner.bind_input(qkv_deq_scale.get());
  runner.bind_input(FusedFaPaGlobalVar::Instance().get_qkv_deq_offset(),
                    phi::DataType::INT32,
                    {qkv_deq_scale->numel()});
}

if (!use_alibi) {
  void *cos_data =
      FusedFaPaGlobalVar::Instance().get_rope_encoder()->rope_emb_cos->data();
  void *sin_data =
      FusedFaPaGlobalVar::Instance().get_rope_encoder()->rope_emb_sin->data();
  runner.bind_input(cos_data, phi::DataType::FLOAT16, {ntokens, head_dim});
  runner.bind_input(sin_data, phi::DataType::FLOAT16, {ntokens, head_dim});
}
if (!use_alibi) {
  void *mask_data = FusedFaPaGlobalVar::Instance().get_casual_mask();
  runner.bind_input(
      mask_data, phi::DataType::FLOAT16, {max_seq_len, max_seq_len});
} else {
  void *mask_data = FusedFaPaGlobalVar::Instance().get_alibi_src_mask();
  // 1, head_num, max_seq_len, max_seq_len
  runner.bind_input(mask_data,
                    phi::DataType::FLOAT16,
                    {batch_size, head_num, max_seq_len, max_seq_len});
}

runner.bind_input(cache_k);
runner.bind_input(cache_v);
runner.bind_input(slots_data, phi::DataType::INT32, {ntokens});
runner.bind_input(
    seqlens_dev, seqlens_host, phi::DataType::INT32, {batch_size});
if (first_or_second_flag) {
  runner.bind_output(
      first_out_data, phi::DataType::FLOAT16, {ntokens, emb_dim});
} else {
  runner.bind_output(
      second_out_data, phi::DataType::FLOAT16, {ntokens, emb_dim});
}
LOG(INFO) << "beging setup  encord**************";
runner.setup(dev_ctx);
atb_layers::TaskQueue::Instance(dev_ctx.GetPlace().GetDeviceId())
    .Commit(std::move(
        std::packaged_task<void(void)>([&] { runner.execute(dev_ctx); })));
}

void FusedFaPaAttnLayerOpDecodingStage(
  const phi::CustomContext &dev_ctx,
  const paddle::Tensor &qkv_weight,
  const paddle::optional<paddle::Tensor> &qkv_bias,
  const paddle::optional<paddle::Tensor> &qkv_deq_scale,
  const paddle::Tensor &cache_k,
  const paddle::Tensor &cache_v,
  const paddle::Tensor &block_tables,
  float qkv_quant_scale,
  int64_t max_seq_len,
  int64_t head_num,
  int64_t kv_head_num,
  int64_t head_dim,
  int64_t emb_dim,
  int64_t ntokens,
  bool trans_qkv,
  bool use_neox_style,
  bool use_alibi) {

bool use_matmul_int8 = qkv_deq_scale.is_initialized();

int64_t batch_size =
    FusedFaPaGlobalVar::Instance().get_seqlens_decoder()->size;
void *slots_data = FusedFaPaGlobalVar::Instance().get_slots_decoder();
void *seqlens_dev =
    FusedFaPaGlobalVar::Instance().get_seqlens_decoder()->dev_ptr;
void *seqlens_host =
    FusedFaPaGlobalVar::Instance().get_seqlens_decoder()->host_ptr;
void *batch_status_data =
    FusedFaPaGlobalVar::Instance().get_batch_status()->data;

void *first_out_data =
    FusedFaPaGlobalVar::Instance().get_out_decoder()->first->data();
void *second_out_data =
    FusedFaPaGlobalVar::Instance().get_out_decoder()->second->data();

auto &runner = *FusedFaPaGlobalVar::Instance().get_decoder_runner(layer_id);

if (runner.is_initialized()) {
  runner.reset_variant_pack();
} else {
  atb_layers::FaPaAttentionParam param;
  param.use_alibi = use_alibi;
  param.trans_qkv_weight = trans_qkv;
  param.rope_neox = use_neox_style;
  param.has_qkv_bias = qkv_bias.is_initialized();
  param.head_num = head_num;
  param.kv_head_num = kv_head_num;
  param.head_dim = head_dim;
  param.is_prefill = false;
  param.qkv_quant_scale = qkv_quant_scale * 127.0f;
  param.use_matmul_int8 = use_matmul_int8;
  runner.create(param);
}

if (first_or_second_flag) {
  runner.bind_input(
      second_out_data, phi::DataType::FLOAT16, {ntokens, emb_dim});
} else {
  runner.bind_input(
      first_out_data, phi::DataType::FLOAT16, {ntokens, emb_dim});
}
runner.bind_input(qkv_weight);
if (qkv_bias.is_initialized()) {
  runner.bind_input(qkv_bias.get());
}
if (qkv_deq_scale.is_initialized()) {
  runner.bind_input(qkv_deq_scale.get());
  runner.bind_input(FusedFaPaGlobalVar::Instance().get_qkv_deq_offset(),
                    phi::DataType::INT32,
                    {qkv_deq_scale->numel()});
}

if (!use_alibi) {
  void *cos_data =
      FusedFaPaGlobalVar::Instance().get_rope_decoder()->rope_emb_cos->data();
  void *sin_data =
      FusedFaPaGlobalVar::Instance().get_rope_decoder()->rope_emb_sin->data();
  runner.bind_input(cos_data, phi::DataType::FLOAT16, {ntokens, head_dim});
  runner.bind_input(sin_data, phi::DataType::FLOAT16, {ntokens, head_dim});
}
if (use_alibi) {
  void *mask_data = FusedFaPaGlobalVar::Instance().get_alibi_tgt_mask();
  // batch, head_num, 1, max_seq_len
  runner.bind_input(mask_data,
                    phi::DataType::FLOAT16,
                    {batch_size, head_num, 1, max_seq_len});
}
runner.bind_input(cache_k);
runner.bind_input(cache_v);
runner.bind_input(slots_data, phi::DataType::INT32, {ntokens});
runner.bind_input(block_tables);
runner.bind_input(
    seqlens_dev, seqlens_host, phi::DataType::INT32, {batch_size});
runner.bind_host_input(batch_status_data, phi::DataType::INT32, {batch_size});
if (first_or_second_flag) {
    runner.bind_output(
        first_out_data, phi::DataType::FLOAT16, {ntokens, emb_dim});
  } else {
    runner.bind_output(
        second_out_data, phi::DataType::FLOAT16, {ntokens, emb_dim});
  }
LOG(INFO) << "beging setup  decord**************";
runner.setup(dev_ctx);
atb_layers::TaskQueue::Instance(dev_ctx.GetPlace().GetDeviceId())
    .Commit(std::move(
        std::packaged_task<void(void)>([&] { runner.execute(dev_ctx); })));
}

std::vector<paddle::Tensor> FusedFaPaAttnOp(
  const paddle::Tensor &hidden,
  const paddle::Tensor &qkv_weight,
  const paddle::optional<paddle::Tensor> &qkv_bias,
  const paddle::optional<paddle::Tensor> &qkv_deq_scale,
  const paddle::Tensor &rope_emb,
  const paddle::Tensor &cache_k,
  const paddle::Tensor &cache_v,
  const paddle::Tensor &seq_lens_encoder,
  const paddle::Tensor &seq_lens_decoder,
  const paddle::Tensor &block_tables,
  int32_t flag,
  int32_t max_seq_len,
  int32_t block_size,
  float qkv_quant_scale,
  bool trans_qkv,
  bool use_neox_style,
  bool use_alibi) {
  LOG(INFO) << "beging **************" << max_seq_len;

bool use_matmul_int8 = qkv_deq_scale.is_initialized();

const auto &hidden_shape = hidden.shape();
const auto &cache_k_shape = cache_k.shape();
const auto &block_tables_shape = block_tables.shape();
// rope: 2, B, S, 1, D, alibi: B,H,S,S + B,H,1,S
const auto &rope_emb_shape = rope_emb.shape();
uint64_t token_num = hidden_shape[0];
uint64_t q_hid_dim = hidden_shape[2];
uint64_t emb_dim = hidden_shape[1];
uint64_t kv_head_num = cache_k_shape[1];
uint64_t head_dim = cache_k_shape[3];
uint64_t head_num = q_hid_dim / head_dim;
uint64_t max_block_num_per_seq = block_tables_shape[1];
uint64_t batch_size = seq_lens_encoder.numel();

LOG(INFO) << "token_num **************" << token_num;
LOG(INFO) << "emb_dim **************" << emb_dim;
LOG(INFO) << "kv_head_num **************" << kv_head_num;
LOG(INFO) << "head_dim **************" << head_dim;
LOG(INFO) << "head_num **************" << head_num;
LOG(INFO) << "max_block_num_per_seq **************" << max_block_num_per_seq;
LOG(INFO) << "batch_size **************" << batch_size;

auto place = hidden.place();
const auto &dev_ctx = *static_cast<const phi::CustomContext *>(
    paddle::experimental::DeviceContextPool::Instance().Get(place));
if (rope_emb_shape.size() == 1) {
  use_alibi = true;
}
if (flag == kFusedFaPaAttnLayerBegin) {
  FusedFaPaGlobalVar::Instance().update_block_tables(dev_ctx, block_tables);
  FusedFaPaGlobalVar::Instance().update_seqlens_encoder(dev_ctx,
                                                        seq_lens_encoder);
  FusedFaPaGlobalVar::Instance().update_seqlens_decoder(dev_ctx,
                                                        seq_lens_decoder);
  if (!use_alibi) {
    FusedFaPaGlobalVar::Instance().update_casual_mask(dev_ctx, max_seq_len);
  } else {
    if (phi::DataType::FLOAT16 != rope_emb.dtype()) {
      PD_THROW("NOT supported data type. Only float16 are supported. ");
    }
    FusedFaPaGlobalVar::Instance().update_alibi_mask(
        const_cast<phi::float16 *>(rope_emb.data<phi::float16>()),
        const_cast<phi::float16 *>(rope_emb.data<phi::float16>() +
                                   batch_size * head_num * max_seq_len *
                                       max_seq_len));
  }

  FusedFaPaGlobalVar::Instance().update_slots_encoder(
      dev_ctx, block_size, max_block_num_per_seq);
  FusedFaPaGlobalVar::Instance().update_slots_decoder(
      dev_ctx, block_size, max_block_num_per_seq);

  if (!use_alibi) {
    if (phi::DataType::FLOAT16 != rope_emb.dtype()) {
      auto rope_emb_fp16 = rope_emb.cast(phi::DataType::FLOAT16);
      FusedFaPaGlobalVar::Instance().update_rope_encoder(
          dev_ctx, rope_emb_fp16, max_seq_len, head_dim);
      FusedFaPaGlobalVar::Instance().update_rope_decoder(
          dev_ctx, rope_emb_fp16, max_seq_len, head_dim);
    } else {
      FusedFaPaGlobalVar::Instance().update_rope_encoder(
          dev_ctx, rope_emb, max_seq_len, head_dim);
      FusedFaPaGlobalVar::Instance().update_rope_decoder(
          dev_ctx, rope_emb, max_seq_len, head_dim);
    }
  }

  FusedFaPaGlobalVar::Instance().update_in_encoder(dev_ctx, hidden);
  FusedFaPaGlobalVar::Instance().update_in_decoder(dev_ctx, hidden);

  if (use_matmul_int8) {
    FusedFaPaGlobalVar::Instance().update_qkv_deq_offset(
        dev_ctx, qkv_deq_scale->numel());
  }

  first_or_second_flag = false;
  layer_id = 0;
} else {
  first_or_second_flag = !first_or_second_flag;
  layer_id++;
}

auto ntokens_encoder =
    FusedFaPaGlobalVar::Instance().get_seqlens_encoder()->ntokens;
auto ntokens_decoder =
    FusedFaPaGlobalVar::Instance().get_seqlens_decoder()->ntokens;

LOG(INFO) << "ntokens_encoder**************" << ntokens_encoder;
LOG(INFO) << "ntokens_decoder**************" << ntokens_decoder;
if (ntokens_encoder > 0) {
  FusedFaPaAttnLayerOpPrefillStage(dev_ctx,
                                    qkv_weight,
                                    qkv_bias,
                                    qkv_deq_scale,
                                    cache_k,
                                    cache_v,
                                    qkv_quant_scale,                                   
                                    max_seq_len,
                                    head_num,
                                    kv_head_num,
                                    head_dim,
                                    emb_dim,                                  
                                    ntokens_encoder, 
                                    trans_qkv,
                                    use_neox_style,                                   
                                    use_alibi);
}
if (ntokens_decoder > 0) {
  FusedFaPaAttnLayerOpDecodingStage(dev_ctx,
                                     qkv_weight, 
                                     qkv_bias,
                                     qkv_deq_scale,
                                     cache_k,
                                     cache_v,
                                     block_tables,
                                     qkv_quant_scale,                                     
                                     max_seq_len,
                                     head_num,
                                     kv_head_num,
                                     head_dim,  
                                     emb_dim,      
                                     use_neox_style,
                                     trans_qkv,                             
                                     ntokens_decoder,                                    
                                     use_alibi);
}

paddle::Tensor out(place);
if (flag == kFusedFaPaAttnLayerEnd) {
  atb_layers::TaskQueue::Instance(dev_ctx.GetPlace().GetDeviceId()).Wait();

  fapa_layers::init_tensor(dev_ctx, phi::DataType::FLOAT16, {batch_size, emb_dim}, &out);
  FusedFaPaGlobalVar::Instance().update_out_encoder(
      dev_ctx, first_or_second_flag, &out);
  FusedFaPaGlobalVar::Instance().update_out_decoder(
      dev_ctx, first_or_second_flag, &out);
} else {
  fapa_layers::init_tensor(dev_ctx, phi::DataType::FLOAT16, {1, emb_dim}, &out);
}
return {out};
}

std::vector<std::vector<int64_t>> FusedFaPaAttnOpInferShape(
  const std::vector<int64_t> &hidden_shape,
  const std::vector<int64_t> &qkv_weight_shape,
  const std::vector<int64_t> &qkv_bias_shape,
  const paddle::optional<std::vector<int64_t>> &qkv_deq_scale_shape,
  const std::vector<int64_t> &rope_emb_shape,
  const std::vector<int64_t> &cache_k_shape,
  const std::vector<int64_t> &cache_v_shape,
  const std::vector<int64_t> &seq_lens_encoder_shape,
  const std::vector<int64_t> &seq_lens_decoder_shape,
  const std::vector<int64_t> &block_tables_shape,
  int32_t flag,
  int32_t max_seq_len,
  int32_t block_size,
  float qkv_quant_scale,
  bool trans_qkv,
  bool use_neox_style,
  bool use_alibi) {
return {{-1, hidden_shape[1]}};
}


PD_BUILD_OP(fused_fapa_attention_op)
    .Inputs({"hidden",
             "qkv_weight", 
             "qkv_bias@OPTIONAL",
             "qkv_deq_scale@OPTIONAL",
             "rope_emb",        
             "cache_k",
             "cache_v",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "block_tables"})
    .Outputs({"attn_out"})
    .Attrs({"flag: int",  // begin: 1, end: 2, other: 0
            "max_seq_len: int",
            "block_size: int",
            "qkv_quant_scale: float",
            "trans_qkv: bool",
            "use_neox_style: bool",
            "use_alibi: bool"})
    .SetKernelFn(PD_KERNEL(FusedFaPaAttnOp))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedFaPaAttnOpInferShape));


#endif
