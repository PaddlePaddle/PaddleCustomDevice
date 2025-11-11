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
#include "fused_fapa_attn_op_utils.h"  // NOLINT

constexpr int32_t kFusedFaPaAttnLayerBegin = 1;
constexpr int32_t kFusedFaPaAttnLayerEnd = 2;

static bool first_or_second_flag = false;
static int layer_id = 0;

void FusedFaPaAttnLayerOpPrefillStage(const phi::CustomContext &dev_ctx,
                                      const paddle::Tensor &qkv_out,
                                      const paddle::Tensor &cos,
                                      const paddle::Tensor &sin,
                                      const paddle::Tensor &cache_k,
                                      const paddle::Tensor &cache_v,
                                      int64_t max_seq_len,
                                      int64_t head_num,
                                      int64_t kv_head_num,
                                      int64_t head_dim,
                                      int64_t emb_dim,
                                      int64_t ntokens,
                                      bool use_neox_style,
                                      bool use_alibi) {
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
    param.rope_neox = use_neox_style;
    param.head_num = head_num;
    param.kv_head_num = kv_head_num;
    param.head_dim = head_dim;
    param.is_prefill = true;
    runner.create(param);
  }

  runner.bind_input(qkv_out);

  if (!use_alibi) {
    runner.bind_input(cos, {ntokens, head_dim});
    runner.bind_input(sin, {ntokens, head_dim});
  }
  if (!use_alibi) {
    void *mask_data = FusedFaPaGlobalVar::Instance().get_casual_mask();
    runner.bind_input(
        mask_data, phi::DataType::BFLOAT16, {max_seq_len, max_seq_len});
  } else {
    void *mask_data = FusedFaPaGlobalVar::Instance().get_alibi_src_mask();
    // 1, head_num, max_seq_len, max_seq_len
    runner.bind_input(mask_data,
                      phi::DataType::BFLOAT16,
                      {batch_size, head_num, max_seq_len, max_seq_len});
  }

  runner.bind_input(cache_k);
  runner.bind_input(cache_v);
  runner.bind_input(slots_data, phi::DataType::INT32, {ntokens});
  runner.bind_input(
      seqlens_dev, seqlens_host, phi::DataType::INT32, {batch_size});
  if (first_or_second_flag) {
    runner.bind_output(
        first_out_data, phi::DataType::BFLOAT16, {ntokens, emb_dim});
  } else {
    runner.bind_output(
        second_out_data, phi::DataType::BFLOAT16, {ntokens, emb_dim});
  }

  runner.setup(dev_ctx);
  atb_layers::TaskQueue::Instance(dev_ctx.GetPlace().GetDeviceId())
      .Commit(std::move(
          std::packaged_task<void(void)>([&] { runner.execute(dev_ctx); })));
}

void FusedFaPaAttnLayerOpDecodingStage(const phi::CustomContext &dev_ctx,
                                       const paddle::Tensor &qkv_out,
                                       const paddle::Tensor &cos,
                                       const paddle::Tensor &sin,
                                       const paddle::Tensor &cache_k,
                                       const paddle::Tensor &cache_v,
                                       const paddle::Tensor &block_tables,
                                       int64_t max_seq_len,
                                       int64_t head_num,
                                       int64_t kv_head_num,
                                       int64_t head_dim,
                                       int64_t emb_dim,
                                       int64_t ntokens,
                                       bool use_neox_style,
                                       bool use_alibi) {
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
    param.rope_neox = use_neox_style;
    param.head_num = head_num;
    param.kv_head_num = kv_head_num;
    param.head_dim = head_dim;
    param.is_prefill = false;
    runner.create(param);
  }

  runner.bind_input(qkv_out);

  if (!use_alibi) {
    runner.bind_input(cos, {ntokens, head_dim});
    runner.bind_input(sin, {ntokens, head_dim});
  }
  if (use_alibi) {
    void *mask_data = FusedFaPaGlobalVar::Instance().get_alibi_tgt_mask();
    // batch, head_num, 1, max_seq_len
    runner.bind_input(mask_data,
                      phi::DataType::BFLOAT16,
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
        first_out_data, phi::DataType::BFLOAT16, {ntokens, emb_dim});
  } else {
    runner.bind_output(
        second_out_data, phi::DataType::BFLOAT16, {ntokens, emb_dim});
  }

  runner.setup(dev_ctx);
  atb_layers::TaskQueue::Instance(dev_ctx.GetPlace().GetDeviceId())
      .Commit(std::move(
          std::packaged_task<void(void)>([&] { runner.execute(dev_ctx); })));
}

std::vector<paddle::Tensor> FusedFaPaAttnOp(
    const paddle::Tensor &qkv_out,
    const paddle::Tensor &cos,
    const paddle::Tensor &sin,
    const paddle::Tensor &cache_k,
    const paddle::Tensor &cache_v,
    const paddle::Tensor &seq_lens_encoder,
    const paddle::Tensor &seq_lens_decoder,
    const paddle::Tensor &block_tables,
    int32_t head_num,
    int32_t kv_head_num,
    int32_t head_dim,
    int32_t flag,
    int32_t max_seq_len,
    int32_t block_size,
    bool use_neox_style,
    bool use_alibi) {
  const auto &cache_k_shape = cache_k.shape();
  const auto &block_tables_shape = block_tables.shape();
  uint64_t emb_dim = head_num * head_dim;
  uint64_t max_block_num_per_seq = block_tables_shape[1];
  uint64_t batch_size = seq_lens_encoder.numel();

  auto place = qkv_out.place();
  const auto &dev_ctx = *static_cast<const phi::CustomContext *>(
      paddle::experimental::DeviceContextPool::Instance().Get(place));
  if (flag == kFusedFaPaAttnLayerBegin) {
    FusedFaPaGlobalVar::Instance().update_block_tables(dev_ctx, block_tables);
    FusedFaPaGlobalVar::Instance().update_seqlens_encoder(dev_ctx,
                                                          seq_lens_encoder);
    FusedFaPaGlobalVar::Instance().update_seqlens_decoder(dev_ctx,
                                                          seq_lens_decoder);

    FusedFaPaGlobalVar::Instance().update_casual_mask(dev_ctx, max_seq_len);

    FusedFaPaGlobalVar::Instance().update_slots_encoder(
        dev_ctx, block_size, max_block_num_per_seq);
    FusedFaPaGlobalVar::Instance().update_slots_decoder(
        dev_ctx, block_size, max_block_num_per_seq);

    FusedFaPaGlobalVar::Instance().update_in_encoder(dev_ctx, qkv_out);
    FusedFaPaGlobalVar::Instance().update_in_decoder(dev_ctx, qkv_out);

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

  int32_t ntokens = -1;
  if (ntokens_encoder > 0) {
    FusedFaPaAttnLayerOpPrefillStage(dev_ctx,
                                     qkv_out,
                                     cos,
                                     sin,
                                     cache_k,
                                     cache_v,
                                     max_seq_len,
                                     head_num,
                                     kv_head_num,
                                     head_dim,
                                     emb_dim,
                                     ntokens_encoder,
                                     use_neox_style,
                                     use_alibi);
    ntokens = ntokens_encoder;
  }
  if (ntokens_decoder > 0) {
    FusedFaPaAttnLayerOpDecodingStage(dev_ctx,
                                      qkv_out,
                                      cos,
                                      sin,
                                      cache_k,
                                      cache_v,
                                      block_tables,
                                      max_seq_len,
                                      head_num,
                                      kv_head_num,
                                      head_dim,
                                      emb_dim,
                                      ntokens_decoder,
                                      use_neox_style,
                                      use_alibi);
    ntokens = ntokens_decoder;
  }

  paddle::Tensor out(place);
  atb_layers::TaskQueue::Instance(dev_ctx.GetPlace().GetDeviceId()).Wait();

  fapa_layers::init_tensor(
      dev_ctx, phi::DataType::BFLOAT16, {ntokens, emb_dim}, &out);
  FusedFaPaGlobalVar::Instance().update_out_encoder(
      dev_ctx, first_or_second_flag, &out);
  FusedFaPaGlobalVar::Instance().update_out_decoder(
      dev_ctx, first_or_second_flag, &out);
  return {out};
}

std::vector<std::vector<int64_t>> FusedFaPaAttnOpInferShape(
    const std::vector<int64_t> &qkv_out_shape,
    const std::vector<int64_t> &cos_shape,
    const std::vector<int64_t> &sin_shape,
    const std::vector<int64_t> &cache_k_shape,
    const std::vector<int64_t> &cache_v_shape,
    const std::vector<int64_t> &seq_lens_encoder_shape,
    const std::vector<int64_t> &seq_lens_decoder_shape,
    const std::vector<int64_t> &block_tables_shape,
    int32_t head_num,
    int32_t kv_head_num,
    int32_t head_dim,
    int32_t flag,
    int32_t max_seq_len,
    int32_t block_size,
    bool use_neox_style,
    bool use_alibi) {
  return {{-1, qkv_out_shape[1]}};
}

PD_BUILD_OP(fused_fapa_attention_op)
    .Inputs({"qkv_out",
             "cos",
             "sin",
             "cache_k",
             "cache_v",
             "seq_lens_encoder",
             "seq_lens_decoder",
             "block_tables"})
    .Outputs({"attn_out"})
    .Attrs({"num_heads: int",
            "kv_num_heads: int",
            "head_dim: int",
            "flag: int",  // begin: 1, end: 2, other: 0
            "max_seq_len: int",
            "block_size: int",
            "use_neox_style: bool",
            "use_alibi: bool"})
    .SetKernelFn(PD_KERNEL(FusedFaPaAttnOp))
    .SetInferShapeFn(PD_INFER_SHAPE(FusedFaPaAttnOpInferShape));

#endif
