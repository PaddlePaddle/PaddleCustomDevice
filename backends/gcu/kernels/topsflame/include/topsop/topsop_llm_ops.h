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

/*
 * Copyright 2022-2023 Enflame. All Rights Reserved.

 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 *     @defgroup ops
 *     @{
 *
 */

/**
 * @file topsop_llm_ops.h
 * @brief topsflame common ops api definitions.
 */

#ifndef TOPSOP_LLM_OPS_H_  // NOLINT
#define TOPSOP_LLM_OPS_H_

#include "tops/tops_runtime.h"
#include "topsop/topsop_define.h"

#if defined(__cplusplus)
extern "C" {
#endif

/** topsopNormMode_t */
typedef enum {
  TOPSOP_LAYERNORM = 0, /**< TOPSOP_LAYERNORM 0 */
  TOPSOP_RMSNORM = 1,   /**< TOPSOP_RMSNORM 1 */
} topsopNormMode_t;

typedef enum {
  TOPSOP_ROTARY_EMB_NO = 0,           /**< no rotary_emb */
  TOPSOP_ROTARY_EMB_1D = 1,           /**< baicuan  */
  TOPSOP_ROTARY_EMB_1D_HALF = 2,      /**< glm2 half tensor rotary elemwise*/
  TOPSOP_ROTARY_EMB_1D_TRANSPOSE = 3, /**< llama facebook  1 */
  TOPSOP_ROTARY_EMB_2D = 4,           /**< glm1 position 2d elewise */
  TOPSOP_ROTARY_EMB_1D_HALF_FP32 = 5, /**< glm3 half rotary with fp32 freq*/
  TOPSOP_ROTARY_EMB_WITH_PCT = 6,     /**< gptneox with rope pct */
  TOPSOP_ROTARY_EMB_1D_FP32 = 7,      /**< baicuan  */
  TOPSOP_ROTARY_EMB_1D_TRANSPOSE_FP32 = 8, /**< llama facebook  1 */
} topsopRotaryEmbeddingMode_t;

/** topsopWeightPreprocessKind_t */
typedef enum {
  TOPSOP_WEIGHT_PAD = 0,
  TOPSOP_WEIGHT_BITCAST = 1,
  TOPSOP_WEIGHT_TRANSPOSE = 2,
} topsopWeightPreprocessKind_t;

typedef enum {
  TOPSOP_LLM_EMB = 0,
  TOPSOP_LLM_MHA = 1,
  TOPSOP_LLM_MLP = 2,
  TOPSOP_LLM_DOT = 3,
} topsopLLMOpType_t;

/** topsopWeightPreprocess_t */
struct topsopWeightPreprocess_t {
  topsopWeightPreprocess_t *next;
  topsopWeightPreprocessKind_t kind;
  int src_shape[4];
  int dst_shape[4];
  // for pad       : args is padding_low,
  //                 dst_shape = src_shape + reshape;
  // for bitcast   : args is INVALID;
  // for transpose : args is transpose layout;
  int args[4];
};

/** topsopDotWeightHelper_t */
struct topsopDotWeightHelper_t {
  int tiling[2];                 // in format [sip_K, sip_N]
  topsopWeightPreprocess_t *wp;  // weight preprocess operation list
};

struct DotPreprocessInfo {
  // padding relevant members
  bool need_padding;
  int64_t n_tiling_dim;  // l2 tiling dim along n
  int64_t k_tiling_dim;  // l2 tiling dim along k
  int64_t padding_shape[4];

  // swap relevant members
  bool need_swap;

  // slice relevant members
  int64_t mc_uses;
  int64_t bitcast_shape[4];
  int64_t
      bank_use_id[4];  // {0, 1, 2, 3} represent used mc bank id = 0, 1, 2, 3
  int64_t dim_per_bank[4];       // dim on each mc, equal across mcs
  int64_t win_pos_per_bank[16];  // sub window offset on mc dim

  // transpose relevant members
  int64_t permute[4];  // transpose layout, default {0, 1, 2, 3}
  int64_t permute_shape[4];

  // tensorhandle params
  int64_t tensor_dims[4];  // shape after padding
  int64_t rank;            // tensor rank
  int64_t stride[4];       // sub window stride across mcs
};

bool DotWeightConsult(int64_t *shape,
                      int64_t dim,
                      topsopDataType_t data_type,
                      DotPreprocessInfo *helper,
                      topsopLLMOpType_t op_type);

/**
 * @brief MLPNorm operator
 *
 * @param output: the output tensor handle of the operator
 * @param input: the input tensor handle of the operator
 * @param weight_A: the weight tensor handle of dot_A
 * @param bias_A: the bias tensor handle of dot_A
 * @param input_contract_dim: The "k-dim" in x
 * @param weight_A_contract_dim: The "k-dim" in weight_A
 * @param weight_B_contract_dim: The "k-dim" in weight_B
 * @param input_batch_dim: The "batch-dim" in input
 * @param weight_B: the weight tensor handle of dot_B
 * @param bias_B: the bias tensor handle of dot_B
 * @param residual_scale: multiply the mlp input with this scale,
 *                        then add to mlp output.
 * @param pre_norm_scale: scale tensor of pre norm.
 * @param pre_norm_bias: bias tensor of pre norm.
 * @param post_norm_scale: scale tensor of post norm.
 * @param post_norm_bias: bias tensor of post norm.
 * @param weight_A_quant_scale: the quant scale of weight A
 * @param weight_B_quant_scale: the quant scale of weight B
 * @param activation_type: choose activation type in MLP
 * @param norm_type: choose Norm type in MLP
 * @param use_residual_add: option to choose if use residual add
 * @param alpha: dummy param, alwayse be 1.0
 * @param beta: dummy param, alwayse be 0.0
 * @param stream: tops stream
 */
topsopStatus_t topsopMLPNorm(topsopTensorHandle_t output,
                             const topsopTensorHandle_t input,
                             const topsopTensorHandle_t weight_A,
                             const topsopTensorHandle_t bias_A,
                             const topsopTensorHandle_t weight_B,
                             const topsopTensorHandle_t bias_B,
                             const topsopTensorHandle_t pre_norm_scale,
                             const topsopTensorHandle_t pre_norm_bias,
                             const topsopTensorHandle_t post_norm_scale,
                             const topsopTensorHandle_t post_norm_bias,
                             const topsopTensorHandle_t weight_A_quant_scale,
                             const topsopTensorHandle_t weight_B_quant_scale,

                             const float residual_scale,
                             const bool use_residual_add,
                             const topsopActivationMode_t activation_type,
                             const topsopNormMode_t norm_type,
                             const int max_seq_len,
                             const int hidden_size,

                             const topsopSize_t *input_contract_dim,
                             const topsopSize_t *weight_A_contract_dim,
                             const topsopSize_t *weight_B_contract_dim,
                             const topsopSize_t *input_batch_dim,

                             const topsopScalar_t alpha,
                             const topsopScalar_t beta,
                             topsStream_t stream);

/**
 * @brief check whether MLPNorm operator is supported or not with
 * current input tensor
 *
 * @param input: the input tensor handle of the operator
 * @param weight_A: the weight tensor handle of dot_A
 * @param bias_A: the bias tensor handle of dot_A
 * @param input_contract_dim: The "k-dim" in x
 * @param weight_A_contract_dim: The "k-dim" in weight_A
 * @param weight_B_contract_dim: The "k-dim" in weight_B
 * @param input_batch_dim: The "batch-dim" in input
 * @param weight_B: the weight tensor handle of dot_B
 * @param bias_B: the bias tensor handle of dot_B
 * @param residual_scale: multiply the mlp input with this scale,
 *                        then add to mlp output.
 * @param pre_norm_scale: scale tensor of pre norm.
 * @param pre_norm_bias: bias tensor of pre norm.
 * @param post_norm_scale: scale tensor of post norm.
 * @param post_norm_bias: bias tensor of post norm.
 * @param weight_A_quant_scale: the quant scale of weight A
 * @param weight_B_quant_scale: the quant scale of weight B
 * @param activation_type: choose activation type in MLP
 * @param norm_type: choose Norm type in MLP
 * @param use_residual_add: option to choose if use residual add
 *
 * @return bool
 */
bool TOPSOP_EXPORT
topsopMLPNormIsSupported(const topsopTensorHandle_t input,
                         const topsopTensorHandle_t weight_A,
                         const topsopTensorHandle_t bias_A,
                         const topsopTensorHandle_t weight_B,
                         const topsopTensorHandle_t bias_B,
                         const topsopTensorHandle_t pre_norm_scale,
                         const topsopTensorHandle_t pre_norm_bias,
                         const topsopTensorHandle_t post_norm_scale,
                         const topsopTensorHandle_t post_norm_bias,
                         const topsopTensorHandle_t weight_A_quant_scale,
                         const topsopTensorHandle_t weight_B_quant_scale,

                         const float residual_scale,
                         const bool use_residual_add,
                         const topsopActivationMode_t activation_type,
                         const topsopNormMode_t norm_type,
                         const int max_seq_len,
                         const int hidden_size,

                         const topsopSize_t *input_contract_dim,
                         const topsopSize_t *weight_A_contract_dim,
                         const topsopSize_t *weight_B_contract_dim,
                         const topsopSize_t *input_batch_dim);

/**
 * @brief get the dimensions and rank of output tensor
 *
 * @param input: the input tensor handle of the operator
 * @param weight_A: the weight tensor handle of dot_A
 * @param bias_A: the bias tensor handle of dot_A
 * @param input_contract_dim: The "k-dim" in x
 * @param weight_A_contract_dim: The "k-dim" in weight_A
 * @param weight_B_contract_dim: The "k-dim" in weight_B
 * @param input_batch_dim: The "batch-dim" in input
 * @param weight_B: the weight tensor handle of dot_B
 * @param bias_B: the bias tensor handle of dot_B
 * @param residual_scale: multiply the mlp input with this scale,
 *                        then add to mlp output.
 * @param pre_norm_scale: scale tensor of pre norm.
 * @param pre_norm_bias: bias tensor of pre norm.
 * @param post_norm_scale: scale tensor of post norm.
 * @param post_norm_bias: bias tensor of post norm.
 * @param weight_A_quant_scale: the quant scale of weight A
 * @param weight_B_quant_scale: the quant scale of weight B
 * @param activation_type: choose activation type in MLP
 * @param norm_type: choose Norm type in MLP
 * @param use_residual_add: option to choose if use residual add
 *
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return bool
 */

topsopStatus_t TOPSOP_EXPORT
topsopMLPNormGetOutputDim(const topsopTensorHandle_t input,
                          const topsopTensorHandle_t weight_A,
                          const topsopTensorHandle_t bias_A,
                          const topsopTensorHandle_t weight_B,
                          const topsopTensorHandle_t bias_B,
                          const topsopTensorHandle_t pre_norm_scale,
                          const topsopTensorHandle_t pre_norm_bias,
                          const topsopTensorHandle_t post_norm_scale,
                          const topsopTensorHandle_t post_norm_bias,
                          const topsopTensorHandle_t weight_A_quant_scale,
                          const topsopTensorHandle_t weight_B_quant_scale,

                          const float residual_scale,
                          const bool use_residual_add,
                          const topsopActivationMode_t activation_type,
                          const topsopNormMode_t norm_type,
                          const int max_seq_len,
                          const int hidden_size,

                          const topsopSize_t *input_contract_dim,
                          const topsopSize_t *weight_A_contract_dim,
                          const topsopSize_t *weight_B_contract_dim,
                          const topsopSize_t *input_batch_dim,

                          int64_t *dims,
                          int64_t *rank);

/**
 * @brief MultiQueryAttentionNorm operator
 *
 * @param output: the output tensor handle of the operator
 * @param cache_K: the cache_K tensor handle of the operator
 * @param cache_V: the cache_V tensor handle of the operator
 * @param input: the input tensor handle of the operator
 *
 * @param weight_Q: the weight tensor handle of Q-dot
 * @param bias_Q: the bias tensor handle of Q-dot
 *
 * @param weight_K: the weight tensor handle of K-dot
 * @param bias_K: the bias tensor handle of K-dot
 *
 * @param weight_V: the weight tensor handle of V-dot
 * @param bias_V: the bias tensor handle of V-dot
 *
 * @param input_contract_dim: The "k-dim" in x
 * @param weight_Q_contract_dim: The "k-dim" in weight_Q
 * @param weight_K_contract_dim: The "k-dim" in weight_K
 * @param weight_V_contract_dim: The "k-dim" in weight_V
 * @param weight_context_contract_dim: The "k-dim" in context_layer_dot_weight
 * @param input_batch_dim: The "batch-dim" in input
 *
 * @param freq: the frequency tensor of rotary_embedding
 * @param posititon_id: the position tensor of rotary_embedding
 *
 * @param mask: mask tensor of softmax in SDP
 * @param weight_Q_quant_scale: the quant scale of weight Q
 * @param weight_K_quant_scale: the quant scale of weight K
 * @param weight_V_quant_scale: the quant scale of weight V
 * @param weight_C_quant_scale: the quant scale of weight C
 * @param qk_scale_coeff: should be layer_id + 1, the scale before softmax
 *        operation and after Q-dot operation
 *
 * @param context_layer_dot_weight: the weight tensor handle of
 context_layer_dot
 * @param context_layer_dot_bias: the bias tensor handle of context_layer_dot
 * @param ContractingDimension_context: The "k-dim" in context_layer_dot
 * @param BatchDimension_context: The "batch-dim" in context_layer_dot(should be
 dummy)
 *
 * @param residual_scale: multiply the qkv input with this scale, then add to
 context_layer_dot output.
 * @param pre_norm_scale: scale tensor of pre norm.
 * @param pre_norm_bias: bias tensor of pre norm.
 * @param post_norm_scale: scale tensor of post norm.
 * @param post_norm_bias: bias tensor of post norm.
 *
 * @param use_residual_add: option to choose if use residual add
 * @param use_qk_rotary_embedding: option to choose if use qk rotary embedding
 * @param norm_type: choose Norm type in MultiQueryAttention
 * @param cache_length: cache length of current KV cache
 *
 * @param alpha: dummy param, alwayse be 1.0
 * @param beta: dummy param, alwayse be 0.0
 * @param stream: tops stream
 *
 */
topsopStatus_t topsopMultiQueryAttentionNormQkbias(
    topsopTensorHandle_t output,
    topsopTensorHandle_t *cache_K,
    topsopTensorHandle_t *cache_V,
    int num_batch,
    const topsopTensorHandle_t input,
    const topsopTensorHandle_t weight_Q,
    const topsopTensorHandle_t bias_Q,
    const topsopTensorHandle_t weight_K,
    const topsopTensorHandle_t bias_K,
    const topsopTensorHandle_t weight_V,
    const topsopTensorHandle_t bias_V,
    const topsopTensorHandle_t context_layer_dot_weight,
    const topsopTensorHandle_t context_layer_dot_bias,
    const topsopTensorHandle_t pre_norm_scale,
    const topsopTensorHandle_t pre_norm_bias,
    const topsopTensorHandle_t post_norm_scale,
    const topsopTensorHandle_t post_norm_bias,
    const topsopTensorHandle_t freq,
    topsopTensorHandle_t *posititon_id,
    int num_pid,
    const topsopTensorHandle_t mask,
    const topsopTensorHandle_t weight_Q_quant_scale,
    const topsopTensorHandle_t weight_K_quant_scale,
    const topsopTensorHandle_t weight_V_quant_scale,
    const topsopTensorHandle_t weight_C_quant_scale,
    const topsopTensorHandle_t qk_alibi_bias,

    const float qk_scale_coeff,
    const float residual_scale,
    const bool use_residual_add,
    const topsopRotaryEmbeddingMode_t use_qk_rotary_embedding,
    const topsopNormMode_t norm_type,
    int *cache_length,
    int num_cache_length,
    const int max_seq_len,
    const int hidden_size,

    const topsopSize_t *input_contract_dim,
    const topsopSize_t *weight_Q_contract_dim,
    const topsopSize_t *weight_K_contract_dim,
    const topsopSize_t *weight_V_contract_dim,
    const topsopSize_t *weight_context_contract_dim,
    const topsopSize_t *input_batch_dim,

    const topsopScalar_t alpha,
    const topsopScalar_t beta,
    topsStream_t stream);

/**
 * @brief MultiQueryAttentionNorm operator
 *
 * @param output: the output tensor handle of the operator
 * @param cache_K: the cache_K tensor handle of the operator
 * @param cache_V: the cache_V tensor handle of the operator
 * @param input: the input tensor handle of the operator
 *
 * @param weight_Q: the weight tensor handle of Q-dot
 * @param bias_Q: the bias tensor handle of Q-dot
 *
 * @param weight_K: the weight tensor handle of K-dot
 * @param bias_K: the bias tensor handle of K-dot
 *
 * @param weight_V: the weight tensor handle of V-dot
 * @param bias_V: the bias tensor handle of V-dot
 *
 * @param input_contract_dim: The "k-dim" in x
 * @param weight_Q_contract_dim: The "k-dim" in weight_Q
 * @param weight_K_contract_dim: The "k-dim" in weight_K
 * @param weight_V_contract_dim: The "k-dim" in weight_V
 * @param weight_context_contract_dim: The "k-dim" in context_layer_dot_weight
 * @param input_batch_dim: The "batch-dim" in input
 *
 * @param freq: the frequency tensor of rotary_embedding
 * @param posititon_id: the position tensor of rotary_embedding
 *
 * @param mask: mask tensor of softmax in SDP
 * @param weight_Q_quant_scale: the quant scale of weight Q
 * @param weight_K_quant_scale: the quant scale of weight K
 * @param weight_V_quant_scale: the quant scale of weight V
 * @param weight_C_quant_scale: the quant scale of weight C
 * @param weight_Q_quant_zero: the quant zero of weight Q
 * @param weight_K_quant_zero: the quant zero of weight K
 * @param weight_V_quant_zero: the quant zero of weight V
 * @param weight_C_quant_zero: the quant zero of weight C
 * @param qk_scale_coeff: should be layer_id + 1, the scale before softmax
 *        operation and after Q-dot operation
 *
 * @param context_layer_dot_weight: the weight tensor handle of
 context_layer_dot
 * @param context_layer_dot_bias: the bias tensor handle of context_layer_dot
 * @param ContractingDimension_context: The "k-dim" in context_layer_dot
 * @param BatchDimension_context: The "batch-dim" in context_layer_dot(should be
 dummy)
 *
 * @param residual_scale: multiply the qkv input with this scale, then add to
 context_layer_dot output.
 * @param pre_norm_scale: scale tensor of pre norm.
 * @param pre_norm_bias: bias tensor of pre norm.
 * @param post_norm_scale: scale tensor of post norm.
 * @param post_norm_bias: bias tensor of post norm.
 *
 * @param use_residual_add: option to choose if use residual add
 * @param use_qk_rotary_embedding: option to choose if use qk rotary embedding
 * @param norm_type: choose Norm type in MultiQueryAttention
 * @param cache_length: cache length of current KV cache
 *
 * @param k_cache_scale: w8a16+ need for k_cache quant
 * @param v_cache_scale: w8a16+ need for v_cache quant
 *
 * @param alpha: dummy param, alwayse be 1.0
 * @param beta: dummy param, alwayse be 0.0
 * @param stream: tops stream
 *
 */
topsopStatus_t topsopMultiQueryAttentionNormQuant(
    topsopTensorHandle_t output,
    topsopTensorHandle_t *cache_K,
    topsopTensorHandle_t *cache_V,
    int num_batch,
    const topsopTensorHandle_t input,
    const topsopTensorHandle_t weight_Q,
    const topsopTensorHandle_t bias_Q,
    const topsopTensorHandle_t weight_K,
    const topsopTensorHandle_t bias_K,
    const topsopTensorHandle_t weight_V,
    const topsopTensorHandle_t bias_V,
    const topsopTensorHandle_t context_layer_dot_weight,
    const topsopTensorHandle_t context_layer_dot_bias,
    const topsopTensorHandle_t pre_norm_scale,
    const topsopTensorHandle_t pre_norm_bias,
    const topsopTensorHandle_t post_norm_scale,
    const topsopTensorHandle_t post_norm_bias,
    const topsopTensorHandle_t freq,
    topsopTensorHandle_t *posititon_id,
    int num_pid,
    const topsopTensorHandle_t mask,
    const topsopTensorHandle_t weight_Q_quant_scale,
    const topsopTensorHandle_t weight_K_quant_scale,
    const topsopTensorHandle_t weight_V_quant_scale,
    const topsopTensorHandle_t weight_C_quant_scale,
    const topsopTensorHandle_t weight_Q_quant_zero,
    const topsopTensorHandle_t weight_K_quant_zero,
    const topsopTensorHandle_t weight_V_quant_zero,
    const topsopTensorHandle_t weight_C_quant_zero,
    const topsopTensorHandle_t qk_alibi_bias,

    const float qk_scale_coeff,
    const float residual_scale,
    const float k_cache_scale,
    const float v_cache_scale,
    const float k_cache_inv_scale,
    const float v_cache_inv_scale,
    const float k_cache_zero,
    const float v_cache_zero,
    const bool use_residual_add,
    const topsopRotaryEmbeddingMode_t use_qk_rotary_embedding,
    const topsopNormMode_t norm_type,
    int *cache_length,
    int num_cache_length,
    const int max_seq_len,
    const int hidden_size,

    const topsopSize_t *input_contract_dim,
    const topsopSize_t *weight_Q_contract_dim,
    const topsopSize_t *weight_K_contract_dim,
    const topsopSize_t *weight_V_contract_dim,
    const topsopSize_t *weight_context_contract_dim,
    const topsopSize_t *input_batch_dim,

    const topsopScalar_t alpha,
    const topsopScalar_t beta,
    topsStream_t stream);

/**
 * @brief MultiQueryAttentionNorm operator
 *
 * @param output: the output tensor handle of the operator
 * @param cache_K: the cache_K tensor handle of the operator
 * @param cache_V: the cache_V tensor handle of the operator
 * @param input: the input tensor handle of the operator
 *
 * @param weight_Q: the weight tensor handle of Q-dot
 * @param bias_Q: the bias tensor handle of Q-dot
 *
 * @param weight_K: the weight tensor handle of K-dot
 * @param bias_K: the bias tensor handle of K-dot
 *
 * @param weight_V: the weight tensor handle of V-dot
 * @param bias_V: the bias tensor handle of V-dot
 *
 * @param input_contract_dim: The "k-dim" in x
 * @param weight_Q_contract_dim: The "k-dim" in weight_Q
 * @param weight_K_contract_dim: The "k-dim" in weight_K
 * @param weight_V_contract_dim: The "k-dim" in weight_V
 * @param weight_context_contract_dim: The "k-dim" in context_layer_dot_weight
 * @param input_batch_dim: The "batch-dim" in input
 *
 * @param freq: the frequency tensor of rotary_embedding
 * @param posititon_id: the position tensor of rotary_embedding
 *
 * @param mask: mask tensor of softmax in SDP
 * @param weight_Q_quant_scale: the quant scale of weight Q
 * @param weight_K_quant_scale: the quant scale of weight K
 * @param weight_V_quant_scale: the quant scale of weight V
 * @param weight_C_quant_scale: the quant scale of weight C
 * @param qk_scale_coeff: should be layer_id + 1, the scale before softmax
 *        operation and after Q-dot operation
 *
 * @param context_layer_dot_weight: the weight tensor handle of
 context_layer_dot
 * @param context_layer_dot_bias: the bias tensor handle of context_layer_dot
 * @param ContractingDimension_context: The "k-dim" in context_layer_dot
 * @param BatchDimension_context: The "batch-dim" in context_layer_dot(should be
 dummy)
 *
 * @param residual_scale: multiply the qkv input with this scale, then add to
 context_layer_dot output.
 * @param pre_norm_scale: scale tensor of pre norm.
 * @param pre_norm_bias: bias tensor of pre norm.
 * @param post_norm_scale: scale tensor of post norm.
 * @param post_norm_bias: bias tensor of post norm.
 *
 * @param use_residual_add: option to choose if use residual add
 * @param use_qk_rotary_embedding: option to choose if use qk rotary embedding
 * @param norm_type: choose Norm type in MultiQueryAttention
 * @param cache_length: cache length of current KV cache
 *
 * @param alpha: dummy param, alwayse be 1.0
 * @param beta: dummy param, alwayse be 0.0
 * @param stream: tops stream
 *
 */
topsopStatus_t topsopMultiQueryAttentionNorm(
    topsopTensorHandle_t output,
    topsopTensorHandle_t *cache_K,
    topsopTensorHandle_t *cache_V,
    int num_batch,
    const topsopTensorHandle_t input,
    const topsopTensorHandle_t weight_Q,
    const topsopTensorHandle_t bias_Q,
    const topsopTensorHandle_t weight_K,
    const topsopTensorHandle_t bias_K,
    const topsopTensorHandle_t weight_V,
    const topsopTensorHandle_t bias_V,
    const topsopTensorHandle_t context_layer_dot_weight,
    const topsopTensorHandle_t context_layer_dot_bias,
    const topsopTensorHandle_t pre_norm_scale,
    const topsopTensorHandle_t pre_norm_bias,
    const topsopTensorHandle_t post_norm_scale,
    const topsopTensorHandle_t post_norm_bias,
    const topsopTensorHandle_t freq,
    topsopTensorHandle_t *posititon_id,
    int num_pid,
    const topsopTensorHandle_t mask,
    const topsopTensorHandle_t weight_Q_quant_scale,
    const topsopTensorHandle_t weight_K_quant_scale,
    const topsopTensorHandle_t weight_V_quant_scale,
    const topsopTensorHandle_t weight_C_quant_scale,

    const float qk_scale_coeff,
    const float residual_scale,
    const bool use_residual_add,
    const topsopRotaryEmbeddingMode_t use_qk_rotary_embedding,
    const topsopNormMode_t norm_type,
    int *cache_length,
    int num_cache_length,
    const int max_seq_len,
    const int hidden_size,

    const topsopSize_t *input_contract_dim,
    const topsopSize_t *weight_Q_contract_dim,
    const topsopSize_t *weight_K_contract_dim,
    const topsopSize_t *weight_V_contract_dim,
    const topsopSize_t *weight_context_contract_dim,
    const topsopSize_t *input_batch_dim,

    const topsopScalar_t alpha,
    const topsopScalar_t beta,
    topsStream_t stream);

/**
 * @brief check whether MultiQueryAttentionNorm operator is supported or not
 with
 * current input tensor
 *
 * @param input: the input tensor handle of the operator
 *
 * @param weight_Q: the weight tensor handle of Q-dot
 * @param bias_Q: the bias tensor handle of Q-dot
 *
 * @param weight_K: the weight tensor handle of K-dot
 * @param bias_K: the bias tensor handle of K-dot
 *
 * @param weight_V: the weight tensor handle of V-dot
 * @param bias_V: the bias tensor handle of V-dot
 *
 * @param input_contract_dim: The "k-dim" in x
 * @param weight_Q_contract_dim: The "k-dim" in weight_Q
 * @param weight_K_contract_dim: The "k-dim" in weight_K
 * @param weight_V_contract_dim: The "k-dim" in weight_V
 * @param weight_context_contract_dim: The "k-dim" in context_layer_dot_weight
 * @param input_batch_dim: The "batch-dim" in input
 *
 * @param freq: the frequency tensor of rotary_embedding
 * @param posititon_id: the position tensor of rotary_embedding
 *
 * @param mask: mask tensor of softmax in SDP
 * @param weight_Q_quant_scale: the quant scale of weight Q
 * @param weight_K_quant_scale: the quant scale of weight K
 * @param weight_V_quant_scale: the quant scale of weight V
 * @param weight_C_quant_scale: the quant scale of weight C
 * @param qk_scale_coeff: should be layer_id + 1, the scale before softmax
 *        operation and after Q-dot operation
 *
 * @param context_layer_dot_weight: the weight tensor handle of
 context_layer_dot
 * @param context_layer_dot_bias: the bias tensor handle of context_layer_dot
 * @param ContractingDimension_context: The "k-dim" in context_layer_dot
 * @param BatchDimension_context: The "batch-dim" in context_layer_dot(should be
 dummy)
 *
 * @param residual_scale: multiply the qkv input with this scale, then add to
 context_layer_dot output.
 * @param pre_norm_scale: scale tensor of pre norm.
 * @param pre_norm_bias: bias tensor of pre norm.
 * @param post_norm_scale: scale tensor of post norm.
 * @param post_norm_bias: bias tensor of post norm.
 *
 * @param use_residual_add: option to choose if use residual add
 * @param use_qk_rotary_embedding: option to choose if use qk rotary embedding
 * @param norm_type: choose Norm type in MultiQueryAttention
 * @param cache_length: cache length of current KV cache
 *
 * @return bool
 */
bool TOPSOP_EXPORT topsopMultiQueryAttentionNormIsSupported(
    const topsopTensorHandle_t input,
    const topsopTensorHandle_t weight_Q,
    const topsopTensorHandle_t bias_Q,
    const topsopTensorHandle_t weight_K,
    const topsopTensorHandle_t bias_K,
    const topsopTensorHandle_t weight_V,
    const topsopTensorHandle_t bias_V,
    const topsopTensorHandle_t context_layer_dot_weight,
    const topsopTensorHandle_t context_layer_dot_bias,
    const topsopTensorHandle_t pre_norm_scale,
    const topsopTensorHandle_t pre_norm_bias,
    const topsopTensorHandle_t post_norm_scale,
    const topsopTensorHandle_t post_norm_bias,
    const topsopTensorHandle_t freq,
    topsopTensorHandle_t *posititon_id,
    int num_pid,
    const topsopTensorHandle_t mask,
    const topsopTensorHandle_t weight_Q_quant_scale,
    const topsopTensorHandle_t weight_K_quant_scale,
    const topsopTensorHandle_t weight_V_quant_scale,
    const topsopTensorHandle_t weight_C_quant_scale,

    const float qk_scale_coeff,
    const float residual_scale,
    const bool use_residual_add,
    const topsopRotaryEmbeddingMode_t use_qk_rotary_embedding,
    const topsopNormMode_t norm_type,
    int *cache_length,
    int num_cache_length,
    const int max_seq_len,
    const int hidden_size,

    const topsopSize_t *input_contract_dim,
    const topsopSize_t *weight_Q_contract_dim,
    const topsopSize_t *weight_K_contract_dim,
    const topsopSize_t *weight_V_contract_dim,
    const topsopSize_t *weight_context_contract_dim,
    const topsopSize_t *input_batch_dim);

/**
 * @brief get the dimensions and rank of output tensor
 *
 * @param input: the input tensor handle of the operator
 *
 * @param weight_Q: the weight tensor handle of Q-dot
 * @param bias_Q: the bias tensor handle of Q-dot
 *
 * @param weight_K: the weight tensor handle of K-dot
 * @param bias_K: the bias tensor handle of K-dot
 *
 * @param weight_V: the weight tensor handle of V-dot
 * @param bias_V: the bias tensor handle of V-dot
 *
 * @param input_contract_dim: The "k-dim" in x
 * @param weight_Q_contract_dim: The "k-dim" in weight_Q
 * @param weight_K_contract_dim: The "k-dim" in weight_K
 * @param weight_V_contract_dim: The "k-dim" in weight_V
 * @param weight_context_contract_dim: The "k-dim" in context_layer_dot_weight
 * @param input_batch_dim: The "batch-dim" in input
 *
 * @param freq: the frequency tensor of rotary_embedding
 * @param posititon_id: the position tensor of rotary_embedding
 *
 * @param mask: mask tensor of softmax in SDP
 * @param weight_Q_quant_scale: the quant scale of weight Q
 * @param weight_K_quant_scale: the quant scale of weight K
 * @param weight_V_quant_scale: the quant scale of weight V
 * @param weight_C_quant_scale: the quant scale of weight C
 * @param qk_scale_coeff: should be layer_id + 1, the scale before softmax
 *        operation and after Q-dot operation
 *
 * @param context_layer_dot_weight: the weight tensor handle of
 context_layer_dot
 * @param context_layer_dot_bias: the bias tensor handle of context_layer_dot
 * @param ContractingDimension_context: The "k-dim" in context_layer_dot
 * @param BatchDimension_context: The "batch-dim" in context_layer_dot(should be
 dummy)
 *
 * @param residual_scale: multiply the qkv input with this scale, then add to
 context_layer_dot output.
 * @param pre_norm_scale: scale tensor of pre norm.
 * @param pre_norm_bias: bias tensor of pre norm.
 * @param post_norm_scale: scale tensor of post norm.
 * @param post_norm_bias: bias tensor of post norm.
 *
 * @param use_residual_add: option to choose if use residual add
 * @param use_qk_rotary_embedding: option to choose if use qk rotary embedding
 * @param norm_type: choose Norm type in MultiQueryAttention
 * @param cache_length: cache length of current KV cache
 *
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return bool
 */

topsopStatus_t TOPSOP_EXPORT topsopMultiQueryAttentionNormGetOutputDim(
    const topsopTensorHandle_t input,
    const topsopTensorHandle_t weight_Q,
    const topsopTensorHandle_t bias_Q,
    const topsopTensorHandle_t weight_K,
    const topsopTensorHandle_t bias_K,
    const topsopTensorHandle_t weight_V,
    const topsopTensorHandle_t bias_V,
    const topsopTensorHandle_t context_layer_dot_weight,
    const topsopTensorHandle_t context_layer_dot_bias,
    const topsopTensorHandle_t pre_norm_scale,
    const topsopTensorHandle_t pre_norm_bias,
    const topsopTensorHandle_t post_norm_scale,
    const topsopTensorHandle_t post_norm_bias,
    const topsopTensorHandle_t freq,
    topsopTensorHandle_t *posititon_id,
    int num_pid,
    const topsopTensorHandle_t mask,
    const topsopTensorHandle_t weight_Q_quant_scale,
    const topsopTensorHandle_t weight_K_quant_scale,
    const topsopTensorHandle_t weight_V_quant_scale,
    const topsopTensorHandle_t weight_C_quant_scale,

    const float qk_scale_coeff,
    const float residual_scale,
    const bool use_residual_add,
    const topsopRotaryEmbeddingMode_t use_qk_rotary_embedding,
    const topsopNormMode_t norm_type,
    int *cache_length,
    int num_cache_length,
    const int max_seq_len,
    const int hidden_size,

    const topsopSize_t *input_contract_dim,
    const topsopSize_t *weight_Q_contract_dim,
    const topsopSize_t *weight_K_contract_dim,
    const topsopSize_t *weight_V_contract_dim,
    const topsopSize_t *weight_context_contract_dim,
    const topsopSize_t *input_batch_dim,

    int64_t *dims,
    int64_t *rank);

/**
 * @brief EmbeddingNorm operator
 *
 * @param output: the output tensor handle of the operator
 * @param input: the input tensor handle of the operator
 * @param lookup_table: the lookup table tensor handle of gather
 * @param post_norm_scale: scale tensor of post norm.
 * @param post_norm_bias: bias tensor of post norm.
 * @param lut_quant_scale: the quant scale of look up table
 * @param norm_type: choose Norm type in this opeator
 * @param alpha: dummy param, alwayse be 1.0
 * @param beta: dummy param, alwayse be 0.0
 * @param stream: tops stream
 *
 */
topsopStatus_t topsopEmbeddingNorm(topsopTensorHandle_t output,
                                   const topsopTensorHandle_t input,
                                   const topsopTensorHandle_t lookup_table,
                                   const topsopTensorHandle_t post_norm_scale,
                                   const topsopTensorHandle_t post_norm_bias,
                                   const topsopTensorHandle_t lut_quant_scale,
                                   const topsopNormMode_t norm_type,

                                   const topsopScalar_t alpha,
                                   const topsopScalar_t beta,
                                   topsStream_t stream);

/**
 * @brief check whether EmbeddingNorm operator is supported or not with
 * current input tensor
 *
 * @param input: the input tensor handle of the operator
 * @param lookup_table: the lookup table tensor handle of gather
 * @param post_norm_scale: scale tensor of post norm.
 * @param post_norm_bias: bias tensor of post norm.
 * @param lut_quant_scale: the quant scale of look up table
 * @param norm_type: choose Norm type in this opeator
 *
 * @return bool
 */
bool TOPSOP_EXPORT
topsopEmbeddingNormIsSupported(const topsopTensorHandle_t input,
                               const topsopTensorHandle_t lookup_table,
                               const topsopTensorHandle_t post_norm_scale,
                               const topsopTensorHandle_t post_norm_bias,
                               const topsopTensorHandle_t lut_quant_scale,
                               const topsopNormMode_t norm_type);

/**
 * @brief get the dimensions and rank of output tensor
 *
 * @param input: the input tensor handle of the operator
 * @param lookup_table: the lookup table tensor handle of gather
 * @param post_norm_scale: scale tensor of post norm.
 * @param post_norm_bias: bias tensor of post norm.
 * @param lut_quant_scale: the quant scale of look up table
 * @param norm_type: choose Norm type in this opeator
 *
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return bool
 */
topsopStatus_t TOPSOP_EXPORT
topsopEmbeddingNormGetOutputDim(const topsopTensorHandle_t input,
                                const topsopTensorHandle_t lookup_table,
                                const topsopTensorHandle_t post_norm_scale,
                                const topsopTensorHandle_t post_norm_bias,
                                const topsopTensorHandle_t lut_quant_scale,
                                const topsopNormMode_t norm_type,
                                int64_t *dims,
                                int64_t *rank);

/**
 * @brief DotNorm operator
 *
 * @param output: the output tensor handle of the operator
 * @param input: the input tensor handle of the operator
 * @param weight: the weight tensor handle of head dot
 * @param norm_type: choose Norm type in this opeator
 * @param pre_norm_scale: scale tensor of pre norm.
 * @param pre_norm_bias: bias tensor of pre norm.
 * @param weight_quant_scale: the quant scale of weight
 * @param alpha: dummy param, alwayse be 1.0
 * @param beta: dummy param, alwayse be 0.0
 * @param stream: tops stream
 *
 */
topsopStatus_t topsopDotNorm(topsopTensorHandle_t output,
                             const topsopTensorHandle_t input,
                             const topsopTensorHandle_t *dot_weights,
                             const int64_t num_dot_weights,
                             const topsopTensorHandle_t *norm_params,
                             const int64_t num_norm_params,
                             const topsopTensorHandle_t *dot_quant_params,
                             const int64_t num_dot_quant_params,
                             const topsopNormMode_t norm_type,
                             const bool use_pre_norm,
                             const bool use_quant,

                             const topsopScalar_t alpha,
                             const topsopScalar_t beta,
                             topsStream_t stream);

/**
 * @brief check whether DotNorm operator is supported or not with
 * current input tensor
 *
 * @param input: the input tensor handle of the operator
 * @param weight: the weight tensor handle of head dot
 * @param bias: the bias tensor handle of head dot
 * @param norm_type: choose Norm type in this opeator
 * @param pre_norm_scale: scale tensor of pre norm.
 * @param pre_norm_bias: bias tensor of pre norm.
 * @param weight_quant_scale: the quant scale of weight
 *
 * @return bool
 */
bool TOPSOP_EXPORT
topsopDotNormIsSupported(const topsopTensorHandle_t input,
                         const topsopTensorHandle_t *dot_weights,
                         const int64_t num_dot_weights,
                         const topsopTensorHandle_t *norm_params,
                         const int64_t num_norm_params,
                         const topsopTensorHandle_t *dot_quant_params,
                         const int64_t num_dot_quant_params,
                         const topsopNormMode_t norm_type,
                         const bool use_pre_norm,
                         const bool use_quant);

/**
 * @brief get the dimensions and rank of output tensor
 *
 * @param input: the input tensor handle of the operator
 * @param weight: the weight tensor handle of head dot
 * @param bias: the bias tensor handle of head dot
 * @param norm_type: choose Norm type in this opeator
 * @param pre_norm_scale: scale tensor of pre norm.
 * @param pre_norm_bias: bias tensor of pre norm.
 * @param weight_quant_scale: the quant scale of weight
 *
 * @param dims the dimensions of output tensor
 * @param rank the rank of output tensor
 * @return bool
 */
topsopStatus_t TOPSOP_EXPORT
topsopDotNormGetOutputDim(const topsopTensorHandle_t input,
                          const topsopTensorHandle_t *dot_weights,
                          const int64_t num_dot_weights,
                          const topsopTensorHandle_t *norm_params,
                          const int64_t num_norm_params,
                          const topsopTensorHandle_t *dot_quant_params,
                          const int64_t num_dot_quant_params,
                          const topsopNormMode_t norm_type,
                          const bool use_pre_norm,
                          const bool use_quant,
                          int64_t *dims,
                          int64_t *rank);

topsopStatus_t topsopLLMDot(topsopTensorHandle_t output,
                            const topsopTensorHandle_t input,
                            const topsopTensorHandle_t weight,
                            const topsopTensorHandle_t bias,
                            topsStream_t stream);
bool TOPSOP_EXPORT topsopLLMDotIsSupported(const topsopTensorHandle_t input,
                                           const topsopTensorHandle_t weight,
                                           const topsopTensorHandle_t bias);

topsopStatus_t TOPSOP_EXPORT
topsopLLMDotGetOutputDim(const topsopTensorHandle_t input,
                         const topsopTensorHandle_t weight,
                         const topsopTensorHandle_t bias,
                         int64_t *dims,
                         int64_t *rank);

topsopStatus_t topsopLLMNorm(topsopTensorHandle_t output,
                             const topsopTensorHandle_t input,
                             const topsopTensorHandle_t scale,
                             const topsopTensorHandle_t bias,
                             const topsopTensorHandle_t buffer,
                             topsStream_t stream);
bool TOPSOP_EXPORT topsopLLMNormIsSupported(const topsopTensorHandle_t input,
                                            const topsopTensorHandle_t scale,
                                            const topsopTensorHandle_t bias,
                                            const topsopTensorHandle_t buffer);
topsopStatus_t TOPSOP_EXPORT
topsopLLMNormGetOutputDim(const topsopTensorHandle_t input,
                          const topsopTensorHandle_t scale,
                          const topsopTensorHandle_t bias,
                          const topsopTensorHandle_t buffer,
                          int64_t *dims,
                          int64_t *rank);

/**
 * @brief partial smapler: sort + mul + softmax
 * sorted_tensor, output_idx = sort(input)
 * output = sotfmax(1/temperature * sort(sorted_tensor))
 *
 * @param output: the output tensor handle of the operator
 * @param output_idx: the output index tensor handle of the elements int in the
 * input
 * @param input: the input tensor handle of the operator
 * @param temperature: temperature
 * @param stream: tops stream
 * @return bool
 */
topsopStatus_t TOPSOP_EXPORT
topsopPartialSample(topsopTensorHandle_t output,
                    topsopTensorHandle_t output_idx,
                    const topsopTensorHandle_t input,
                    const float temperature,
                    const topsStream_t stream);

topsopStatus_t topsopDotWeightHelper(topsopDotWeightHelper_t *dwh_p,
                                     const topsopTensorHandle_t dot_weights,
                                     const topsDeviceProp_t target_prop,
                                     const bool enable_scatter_memory);

topsopStatus_t topsopDotWeightHelperFree(topsopDotWeightHelper_t *dwh_p);

topsopStatus_t topsopDotWeightHelperPrint(const topsopDotWeightHelper_t dwh);
/**
 * @brief MultiQueryAttentionNormCache4mc operator
 *
 * @param output: the output tensor handle of the operator
 * @param cache_K: the cache_K tensor handle of the operator
 * @param cache_V: the cache_V tensor handle of the operator
 * @param input: the input tensor handle of the operator
 *
 * @param weight_Q: the weight tensor handle of Q-dot
 * @param bias_Q: the bias tensor handle of Q-dot
 *
 * @param weight_K: the weight tensor handle of K-dot
 * @param bias_K: the bias tensor handle of K-dot
 *
 * @param weight_V: the weight tensor handle of V-dot
 * @param bias_V: the bias tensor handle of V-dot
 *
 * @param input_contract_dim: The "k-dim" in x
 * @param weight_Q_contract_dim: The "k-dim" in weight_Q
 * @param weight_K_contract_dim: The "k-dim" in weight_K
 * @param weight_V_contract_dim: The "k-dim" in weight_V
 * @param weight_context_contract_dim: The "k-dim" in context_layer_dot_weight
 * @param input_batch_dim: The "batch-dim" in input
 *
 * @param freq: the frequency tensor of rotary_embedding
 * @param posititon_id: the position tensor of rotary_embedding
 *
 * @param mask: mask tensor of softmax in SDP
 * @param weight_Q_quant_scale: the quant scale of weight Q
 * @param weight_K_quant_scale: the quant scale of weight K
 * @param weight_V_quant_scale: the quant scale of weight V
 * @param weight_C_quant_scale: the quant scale of weight C
 * @param qk_scale_coeff: should be layer_id + 1, the scale before softmax
 *        operation and after Q-dot operation
 *
 * @param context_layer_dot_weight: the weight tensor handle of
 context_layer_dot
 * @param context_layer_dot_bias: the bias tensor handle of context_layer_dot
 * @param ContractingDimension_context: The "k-dim" in context_layer_dot
 * @param BatchDimension_context: The "batch-dim" in context_layer_dot(should be
 dummy)
 *
 * @param residual_scale: multiply the qkv input with this scale, then add to
 context_layer_dot output.
 * @param pre_norm_scale: scale tensor of pre norm.
 * @param pre_norm_bias: bias tensor of pre norm.
 * @param post_norm_scale: scale tensor of post norm.
 * @param post_norm_bias: bias tensor of post norm.
 *
 * @param use_residual_add: option to choose if use residual add
 * @param use_qk_rotary_embedding: option to choose if use qk rotary embedding
 * @param norm_type: choose Norm type in MultiQueryAttention
 * @param cache_length: cache length of current KV cache
 * @param max_seq_len: max sequence length
 * @param hidden_size: hidden size
 * @param stream: tops stream
 *
 */
topsopStatus_t topsopMultiQueryAttentionNormCache4mc(
    topsopTensorHandle_t output,
    topsopTensorHandle_t cache_K,
    topsopTensorHandle_t cache_V,
    int num_batch,
    const topsopTensorHandle_t batch_id,
    const topsopTensorHandle_t input,
    const topsopTensorHandle_t weight_Q,
    const topsopTensorHandle_t bias_Q,
    const topsopTensorHandle_t weight_K,
    const topsopTensorHandle_t bias_K,
    const topsopTensorHandle_t weight_V,
    const topsopTensorHandle_t bias_V,
    const topsopTensorHandle_t context_layer_dot_weight,
    const topsopTensorHandle_t context_layer_dot_bias,
    const topsopTensorHandle_t pre_norm_scale,
    const topsopTensorHandle_t pre_norm_bias,
    const topsopTensorHandle_t post_norm_scale,
    const topsopTensorHandle_t post_norm_bias,
    const topsopTensorHandle_t freq,
    const topsopTensorHandle_t position_id,
    const topsopTensorHandle_t mask,
    const topsopTensorHandle_t weight_Q_quant_scale,
    const topsopTensorHandle_t weight_K_quant_scale,
    const topsopTensorHandle_t weight_V_quant_scale,
    const topsopTensorHandle_t weight_C_quant_scale,

    const float qk_scale_coeff,
    const float residual_scale,
    const bool use_residual_add,
    const topsopRotaryEmbeddingMode_t use_qk_rotary_embedding,
    const topsopNormMode_t norm_type,
    const topsopTensorHandle_t cache_length,
    const int max_seq_len,
    const int hidden_size,
    topsStream_t stream);

#if defined(__cplusplus)
}
#endif

#endif /* TOPSOP_LLM_OPS_H_ */  // NOLINT

// Doxygen end group topsop_llm_ops.h
/** @} */
