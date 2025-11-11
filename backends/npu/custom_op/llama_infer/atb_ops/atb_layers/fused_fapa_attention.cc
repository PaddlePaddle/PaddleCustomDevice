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

#include "fused_fapa_attention.h"  // NOLINT

#include <cmath>

#include "glog/logging.h"
#include "linear.h"     // NOLINT
#include "qkv_split.h"  // NOLINT

namespace atb_layers {

void CreateFaPaAttention(const FaPaAttentionParam &param,
                         atb::Operation **operation) {
  uint64_t TENSOR_ID = 0;

  uint64_t INPUT_QKV_OUT = TENSOR_ID++;

  uint64_t INPUT_COS = param.use_alibi ? 0 : TENSOR_ID++;
  uint64_t INPUT_SIN = param.use_alibi ? 0 : TENSOR_ID++;
  uint64_t INPUT_MASK = param.is_prefill || param.use_alibi ? TENSOR_ID++ : 0;
  uint64_t INPUT_CACHE_K = TENSOR_ID++;
  uint64_t INPUT_CACHE_V = TENSOR_ID++;
  uint64_t INPUT_SLOTS = TENSOR_ID++;
  uint64_t INPUT_BLOCK_TABLES = !param.is_prefill ? TENSOR_ID++ : 0;
  uint64_t INPUT_SEQLEN = TENSOR_ID++;
  uint64_t INPUT_BATCH_STATUS = !param.is_prefill ? TENSOR_ID++ : INPUT_SEQLEN;

  uint64_t OUTPUT = TENSOR_ID++;

  uint64_t INTERMEDIATE_Q = TENSOR_ID++;
  uint64_t INTERMEDIATE_K = TENSOR_ID++;
  uint64_t INTERMEDIATE_V = TENSOR_ID++;
  uint64_t INTERMEDIATE_EMB_Q = TENSOR_ID++;
  uint64_t INTERMEDIATE_EMB_K = TENSOR_ID++;

  uint64_t nodeIdx = 0;
  atb::GraphParam opGraph;
  opGraph.name = "FaPaAttentionOperation";
  opGraph.inTensorNum = INPUT_BATCH_STATUS - INPUT_QKV_OUT + 1;
  opGraph.outTensorNum = 1;
  opGraph.internalTensorNum = INTERMEDIATE_EMB_K - INTERMEDIATE_Q + 1;
  if (param.use_alibi) {
    opGraph.nodes.resize(3);
  } else {
    opGraph.nodes.resize(4);
  }

  // split q,k,v
  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb_layers::QKVSplitParam opParam;
    opParam.head_num = param.head_num;
    opParam.kv_head_num = param.kv_head_num;
    opParam.head_dim = param.head_dim;
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {INPUT_QKV_OUT};
    opNode.outTensorIds = {INTERMEDIATE_Q, INTERMEDIATE_K, INTERMEDIATE_V};
  }

  // rope
  if (!param.use_alibi) {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::RopeParam opParam;
    opParam.rotaryCoeff = param.rope_neox ? param.head_dim : 2;
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {
        INTERMEDIATE_Q, INTERMEDIATE_K, INPUT_COS, INPUT_SIN, INPUT_SEQLEN};
    opNode.outTensorIds = {INTERMEDIATE_EMB_Q, INTERMEDIATE_EMB_K};
  }

  // write kv
  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::ReshapeAndCacheParam opParam;
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {INTERMEDIATE_EMB_K,
                          INTERMEDIATE_V,
                          INPUT_CACHE_K,
                          INPUT_CACHE_V,
                          INPUT_SLOTS};
    opNode.outTensorIds = {INPUT_CACHE_K, INPUT_CACHE_V};  // write in place
    opNode.inTensorReshapeFuncs.resize(opNode.inTensorIds.size());
    opNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape,
                                         atb::Dims &newShape) {
      newShape.dimNum = 3;
      newShape.dims[0] = oldShape.dims[0];
      newShape.dims[1] = param.kv_head_num;
      newShape.dims[2] = param.head_dim;
    };
    opNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape,
                                         atb::Dims &newShape) {
      newShape.dimNum = 3;
      newShape.dims[0] = oldShape.dims[0];
      newShape.dims[1] = param.kv_head_num;
      newShape.dims[2] = param.head_dim;
    };
    opNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape,
                                         atb::Dims &newShape) {
      newShape.dimNum = 4;
      newShape.dims[0] = oldShape.dims[0];
      newShape.dims[1] = oldShape.dims[2];
      newShape.dims[2] = oldShape.dims[1];
      newShape.dims[3] = oldShape.dims[3];
    };
    opNode.inTensorReshapeFuncs[3] = [=](const atb::Dims &oldShape,
                                         atb::Dims &newShape) {
      newShape.dimNum = 4;
      newShape.dims[0] = oldShape.dims[0];
      newShape.dims[1] = oldShape.dims[2];
      newShape.dims[2] = oldShape.dims[1];
      newShape.dims[3] = oldShape.dims[3];
    };
  }

  if (param.is_prefill) {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::SelfAttentionParam opParam;
    opParam.headNum = param.head_num;
    opParam.kvHeadNum = param.kv_head_num;
    opParam.qkScale = 1.0f / sqrt(param.head_dim);
    opParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
    opParam.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM;
    if (param.use_alibi) {
      opParam.isTriuMask = 0;
      opParam.maskType =
          atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_ALIBI;
    } else {
      opParam.isTriuMask = 1;
    }
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {INTERMEDIATE_EMB_Q,
                          INTERMEDIATE_EMB_K,
                          INTERMEDIATE_V,
                          INPUT_MASK,
                          INPUT_SEQLEN};
    LOG(INFO) << "OUTPUT fa **************" << OUTPUT;
    opNode.outTensorIds = {OUTPUT};
    opNode.inTensorReshapeFuncs.resize(opNode.inTensorIds.size());
  } else {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::PagedAttentionParam opParam;
    opParam.headNum = param.head_num;
    opParam.qkScale = 1.0f / sqrt(param.head_dim);
    opParam.kvHeadNum = param.kv_head_num;
    if (param.use_alibi) {
      opParam.maskType =
          atb::infer::PagedAttentionParam::MaskType::MASK_TYPE_ALIBI;
    } else {
      opParam.maskType = atb::infer::PagedAttentionParam::MaskType::UNDEFINED;
    }
    opParam.batchRunStatusEnable = true;

    atb::CreateOperation(opParam, &opNode.operation);

    if (param.use_alibi) {
      opNode.inTensorIds = {INTERMEDIATE_EMB_Q,
                            INPUT_CACHE_K,
                            INPUT_CACHE_V,
                            INPUT_BLOCK_TABLES,
                            INPUT_SEQLEN,
                            INPUT_MASK,
                            INPUT_BATCH_STATUS};
    } else {
      opNode.inTensorIds = {INTERMEDIATE_EMB_Q,
                            INPUT_CACHE_K,
                            INPUT_CACHE_V,
                            INPUT_BLOCK_TABLES,
                            INPUT_SEQLEN,
                            INPUT_BATCH_STATUS};
    }

    opNode.outTensorIds = {OUTPUT};
    opNode.inTensorReshapeFuncs.resize(opNode.inTensorIds.size());
    opNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape,
                                         atb::Dims &newShape) {
      newShape.dimNum = 3;
      newShape.dims[0] = oldShape.dims[0];
      newShape.dims[1] = param.head_num;
      newShape.dims[2] = param.head_dim;
    };
  }

  atb::CreateOperation(opGraph, operation);
}

}  // namespace atb_layers

#endif
