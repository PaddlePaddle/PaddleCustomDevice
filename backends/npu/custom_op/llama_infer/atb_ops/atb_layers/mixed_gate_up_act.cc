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

#ifdef PADDLE_WITH_ATB

#include "mixed_gate_up_act.h"  // NOLINT

#include <iostream>
#include <numeric>

#include "linear.h"  // NOLINT

namespace atb_layers {

// matmul: [bs, seqlen, emb_dim] , [emb_dim, 2 * head_num * head_dim] -> [bs,
// seqlen, 2 * head_num * head_dim] split: [bs, seqlen, head_num * head_dim]
// mul: [bs, seqlen, head_num * head_dim]
void CreateMixedGateUpAct(const MixedGateUpActParam &param,
                          atb::Operation **operation) {
  uint64_t nodeIdx = 0;
  uint64_t TENSOR_ID = 0;
  uint64_t INPUT_HIDDEN_STATES = TENSOR_ID++;
  uint64_t INPUT_MIXED_GATE_UP_WEIGHT = TENSOR_ID++;
  uint64_t INPUT_MIXED_GATE_UP_DEQSCALE =
      param.linear_quant ? TENSOR_ID++ : INPUT_MIXED_GATE_UP_WEIGHT;
  uint64_t INPUT_MIXED_GATE_UP_DEQBIAS =
      param.linear_quant && param.has_dequant_offset
          ? TENSOR_ID++
          : INPUT_MIXED_GATE_UP_DEQSCALE;
  uint64_t INPUT_MIXED_GATE_UP_SMOOTH =
      param.input_smooth_quant ? TENSOR_ID++ : INPUT_MIXED_GATE_UP_DEQBIAS;
  uint64_t INPUT_MIXED_GATE_UP_SHIFT =
      param.input_smooth_quant ? TENSOR_ID++ : INPUT_MIXED_GATE_UP_SMOOTH;
  uint64_t OUTPUT = TENSOR_ID++;
  uint64_t INTERMEDIATE_FFN_GATE_UP_OUT = TENSOR_ID++;

  uint64_t kInputTensorNum = INPUT_MIXED_GATE_UP_SHIFT + 1;
  uint64_t kOutputTensorNum = 1;
  uint64_t kInternalTensorNum = 1;

  atb::GraphParam opGraph;
  opGraph.name = "MixedGateUpAct";
  opGraph.inTensorNum = kInputTensorNum;
  opGraph.outTensorNum = kOutputTensorNum;
  opGraph.internalTensorNum = kInternalTensorNum;
  opGraph.nodes.resize(2);

  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    if (param.linear_quant) {
      atb_layers::LinearParam opParam;
      opParam.trans_weight = param.trans_weight;
      opParam.has_bias = false;
      opParam.input_quant = param.input_quant;
      opParam.input_quant_scale = param.input_quant_scale;
      opParam.input_quant_offset = param.input_quant_offset;
      opParam.input_smooth_quant = param.input_smooth_quant;
      opParam.has_dequant_offset = param.has_dequant_offset;
      atb::CreateOperation(opParam, &opNode.operation);
      if (param.input_smooth_quant && !param.has_dequant_offset) {
        opNode.inTensorIds = {INPUT_HIDDEN_STATES,
                              INPUT_MIXED_GATE_UP_WEIGHT,
                              INPUT_MIXED_GATE_UP_DEQSCALE,
                              INPUT_MIXED_GATE_UP_SMOOTH,
                              INPUT_MIXED_GATE_UP_SHIFT};
      } else if (param.input_smooth_quant && param.has_dequant_offset) {
        opNode.inTensorIds = {INPUT_HIDDEN_STATES,
                              INPUT_MIXED_GATE_UP_WEIGHT,
                              INPUT_MIXED_GATE_UP_DEQSCALE,
                              INPUT_MIXED_GATE_UP_DEQBIAS,
                              INPUT_MIXED_GATE_UP_SMOOTH,
                              INPUT_MIXED_GATE_UP_SHIFT};
      } else if (!param.input_smooth_quant && !param.has_dequant_offset) {
        opNode.inTensorIds = {INPUT_HIDDEN_STATES,
                              INPUT_MIXED_GATE_UP_WEIGHT,
                              INPUT_MIXED_GATE_UP_DEQSCALE};
      } else {
        opNode.inTensorIds = {INPUT_HIDDEN_STATES,
                              INPUT_MIXED_GATE_UP_WEIGHT,
                              INPUT_MIXED_GATE_UP_DEQSCALE,
                              INPUT_MIXED_GATE_UP_DEQBIAS};
      }
    } else {
      atb::infer::LinearParam opParam;
      opParam.transposeA = false;
      opParam.transposeB = param.trans_weight;
      opParam.hasBias = false;
      atb::CreateOperation(opParam, &opNode.operation);
      opNode.inTensorIds = {INPUT_HIDDEN_STATES, INPUT_MIXED_GATE_UP_WEIGHT};
    }
    opNode.outTensorIds = {INTERMEDIATE_FFN_GATE_UP_OUT};
  }

  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::ActivationParam opParam;
    opParam.activationType =
        atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
    opParam.scale = param.scale;
    opParam.dim = -1;

    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {INTERMEDIATE_FFN_GATE_UP_OUT};
    opNode.outTensorIds = {OUTPUT};
  }

  opGraph
      .inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                            atb::SVector<atb::TensorDesc> &outTensorDescs) {
    outTensorDescs.resize(kOutputTensorNum);
    outTensorDescs.at(0) = inTensorDescs.at(INPUT_HIDDEN_STATES);
    if (param.linear_quant) {
      outTensorDescs.at(0).dtype = ACL_FLOAT16;
    }
    bool trans_weight = param.trans_weight;
    if (trans_weight) {
      outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] =
          inTensorDescs.at(INPUT_MIXED_GATE_UP_WEIGHT).shape.dims[0] / 2;
    } else {
      outTensorDescs.at(0).shape.dims[outTensorDescs.at(0).shape.dimNum - 1] =
          inTensorDescs.at(INPUT_MIXED_GATE_UP_WEIGHT).shape.dims[1] / 2;
    }
    return atb::NO_ERROR;
  };

  atb::CreateOperation(opGraph, operation);
}
}  // namespace atb_layers

#endif
