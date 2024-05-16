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

#include "linear.h"  // NOLINT

#include "smooth_quant.h"  // NOLINT

namespace atb_layers {

void CreateLinearQuant(const LinearParam &param, atb::Operation **operation) {
  uint64_t TENSOR_ID = 0;
  uint64_t INPUT = TENSOR_ID++;
  uint64_t INPUT_WEIGHT = TENSOR_ID++;
  uint64_t INPUT_BIAS = param.has_bias ? TENSOR_ID++ : INPUT_WEIGHT;
  uint64_t INPUT_DEQSCALE = TENSOR_ID++;
  uint64_t INPUT_DEQBIAS =
      param.has_dequant_offset ? TENSOR_ID++ : INPUT_DEQSCALE;
  uint64_t INPUT_SMOOTH =
      param.input_smooth_quant ? TENSOR_ID++ : INPUT_DEQBIAS;
  uint64_t INPUT_SHIFT = param.input_smooth_quant ? TENSOR_ID++ : INPUT_SMOOTH;
  uint64_t OUTPUT = TENSOR_ID++;
  uint64_t INTERMEDIATE_INPUT_QUANT = param.input_quant ? TENSOR_ID++ : INPUT;
  uint64_t INTERMEDIATE_LINEAR_OUT = param.has_bias ? TENSOR_ID++ : OUTPUT;

  uint64_t nodeIdx = 0;
  atb::GraphParam opGraph;
  opGraph.name = "Linear";
  opGraph.inTensorNum = INPUT_SHIFT + 1;
  opGraph.outTensorNum = 1;

  if (param.input_quant && param.has_bias) {
    opGraph.internalTensorNum = 2;
    opGraph.nodes.resize(3);
  } else if (param.input_quant) {
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);
  } else if (param.has_bias) {
    opGraph.internalTensorNum = 1;
    opGraph.nodes.resize(2);
  } else {
    opGraph.internalTensorNum = 0;
    opGraph.nodes.resize(1);
  }

  if (param.input_quant) {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb_layers::SmoothQuantParam opParam;
    opParam.scale = param.input_quant_scale;
    opParam.offset = param.input_quant_offset;
    opParam.smooth_quant = param.input_smooth_quant;
    atb::CreateOperation(opParam, &opNode.operation);
    if (opParam.smooth_quant) {
      opNode.inTensorIds = {INPUT, INPUT_SMOOTH, INPUT_SHIFT};
    } else {
      opNode.inTensorIds = {INPUT};
    }
    opNode.outTensorIds = {INTERMEDIATE_INPUT_QUANT};
  }

  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::LinearParam opParam;
    opParam.outDataType = ACL_FLOAT16;
    opParam.transposeA = false;
    opParam.transposeB = param.trans_weight;
    opParam.hasBias = param.has_dequant_offset;
    atb::CreateOperation(opParam, &opNode.operation);
    if (param.has_dequant_offset) {
      opNode.inTensorIds = {INTERMEDIATE_INPUT_QUANT,
                            INPUT_WEIGHT,
                            INPUT_DEQBIAS,
                            INPUT_DEQSCALE};
    } else {
      opNode.inTensorIds = {
          INTERMEDIATE_INPUT_QUANT, INPUT_WEIGHT, INPUT_DEQSCALE};
    }
    opNode.outTensorIds = {INTERMEDIATE_LINEAR_OUT};
  }

  if (param.has_bias) {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::ElewiseParam opParam;
    opParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {INTERMEDIATE_LINEAR_OUT, INPUT_BIAS};
    opNode.outTensorIds = {OUTPUT};
  }
  atb::CreateOperation(opGraph, operation);
}

void CreateLinear(const LinearParam &param, atb::Operation **operation) {
  if (param.input_quant) {
    CreateLinearQuant(param, operation);
  } else {
    atb::infer::LinearParam opParam;
    opParam.transposeA = false;
    opParam.transposeB = param.trans_weight;
    opParam.hasBias = param.has_bias;
    atb::CreateOperation(opParam, operation);
  }
}

}  // namespace atb_layers

#endif
