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

#include "fused_rms_norm.h"  // NOLINT

namespace atb_layers {

void CreateRmsNorm(const RmsNormParam &param, atb::Operation **operation) {
  uint64_t TENSOR_ID = 0;
  uint64_t INPUT = TENSOR_ID++;
  uint64_t INPUT_WEIGHT = TENSOR_ID++;
  uint64_t INPUT_RESIDUAL = param.has_residual ? TENSOR_ID++ : INPUT_WEIGHT;
  uint64_t OUTPUT = TENSOR_ID++;
  uint64_t OUTPUT_RESIDUAL = param.has_residual ? TENSOR_ID++ : OUTPUT;

  uint64_t nodeIdx = 0;
  atb::GraphParam opGraph;
  opGraph.name = "RmsNormOperation";
  opGraph.internalTensorNum = 0;

  if (param.has_residual) {
    opGraph.inTensorNum = 3;
    opGraph.outTensorNum = 2;
    opGraph.nodes.resize(2);
  } else {
    opGraph.inTensorNum = 2;
    opGraph.outTensorNum = 1;
    opGraph.nodes.resize(1);
  }

  if (param.has_residual) {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::ElewiseParam opParam;
    opParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {INPUT, INPUT_RESIDUAL};
    opNode.outTensorIds = {OUTPUT_RESIDUAL};
  }

  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::RmsNormParam opParam;
    opParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    opParam.normParam.epsilon = param.epsilon;
    atb::CreateOperation(opParam, &opNode.operation);
    if (param.has_residual) {
      opNode.inTensorIds = {OUTPUT_RESIDUAL, INPUT_WEIGHT};
    } else {
      opNode.inTensorIds = {INPUT, INPUT_WEIGHT};
    }
    opNode.outTensorIds = {OUTPUT};
  }

  atb::CreateOperation(opGraph, operation);
}

}  // namespace atb_layers

#endif
