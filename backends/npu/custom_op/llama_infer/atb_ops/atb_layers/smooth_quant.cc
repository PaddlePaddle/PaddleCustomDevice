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

#include "smooth_quant.h"  // NOLINT

namespace atb_layers {

void CreateSmoothQuant(const SmoothQuantParam &param,
                       atb::Operation **operation) {
  uint64_t TENSOR_ID = 0;
  uint64_t INPUT = TENSOR_ID++;
  uint64_t INPUT_SMOOTH = param.smooth_quant ? TENSOR_ID++ : INPUT;
  uint64_t INPUT_SHIFT = param.smooth_quant ? TENSOR_ID++ : INPUT_SMOOTH;
  uint64_t OUTPUT = TENSOR_ID++;
  uint64_t INTERMEDIATE_SHIFT = param.smooth_quant ? TENSOR_ID++ : 0;
  uint64_t INTERMEDIATE_SMOOTH = param.smooth_quant ? TENSOR_ID++ : 0;

  uint64_t nodeIdx = 0;
  atb::GraphParam opGraph;
  opGraph.name = "SmoothQuant";
  opGraph.inTensorNum = INPUT_SHIFT + 1;
  opGraph.outTensorNum = 1;
  opGraph.internalTensorNum = param.smooth_quant ? 2 : 0;
  opGraph.nodes.resize(param.smooth_quant ? 3 : 1);
  if (param.smooth_quant) {
    {
      atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
      atb::infer::ElewiseParam opParam;
      opParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
      atb::CreateOperation(opParam, &opNode.operation);
      opNode.inTensorIds = {INPUT, INPUT_SHIFT};
      opNode.outTensorIds = {INTERMEDIATE_SHIFT};
    }
    {
      atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
      atb::infer::ElewiseParam opParam;
      opParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
      atb::CreateOperation(opParam, &opNode.operation);
      opNode.inTensorIds = {INTERMEDIATE_SHIFT, INPUT_SMOOTH};
      opNode.outTensorIds = {INTERMEDIATE_SMOOTH};
    }
    {
      atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
      atb::infer::ElewiseParam opParam;
      opParam.elewiseType =
          atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
      opParam.quantParam.inputScale = param.scale;
      opParam.quantParam.inputOffset = param.offset;
      atb::CreateOperation(opParam, &opNode.operation);
      opNode.inTensorIds = {INTERMEDIATE_SMOOTH};
      opNode.outTensorIds = {OUTPUT};
    }
  } else {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::ElewiseParam opParam;
    opParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
    opParam.quantParam.inputScale = param.scale;
    opParam.quantParam.inputOffset = param.offset;
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {INPUT};
    opNode.outTensorIds = {OUTPUT};
  }
  opGraph.inferShapeFunc =
      [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
          atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.resize(1);
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).dtype = ACL_INT8;
        return atb::NO_ERROR;
      };
  atb::CreateOperation(opGraph, operation);
}

}  // namespace atb_layers

#endif
