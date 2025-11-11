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

#include "fused_lm_head_layer.h"  // NOLINT

#include <cmath>

namespace atb_layers {

void CreateFusedLmHeadLayer(const FusedLmHeadLayerParam &param,
                            atb::Operation **operation) {
  uint64_t TENSOR_ID = 0;
  uint64_t INPUT_X = TENSOR_ID++;
  uint64_t INPUT_NORM_WEIGHT = TENSOR_ID++;
  uint64_t INPUT_LINEAR_WEIGHT = TENSOR_ID++;
  uint64_t OUTPUT = TENSOR_ID++;
  uint64_t INTERMEDIATE_NORM_OUT = TENSOR_ID++;

  uint64_t nodeIdx = 0;
  atb::GraphParam opGraph;
  opGraph.name = "FusedLmHeadLayer";
  opGraph.inTensorNum = INPUT_LINEAR_WEIGHT - INPUT_X + 1;
  opGraph.outTensorNum = 1;
  opGraph.internalTensorNum = 1;
  opGraph.nodes.resize(2);

  // rms_norm
  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::RmsNormParam opParam;
    opParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    opParam.normParam.epsilon = param.epsilon;
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {INPUT_X, INPUT_NORM_WEIGHT};
    opNode.outTensorIds = {INTERMEDIATE_NORM_OUT};
  }

  // matmul
  {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::LinearParam opParam;
    opParam.transposeA = false;
    opParam.transposeB = param.trans_weight;
    opParam.hasBias = false;
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {INTERMEDIATE_NORM_OUT, INPUT_LINEAR_WEIGHT};
    opNode.outTensorIds = {OUTPUT};
  }
  opGraph.inferShapeFunc =
      [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
          atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.resize(0);
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 2;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        bool trans_weight = param.trans_weight;
        if (trans_weight) {
          outTensorDescs.at(0).shape.dims[1] =
              inTensorDescs.at(2).shape.dims[0] * param.nranks;
        } else {
          outTensorDescs.at(0).shape.dims[1] =
              inTensorDescs.at(2).shape.dims[1] * param.nranks;
        }
        return atb::NO_ERROR;
      };
  atb::CreateOperation(opGraph, operation);
}

}  // namespace atb_layers

#endif
