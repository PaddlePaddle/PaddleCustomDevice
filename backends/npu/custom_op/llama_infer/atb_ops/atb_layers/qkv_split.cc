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

#include "qkv_split.h"  // NOLINT

namespace atb_layers {

void CreateQKVSplit(const QKVSplitParam &param, atb::Operation **operation) {
  uint64_t TENSOR_ID = 0;
  uint64_t INPUT_MIXED_QKV =
      TENSOR_ID++;  // [ntokens, 3 * head_num * head_dim] or [ntokens,
                    // (head_num + 2 * kv_head_num) * head_dim]
  uint64_t OUTPUT_Q = TENSOR_ID++;  // [ntokens, head_num * head_dim]
  uint64_t OUTPUT_K = TENSOR_ID++;
  uint64_t OUTPUT_V = TENSOR_ID++;

  auto kv_head_num =
      (param.kv_head_num > 0 && param.kv_head_num != param.head_num)
          ? param.kv_head_num
          : 0;

  uint64_t nodeIdx = 0;
  atb::GraphParam opGraph;
  opGraph.name = "QKVSplit";
  opGraph.inTensorNum = 1;
  opGraph.outTensorNum = 3;
  opGraph.internalTensorNum = 0;
  opGraph.nodes.resize(kv_head_num > 0 ? 3 : 1);

  if (kv_head_num > 0) {
    {
      atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
      atb::infer::SliceParam opParam;
      opParam.offsets.resize(2);
      opParam.size.resize(2);
      opParam.offsets[0] = 0;
      opParam.offsets[1] = 0;
      opParam.size[0] = -1;
      opParam.size[1] = param.head_num * param.head_dim;
      atb::CreateOperation(opParam, &opNode.operation);
      opNode.inTensorIds = {INPUT_MIXED_QKV};
      opNode.outTensorIds = {OUTPUT_Q};
    }
    {
      atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
      atb::infer::SliceParam opParam;
      opParam.offsets.resize(2);
      opParam.size.resize(2);
      opParam.offsets[0] = 0;
      opParam.offsets[1] = param.head_num * param.head_dim;
      opParam.size[0] = -1;
      opParam.size[1] = param.kv_head_num * param.head_dim;
      atb::CreateOperation(opParam, &opNode.operation);
      opNode.inTensorIds = {INPUT_MIXED_QKV};
      opNode.outTensorIds = {OUTPUT_K};
    }
    {
      atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
      atb::infer::SliceParam opParam;
      opParam.offsets.resize(2);
      opParam.size.resize(2);
      opParam.offsets[0] = 0;
      opParam.offsets[1] =
          (param.head_num + param.kv_head_num) * param.head_dim;
      opParam.size[0] = -1;
      opParam.size[1] = param.kv_head_num * param.head_dim;
      atb::CreateOperation(opParam, &opNode.operation);
      opNode.inTensorIds = {INPUT_MIXED_QKV};
      opNode.outTensorIds = {OUTPUT_V};
    }
    opGraph.inferShapeFunc =
        [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
            atb::SVector<atb::TensorDesc> &outTensorDescs) {
          outTensorDescs.resize(3);
          outTensorDescs.at(0) = inTensorDescs.at(0);
          outTensorDescs.at(0).shape.dims[0] =
              inTensorDescs.at(0).shape.dims[0];
          outTensorDescs.at(0).shape.dims[1] = param.head_num * param.head_dim;
          outTensorDescs.at(1) = inTensorDescs.at(0);
          outTensorDescs.at(1).shape.dims[0] =
              inTensorDescs.at(0).shape.dims[0];
          outTensorDescs.at(1).shape.dims[1] =
              param.kv_head_num * param.head_dim;
          outTensorDescs.at(2) = inTensorDescs.at(0);
          outTensorDescs.at(2).shape.dims[0] =
              inTensorDescs.at(0).shape.dims[0];
          outTensorDescs.at(2).shape.dims[1] =
              param.kv_head_num * param.head_dim;
          return atb::NO_ERROR;
        };
  } else {
    atb::Node &opNode = opGraph.nodes.at(nodeIdx++);
    atb::infer::SplitParam opParam;
    opParam.splitDim = 1;
    opParam.splitNum = 3;  // only fp16
    atb::CreateOperation(opParam, &opNode.operation);
    opNode.inTensorIds = {INPUT_MIXED_QKV};
    opNode.outTensorIds = {OUTPUT_Q, OUTPUT_K, OUTPUT_V};
    opGraph.inferShapeFunc =
        [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
            atb::SVector<atb::TensorDesc> &outTensorDescs) {
          outTensorDescs.resize(3);
          outTensorDescs.at(0) = inTensorDescs.at(0);
          outTensorDescs.at(0).shape.dims[0] =
              inTensorDescs.at(0).shape.dims[0];
          outTensorDescs.at(0).shape.dims[1] = param.head_num * param.head_dim;
          outTensorDescs.at(1) = inTensorDescs.at(0);
          outTensorDescs.at(1).shape.dims[0] =
              inTensorDescs.at(0).shape.dims[0];
          outTensorDescs.at(1).shape.dims[1] = param.head_num * param.head_dim;
          outTensorDescs.at(2) = inTensorDescs.at(0);
          outTensorDescs.at(2).shape.dims[0] =
              inTensorDescs.at(0).shape.dims[0];
          outTensorDescs.at(2).shape.dims[1] = param.head_num * param.head_dim;
          return atb::NO_ERROR;
        };
  }
  atb::CreateOperation(opGraph, operation);
}

}  // namespace atb_layers

#endif
