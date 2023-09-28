/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <iostream>
#include "llama_lmhead_operation.h"

static const uint64_t IN_TENSOR_COUNT = 3;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 3;
static const uint64_t NODE_COUNT = 4;

enum LlamaLmheadTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_MATMULWEIGHT,
    OUT_LMHEADOUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_LINEAR_OUT,
    INTERMIDATE_GATHER_OUT
};

atb::Status LlamaLmheadOperation(const LlamaLmheadParam &param,
                                 atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputNormNode  = opGraph.nodes.at(nodeId++);
    atb::Node &linearNode  = opGraph.nodes.at(nodeId++);
    atb::Node &allgatherNode = opGraph.nodes.at(nodeId++);
    atb::Node &transposeNode = opGraph.nodes.at(nodeId++);

    // [bs, seq_len, hidden_size]
    atb::infer::RmsNormParam inputNormParam;
    inputNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    inputNormParam.normParam.epsilon = param.rmsNormEps;
    atb::CreateOperation(inputNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    // [bs, seq_len, hidden_size] * [hidden_size, vocsize_pre_card_num] -> [bs * seq_len, vocsize_pre_card_num]
    atb::infer::MatmulParam linearParam = {false, param.transpose};
    atb::CreateOperation(linearParam, &linearNode.operation);
    linearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_MATMULWEIGHT};
    linearNode.outTensorIds = {INTERMIDATE_LINEAR_OUT};
    linearNode.inTensorReshapeFuncs.resize(linearNode.inTensorIds.size());
    linearNode.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
      newShape.dimNum = 2; // dimNum: 2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };

    // [bs * seq_len, vocsize_pre_card_num] -> [card_num, bs * seq_len, vocsize_pre_card_num]
    atb::infer::AllGatherParam allgatherParam = {param.rank, param.rankSize, 0, "hccl", param.hcclComm};
    atb::CreateOperation(allgatherParam, &allgatherNode.operation);
    allgatherNode.inTensorIds = {INTERMIDATE_LINEAR_OUT};
    allgatherNode.outTensorIds = {INTERMIDATE_GATHER_OUT};
    
    // [card_num, bs * seq_len, vocsize_pre_card_num] -> [bs * seq_len, card_num, vocsize_pre_card_num]
    atb::infer::TransposeParam transposeParam = {{1, 0, 2}};
    atb::CreateOperation(transposeParam, &transposeNode.operation);
    transposeNode.inTensorIds = {INTERMIDATE_GATHER_OUT};
    transposeNode.outTensorIds = {OUT_LMHEADOUT};

    // [bs * seq_len, vocsize]
    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 2;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0] * inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(2).shape.dims[1] * param.rankSize;
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);

    return atb::NO_ERROR;
}
