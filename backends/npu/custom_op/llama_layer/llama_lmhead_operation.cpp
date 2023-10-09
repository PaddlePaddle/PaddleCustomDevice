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
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 2;

enum LlamaLmheadTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_MATMULWEIGHT,
    OUT_LMHEADOUT,
    INTERMIDATE_INPUTNORMOUT,
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
    linearNode.outTensorIds = {OUT_LMHEADOUT};
    linearNode.inTensorReshapeFuncs.resize(linearNode.inTensorIds.size());
    linearNode.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };

    // [bs, seq_len, vocsize_pre_card_num]
    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        size_t dimNum = outTensorDescs.at(0).shape.dimNum;
        if (param.transpose) {
            outTensorDescs.at(0).shape.dims[dimNum - 1] = inTensorDescs.at(2).shape.dims[0];
        } else {
            outTensorDescs.at(0).shape.dims[dimNum - 1] = inTensorDescs.at(2).shape.dims[1];
        }

        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);

    return atb::NO_ERROR;
}
