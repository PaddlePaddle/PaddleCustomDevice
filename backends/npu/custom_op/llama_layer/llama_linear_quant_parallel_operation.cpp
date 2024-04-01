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
#include "llama_linear_quant_parallel_operation.h"
#include <numeric>
#include <cmath>
#include <atb/atb_infer.h>

enum LlamaLinearQuantParallelTensorId {
    IN_INPUT = 0,
    IN_WEIGHT,
    IN_DEQSCALE,
    OUT_LINEAROUT,
    INTERMIDATE_LINEAROUT,
};
 
static const uint64_t IN_TENSOR_COUNT = 3;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 2;
 
atb::Status CreateLlamaLinearQuantParallelOperation(const llamaLinearQuantParallelParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
 
    size_t nodeId = 0;
    atb::Node &linearQuantNode = opGraph.nodes.at(nodeId++);
    atb::Node &allReduceNode = opGraph.nodes.at(nodeId++);
    // atb::Node &dequantAddNode = opGraph.nodes.at(nodeId++);

    atb::infer::LinearParam linearQuantParam;
    linearQuantParam.linearType = atb::infer::LinearType::LINEAR_INT8INT8_INT32_FP16;
    linearQuantParam.transposeA = false;
    linearQuantParam.transposeB = !param.transWeight;
    linearQuantParam.hasBias = false;
    CreateOperation(linearQuantParam, &linearQuantNode.operation);
    linearQuantNode.inTensorIds = {IN_INPUT, IN_WEIGHT, IN_DEQSCALE};
    linearQuantNode.outTensorIds = {INTERMIDATE_LINEAROUT};
 
    atb::infer::AllReduceParam allReduceParam = { param.rank, param.rankSize, param.rankRoot, "sum",
                                                 param.backend, param.hcclComm };
    CreateOperation(allReduceParam, &allReduceNode.operation);
    allReduceNode.inTensorIds = {INTERMIDATE_LINEAROUT};
    allReduceNode.outTensorIds = {OUT_LINEAROUT};
    // allReduceNode.inTensorReshapeFuncs.resize(allReduceNode.inTensorIds.size());
    // allReduceNode.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
    //     newShape.dimNum = 2; // dimNum: 2
    //     newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    //     newShape.dims[1] = oldShape.dims[2];
    // };
 
    
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
 
        outTensorDescs.at(0).dtype = ACL_FLOAT16;
        outTensorDescs.at(0).format = inTensorDescs.at(0).format;
        outTensorDescs.at(0).shape = inTensorDescs.at(0).shape;
        auto outDimSize = outTensorDescs.at(0).shape.dimNum;
        if (!param.transWeight) {
            outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(1).shape.dims[0];
        } else {
            outTensorDescs.at(0).shape.dims[outDimSize - 1] = inTensorDescs.at(1).shape.dims[1];
        }
 
        return atb::NO_ERROR;
    };
 
    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}