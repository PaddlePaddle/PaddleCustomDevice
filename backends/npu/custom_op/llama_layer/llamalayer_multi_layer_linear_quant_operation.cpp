/*
 * Copyright (c) Huawei Ties Co., Ltd. 2023. All rights reserved.
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
#include "llamalayer_multi_layer_linear_quant_operation.h"
#include <atb/atb_infer.h>
 
 
enum LlamamultiLayerLinearQuantTensorId {
    IN_INPUTTENSOR = 0,
    IN_WEIGHTTENSOR,
    IN_DEQSCALE,
    IN_BLANK_BIAS,
    OUT_MATMULRESULTQTENSOR,
    OUT_MATMULRESULTKTENSOR,
    OUT_MATMULRESULTVTENSOR,
    INTERMIDATE_LINEAR_OUT,
};
 
static const uint64_t IN_TENSOR_COUNT = 4;
static const uint64_t OUT_TENSOR_COUNT = 3;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 2;
static uint64_t DIM3 = 3;
 
atb::Status CreateLlamaMultiLayerLinearQuantOperation(const MultiLayerLinearQuantParam &param, atb::Operation **operation)
{
 
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
 
    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::Node &splitNode = opGraph.nodes.at(nodeId++);
    
    atb::infer::LinearQuantParam linearQuantParam = {false, param.transpose, true};
    CreateOperation(linearQuantParam, &linearNode.operation);
    linearNode.inTensorIds = {IN_INPUTTENSOR, IN_WEIGHTTENSOR, IN_BLANK_BIAS, IN_DEQSCALE};
    linearNode.outTensorIds = {INTERMIDATE_LINEAR_OUT};
    linearNode.inTensorReshapeFuncs.resize(linearNode.inTensorIds.size());
    linearNode.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
      newShape.dimNum = 2; // dimNum: 2
      newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
      newShape.dims[1] = oldShape.dims[2];
    };

    atb::infer::SplitParam splitParam = {1, 3};
    CreateOperation(splitParam, &splitNode.operation);
    splitNode.inTensorIds = {INTERMIDATE_LINEAR_OUT};
    splitNode.outTensorIds = {OUT_MATMULRESULTQTENSOR, OUT_MATMULRESULTKTENSOR, OUT_MATMULRESULTVTENSOR};
 
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {                                   
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = DIM3;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(0).dtype = ACL_FLOAT16;

        outTensorDescs.at(1) = inTensorDescs.at(0);
        outTensorDescs.at(1).shape.dimNum = DIM3;
        outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(1).dtype = ACL_FLOAT16;

        outTensorDescs.at(2) = inTensorDescs.at(0);
        outTensorDescs.at(2).shape.dimNum = DIM3;
        outTensorDescs.at(2).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(2).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(2).dtype = ACL_FLOAT16;
        if (param.transpose == true) {
            outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(1).shape.dims[0] / DIM3;
 
            outTensorDescs.at(1).shape.dims[2] = inTensorDescs.at(1).shape.dims[0] / DIM3;
 
            outTensorDescs.at(2).shape.dims[2] = inTensorDescs.at(1).shape.dims[0] / DIM3;
        } else {
            outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(1).shape.dims[1] / DIM3;
 
            outTensorDescs.at(1).shape.dims[2] = inTensorDescs.at(1).shape.dims[1] / DIM3;
 
            outTensorDescs.at(2).shape.dims[2] = inTensorDescs.at(1).shape.dims[1] / DIM3;
        }
 
        return atb::NO_ERROR;
    };
 
    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}