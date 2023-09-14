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
#include "llama_position_embedding_1d_split_fusion_operation.h"
#include <atb/atb_infer.h>

enum LlamaPositionEmbedding1DSplitFusionTensorId {
    IN_QLAYERTENSOR = 0,
    IN_KLAYERTENSOR,
    IN_POSITIONIDSTENSOR,
    IN_COSTABLETENSOR,
    IN_SINTABLETENSOR,
    IN_SEQLENTENSOR,
    OUT_QEMBEDDEDTENSOR,
    OUT_KEMBEDDEDTENSOR,
    INTERMIDATE_COS,
    INTERMIDATE_SIN,
};

static const uint64_t IN_TENSOR_COUNT = 6;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 2;
static const uint64_t NODE_COUNT = 3;
static uint64_t DIM3 = 3;

atb::Status CreateLlamaPositionEmbedding1DSplitFusionOperation(const llamaPositionEmbedding1DSplitFusionParam &param, atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &embedding0Node = opGraph.nodes.at(nodeId++);
    atb::Node &embedding1Node = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode = opGraph.nodes.at(nodeId++);


    atb::infer::GatherParam embedding0NodeParam;
    embedding0NodeParam.axis = 0;
    CreateOp(embedding0NodeParam, &embedding0Node.op);
    embedding0Node.inTensorIds = {IN_COSTABLETENSOR, IN_POSITIONIDSTENSOR};
    embedding0Node.outTensorIds = {INTERMIDATE_COS};
    embedding0Node.inTensorReshapeFuncs.resize(embedding0Node.inTensorIds.size());
    embedding0Node.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        if (oldShape.dims[0] == 1) {
            newShape.dimNum = oldShape.dimNum - 2;
            for (size_t i = 0; i < newShape.dimNum; i++) {
                newShape.dims[i] = oldShape.dims[i + 2];
            }
        } else {
            newShape = oldShape;
        }
    };

    atb::infer::GatherParam embedding1NodeParam;
    embedding1NodeParam.axis = 0;
    CreateOp(embedding1NodeParam, &embedding1Node.op);
    embedding1Node.inTensorIds = {IN_SINTABLETENSOR, IN_POSITIONIDSTENSOR};
    embedding1Node.outTensorIds = {INTERMIDATE_SIN};
    embedding1Node.inTensorReshapeFuncs.resize(embedding1Node.inTensorIds.size());
    embedding1Node.inTensorReshapeFuncs[0] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        if (oldShape.dims[0] == 1) {
            newShape.dimNum = oldShape.dimNum - 2;
            for (size_t i = 0; i < newShape.dimNum; i++) {
                newShape.dims[i] = oldShape.dims[i + 2];
            }
        } else {
            newShape = oldShape;
        }
    };

    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = 2;
    CreateOp(ropeParam, &ropeNode.op);
    ropeNode.inTensorIds = {IN_QLAYERTENSOR, IN_KLAYERTENSOR, INTERMIDATE_COS, INTERMIDATE_SIN, IN_SEQLENTENSOR};
    ropeNode.outTensorIds = {OUT_QEMBEDDEDTENSOR, OUT_KEMBEDDEDTENSOR};
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs[2] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };
    ropeNode.inTensorReshapeFuncs[3] = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };


    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {

        outTensorDescs.resize(2);
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).shape.dimNum = 2;
        outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
        outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
        outTensorDescs.at(1) = inTensorDescs.at(1);
        outTensorDescs.at(1).shape.dimNum = 2;
        outTensorDescs.at(1).shape.dims[0] = inTensorDescs.at(1).shape.dims[0];
        outTensorDescs.at(1).shape.dims[1] = inTensorDescs.at(1).shape.dims[1];

        return atb::NO_ERROR;
    };

    atb::CreateOp(opGraph, operation);
    return atb::NO_ERROR;
}
