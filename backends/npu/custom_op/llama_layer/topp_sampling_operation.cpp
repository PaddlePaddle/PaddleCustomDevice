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
#if defined (USE_ATB_TOPP)
#include "topp_sampling_operation.h"

static const uint64_t IN_TENSOR_COUNT = 2;
static const uint64_t OUT_TENSOR_COUNT = 2;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 2;
static const uint64_t NODE_COUNT = 3;

enum ToppSamplingTensorId {
    IN_PROBS = 0,
    IN_TOPPS,
    OUT_CAST_TOPP_IDS,
    OUT_CAST_TOPP_PROBS,
    INTERMIDATE_PROBS_CAST,
    INTERMIDATE_TOPPS_CAST,
};

atb::Status CreateToppSamplingOperation(const ToppSamplingParam &param,
                                        atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &castProbsNode  = opGraph.nodes.at(nodeId++);
    atb::Node &castToppsNode  = opGraph.nodes.at(nodeId++);
    atb::Node &topPNode = opGraph.nodes.at(nodeId++);

    atb::infer::ElewiseParam castProbsParam;
    castProbsParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castProbsParam.outTensorType = ACL_FLOAT16;
    atb::CreateOperation(castProbsParam, &castProbsNode.operation);
    castProbsNode.inTensorIds = {IN_PROBS};
    castProbsNode.outTensorIds = {INTERMIDATE_PROBS_CAST};

    atb::infer::ElewiseParam castToppsParam;
    castToppsParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castToppsParam.outTensorType = ACL_FLOAT16;
    atb::CreateOperation(castToppsParam, &castToppsNode.operation);
    castToppsNode.inTensorIds = {IN_TOPPS};
    castToppsNode.outTensorIds = {INTERMIDATE_TOPPS_CAST};

    atb::infer::TopkToppSamplingParam TopParam = {param.randSeed, param.topk};
    atb::CreateOperation(TopParam, &topPNode.operation);
    topPNode.inTensorIds = {INTERMIDATE_PROBS_CAST, INTERMIDATE_TOPPS_CAST};
    topPNode.outTensorIds = {OUT_CAST_TOPP_IDS, OUT_CAST_TOPP_PROBS};

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        size_t dimNum = outTensorDescs.at(0).shape.dimNum;
        outTensorDescs.at(0).shape.dims[dimNum - 1] = 1;

        outTensorDescs.at(1) = inTensorDescs.at(0);

        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);

    return atb::NO_ERROR;
}
#endif