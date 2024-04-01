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
#include "llamalayer_mlp_dequant_operation.h"
#include <atb/atb_infer.h>
#include <memory>
 
enum LlamaMlpDequantTensorId {
    IN_HIDDENSTATUS = 0,
    IN_WEIGHTTENSOR,
    IN_DEQSCALE,
    OUT_MLPRESULTSTENSOR,
    INTERMIDATE_MATMUL_ALL_OUT,
};
 
static const uint64_t IN_TENSOR_COUNT = 3;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 1;
static const uint64_t NODE_COUNT = 2;
static uint64_t DIM3 = 3;
 
atb::Status CreateLlamaMlpDequantOperation(const LlamaMlpDequantParam &param, atb::Operation **operation)
{
 
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);
 
    size_t nodeId = 0;
    atb::Node &linearNode = opGraph.nodes.at(nodeId++);
    atb::Node &swishNode = opGraph.nodes.at(nodeId++);

    atb::infer::LinearParam linearQuantParam;
    linearQuantParam.linearType = atb::infer::LinearType::LINEAR_INT8INT8_INT32_FP16;
    linearQuantParam.transposeA = false;
    linearQuantParam.transposeB = param.transpose;
    linearQuantParam.hasBias = false;
    CreateOperation(linearQuantParam, &linearNode.operation);
    linearNode.inTensorIds = {IN_HIDDENSTATUS, IN_WEIGHTTENSOR, IN_DEQSCALE};
    linearNode.outTensorIds = {INTERMIDATE_MATMUL_ALL_OUT};
 
    atb::infer::ActivationParam activationParam;
    activationParam.activationType = atb::infer::ActivationType::ACTIVATION_SWIGLU_FORWARD;
    activationParam.dim = -1;
    CreateOperation(activationParam, &swishNode.operation);
    swishNode.inTensorIds = {INTERMIDATE_MATMUL_ALL_OUT};
    swishNode.outTensorIds = {OUT_MLPRESULTSTENSOR};
  
    opGraph.inferShapeFunc = [=](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        outTensorDescs.at(0).dtype = ACL_FLOAT16; // 修改为float16 type
        if (param.transpose == true) {
            outTensorDescs.at(0).shape.dimNum = DIM3;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
            outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(1).shape.dims[0] / 2;
        } else {
            outTensorDescs.at(0).shape.dimNum = DIM3;
            outTensorDescs.at(0).shape.dims[0] = inTensorDescs.at(0).shape.dims[0];
            outTensorDescs.at(0).shape.dims[1] = inTensorDescs.at(0).shape.dims[1];
            outTensorDescs.at(0).shape.dims[2] = inTensorDescs.at(1).shape.dims[1] / 2;
        }
        return atb::NO_ERROR;
    };
 
    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}