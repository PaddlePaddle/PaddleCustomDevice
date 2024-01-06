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
#include <memory>
#include "llama_blockattn_smooth_parallel_operation.h"
#include "llamalayer_multi_layer_linear_quant_operation.h"
#include "llamalayer_mlp_dequant_operation.h"
#include "llama_linear_quant_parallel_operation.h"

static const uint64_t IN_TENSOR_COUNT = 26;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 18;
static const uint64_t NODE_COUNT = 17;

static void reshapeHeads(const atb::Dims &oldShape, atb::Dims &newShape, int headNum)
{
    newShape.dimNum = 3; // dimNum: 3
    if (oldShape.dimNum == 2) {
        newShape.dims[0] = oldShape.dims[0]; // 0 dim: n tokens
        newShape.dims[2] = oldShape.dims[1] / headNum;  // 1 dim: head size
    } else {
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1]; // 0 dim: n tokens
        newShape.dims[2] = oldShape.dims[2] / headNum;  // 1 dim: head size
    }

    newShape.dims[1] = headNum;  // 1 dim: head num
}

atb::Status LlamaBlockAttnSmoothParallelOperation(const LlamaBlockAttnSmoothParallelParam &param,
                                            atb::Operation **operation)
{
    std::shared_ptr<int64_t> seqLenPtr = std::make_shared<int64_t>(0);

    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputNormNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mixdQKVLinearNode  = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode  = opGraph.nodes.at(nodeId++);
    atb::Node &reshapeAndCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &attentionNode  = opGraph.nodes.at(nodeId++);
    atb::Node &outShiftAddNode  = opGraph.nodes.at(nodeId++);
    atb::Node &outSmoothMulNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutQuantNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearParallelNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mlpShiftAddNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mlpSmoothMulNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mlpQuantNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpLinearParallelNode   = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode   = opGraph.nodes.at(nodeId++);

    // 全量:[bs, seq_len, hidden_size]
    atb::infer::RmsNormParam inputNormParam;
    inputNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    inputNormParam.normParam.epsilon = param.rmsNormEps;
    inputNormParam.normParam.quantInputScale = param.inputNormQuantInputScale;
    inputNormParam.normParam.quantInputOffset = param.inputNormQuantInputOffset;
    inputNormParam.normParam.quantType = atb::infer::QUANT_INT8;
    atb::CreateOperation(inputNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES_SMOOTH, IN_NORMWEIGHT_SMOOTH, IN_NORM_BLANK_BIAS_SMOOTH};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT_SMOOTH};
    inputNormNode.inTensorReshapeFuncs.resize(inputNormNode.inTensorIds.size());
    inputNormNode.inTensorReshapeFuncs.at(0) = [seqLenPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape = oldShape;
        *seqLenPtr = oldShape.dims[1]; // 获取一下seqLen大小，帮助后面infer shape
    };

    // [bs, seq_len, hidden_size] * [3 * hidden_size / card_num, hidden_size] -> [bs，seq_len, hidden_size / card_num]
    MultiLayerLinearQuantParam multiLayerLinearQuantParam;
    multiLayerLinearQuantParam.transpose = param.transpose;
    CreateLlamaMultiLayerLinearQuantOperation(multiLayerLinearQuantParam, &mixdQKVLinearNode.operation);
    mixdQKVLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT_SMOOTH, IN_QKVMIXDWEIGHT_SMOOTH, IN_QKVDEQSCALE_SMOOTH};
    mixdQKVLinearNode.outTensorIds = {INTERMIDATE_MIXEDQ_SMOOTH, INTERMIDATE_MIXEDK_SMOOTH, INTERMIDATE_MIXEDV_SMOOTH};

    // output:[bs * seq_len, head_dim * head_num_pre_card]
    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = param.rotaryCoeff;
    atb::CreateOperation(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMIDATE_MIXEDQ_SMOOTH, INTERMIDATE_MIXEDK_SMOOTH, IN_COS_TABLE_SMOOTH, IN_SIN_TABLE_SMOOTH, IN_SEQLEN_SMOOTH};
    ropeNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ_SMOOTH, INTERMIDATE_POSITIONEMBEDK_SMOOTH};
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs.at(0) = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };
    ropeNode.inTensorReshapeFuncs.at(1) = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };
    ropeNode.inTensorReshapeFuncs.at(4) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1;
        newShape.dims[0] = oldShape.dims[0];
    };

    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CreateOperation(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDK_SMOOTH, INTERMIDATE_MIXEDV_SMOOTH,
                                        IN_CACHE_K_SMOOTH, IN_CACHE_V_SMOOTH, IN_SLOTS_SMOOTH};
    reshapeAndCacheNode.outTensorIds = {};
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
    reshapeAndCacheNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.headNum);
    };
    reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.headNum);
    };
    reshapeAndCacheNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {

        newShape = oldShape;
        newShape.dims[1] = oldShape.dims[2];
        newShape.dims[2] = oldShape.dims[1];
    };
    reshapeAndCacheNode.inTensorReshapeFuncs[3] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape = oldShape;
        newShape.dims[1] = oldShape.dims[2];
        newShape.dims[2] = oldShape.dims[1];
    };

    if (param.isPrefill) {
        atb::infer::SelfAttentionParam faEnParam;
        faEnParam.headDim = param.headDim;
        faEnParam.headNum = param.headNum;
        faEnParam.qkScale = param.qkScale;
        faEnParam.kvHeadNum = param.headNum;
        faEnParam.isEncoder = true; // use encoder when decoder is pagedAttention
        faEnParam.maskType = atb::infer::SelfAttentionParam::MASK_TYPE_NORM;
        faEnParam.isTriuMask = 1;
        CreateOperation(faEnParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ_SMOOTH, INTERMIDATE_POSITIONEMBEDK_SMOOTH, INTERMIDATE_MIXEDV_SMOOTH,
                                     IN_ATTENTIONMASK_SMOOTH, IN_SEQLEN_SMOOTH};
        attentionNode.outTensorIds = {INTERMIDATE_SELFOUT_SMOOTH};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, param.headNum);
        };
        attentionNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, param.headNum);
        };
        attentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, param.headNum);
        };
    } else {
        atb::infer::PagedAttentionParam paDeParam;
        paDeParam.headNum = param.headNum;
        paDeParam.qkScale = param.qkScale;
        paDeParam.kvHeadNum = param.headNum;
        paDeParam.maskType = atb::infer::PagedAttentionParam::UNDEFINED;
        paDeParam.isSupportAlibi = true;
        paDeParam.batchRunStatusEnable = true;
        CreateOperation(paDeParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ_SMOOTH, IN_CACHE_K_SMOOTH, IN_CACHE_V_SMOOTH,
                                     IN_BLOCK_TABLES_SMOOTH, IN_SEQLEN_SMOOTH, IN_BATCH_STATUS_SMOOTH};
        attentionNode.outTensorIds = {INTERMIDATE_SELFOUT_SMOOTH};
        attentionNode.inTensorReshapeFuncs.resize(attentionNode.inTensorIds.size());
        attentionNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            reshapeHeads(oldShape, newShape, param.headNum);
        };
        // 传过来[max_block_num, head_num, block_size, head_dim]
        // 加速库需要[max_block_num, block_size, head_num, head_dim]
        attentionNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape = oldShape;
            newShape.dims[1] = oldShape.dims[2];
            newShape.dims[2] = oldShape.dims[1];
        };
        attentionNode.inTensorReshapeFuncs[2] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape = oldShape;
            newShape.dims[1] = oldShape.dims[2];
            newShape.dims[2] = oldShape.dims[1];
        };
    }

    atb::infer::ElewiseParam outShiftAddParam;
    outShiftAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(outShiftAddParam, &outShiftAddNode.operation);
    outShiftAddNode.inTensorIds = {INTERMIDATE_SELFOUT_SMOOTH, IN_SELFOUTLINEARSHIFT_SMOOTH};
    outShiftAddNode.outTensorIds = {IINTERMIDATE_SELFOUTLINEARADDSHIFTOUT_SMOOTH};
    outShiftAddNode.inTensorReshapeFuncs.resize(outShiftAddNode.inTensorIds.size());
    outShiftAddNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;  // dimNum is 2
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2];
    };

    atb::infer::ElewiseParam outSmoothMulParam;
    outSmoothMulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::CreateOperation(outSmoothMulParam, &outSmoothMulNode.operation);
    outSmoothMulNode.inTensorIds = {IINTERMIDATE_SELFOUTLINEARADDSHIFTOUT_SMOOTH, IN_SELFOUTLINEARSMOOTH_SMOOTH};
    outSmoothMulNode.outTensorIds = {IINTERMIDATE_SELFOUTLINEARMULSMOOTHOUT_SMOOTH};

    atb::infer::ElewiseParam selfOutLinearParallelQuantParam;
    selfOutLinearParallelQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
    selfOutLinearParallelQuantParam.quantParam.inputScale = param.selfOutLinearParallelQuantScale;
    selfOutLinearParallelQuantParam.quantParam.inputOffset = param.selfOutLinearParallelQuantOffset;
    CreateOperation(selfOutLinearParallelQuantParam, &selfOutQuantNode.operation);
    selfOutQuantNode.inTensorIds = {IINTERMIDATE_SELFOUTLINEARMULSMOOTHOUT_SMOOTH};
    selfOutQuantNode.outTensorIds = {INTERMIDATE_SELFOUT_QUANT_SMOOTH};

    // [1, 1, 512] * [512, 4096] -> [1, 1, 4096]
    llamaLinearQuantParallelParam selfOutLinearParallelParam;
    selfOutLinearParallelParam.transWeight = false;
    selfOutLinearParallelParam.rank = param.rank;
    selfOutLinearParallelParam.rankSize = param.rankSize;
    selfOutLinearParallelParam.rankRoot = 0;
    selfOutLinearParallelParam.bias = "None";
    selfOutLinearParallelParam.parallelType = "RowParallel";
    selfOutLinearParallelParam.backend = "lccl";
    selfOutLinearParallelParam.hcclComm = param.hcclComm;
    CreateLlamaLinearQuantParallelOperation(selfOutLinearParallelParam, &selfOutLinearParallelNode.operation);
    selfOutLinearParallelNode.inTensorIds = {INTERMIDATE_SELFOUT_QUANT_SMOOTH, IN_SELFOUTLINEARWEIGHT_SMOOTH, IN_SELFOUTLINEARDEQSCALE_SMOOTH};
    selfOutLinearParallelNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT_SMOOTH};
    atb::infer::ElewiseParam selfResidualAddParam;
    selfResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(selfResidualAddParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES_SMOOTH, INTERMIDATE_SELFLINEAROUT_SMOOTH};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT_SMOOTH};
    selfResidualAddNode.inTensorReshapeFuncs.resize(selfResidualAddNode.inTensorIds.size());


    atb::infer::RmsNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    selfNormParam.normParam.epsilon = param.rmsNormEps;
    selfNormParam.normParam.quantInputScale = param.selfNormQuantInputScale;
    selfNormParam.normParam.quantInputOffset = param.selfNormQuantInputOffset;
    selfNormParam.normParam.quantType = atb::infer::QUANT_INT8;
    atb::CreateOperation(selfNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT_SMOOTH, IN_SELFOUTNORMWEIGHT_SMOOTH, IN_SELFOUTNORM_BLANK_BIAS_SMOOTH};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT_SMOOTH};

    LlamaMlpDequantParam llamaMlpDequantParam;
    llamaMlpDequantParam.transpose = true;
    CreateLlamaMlpDequantOperation(llamaMlpDequantParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT_SMOOTH, IN_MLPGATEUPWEIGHT_SMOOTH, IN_MLPDEQSCALE_SMOOTH};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT_SMOOTH};


    atb::infer::ElewiseParam mlpShiftAddParam;
    mlpShiftAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(mlpShiftAddParam, &mlpShiftAddNode.operation);
    mlpShiftAddNode.inTensorIds = {INTERMIDATE_MLPOUT_SMOOTH, IN_MLPDOWNSHIFT_SMOOTH};
    mlpShiftAddNode.outTensorIds = {IINTERMIDATE_MLPDOWNADDSHIFTOUT_SMOOTH};
    mlpShiftAddNode.inTensorReshapeFuncs.resize(mlpShiftAddNode.inTensorIds.size());

    atb::infer::ElewiseParam mlpSmoothMulParam;
    mlpSmoothMulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::CreateOperation(mlpSmoothMulParam, &mlpSmoothMulNode.operation);
    mlpSmoothMulNode.inTensorIds = {IINTERMIDATE_MLPDOWNADDSHIFTOUT_SMOOTH, IN_MLPDOWNSMOOTH_SMOOTH};
    mlpSmoothMulNode.outTensorIds = {IINTERMIDATE_MLPDOWNMULSMOOTHOUT_SMOOTH};

    atb::infer::ElewiseParam mlpLinearParallelQuantParam;
    mlpLinearParallelQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
    mlpLinearParallelQuantParam.quantParam.inputScale = param.mlpLinearParallelQuantScale;
    mlpLinearParallelQuantParam.quantParam.inputOffset = param.mlpLinearParallelQuantOffset;
    CreateOperation(mlpLinearParallelQuantParam, &mlpQuantNode.operation);
    mlpQuantNode.inTensorIds = {IINTERMIDATE_MLPDOWNMULSMOOTHOUT_SMOOTH};
    mlpQuantNode.outTensorIds = {INTERMIDATE_MLPOUT_QUANT_SMOOTH};

    llamaLinearQuantParallelParam mlpLinearParallelParam;
    mlpLinearParallelParam.transWeight = false;
    mlpLinearParallelParam.rank = param.rank;
    mlpLinearParallelParam.rankSize = param.rankSize;
    mlpLinearParallelParam.rankRoot = 0;
    mlpLinearParallelParam.bias = "None";
    mlpLinearParallelParam.parallelType = "RowParallel";
    mlpLinearParallelParam.backend = "lccl";
    mlpLinearParallelParam.hcclComm = param.hcclComm;
    CreateLlamaLinearQuantParallelOperation(mlpLinearParallelParam, &mlpLinearParallelNode.operation);
    mlpLinearParallelNode.inTensorIds = {INTERMIDATE_MLPOUT_QUANT_SMOOTH, IN_MLPDOWNWEIGHT_SMOOTH, IN_MLPDOWNDEQSCALE_SMOOTH};
    mlpLinearParallelNode.outTensorIds = {INTERMIDATE_MLPLINEARPARALLELOUT_SMOOTH};

    atb::infer::ElewiseParam mlpResidualAddParam;
    mlpResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(mlpResidualAddParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT_SMOOTH, INTERMIDATE_MLPLINEARPARALLELOUT_SMOOTH};
    mlpResidualAddNode.outTensorIds = {OUT_LLAMALAYEROUT_SMOOTH};

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
