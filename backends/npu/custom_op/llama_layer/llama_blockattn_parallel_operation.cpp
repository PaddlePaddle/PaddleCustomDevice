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
#include "llama_blockattn_parallel_operation.h"
#include "llamalayer_multi_layer_linear_quant_operation.h"
#include "llamalayer_mlp_dequant_operation.h"
#include "llama_linear_quant_parallel_operation.h"

static const uint64_t IN_TENSOR_COUNT = 36;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 18;
static const uint64_t NODE_COUNT = 17;

void reshapeHeads(const atb::Dims &oldShape, atb::Dims &newShape, int headNum)
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

atb::Status LlamaBlockAttnParallelOperation(const LlamaBlockAttnParallelParam &param,
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
    atb::Node &kValueQuantNode  = opGraph.nodes.at(nodeId++);
    atb::Node &vValueQuantNode  = opGraph.nodes.at(nodeId++);
    atb::Node &reshapeAndCacheNode = opGraph.nodes.at(nodeId++);
    atb::Node &attentionNode  = opGraph.nodes.at(nodeId++);
    atb::Node &outSmoothMulNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutQuantNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearParallelNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mlpSmoothMulNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mlpQuantNode = opGraph.nodes.at(nodeId++);
    atb::Node &mlpLinearParallelNode   = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode   = opGraph.nodes.at(nodeId++);

    // 全量:[bs, seq_len, hidden_size]
    atb::infer::RmsNormParam inputNormParam;
    inputNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    inputNormParam.normParam.epsilon = param.rmsNormEps;
    inputNormParam.normParam.quantType = atb::infer::QUANT_INT8;
    atb::CreateOperation(inputNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT, IN_NORM_BLANK_BIAS, IN_INPUT_RMSNORM_SCALE, IN_NORM_BLANK_OFFSET};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};
    inputNormNode.inTensorReshapeFuncs.resize(inputNormNode.inTensorIds.size());
    inputNormNode.inTensorReshapeFuncs.at(0) = [seqLenPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape = oldShape;
        *seqLenPtr = oldShape.dims[1]; // 获取一下seqLen大小，帮助后面infer shape
    };

    // [bs, seq_len, hidden_size] * [3 * hidden_size / card_num, hidden_size] -> [bs，seq_len, hidden_size / card_num]
    MultiLayerLinearQuantParam multiLayerLinearQuantParam;
    multiLayerLinearQuantParam.transpose = param.transpose;
    CreateLlamaMultiLayerLinearQuantOperation(multiLayerLinearQuantParam, &mixdQKVLinearNode.operation);
    mixdQKVLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXDWEIGHT, IN_QKVDEQSCALE};
    mixdQKVLinearNode.outTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, INTERMIDATE_MIXEDV};

    // output:[bs * seq_len, head_dim * head_num_pre_card]
    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = param.rotaryCoeff;
    atb::CreateOperation(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, IN_COS_TABLE, IN_SIN_TABLE, IN_SEQLEN};
    ropeNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK};
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

    atb::infer::ElewiseParam kValueQuantParam;
    kValueQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
    CreateOperation(kValueQuantParam, &kValueQuantNode.operation);
    kValueQuantNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDK, IN_K_QUANT_SCALES, IN_EMPTY_OFFSET};
    kValueQuantNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDK_INT8};

    atb::infer::ElewiseParam vValueQuantParam;
    vValueQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT_PER_CHANNEL;
    CreateOperation(vValueQuantParam, &vValueQuantNode.operation);
    vValueQuantNode.inTensorIds = {INTERMIDATE_MIXEDV, IN_V_QUANT_SCALES, IN_EMPTY_OFFSET};
    vValueQuantNode.outTensorIds = {INTERMIDATE_MIXEDV_INT8};

    // output: [bs, seqlen, head_dim * head_num_pre_card]
    // Q:[bs * seq_len, head_dim * head_num_pre_car]
    // K:[bs * seq_len, head_dim * head_num_pre_car]
    // V:[bs，seq_len, hidden_size / head_num_pre_car]
    // CacheK:[1, max_bs, head_num / card_num, max_length, head_dim]
    // CacheV:[1, max_bs, head_num / card_num, max_length, head_dim]
    // attention_mask: [max_bs, 1, max_len, max_len]
    // tokenoffset:[bs, 1]
    atb::infer::ReshapeAndCacheParam reshapeCacheParm;
    CreateOperation(reshapeCacheParm, &reshapeAndCacheNode.operation);
    reshapeAndCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDK_INT8, INTERMIDATE_MIXEDV_INT8,
                                        IN_CACHE_K, IN_CACHE_V, IN_SLOTS};
    reshapeAndCacheNode.outTensorIds = {IN_CACHE_K, IN_CACHE_V};
    reshapeAndCacheNode.inTensorReshapeFuncs.resize(reshapeAndCacheNode.inTensorIds.size());
    reshapeAndCacheNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.headNum);
    };
    reshapeAndCacheNode.inTensorReshapeFuncs[1] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        reshapeHeads(oldShape, newShape, param.headNum);
    };
    // 传过来[max_block_num, head_num, block_size, head_dim]
    // 加速库需要[max_block_num, block_size, head_num, head_dim]
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
        faEnParam.headNum = param.headNum;
        faEnParam.qkScale = param.qkScale;
        faEnParam.kvHeadNum = param.headNum;
        faEnParam.calcType = atb::infer::SelfAttentionParam::CalcType::PA_ENCODER;
        faEnParam.maskType = atb::infer::SelfAttentionParam::MaskType::MASK_TYPE_NORM;
        faEnParam.isTriuMask = 1;
        CreateOperation(faEnParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK, INTERMIDATE_MIXEDV,
                                     IN_ATTENTIONMASK, IN_SEQLEN};
        attentionNode.outTensorIds = {INTERMIDATE_SELFOUT};
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
        paDeParam.quantType = atb::infer::PagedAttentionParam::TYPE_DEQUANT_FUSION;
        paDeParam.hasQuantOffset  = false;
		paDeParam.batchRunStatusEnable = true;
        CreateOperation(paDeParam, &attentionNode.operation);
        attentionNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ, IN_CACHE_K, IN_CACHE_V, 
                                     IN_BLOCK_TABLES, IN_SEQLEN, IN_BATCH_STATUS, IN_K_DEQUANT_SCALES, IN_V_DEQUANT_SCALES};
        attentionNode.outTensorIds = {INTERMIDATE_SELFOUT};
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

    atb::infer::ElewiseParam outSmoothMulParam;
    outSmoothMulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::CreateOperation(outSmoothMulParam, &outSmoothMulNode.operation);
    outSmoothMulNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARSMOOTH};
    outSmoothMulNode.outTensorIds = {IINTERMIDATE_SELFOUTLINEARMULSMOOTHOUT};
    outSmoothMulNode.inTensorReshapeFuncs.resize(outSmoothMulNode.inTensorIds.size());
    outSmoothMulNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;  // dimNum is 2
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2];
    };

    atb::infer::ElewiseParam selfOutLinearParallelQuantParam;
    selfOutLinearParallelQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
    selfOutLinearParallelQuantParam.quantParam.inputScale = param.selfOutLinearParallelQuantScale;
    selfOutLinearParallelQuantParam.quantParam.inputOffset = param.selfOutLinearParallelQuantOffset;
    CreateOperation(selfOutLinearParallelQuantParam, &selfOutQuantNode.operation);
    selfOutQuantNode.inTensorIds = {IINTERMIDATE_SELFOUTLINEARMULSMOOTHOUT};
    selfOutQuantNode.outTensorIds = {INTERMIDATE_SELFOUT_QUANT};

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
    selfOutLinearParallelNode.inTensorIds = {INTERMIDATE_SELFOUT_QUANT, IN_SELFOUTLINEARWEIGHT, IN_SELFOUTLINEARDEQSCALE};
    selfOutLinearParallelNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};
    // selfOutLinearParallelNode.inTensorReshapeFuncs.resize(selfOutLinearParallelNode.inTensorIds.size());
    // selfOutLinearParallelNode.inTensorReshapeFuncs[0] = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
    //     newShape.dimNum = 2;  // dimNum is 2
    //     newShape.dims[0] = oldShape.dims[0];
    //     newShape.dims[1] = oldShape.dims[1] * oldShape.dims[2];
    // };
    // [bs * seq_len, hidden_size] + [1, 1, 4096]
    atb::infer::ElewiseParam selfResidualAddParam;
    selfResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(selfResidualAddParam, &selfResidualAddNode.operation);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};
    selfResidualAddNode.inTensorReshapeFuncs.resize(selfResidualAddNode.inTensorIds.size());
    // selfResidualAddNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
    //     newShape.dimNum = 2; // dimNum: 3
    //     newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    //     newShape.dims[1] = oldShape.dims[2];
    // };

    atb::infer::RmsNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    selfNormParam.normParam.epsilon = param.rmsNormEps;
    selfNormParam.normParam.quantType = atb::infer::QUANT_INT8;
    atb::CreateOperation(selfNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT, IN_SELFOUTNORM_BLANK_BIAS, IN_SELF_RMSNORM_SCALE, IN_NORM_BLANK_OFFSET};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    LlamaMlpDequantParam llamaMlpDequantParam;
    llamaMlpDequantParam.transpose = true;
    CreateLlamaMlpDequantOperation(llamaMlpDequantParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPGATEUPWEIGHT, IN_MLPDEQSCALE};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    atb::infer::ElewiseParam mlpSmoothMulParam;
    mlpSmoothMulParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_MUL;
    atb::CreateOperation(mlpSmoothMulParam, &mlpSmoothMulNode.operation);
    mlpSmoothMulNode.inTensorIds = {INTERMIDATE_MLPOUT, IN_MLPDOWNSMOOTH};
    mlpSmoothMulNode.outTensorIds = {IINTERMIDATE_MLPDOWNMULSMOOTHOUT};


    atb::infer::ElewiseParam mlpLinearParallelQuantParam;
    mlpLinearParallelQuantParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_QUANT;
    mlpLinearParallelQuantParam.quantParam.inputScale = param.mlpLinearParallelQuantScale;
    mlpLinearParallelQuantParam.quantParam.inputOffset = param.mlpLinearParallelQuantOffset;
    CreateOperation(mlpLinearParallelQuantParam, &mlpQuantNode.operation);
    mlpQuantNode.inTensorIds = {IINTERMIDATE_MLPDOWNMULSMOOTHOUT};
    mlpQuantNode.outTensorIds = {INTERMIDATE_MLPOUT_QUANT};

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
    mlpLinearParallelNode.inTensorIds = {INTERMIDATE_MLPOUT_QUANT, IN_MLPDOWNWEIGHT, IN_MLPDOWNDEQSCALE};
    mlpLinearParallelNode.outTensorIds = {INTERMIDATE_MLPLINEARPARALLELOUT};

    atb::infer::ElewiseParam mlpResidualAddParam;
    mlpResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(mlpResidualAddParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPLINEARPARALLELOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LLAMALAYEROUT};

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
