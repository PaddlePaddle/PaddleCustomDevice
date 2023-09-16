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
#include "llama_self_attention_operation.h"
#include "llama_fusion_parallel_operation.h"
#include "llama_multi_layer_operation.h"
#include "llama_mlp_operation.h"
#include "llama_position_embedding_1d_split_fusion_operation.h"

static const uint64_t IN_TENSOR_COUNT = 17;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 14;
static const uint64_t NODE_COUNT = 12;

atb::Status LlamaLayerFusionParallelOperation(const LlamaLayerFusionParallelParam &param,
                                                        atb::Operation **operation)
{
    atb::GraphParam opGraph;
    opGraph.inTensorNum = IN_TENSOR_COUNT;
    opGraph.outTensorNum = OUT_TENSOR_COUNT;
    opGraph.internalTensorNum = INTERMEDIATE_TENSOR_COUNT;
    opGraph.nodes.resize(NODE_COUNT);

    size_t nodeId = 0;
    atb::Node &inputNormNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mixdQKVLinearNode  = opGraph.nodes.at(nodeId++);
    atb::Node &castCosNode = opGraph.nodes.at(nodeId++);
    atb::Node &castSinNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionKvCacheNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearParallelNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mlpLinearParallelNode   = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode   = opGraph.nodes.at(nodeId++);

    atb::infer::RmsNormParam inputNormParam;
    inputNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    inputNormParam.normParam.epsilon = param.rmsNormEps;
    CreateOp(inputNormParam, &inputNormNode.op);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    MultiLayerLinearParam multiLayerLinearParam;
    multiLayerLinearParam.transpose = param.transpose;
    CreateLlamaMultiLayerLinearOperation(multiLayerLinearParam, &mixdQKVLinearNode.op);
    mixdQKVLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXDWEIGHT};
    mixdQKVLinearNode.outTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, INTERMIDATE_MIXEDV};

    atb::infer::ElewiseParam castCosParam;
    castCosParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    CreateOp(castCosParam, &castCosNode.op);
    castCosNode.inTensorIds = {IN_COSTABLE};
    castCosNode.outTensorIds = {INTERMIDATE_CASTCOS};

    atb::infer::ElewiseParam castSinParam;
    castSinParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    CreateOp(castSinParam, &castSinNode.op);
    castSinNode.inTensorIds = {IN_SINTABLE};
    castSinNode.outTensorIds = {INTERMIDATE_CASTSIN};

    llamaPositionEmbedding1DSplitFusionParam positionEmbedding1dFusionParam;
    positionEmbedding1dFusionParam.headNum = param.headNum;
    positionEmbedding1dFusionParam.rotaryCoeff = param.rotaryCoeff;
    CreateLlamaPositionEmbedding1DSplitFusionOperation(positionEmbedding1dFusionParam, &ropeNode.op);
    ropeNode.inTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, IN_POSITIONIDS, INTERMIDATE_CASTCOS, INTERMIDATE_CASTSIN, IN_SEQLEN};
    ropeNode.outTensorIds = {INTERMIDATE_POSITIONEMBEDQ, INTERMIDATE_POSITIONEMBEDK};
    ropeNode.inTensorReshapeFuncs.resize(ropeNode.inTensorIds.size());
    ropeNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };
    ropeNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 2
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
    };

    atb::infer::SelfAttentionParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.headDim = param.headDim;
    selfAttentionKvCacheParam.headNum = param.headNum;
    selfAttentionKvCacheParam.qScale = param.layerId; // TODO：
    CreateOp(selfAttentionKvCacheParam, &selfAttentionKvCacheNode.op);
    selfAttentionKvCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ,
                                            INTERMIDATE_POSITIONEMBEDK,
                                            INTERMIDATE_MIXEDV,
                                            IN_CACHEK,
                                            IN_CACHEV,
                                            IN_ATTENTIONMASK,
                                            IN_TOKENOFFSET,
                                            IN_SEQLEN,
                                            IN_LAYERID};
    selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT};
    selfAttentionKvCacheNode.inTensorReshapeFuncs.resize(selfAttentionKvCacheNode.inTensorIds.size());
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // dimNum: 4
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[0];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[1] / param.headNum;
    };
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // dimNum: 4
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[0];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[1] / param.headNum;
    };
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // dimNum: 4
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };

    atb::infer::LinearParallelParam selfOutLinearParallelParam;
    selfOutLinearParallelParam.transWeight = true;
    selfOutLinearParallelParam.rank = param.rank;
    selfOutLinearParallelParam.rankSize = param.rankSize;
    selfOutLinearParallelParam.rankRoot = 0;
    selfOutLinearParallelParam.bias = "None";
    selfOutLinearParallelParam.parallelType = "RowParallel";
    selfOutLinearParallelParam.backend = "hccl";
    selfOutLinearParallelParam.hcclComm = param.hcclComm;
    CreateOp(selfOutLinearParallelParam, &selfOutLinearParallelNode.op);
    selfOutLinearParallelNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT};
    selfOutLinearParallelNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

    atb::infer::ElewiseParam selfResidualAddParam;
    selfResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOp(selfResidualAddParam, &selfResidualAddNode.op);
    selfResidualAddNode.inTensorIds = {IN_HIDDENSTATES, INTERMIDATE_SELFLINEAROUT};
    selfResidualAddNode.outTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT};
    selfResidualAddNode.inTensorReshapeFuncs.resize(selfResidualAddNode.inTensorIds.size());
    selfResidualAddNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 3; // dimNum: 3
        newShape.dims[0] = oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[0];
        newShape.dims[2] = oldShape.dims[2];
    };

    atb::infer::RmsNormParam selfNormParam;
    selfNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    selfNormParam.normParam.epsilon = param.rmsNormEps;
    CreateOp(selfNormParam, &selfNormNode.op);
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    LlamaMlpParam llamaMlpParam;
    llamaMlpParam.transpose = param.transpose;
    CreateLlamaMlpOperation(llamaMlpParam, &mlpNode.op);
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPGATEWEIGHT, IN_MLPUPWEIGHT};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    atb::infer::LinearParallelParam mlpLinearParallelParam;
    mlpLinearParallelParam.transWeight = true;
    mlpLinearParallelParam.rank = param.rank;
    mlpLinearParallelParam.rankSize = param.rankSize;
    mlpLinearParallelParam.rankRoot = 0;
    mlpLinearParallelParam.bias = "None";
    mlpLinearParallelParam.parallelType = "RowParallel";
    mlpLinearParallelParam.backend = "hccl";
    mlpLinearParallelParam.hcclComm = param.hcclComm;
    CreateOp(mlpLinearParallelParam, &mlpLinearParallelNode.op);
    mlpLinearParallelNode.inTensorIds = {INTERMIDATE_MLPOUT, IN_MLPDOWNWEIGHT};
    mlpLinearParallelNode.outTensorIds = {INTERMIDATE_MLPLINEARPARALLELOUT};

    atb::infer::ElewiseParam mlpResidualAddParam;
    mlpResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    CreateOp(mlpResidualAddParam, &mlpResidualAddNode.op);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPLINEARPARALLELOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LLAMA13BLAYEROUT};

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOp(opGraph, operation);
    return atb::NO_ERROR;
}
