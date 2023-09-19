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

static const uint64_t IN_TENSOR_COUNT = 14;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 17;
static const uint64_t NODE_COUNT = 13;

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
    atb::Node &castInNode = opGraph.nodes.at(nodeId++);
    atb::Node &cosSinSplitNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode  = opGraph.nodes.at(nodeId++);
    atb::Node &cacheKVSplitNode = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionKvCacheNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearParallelNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mlpLinearParallelNode   = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode   = opGraph.nodes.at(nodeId++);

    // [bs, seq_len, hidden_size]
    atb::infer::RmsNormParam inputNormParam;
    inputNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    inputNormParam.normParam.epsilon = param.rmsNormEps;
    atb::CreateOperation(inputNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};
    inputNormNode.inTensorReshapeFuncs.resize(inputNormNode.inTensorIds.size());
    inputNormNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        if (oldShape.dimNum == 3) {
          newShape = oldShape;
        } else if (oldShape.dimNum == 2) {
          newShape.dimNum = 3; // 增量阶段
          newShape.dims[0] = oldShape.dims[0];
          newShape.dims[1] = 1;
          newShape.dims[2] = oldShape.dims[1];
        }
    };

    // [bs, seq_len, hidden_size] * [3 * hidden_size / card_num, hidden_size] -> [bs，seq_len, hidden_size / card_num]
    MultiLayerLinearParam multiLayerLinearParam;
    multiLayerLinearParam.transpose = param.transpose;
    CreateLlamaMultiLayerLinearOperation(multiLayerLinearParam, &mixdQKVLinearNode.operation);
    mixdQKVLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXDWEIGHT};
    mixdQKVLinearNode.outTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, INTERMIDATE_MIXEDV};

    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    CreateOperation(castParam, &castInNode.operation);
    castInNode.inTensorIds = {IN_COS_SIN_TABLE};
    castInNode.outTensorIds = {INTERNAL_CAST_COS_SIN_TABLE};

    // [2, head_dim, 1, seq_len, 1] ? [1, head_dim, 1, seq_len, 1]
    atb::infer::SplitParam splitParam = {0, 2};
    atb::CreateOperation(splitParam, &cosSinSplitNode.operation);
    cosSinSplitNode.inTensorIds = {INTERNAL_CAST_COS_SIN_TABLE};
    cosSinSplitNode.outTensorIds = {INTERMIDATE_CASTCOS, INTERMIDATE_CASTSIN};

    // 全量：
    llamaPositionEmbedding1DSplitFusionParam positionEmbedding1dFusionParam;
    positionEmbedding1dFusionParam.headNum = param.headNum;
    positionEmbedding1dFusionParam.rotaryCoeff = param.rotaryCoeff;
    CreateLlamaPositionEmbedding1DSplitFusionOperation(positionEmbedding1dFusionParam, &ropeNode.operation);
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
    ropeNode.inTensorReshapeFuncs.at(3) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // dimNum: 4
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
        newShape.dims[2] = oldShape.dims[3];
        newShape.dims[3] = oldShape.dims[4];
    };
    ropeNode.inTensorReshapeFuncs.at(4) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // dimNum: 4
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
        newShape.dims[1] = oldShape.dims[2];
        newShape.dims[2] = oldShape.dims[3];
        newShape.dims[3] = oldShape.dims[4];
    };
    ropeNode.inTensorReshapeFuncs.at(5) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1; // dimNum: 4
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };

    // [2, 1, head_num / card_num, max_length, head_dim]
    atb::infer::SplitParam splitKVParam = {0, 2};
    atb::CreateOperation(splitKVParam, &cacheKVSplitNode.operation);
    cacheKVSplitNode.inTensorIds = {IN_CACHE_KV};
    cacheKVSplitNode.outTensorIds = {INTERMIDATE_CACHEK, INTERMIDATE_CACHEV};
    //全量 [1, 1, 4, 128] [1, 1, 4, 128] [1, 1, 4, 128] [1, 4, 2048, 128] [1, 4, 2048, 128] [1, 4, 2048, 2048] [1, 1] [1, 1] [1]
    atb::infer::SelfAttentionParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.headDim = param.headDim;
    selfAttentionKvCacheParam.headNum = param.headNum;
    selfAttentionKvCacheParam.qScale = param.layerId; // TODO：
    atb::CreateOperation(selfAttentionKvCacheParam, &selfAttentionKvCacheNode.operation);
    selfAttentionKvCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ,
                                            INTERMIDATE_POSITIONEMBEDK,
                                            INTERMIDATE_MIXEDV,
                                            INTERMIDATE_CACHEK,
                                            INTERMIDATE_CACHEV,
                                            IN_ATTENTIONMASK,
                                            IN_TOKENOFFSET,
                                            IN_SEQLEN,
                                            IN_LAYERID};
    selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT};
    selfAttentionKvCacheNode.inTensorReshapeFuncs.resize(selfAttentionKvCacheNode.inTensorIds.size());
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // dimNum: 4
        newShape.dims[0] = 1;
        newShape.dims[1] = oldShape.dims[0];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[1] / param.headNum;
    };
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // dimNum: 4
        newShape.dims[0] = 1;
        newShape.dims[1] = oldShape.dims[0];
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[1] / param.headNum;
    };
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(2) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // dimNum: 4
        newShape.dims[0] = oldShape.dims[0];
        newShape.dims[1] = oldShape.dims[1]; // TODO: how to get seq_len
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[2] / param.headNum;
    };
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(3) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
      // 生成的是[1, max_batch_size, head_num, max_len, head_dim]
      // 加速库需要[layer, max_batch_size, max_len, head_size], 理论应有transpose完成，但读写都为加速库使用，故直接reshape规避
      newShape.dimNum = 4; // dimNum: 4
      newShape.dims[0] = 1;
      newShape.dims[1] = oldShape.dims[1];
      newShape.dims[2] = oldShape.dims[3];
      newShape.dims[3] = oldShape.dims[2] * oldShape.dims[4];
    };
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(4) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
      // 生成的是[1, max_batch_size, head_num, max_len, head_dim]
      // 加速库需要[layer, max_batch_size, max_len, head_size], 理论应有transpose完成，但读写都为加速库使用，故直接reshape规避
      newShape.dimNum = 4; // dimNum: 4
      newShape.dims[0] = 1;
      newShape.dims[1] = oldShape.dims[1];
      newShape.dims[2] = oldShape.dims[3];
      newShape.dims[3] = oldShape.dims[2] * oldShape.dims[4];
    };
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(5) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2; // dimNum: 4
        newShape.dims[0] = oldShape.dims[2];
        newShape.dims[1] = oldShape.dims[3];
    };
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(6) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1; // dimNum: 1
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(7) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1; // dimNum: 1
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1];
    };

    // [1, 1, 512] * [512, 4096] -> [1, 1, 4096]
    atb::infer::LinearParallelParam selfOutLinearParallelParam;
    selfOutLinearParallelParam.transWeight = true;
    selfOutLinearParallelParam.rank = param.rank;
    selfOutLinearParallelParam.rankSize = param.rankSize;
    selfOutLinearParallelParam.rankRoot = 0;
    selfOutLinearParallelParam.bias = "None";
    selfOutLinearParallelParam.parallelType = "RowParallel";
    selfOutLinearParallelParam.backend = "hccl";
    selfOutLinearParallelParam.hcclComm = param.hcclComm;
    atb::CreateOperation(selfOutLinearParallelParam, &selfOutLinearParallelNode.operation);
    selfOutLinearParallelNode.inTensorIds = {INTERMIDATE_SELFOUT, IN_SELFOUTLINEARWEIGHT};
    selfOutLinearParallelNode.outTensorIds = {INTERMIDATE_SELFLINEAROUT};

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
    atb::CreateOperation(selfNormParam, &selfNormNode.operation);
    selfNormNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, IN_SELFOUTNORMWEIGHT};
    selfNormNode.outTensorIds = {INTERMIDATE_SELFNORMOUT};

    LlamaMlpParam llamaMlpParam;
    llamaMlpParam.transpose = param.transpose;
    CreateLlamaMlpOperation(llamaMlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPGATEUPWEIGHT};
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
    atb::CreateOperation(mlpLinearParallelParam, &mlpLinearParallelNode.operation);
    mlpLinearParallelNode.inTensorIds = {INTERMIDATE_MLPOUT, IN_MLPDOWNWEIGHT};
    mlpLinearParallelNode.outTensorIds = {INTERMIDATE_MLPLINEARPARALLELOUT};

    atb::infer::ElewiseParam mlpResidualAddParam;
    mlpResidualAddParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_ADD;
    atb::CreateOperation(mlpResidualAddParam, &mlpResidualAddNode.operation);
    mlpResidualAddNode.inTensorIds = {INTERMIDATE_SELFRESIDUALADDOUT, INTERMIDATE_MLPLINEARPARALLELOUT};
    mlpResidualAddNode.outTensorIds = {OUT_LLAMA13BLAYEROUT};

    opGraph.inferShapeFunc = [&](const atb::SVector<atb::TensorDesc> &inTensorDescs,
                                 atb::SVector<atb::TensorDesc> &outTensorDescs) {
        outTensorDescs.at(0) = inTensorDescs.at(0);
        return atb::NO_ERROR;
    };

    atb::CreateOperation(opGraph, operation);
    return atb::NO_ERROR;
}
