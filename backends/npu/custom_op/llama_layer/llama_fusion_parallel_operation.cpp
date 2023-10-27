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
#include "llama_self_attention_operation.h"
#include "llama_fusion_parallel_operation.h"
#include "llama_multi_layer_operation.h"
#include "llama_mlp_operation.h"
#include "llama_position_embedding_1d_split_fusion_operation.h"

static const uint64_t IN_TENSOR_COUNT = 16;
static const uint64_t OUT_TENSOR_COUNT = 1;
static const uint64_t INTERMEDIATE_TENSOR_COUNT = 15;
static const uint64_t NODE_COUNT = 12;

atb::Status LlamaLayerFusionParallelOperation(const LlamaLayerFusionParallelParam &param,
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
    atb::Node &castInNode = opGraph.nodes.at(nodeId++);
    atb::Node &cosSinSplitNode = opGraph.nodes.at(nodeId++);
    atb::Node &ropeNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfAttentionKvCacheNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfOutLinearParallelNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfResidualAddNode  = opGraph.nodes.at(nodeId++);
    atb::Node &selfNormNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mlpNode  = opGraph.nodes.at(nodeId++);
    atb::Node &mlpLinearParallelNode   = opGraph.nodes.at(nodeId++);
    atb::Node &mlpResidualAddNode   = opGraph.nodes.at(nodeId++);

    // 全量:[bs, seq_len, hidden_size] 
    atb::infer::RmsNormParam inputNormParam;
    inputNormParam.layerType = atb::infer::RmsNormParam::RmsNormType::RMS_NORM_NORM;
    inputNormParam.normParam.epsilon = param.rmsNormEps;
    atb::CreateOperation(inputNormParam, &inputNormNode.operation);
    inputNormNode.inTensorIds = {IN_HIDDENSTATES, IN_NORMWEIGHT};
    inputNormNode.outTensorIds = {INTERMIDATE_INPUTNORMOUT};

    // [bs, seq_len, hidden_size] * [3 * hidden_size / card_num, hidden_size] -> [bs，seq_len, hidden_size / card_num]
    MultiLayerLinearParam multiLayerLinearParam;
    multiLayerLinearParam.transpose = param.transpose;
    CreateLlamaMultiLayerLinearOperation(multiLayerLinearParam, &mixdQKVLinearNode.operation);
    mixdQKVLinearNode.inTensorIds = {INTERMIDATE_INPUTNORMOUT, IN_QKVMIXDWEIGHT};
    mixdQKVLinearNode.outTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, INTERMIDATE_MIXEDV};

    atb::infer::ElewiseParam castParam;
    castParam.elewiseType = atb::infer::ElewiseParam::ElewiseType::ELEWISE_CAST;
    castParam.outTensorType = ACL_FLOAT16;
    atb::CreateOperation(castParam, &castInNode.operation);
    castInNode.inTensorIds = {IN_COS_SIN_TABLE};
    castInNode.outTensorIds = {INTERNAL_CAST_COS_SIN_TABLE};

     // [2, max_bs, 1, seq_len, head_dim] -> [1, max_bs, 1, seq_len, head_dim]
    atb::infer::SplitParam splitParam = {0, 2};
    atb::CreateOperation(splitParam, &cosSinSplitNode.operation);
    cosSinSplitNode.inTensorIds = {INTERNAL_CAST_COS_SIN_TABLE};
    cosSinSplitNode.outTensorIds = {INTERMIDATE_CASTCOS, INTERMIDATE_CASTSIN};
    cosSinSplitNode.inTensorReshapeFuncs.resize(cosSinSplitNode.inTensorIds.size());
    cosSinSplitNode.inTensorReshapeFuncs.at(0) = [seqLenPtr](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape = oldShape;
        *seqLenPtr = oldShape.dims[3]; // 获取一下seqLen大小，帮助后面infer shape
    };

    // output:[bs * seq_len, head_dim * head_num_pre_card]
    atb::infer::RopeParam ropeParam;
    ropeParam.rotaryCoeff = param.rotaryCoeff;
    atb::CreateOperation(ropeParam, &ropeNode.operation);
    ropeNode.inTensorIds = {INTERMIDATE_MIXEDQ, INTERMIDATE_MIXEDK, INTERMIDATE_CASTCOS, INTERMIDATE_CASTSIN, IN_SEQLEN};
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
    ropeNode.inTensorReshapeFuncs.at(2) = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1] * oldShape.dims[2] * oldShape.dims[3];
        newShape.dims[1] = oldShape.dims[4];
    };
    ropeNode.inTensorReshapeFuncs.at(3) = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 2;
        newShape.dims[0] = oldShape.dims[0] * oldShape.dims[1] * oldShape.dims[2] * oldShape.dims[3];
        newShape.dims[1] = oldShape.dims[4];
    };
    ropeNode.inTensorReshapeFuncs.at(4) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1;
        newShape.dims[0] = oldShape.dims[0];
    };

    // output: [bs, seqlen, head_dim * head_num_pre_card]
    // Q:[bs * seq_len, head_dim * head_num_pre_car]
    // K:[bs * seq_len, head_dim * head_num_pre_car]
    // V:[bs，seq_len, hidden_size / head_num_pre_car]
    // CacheK:[1, max_bs, head_num / card_num, max_length, head_dim]
    // CacheV:[1, max_bs, head_num / card_num, max_length, head_dim]
    // attention_mask: [max_bs, 1, max_len, max_len]
    // tokenoffset:[bs, 1]
    atb::infer::SelfAttentionParam selfAttentionKvCacheParam;
    selfAttentionKvCacheParam.headDim = param.headDim;
    selfAttentionKvCacheParam.headNum = param.headNum;
    selfAttentionKvCacheParam.qkScale = param.qkScale;
    selfAttentionKvCacheParam.batchRunStatusEnable = param.batchRunStatusEnable;
    atb::CreateOperation(selfAttentionKvCacheParam, &selfAttentionKvCacheNode.operation);
    if (param.batchRunStatusEnable) {
        selfAttentionKvCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ,
                                                INTERMIDATE_POSITIONEMBEDK,
                                                INTERMIDATE_MIXEDV,
                                                IN_CACHE_K,
                                                IN_CACHE_V,
                                                IN_ATTENTIONMASK,
                                                IN_TOKENOFFSET,
                                                IN_SEQLEN,
                                                IN_LAYERID,
                                                IN_BATCH_STATUS};
    } else {
        selfAttentionKvCacheNode.inTensorIds = {INTERMIDATE_POSITIONEMBEDQ,
                                                INTERMIDATE_POSITIONEMBEDK,
                                                INTERMIDATE_MIXEDV,
                                                IN_CACHE_K,
                                                IN_CACHE_V,
                                                IN_ATTENTIONMASK,
                                                IN_TOKENOFFSET,
                                                IN_SEQLEN,
                                                IN_LAYERID};
    }

    selfAttentionKvCacheNode.outTensorIds = {INTERMIDATE_SELFOUT};
    selfAttentionKvCacheNode.inTensorReshapeFuncs.resize(selfAttentionKvCacheNode.inTensorIds.size());
    // 当前selfAttentionKvCache输入需要4维
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(0) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // dimNum: 4
        newShape.dims[0] = oldShape.dims[0] / (*seqLenPtr);
        newShape.dims[1] = (*seqLenPtr);
        newShape.dims[2] = param.headNum;
        newShape.dims[3] = oldShape.dims[1] / param.headNum;
    };
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(1) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 4; // dimNum: 4
        newShape.dims[0] = oldShape.dims[0] / (*seqLenPtr);
        newShape.dims[1] = (*seqLenPtr);
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

    if (param.batchRunStatusEnable) {
        selfAttentionKvCacheNode.inTensorReshapeFuncs.at(3) = [](const atb::Dims &oldShape, atb::Dims &newShape) {
            // 生成的是[1, max_batch_size, head_num, max_len, head_dim]
            // 加速库需要[max_batch_size, max_len, hidden_size], 理论应由transpose完成，但KVCache读写都为加速库完成，故直接reshape规避
            newShape.dimNum = 3; // dimNum: 4
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = oldShape.dims[2];
            newShape.dims[2] = oldShape.dims[1] * oldShape.dims[3];
        };
        selfAttentionKvCacheNode.inTensorReshapeFuncs.at(4) = [](const atb::Dims &oldShape, atb::Dims &newShape) {
            // 生成的是[1, max_batch_size, head_num, max_len, head_dim]
            // 加速库需要[max_batch_size, max_len, hidden_size], 理论应由transpose完成，但KVCache读写都为加速库完成，故直接reshape规避
            newShape.dimNum = 3; // dimNum: 4
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = oldShape.dims[2];
            newShape.dims[2] = oldShape.dims[1] * oldShape.dims[3];
        };
        selfAttentionKvCacheNode.inTensorReshapeFuncs.at(5) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 3;
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = oldShape.dims[2];
            newShape.dims[2] = oldShape.dims[3];
        };
    } else {
        selfAttentionKvCacheNode.inTensorReshapeFuncs.at(3) = [](const atb::Dims &oldShape, atb::Dims &newShape) {
            // 生成的是[1, max_batch_size, head_num, max_len, head_dim]
            // 加速库需要[layer, max_batch_size, max_len, hidden_size], 理论应由transpose完成，但读写都为加速库使用，故直接reshape规避
            newShape.dimNum = 4; // dimNum: 4
            newShape.dims[0] = 1;
            newShape.dims[1] = oldShape.dims[0];
            newShape.dims[2] = oldShape.dims[2];
            newShape.dims[3] = oldShape.dims[1] * oldShape.dims[3];
        };
        selfAttentionKvCacheNode.inTensorReshapeFuncs.at(4) = [](const atb::Dims &oldShape, atb::Dims &newShape) {
            // 生成的是[1, max_batch_size, head_num, max_len, head_dim]
            // 加速库需要[layer, max_batch_size, max_len, hidden_size 理论应由transpose完成，但读写都为加速库使用，故直接reshape规避
            newShape.dimNum = 4; // dimNum: 4
            newShape.dims[0] = 1;
            newShape.dims[1] = oldShape.dims[0];
            newShape.dims[2] = oldShape.dims[2];
            newShape.dims[3] = oldShape.dims[1] * oldShape.dims[3];
        };
        // attention mask: [bs, 1, max_len, max_len]
        selfAttentionKvCacheNode.inTensorReshapeFuncs.at(5) = [=](const atb::Dims &oldShape, atb::Dims &newShape) {
            newShape.dimNum = 3; // dimNum: 4
            newShape.dims[0] = oldShape.dims[0];
            newShape.dims[1] = oldShape.dims[2];
            newShape.dims[2] = oldShape.dims[3];
        };
    }
    // kv_seq_len: [bs, 1]
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(6) = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1; // dimNum: 1
        newShape.dims[0] = oldShape.dims[0];
    };
    // q_seq_len: [bs, 1]
    selfAttentionKvCacheNode.inTensorReshapeFuncs.at(7) = [](const atb::Dims &oldShape, atb::Dims &newShape) {
        newShape.dimNum = 1; // dimNum: 1
        newShape.dims[0] = oldShape.dims[0];
    };

    // [1, 1, 512] * [512, 4096] -> [1, 1, 4096]
    atb::infer::LinearParallelParam selfOutLinearParallelParam;
    selfOutLinearParallelParam.transWeight = false;
    selfOutLinearParallelParam.rank = param.rank;
    selfOutLinearParallelParam.rankSize = param.rankSize;
    selfOutLinearParallelParam.rankRoot = 0;
    selfOutLinearParallelParam.bias = "None";
    selfOutLinearParallelParam.parallelType = "RowParallel";
    selfOutLinearParallelParam.backend = "lccl";
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
    llamaMlpParam.transpose = true;
    CreateLlamaMlpOperation(llamaMlpParam, &mlpNode.operation);
    mlpNode.inTensorIds = {INTERMIDATE_SELFNORMOUT, IN_MLPGATEUPWEIGHT};
    mlpNode.outTensorIds = {INTERMIDATE_MLPOUT};

    atb::infer::LinearParallelParam mlpLinearParallelParam;
    mlpLinearParallelParam.transWeight = false;
    mlpLinearParallelParam.rank = param.rank;
    mlpLinearParallelParam.rankSize = param.rankSize;
    mlpLinearParallelParam.rankRoot = 0;
    mlpLinearParallelParam.bias = "None";
    mlpLinearParallelParam.parallelType = "RowParallel";
    mlpLinearParallelParam.backend = "lccl";
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
