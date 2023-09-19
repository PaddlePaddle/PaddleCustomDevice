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
#pragma once

#include <atb/atb_infer.h>
#include <atb/svector.h>

enum LlamaLayerFusionParallelTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_QKVMIXDWEIGHT,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPGATEUPWEIGHT,
    IN_MLPDOWNWEIGHT,
    IN_POSITIONIDS,
    IN_COS_SIN_TABLE,
    IN_ATTENTIONMASK,
    IN_CACHE_KV,
    IN_TOKENOFFSET,
    IN_SEQLEN,
    IN_LAYERID,
    IN_BATCH_STATUS,
    OUT_LLAMA13BLAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_MIXEDQ,
    INTERMIDATE_MIXEDK,
    INTERMIDATE_MIXEDV,
    INTERNAL_CAST_COS_SIN_TABLE,
    INTERMIDATE_CASTCOS,
    INTERMIDATE_CASTSIN,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_CACHEK,
    INTERMIDATE_CACHEV,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_MLPOUT,
    INTERMIDATE_MLPLINEARPARALLELOUT,
};

struct LlamaLayerFusionParallelParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int headDim = 0;
    int rank = 0;
    int rankSize = 1;
    float qkScale = 1.0;
    int rotaryCoeff = 2;
    bool transpose = true;
    void *hcclComm = nullptr; // only effect when hcclComm is not null
    bool batchRunStatusEnable = false; // enable dynamic batch
};

atb::Status LlamaLayerFusionParallelOperation(const LlamaLayerFusionParallelParam &param,
                                                        atb::Operation **operation);
