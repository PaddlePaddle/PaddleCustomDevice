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

enum LlamaBlockAttnSmoothParallelTensorId {
    IN_HIDDENSTATES_SMOOTH = 0,
    IN_NORMWEIGHT_SMOOTH,
    IN_NORM_BLANK_BIAS_SMOOTH,
    IN_QKVMIXDWEIGHT_SMOOTH,
    IN_QKVDEQSCALE_SMOOTH,
    IN_SELFOUTLINEARWEIGHT_SMOOTH,
    IN_SELFOUTLINEARSHIFT_SMOOTH,
    IN_SELFOUTLINEARSMOOTH_SMOOTH,
    IN_SELFOUTLINEARDEQSCALE_SMOOTH,
    IN_SELFOUTNORMWEIGHT_SMOOTH,
    IN_SELFOUTNORM_BLANK_BIAS_SMOOTH,
    IN_MLPGATEUPWEIGHT_SMOOTH,
    IN_MLPDEQSCALE_SMOOTH,
    IN_MLPDOWNWEIGHT_SMOOTH,
    IN_MLPDOWNSHIFT_SMOOTH,
    IN_MLPDOWNSMOOTH_SMOOTH,
    IN_MLPDOWNDEQSCALE_SMOOTH,
    IN_COS_TABLE_SMOOTH,
    IN_SIN_TABLE_SMOOTH,
    IN_ATTENTIONMASK_SMOOTH,
    IN_CACHE_K_SMOOTH,
    IN_CACHE_V_SMOOTH,
    IN_NORM_BLANK_OFFSET_SMOOTH,
    IN_INPUT_RMSNORM_SCALE_SMOOTH,
    IN_SELF_RMSNORM_SCALE_SMOOTH,
    IN_SELF_QUANT_SCALE_SMOOTH,
    IN_MLP_QUANT_SCALE_SMOOTH,
    IN_SEQLEN_SMOOTH,
    IN_BLOCK_TABLES_SMOOTH,
    IN_SLOTS_SMOOTH,
    IN_BATCH_STATUS_SMOOTH,
    OUT_LLAMALAYEROUT_SMOOTH,
    INTERMIDATE_INPUTNORMOUT_SMOOTH,
    INTERMIDATE_MIXEDQ_SMOOTH,
    INTERMIDATE_MIXEDK_SMOOTH,
    INTERMIDATE_MIXEDV_SMOOTH,
    INTERMIDATE_POSITIONEMBEDQ_SMOOTH,
    INTERMIDATE_POSITIONEMBEDK_SMOOTH,
    INTERMIDATE_SELFOUT_SMOOTH,
    INTERMIDATE_SELFOUT_QUANT_SMOOTH,
    INTERMIDATE_SELFLINEAROUT_SMOOTH,
    INTERMIDATE_SELFRESIDUALADDOUT_SMOOTH,
    INTERMIDATE_SELFNORMOUT_SMOOTH,
    INTERMIDATE_MLPOUT_SMOOTH,
    INTERMIDATE_MLPOUT_QUANT_SMOOTH,
    INTERMIDATE_MLPLINEARPARALLELOUT_SMOOTH,
    IINTERMIDATE_SELFOUTLINEARMULSMOOTHOUT_SMOOTH,
    IINTERMIDATE_MLPDOWNMULSMOOTHOUT_SMOOTH,
};

struct LlamaBlockAttnSmoothParallelParam {
    float rmsNormEps = 0;
    int headNum = 0;
    int headDim = 0;
    int rank = 0;
    int rankSize = 1;
    float qkScale = 1.0;
    int rotaryCoeff = 2;
    bool transpose = true;
    void *hcclComm = nullptr; // only effect when hcclComm is not null
    bool isPrefill = false;
    
    // new params for quantization: 
    float selfNormQuantInputScale = 0;
    int selfNormQuantInputOffset = 0;
    float selfOutLinearParallelQuantScale = 0;
    int selfOutLinearParallelQuantOffset = 0;
    float inputNormQuantInputScale = 0;
    int inputNormQuantInputOffset = 0;
    float mlpLinearParallelQuantScale = 0;
    int mlpLinearParallelQuantOffset = 0;
};

atb::Status LlamaBlockAttnSmoothParallelOperation(const LlamaBlockAttnSmoothParallelParam &param,
                                                        atb::Operation **operation);
