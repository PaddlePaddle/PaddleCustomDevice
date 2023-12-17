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

enum LlamaBlockAttnParallelTensorId {
    IN_HIDDENSTATES = 0,
    IN_NORMWEIGHT,
    IN_INPUTNORM_BETA,
    IN_SELFNORM_BETA,
    IN_QKVMIXDWEIGHT,
    IN_QKVDEQBIAS,
    IN_QKVDEQSCALE,
    IN_QKVDEQBLANKBIAS,
    IN_SELFOUTLINEARWEIGHT,
    IN_SELFOUTLINEARDEQBIAS,
    IN_SELFOUTLINEARDEQSCALE,
    IN_SELFOUTLINEARDEQBLANKBIAS,
    IN_SELFOUTNORMWEIGHT,
    IN_MLPGATEUPWEIGHT,
    IN_MLPDEQBIAS,
    IN_MLPDEQSCALE,
    IN_MLPDEQBLANKBIAS,
    IN_MLPDOWNWEIGHT,
    IN_MLPDOWNDEQBIAS,
    IN_MLPDOWNDEQSCALE,
    IN_MLPDOWNDEQBLANKBIAS,
    IN_COS_TABLE,
    IN_SIN_TABLE,
    IN_ATTENTIONMASK,
    IN_CACHE_K,
    IN_CACHE_V,
    IN_SEQLEN,
    IN_BLOCK_TABLES,
    IN_SLOTS,
    OUT_LLAMALAYEROUT,
    INTERMIDATE_INPUTNORMOUT,
    INTERMIDATE_INPUTNORMQUANTOUT,
    INTERMIDATE_INPUTNORMQUANTADDOUT,
    INTERMIDATE_MIXEDQ,
    INTERMIDATE_MIXEDK,
    INTERMIDATE_MIXEDV,
    INTERMIDATE_POSITIONEMBEDQ,
    INTERMIDATE_POSITIONEMBEDK,
    INTERMIDATE_SELFOUT,
    INTERMIDATE_SELFOUT_QUANT,
    INTERMIDATE_SELFLINEAROUT,
    INTERMIDATE_SELFRESIDUALADDOUT,
    INTERMIDATE_SELFNORMOUT,
    INTERMIDATE_SELFNORMQUANTOUT,
    INTERMIDATE_SELFNORMQUANTADDOUT,
    INTERMIDATE_MLPOUT,
    INTERMIDATE_MLPOUT_QUANT,
    INTERMIDATE_MLPLINEARPARALLELOUT,
};

struct LlamaBlockAttnParallelParam {
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
    float selfNormQuantInputOffset = 0;
    float selfOutLinearParallelQuantScale = 0;
    float selfOutLinearParallelQuantOffset = 0;
    float inputNormQuantInputScale = 0;
    float inputNormQuantInputOffset = 0;
    float mlpLinearParallelQuantScale = 0;
    float mlpLinearParallelQuantOffset = 0;
};

atb::Status LlamaBlockAttnParallelOperation(const LlamaBlockAttnParallelParam &param,
                                                        atb::Operation **operation);
