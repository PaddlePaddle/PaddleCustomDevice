/**
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
#include "token_penalty_multi_scores_tiling.h"

#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

using namespace std;

namespace {
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
}

namespace optiling {
    static ge::graphStatus TokenPenaltyMultiScoresTilingFunc(gert::TilingContext *context)
    {
        TokenPenaltyMultiScoresTilingData tiling;
        const gert::StorageShape* preIdsShape = context->GetInputShape(0);
        const gert::StorageShape* logitsShape = context->GetInputShape(1);
        const gert::StorageShape* endTokenIdShape = context->GetInputShape(8);

        int bs = logitsShape->GetStorageShape().GetDim(0);
        int vs = logitsShape->GetStorageShape().GetDim(1);
        int seqLen = preIdsShape->GetStorageShape().GetDim(1);
        int etiLength = endTokenIdShape->GetStorageShape().GetDim(0);

        auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
        int coreNum = ascendcPlatform.GetCoreNumAiv();
        int coreLoop = (bs + coreNum - 1) / coreNum;
        int blockDim = (bs + coreLoop - 1) / coreLoop;

        int blockElements = 256 / sizeof(float); // 256 bytes for select
        if (vs % blockElements != 0) {
            std::cout << "[ERROR] TokenPenaltyMultiScores voc_size " << vs << " is invalid" << std::endl;
            return ge::GRAPH_FAILED;
        }
        int vsBlockNum = vs / blockElements;
        int vsBlockBase = blockElements;
        int vsBlock = vsBlockBase;
        for (int i = 2; i < vsBlockNum; i++) {
            if (vsBlockNum % i == 0) {
                int vsBlockTmp = vsBlockBase * i;
                if (vsBlockTmp > 2048) { // 2048 is max vsBlock
                    break;
                }
                vsBlock = vsBlockTmp;
            }
        }

        std::cout << "[INFO] TokenPenaltyMultiScores tiling result, vs " << vs
            << ", vsBlock " << vsBlock << ", seqLen " << seqLen << ", etil " << etiLength
            << ", bs " << bs << ", bsBlock " << coreLoop << std::endl;

        tiling.set_vs(vs);
        tiling.set_vsBlock(vsBlock);
        tiling.set_seqLen(seqLen);
        tiling.set_etil(etiLength);
        tiling.set_bs(bs);
        tiling.set_bsBlock(coreLoop);

        context->SetBlockDim(blockDim);

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (context->GetWorkspaceNum() <= 0) {
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = MINIMAL_WORKSPACE;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static ge::graphStatus TokenPenaltyMultiScoresInferShape(gert::InferShapeContext *context)
    {
        const gert::Shape* x3Shape = context->GetInputShape(1);
        
        gert::Shape* y0Shape = context->GetOutputShape(0);

        *y0Shape = *x3Shape;

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus TokenPenaltyMultiScoresInferShapeRange(gert::InferShapeRangeContext *context)
    {
        const gert::Range<gert::Shape> *inputXShapeRange = context->GetInputShapeRange(1);
        gert::Range<gert::Shape> *y0ShapeRange = context->GetOutputShapeRange(0);

        *y0ShapeRange = *inputXShapeRange;

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus TokenPenaltyMultiScoresInferDataType(gert::InferDataTypeContext *context)
    {
        const ge::DataType x2DataType = context->GetInputDataType(1);

        context->SetOutputDataType(0, x2DataType);

        return GRAPH_SUCCESS;
    }
}

namespace ops {
    class TokenPenaltyMultiScores : public OpDef {
    public:
        explicit TokenPenaltyMultiScores(const char *name) : OpDef(name)
        {            
            this->Input("preIds")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("logits")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("repeatTimes")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("penaltyScores")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("frequencyScores")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("presenceScores")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("curLen")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("minLen")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("eosTokenId")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Output("logitsOut")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->SetInferShape(ge::TokenPenaltyMultiScoresInferShape)
                .SetInferShapeRange(ge::TokenPenaltyMultiScoresInferShapeRange)
                .SetInferDataType(ge::TokenPenaltyMultiScoresInferDataType);

            this->AICore()
                .SetTiling(optiling::TokenPenaltyMultiScoresTilingFunc);

            this->AICore().AddConfig("ascend310p");
            this->AICore().AddConfig("ascend910");
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(TokenPenaltyMultiScores);
}
