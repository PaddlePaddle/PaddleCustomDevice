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
#include "set_value_by_flags_and_idx_v2_tiling.h"
#include "register/op_def_registry.h"

using namespace std;

namespace {
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
}

namespace optiling {
    static ge::graphStatus SetValueByFlagsAndIdxV2TilingFunc(gert::TilingContext *context)
    {
        SetValueByFlagsAndIdxV2TilingData tiling;
        const gert::StorageShape* seqLensThisTimeShape = context->GetInputShape(2);
        const gert::StorageShape* preIdsAllShape = context->GetInputShape(0);
        const gert::StorageShape* inputIdsShape = context->GetInputShape(1);

        int bs = seqLensThisTimeShape->GetStorageShape().GetDim(0);
        int length = preIdsAllShape->GetStorageShape().GetDim(1);
        int lengthInput = inputIdsShape->GetStorageShape().GetDim(1);

        tiling.set_bs(bs);
        tiling.set_length(length);
        tiling.set_lengthInput(lengthInput);

        int32_t blockDims = 1;
        context->SetBlockDim(blockDims);

        tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
        context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());

        size_t *currentWorkspace = context->GetWorkspaceSizes(1);
        if (context->GetWorkspaceNum() <= 0) {
            std::cout << "GetWorkspaceNum Failed" << std::endl;
            return ge::GRAPH_FAILED;
        }
        currentWorkspace[0] = MINIMAL_WORKSPACE;
        return ge::GRAPH_SUCCESS;
    }
}

namespace ge {
    static ge::graphStatus SetValueByFlagsAndIdxV2InferShape(gert::InferShapeContext *context)
    {
        const gert::Shape* x0Shape = context->GetInputShape(0);
        
        gert::Shape* y0Shape = context->GetOutputShape(0);

        *y0Shape = *x0Shape;

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus SetValueByFlagsAndIdxV2InferShapeRange(gert::InferShapeRangeContext *context)
    {
        const gert::Range<gert::Shape> *inputXShapeRange = context->GetInputShapeRange(0);
        
        gert::Range<gert::Shape> *y0ShapeRange = context->GetOutputShapeRange(0);

        *y0ShapeRange = *inputXShapeRange;

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus SetValueByFlagsAndIdxV2InferDataType(gert::InferDataTypeContext *context)
    {
        const ge::DataType x0DataType = context->GetInputDataType(0);

        context->SetOutputDataType(0, x0DataType);

        return GRAPH_SUCCESS;
    }
}

namespace ops {
    class SetValueByFlagsAndIdxV2 : public OpDef {
    public:
        explicit SetValueByFlagsAndIdxV2(const char *name) : OpDef(name)
        {            
            this->Input("preIdsAll")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("inputIds")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("seqLensThisTime")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("seqLensEncoder")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("seqLensDecoder")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("stepIdx")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("stopFlags")
                .ParamType(REQUIRED)
                .DataType({ge::DT_BOOL})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Output("preIdsAllOut")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->SetInferShape(ge::SetValueByFlagsAndIdxV2InferShape)
                .SetInferShapeRange(ge::SetValueByFlagsAndIdxV2InferShapeRange)
                .SetInferDataType(ge::SetValueByFlagsAndIdxV2InferDataType);

            this->AICore()
                .SetTiling(optiling::SetValueByFlagsAndIdxV2TilingFunc);

            this->AICore().AddConfig("ascend310p");
            this->AICore().AddConfig("ascend910");
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(SetValueByFlagsAndIdxV2);
}