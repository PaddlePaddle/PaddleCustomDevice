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
#include "set_stop_value_multi_ends_v2_tiling.h"
#include "register/op_def_registry.h"

namespace {
    constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
}
namespace optiling {
    static ge::graphStatus SetStopValueMultiEndsV2TilingFunc(gert::TilingContext* context)
    {
        SetStopValueMultiEndsV2TilingData tiling;
        const gert::StorageShape* topkIdsShape = context->GetInputShape(0);
        const gert::StorageShape* endIdsShape = context->GetInputShape(3);

        int32_t bs = topkIdsShape->GetStorageShape().GetDim(0);
        int32_t length = endIdsShape->GetStorageShape().GetDim(0);

        tiling.set_bs(bs);
        tiling.set_length(length);

        int32_t blockSize = 1;
        context->SetBlockDim(blockSize);

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
    static ge::graphStatus SetStopValueMultiEndsV2InferShape(gert::InferShapeContext* context)
    {
        const gert::Shape* x0_shape = context->GetInputShape(0);
        const gert::Shape* x1_shape = context->GetInputShape(1);
        const gert::Shape* x4_shape = context->GetInputShape(4);

        gert::Shape* y0_shape = context->GetOutputShape(0);
        gert::Shape* y1_shape = context->GetOutputShape(1);
        gert::Shape* y2_shape = context->GetOutputShape(2);

        *y0_shape = *x0_shape;
        *y1_shape = *x1_shape;
        *y2_shape = *x4_shape;

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus SetStopValueMultiEndsV2InferShapeRange(gert::InferShapeRangeContext *context)
    {
        const gert::Range<gert::Shape> *inputX0ShapeRange = context->GetInputShapeRange(0);
        const gert::Range<gert::Shape> *inputX1ShapeRange = context->GetInputShapeRange(1);
        const gert::Range<gert::Shape> *inputX4ShapeRange = context->GetInputShapeRange(4);
        
        gert::Range<gert::Shape> *y0ShapeRange = context->GetOutputShapeRange(0);
        gert::Range<gert::Shape> *y1ShapeRange = context->GetOutputShapeRange(1);
        gert::Range<gert::Shape> *y2ShapeRange = context->GetOutputShapeRange(2);

        *y0ShapeRange = *inputX0ShapeRange;
        *y1ShapeRange = *inputX1ShapeRange;
        *y2ShapeRange = *inputX4ShapeRange;

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus SetStopValueMultiEndsV2InferDataType(gert::InferDataTypeContext *context)
    {
        const ge::DataType x0DataType = context->GetInputDataType(0);
        const ge::DataType x1DataType = context->GetInputDataType(1);
        const ge::DataType x4DataType = context->GetInputDataType(4);

        context->SetOutputDataType(0, x0DataType);
        context->SetOutputDataType(1, x1DataType);
        context->SetOutputDataType(2, x4DataType);

        return GRAPH_SUCCESS;
    }
}

namespace ops {
class SetStopValueMultiEndsV2 : public OpDef {
public:
    explicit SetStopValueMultiEndsV2(const char* name) : OpDef(name)
    {
        this->Input("topkIds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("stopFlags")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("seqLens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT32})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("endIds")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Input("nextTokens")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("topkIdsOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("stopFlagsOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->Output("nextTokensOut")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::SetStopValueMultiEndsV2InferShape)
            .SetInferShapeRange(ge::SetStopValueMultiEndsV2InferShapeRange)
            .SetInferDataType(ge::SetStopValueMultiEndsV2InferDataType);;

        this->AICore()
            .SetTiling(optiling::SetStopValueMultiEndsV2TilingFunc);

        this->AICore().AddConfig("ascend310p");
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SetStopValueMultiEndsV2);
}