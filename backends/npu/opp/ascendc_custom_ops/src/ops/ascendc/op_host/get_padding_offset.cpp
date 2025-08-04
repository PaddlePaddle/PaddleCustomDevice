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
#include "get_padding_offset_tiling.h"
#include "register/op_def_registry.h"
namespace {
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
}

namespace optiling {
static ge::graphStatus GetPaddingOffsetTilingFunc(gert::TilingContext *context)
{
    GetPaddingOffsetTilingData tiling;

    const gert::StorageShape *input_data_shape = context->GetInputShape(0);

    int32_t batch = input_data_shape->GetStorageShape().GetDim(0);
    int32_t padLength = input_data_shape->GetStorageShape().GetDim(1);

    tiling.set_batch(batch);
    tiling.set_padLength(padLength);

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
static ge::graphStatus GetPaddingOffsetInferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x0_shape = context->GetInputShape(0);
    const gert::Shape *x1_shape = context->GetInputShape(1);
    gert::Shape *y0_shape = context->GetOutputShape(0);
    gert::Shape *y1_shape = context->GetOutputShape(1);
    gert::Shape *y2_shape = context->GetOutputShape(2);
    gert::Shape *y3_shape = context->GetOutputShape(3);
    gert::Shape *y4_shape = context->GetOutputShape(4);

    y0_shape -> SetDimNum(1);
    y0_shape -> SetDim(0, x0_shape -> GetDim(0) * x0_shape -> GetDim(1));

    y1_shape -> SetDimNum(1);
    y1_shape -> SetDim(0, x1_shape -> GetDim(0));

    y2_shape -> SetDimNum(1);
    y2_shape -> SetDim(0, x0_shape -> GetDim(0) * x0_shape -> GetDim(1));

    y3_shape -> SetDimNum(1);
    y3_shape -> SetDim(0, x1_shape -> GetDim(0) + 1);

    y4_shape -> SetDimNum(1);
    y4_shape -> SetDim(0, x1_shape -> GetDim(0) + 1);
    return GRAPH_SUCCESS;
}

ge::graphStatus GetPaddingOffsetInferShapeRange(gert::InferShapeRangeContext *context)
{
    const gert::Range<gert::Shape> *inputXShapeRange = context->GetInputShapeRange(2);
    gert::Range<gert::Shape> *outputShapeRange0 = context->GetOutputShapeRange(0);
    gert::Range<gert::Shape> *outputShapeRange1 = context->GetOutputShapeRange(1);
    gert::Range<gert::Shape> *outputShapeRange2 = context->GetOutputShapeRange(2);
    gert::Range<gert::Shape> *outputShapeRange3 = context->GetOutputShapeRange(3);
    gert::Range<gert::Shape> *outputShapeRange4 = context->GetOutputShapeRange(4);
    *outputShapeRange0 = *inputXShapeRange;
    *outputShapeRange1 = *inputXShapeRange;
    *outputShapeRange2 = *inputXShapeRange;
    *outputShapeRange3 = *inputXShapeRange;
    *outputShapeRange4 = *inputXShapeRange;

    return GRAPH_SUCCESS;
}

ge::graphStatus GetPaddingOffsetInferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType x0DataType = context->GetInputDataType(0);
    const ge::DataType x1DataType = context->GetInputDataType(1);

    context->SetOutputDataType(0, x0DataType);
    context->SetOutputDataType(1, x1DataType);
    context->SetOutputDataType(2, x1DataType);
    context->SetOutputDataType(3, x1DataType);
    context->SetOutputDataType(4, x1DataType);

    return GRAPH_SUCCESS;
}
}

namespace ops {
class GetPaddingOffset : public OpDef {
public:
    GetPaddingOffset(const char *name) : OpDef(name)
    {
        this->Input("input_ids")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT64 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("cum_offsets")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32  })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("token_num")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT64 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("seq_len")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("x_remove_padding")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT64 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("cum_offsets_out")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("padding_offset")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("cu_seqlens_q")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("cu_seqlens_k")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });

        this->SetInferShape(ge::GetPaddingOffsetInferShape);

        this->AICore().SetTiling(optiling::GetPaddingOffsetTilingFunc);

        this->AICore().AddConfig("ascend310p");
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(GetPaddingOffset);
}