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
#include "rebuild_padding_tiling.h"
#include "register/op_def_registry.h"

using namespace std;

namespace {
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
}

namespace optiling {
    static ge::graphStatus RebuildPaddingTilingFunc(gert::TilingContext *context)
    {
        RebuildPaddingTilingData tiling;
        const gert::StorageShape* tmpOut = context->GetInputShape(0);
        const gert::StorageShape* cumOffsetsShape = context->GetInputShape(1);
        
        int bs = cumOffsetsShape->GetStorageShape().GetDim(0);
        int token_num = tmpOut->GetStorageShape().GetDim(0);
        int dim_embed = tmpOut->GetStorageShape().GetDim(1);
        
        auto attrs = context->GetAttrs();
        auto max_input_length_ptr = attrs->GetAttrPointer<int32_t>(0);
        auto max_input_length = *max_input_length_ptr;

        tiling.set_bs(bs);
        tiling.set_token_num(token_num);
        tiling.set_dim_embed(dim_embed);
        tiling.set_max_input_length(max_input_length);

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
    static ge::graphStatus RebuildPaddingInferShape(gert::InferShapeContext *context)
    {
        const gert::Shape* x0Shape = context->GetInputShape(0);
        // const gert::Shape* x1Shape = context->GetInputShape(1);
        const gert::Shape* x3Shape = context->GetInputShape(3);
        int x0num = x0Shape->GetDimNum();
        
        gert::Shape* y0Shape = context->GetOutputShape(0);

        y0Shape->SetDimNum(2);
        // y0Shape->SetDim(0,x1Shape->GetDim(0));
        y0Shape->SetDim(0,x3Shape->GetDim(0));
        y0Shape->SetDim(1,x0Shape->GetDim(x0num-1));

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus RebuildPaddingInferShapeRange(gert::InferShapeRangeContext *context)
    {
        gert::Range<gert::Shape> *y0ShapeRange = context->GetOutputShapeRange(0);
        gert::Shape min = gert::Shape({1});
        gert::Shape max = gert::Shape({-1});
        y0ShapeRange->SetMin(&min);
        y0ShapeRange->SetMax(&max);

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus RebuildPaddingInferDataType(gert::InferDataTypeContext *context)
    {
        const ge::DataType x0DataType = context->GetInputDataType(0);

        context->SetOutputDataType(0, x0DataType);

        return GRAPH_SUCCESS;
    }
}

namespace ops {
    class RebuildPadding : public OpDef {
    public:
        explicit RebuildPadding(const char *name) : OpDef(name)
        {            
            this->Input("tmpOut")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("cumOffsets")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("seqLensDecoder")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("seqLensEncoder")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Output("out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_FLOAT16})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->SetInferShape(ge::RebuildPaddingInferShape)
                .SetInferShapeRange(ge::RebuildPaddingInferShapeRange)
                .SetInferDataType(ge::RebuildPaddingInferDataType);

            this->Attr("max_input_length").AttrType(REQUIRED).Int();

            this->AICore()
                .SetTiling(optiling::RebuildPaddingTilingFunc);

            this->AICore().AddConfig("ascend310p");
            this->AICore().AddConfig("ascend910");
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(RebuildPadding);
}