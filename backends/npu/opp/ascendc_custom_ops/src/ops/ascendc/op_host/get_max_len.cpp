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
#include "get_max_len_tiling.h"
#include "register/op_def_registry.h"

using namespace std;

namespace {
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
}

namespace optiling {
    static ge::graphStatus GetMaxLenTilingFunc(gert::TilingContext *context)
    {
        GetMaxLenTilingData tiling;
        const gert::StorageShape* seqLensEncoder = context->GetInputShape(0);

        int bs = seqLensEncoder->GetStorageShape().GetDim(0);
        tiling.set_bs(bs);

        int32_t blockDims = 1;
        context->SetBlockDim(blockDims);

        std::cout << "GetMaxLenTilingFunc bs " << bs << std::endl;

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
    static ge::graphStatus GetMaxLenInferShape(gert::InferShapeContext *context)
    {
        const gert::Shape* x0Shape = context->GetInputShape(0);
        const gert::Shape* x1Shape = context->GetInputShape(1);
        
        gert::Shape* y0Shape = context->GetOutputShape(0);
        gert::Shape* y1Shape = context->GetOutputShape(1);
        
        y0Shape->SetDimNum(1);
        y0Shape->SetDim(0, 8);
        y1Shape->SetDimNum(1);
        y1Shape->SetDim(0, 8);

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus GetMaxLenInferShapeRange(gert::InferShapeRangeContext *context)
    {
        const gert::Range<gert::Shape> *inputX0ShapeRange = context->GetInputShapeRange(0);
        const gert::Range<gert::Shape> *inputX1ShapeRange = context->GetInputShapeRange(1);
        gert::Range<gert::Shape> *y0ShapeRange = context->GetOutputShapeRange(0);
        gert::Range<gert::Shape> *y1ShapeRange = context->GetOutputShapeRange(1);

        *y0ShapeRange = *inputX0ShapeRange;
        *y1ShapeRange = *inputX1ShapeRange;

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus GetMaxLenInferDataType(gert::InferDataTypeContext *context)
    {
        const ge::DataType x0DataType = context->GetInputDataType(0);

        context->SetOutputDataType(0, x0DataType);
        context->SetOutputDataType(1, x0DataType);

        return GRAPH_SUCCESS;
    }
}

namespace ops {
    class GetMaxLen : public OpDef {
    public:
        explicit GetMaxLen(const char *name) : OpDef(name)
        {            
            this->Input("seq_lens_encoder")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("seq_lens_decoder")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Output("seq_lens_encoder_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Output("seq_lens_decoder_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->SetInferShape(ge::GetMaxLenInferShape)
                .SetInferShapeRange(ge::GetMaxLenInferShapeRange)
                .SetInferDataType(ge::GetMaxLenInferDataType);

            this->AICore()
                .SetTiling(optiling::GetMaxLenTilingFunc);

            this->AICore().AddConfig("ascend310p");
            this->AICore().AddConfig("ascend910");
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(GetMaxLen);
}
