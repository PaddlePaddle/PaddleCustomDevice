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
#include "update_inputs_tiling.h"
#include "register/op_def_registry.h"
#include <iostream>

using namespace std;

namespace {
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
}

namespace optiling {
    static ge::graphStatus UpdateInputsTilingFunc(gert::TilingContext *context)
    {
        UpdateInputsTilingData tiling;
        const gert::StorageShape* stopFlagsShape = context->GetInputShape(0);
        const gert::StorageShape* seqLensThisTimeShape = context->GetInputShape(2);
        const gert::StorageShape* inputIdsShape = context->GetInputShape(5);
        int max_bs = stopFlagsShape->GetStorageShape().GetDim(0);
        int bs = seqLensThisTimeShape->GetStorageShape().GetDim(0);
        int length = inputIdsShape->GetStorageShape().GetDim(1);

        tiling.set_bs(bs);
        tiling.set_max_bs(max_bs);
        tiling.set_length(length);

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
    static ge::graphStatus UpdateInputsInferShape(gert::InferShapeContext *context)
    {
        
        const gert::Shape* x0Shape = context->GetInputShape(1);
        const gert::Shape* x1Shape = context->GetInputShape(2);
        const gert::Shape* x2Shape = context->GetInputShape(3);
        const gert::Shape* x3Shape = context->GetInputShape(4);
        const gert::Shape* x4Shape = context->GetInputShape(5);
        
        gert::Shape* y0Shape = context->GetOutputShape(0);
        
        gert::Shape* y1Shape = context->GetOutputShape(1);
        
        gert::Shape* y2Shape = context->GetOutputShape(2);
        
        gert::Shape* y3Shape = context->GetOutputShape(3);
        
        gert::Shape* y4Shape = context->GetOutputShape(4);

        *y0Shape = *x0Shape;
        
        *y1Shape = *x1Shape;
        
        *y2Shape = *x2Shape;
        
        *y3Shape = *x3Shape;
        
        *y4Shape = *x4Shape;

        
        return GRAPH_SUCCESS;
    }

    static ge::graphStatus UpdateInputsInferShapeRange(gert::InferShapeRangeContext *context)
    {
        const gert::Range<gert::Shape> *inputX0ShapeRange = context->GetInputShapeRange(1);
        const gert::Range<gert::Shape> *inputX1ShapeRange = context->GetInputShapeRange(2);
        const gert::Range<gert::Shape> *inputX2ShapeRange = context->GetInputShapeRange(3);
        const gert::Range<gert::Shape> *inputX3ShapeRange = context->GetInputShapeRange(4);
        const gert::Range<gert::Shape> *inputX4ShapeRange = context->GetInputShapeRange(5);

        gert::Range<gert::Shape> *y0ShapeRange = context->GetOutputShapeRange(0);
        gert::Range<gert::Shape> *y1ShapeRange = context->GetOutputShapeRange(1);
        gert::Range<gert::Shape> *y2ShapeRange = context->GetOutputShapeRange(2);
        gert::Range<gert::Shape> *y3ShapeRange = context->GetOutputShapeRange(3);
        gert::Range<gert::Shape> *y4ShapeRange = context->GetOutputShapeRange(4);

        

        *y0ShapeRange = *inputX0ShapeRange;
        *y1ShapeRange = *inputX1ShapeRange;
        *y2ShapeRange = *inputX2ShapeRange;
        *y3ShapeRange = *inputX3ShapeRange;
        *y4ShapeRange = *inputX4ShapeRange;
        
        return GRAPH_SUCCESS;
    }

    static ge::graphStatus UpdateInputsInferDataType(gert::InferDataTypeContext *context)
    {
        
        const ge::DataType x0DataType = context->GetInputDataType(1);
        const ge::DataType x1DataType = context->GetInputDataType(2);
        const ge::DataType x2DataType = context->GetInputDataType(3);
        const ge::DataType x3DataType = context->GetInputDataType(4);
        const ge::DataType x4DataType = context->GetInputDataType(5);


        context->SetOutputDataType(0, x0DataType);
        context->SetOutputDataType(1, x1DataType);
        context->SetOutputDataType(2, x2DataType);
        context->SetOutputDataType(3, x3DataType);
        context->SetOutputDataType(4, x4DataType);
        
        return GRAPH_SUCCESS;
    }
}

namespace ops {
    class UpdateInputs : public OpDef {
    public:
        explicit UpdateInputs(const char *name) : OpDef(name)
        {       
            this->Input("stop_flags")
                .ParamType(REQUIRED)
                .DataType({ge::DT_BOOL})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("not_need_stop")
                .ParamType(REQUIRED)
                .DataType({ge::DT_BOOL})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("seq_lens_this_time")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
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
            
            this->Input("input_ids")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("stop_nums")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("next_tokens")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("is_block_step")
                .ParamType(REQUIRED)
                .DataType({ge::DT_BOOL})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Output("not_need_stop_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_BOOL})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Output("seq_lens_this_time_out")
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
            
            this->Output("input_ids_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->SetInferShape(ge::UpdateInputsInferShape)
                .SetInferShapeRange(ge::UpdateInputsInferShapeRange)
                .SetInferDataType(ge::UpdateInputsInferDataType);

            this->AICore()
                .SetTiling(optiling::UpdateInputsTilingFunc);

            this->AICore().AddConfig("ascend310p");
            this->AICore().AddConfig("ascend910");
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(UpdateInputs);
}