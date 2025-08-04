#include "step_paddle_tiling.h"
#include "register/op_def_registry.h"

using namespace std;

namespace {
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
}

namespace optiling {
    static ge::graphStatus StepPaddleTilingFunc(gert::TilingContext *context)
    {
        GetStepPaddleTilingData tiling;
        const gert::StorageShape* seqLensThisTimeShape = context->GetInputShape(1);
        const gert::StorageShape* blockTablesShape = context->GetInputShape(5);
        const gert::StorageShape* inputIdsShape = context->GetInputShape(17);
        const gert::StorageShape* preIdsShape = context->GetInputShape(18);

        int bsz = seqLensThisTimeShape->GetStorageShape().GetDim(0);
        int block_num_per_seq = blockTablesShape->GetStorageShape().GetDim(1);
        int length = inputIdsShape->GetStorageShape().GetDim(1);
        int pre_id_length = preIdsShape->GetStorageShape().GetDim(1);

        auto attrs = context->GetAttrs();
        auto block_size_ptr = attrs->GetAttrPointer<int32_t>(0);
        int block_size = *block_size_ptr;
        auto encoder_decoder_block_num_ptr = attrs->GetAttrPointer<int32_t>(1);
        int encoder_decoder_block_num = *encoder_decoder_block_num_ptr;
        int max_decoder_block_num = pre_id_length / block_size - encoder_decoder_block_num;
        auto first_token_id_ptr = attrs->GetAttrPointer<int32_t>(2);
        auto first_token_id = *first_token_id_ptr;

        tiling.set_bsz(bsz);
        tiling.set_block_size(block_size);
        tiling.set_block_num_per_seq(block_num_per_seq);
        tiling.set_max_decoder_block_num(max_decoder_block_num);
        tiling.set_length(length);
        tiling.set_pre_id_length(pre_id_length);
        tiling.set_first_token_id(first_token_id);

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
    static ge::graphStatus StepPaddleInferShape(gert::InferShapeContext *context)
    {
        const gert::Shape* x0Shape = context->GetInputShape(0);
        const gert::Shape* x1Shape = context->GetInputShape(1);
        const gert::Shape* x3Shape = context->GetInputShape(3);
        const gert::Shape* x4Shape = context->GetInputShape(4);
        const gert::Shape* x5Shape = context->GetInputShape(5);
        const gert::Shape* x6Shape = context->GetInputShape(6);
        const gert::Shape* x7Shape = context->GetInputShape(7);
        const gert::Shape* x8Shape = context->GetInputShape(8);
        const gert::Shape* x9Shape = context->GetInputShape(9);
        const gert::Shape* x10Shape = context->GetInputShape(10);
        const gert::Shape* x11Shape = context->GetInputShape(11);
        const gert::Shape* x12Shape = context->GetInputShape(12);
        const gert::Shape* x13Shape = context->GetInputShape(13);
        const gert::Shape* x14Shape = context->GetInputShape(14);
        const gert::Shape* x15Shape = context->GetInputShape(15);
        const gert::Shape* x16Shape = context->GetInputShape(16);
        const gert::Shape* x17Shape = context->GetInputShape(17);
        
        gert::Shape* y0Shape = context->GetOutputShape(0);
        gert::Shape* y1Shape = context->GetOutputShape(1);
        gert::Shape* y2Shape = context->GetOutputShape(2);
        gert::Shape* y3Shape = context->GetOutputShape(3);
        gert::Shape* y4Shape = context->GetOutputShape(4);
        gert::Shape* y5Shape = context->GetOutputShape(5);
        gert::Shape* y6Shape = context->GetOutputShape(6);
        gert::Shape* y7Shape = context->GetOutputShape(7);
        gert::Shape* y8Shape = context->GetOutputShape(8);
        gert::Shape* y9Shape = context->GetOutputShape(9);
        gert::Shape* y10Shape = context->GetOutputShape(10);
        gert::Shape* y11Shape = context->GetOutputShape(11);
        gert::Shape* y12Shape = context->GetOutputShape(12);
        gert::Shape* y13Shape = context->GetOutputShape(13);
        gert::Shape* y14Shape = context->GetOutputShape(14);
        gert::Shape* y15Shape = context->GetOutputShape(15);
        gert::Shape* y16Shape = context->GetOutputShape(16);

        *y0Shape = *x0Shape;
        *y1Shape = *x1Shape;
        *y2Shape = *x3Shape;
        *y3Shape = *x4Shape;
        *y4Shape = *x5Shape;
        *y5Shape = *x6Shape;
        *y6Shape = *x7Shape;
        *y7Shape = *x8Shape;
        *y8Shape = *x9Shape;
        *y9Shape = *x10Shape;
        *y10Shape = *x11Shape;
        *y11Shape = *x12Shape;
        *y12Shape = *x13Shape;
        *y13Shape = *x14Shape;
        *y14Shape = *x15Shape;
        *y15Shape = *x16Shape;
        *y16Shape = *x17Shape;

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus StepPaddleInferShapeRange(gert::InferShapeRangeContext *context)
    {
        const gert::Range<gert::Shape> *inputX0ShapeRange = context->GetInputShapeRange(0);
        const gert::Range<gert::Shape> *inputX1ShapeRange = context->GetInputShapeRange(1);
        const gert::Range<gert::Shape> *inputX3ShapeRange = context->GetInputShapeRange(3);
        const gert::Range<gert::Shape> *inputX4ShapeRange = context->GetInputShapeRange(4);
        const gert::Range<gert::Shape> *inputX5ShapeRange = context->GetInputShapeRange(5);
        const gert::Range<gert::Shape> *inputX6ShapeRange = context->GetInputShapeRange(6);
        const gert::Range<gert::Shape> *inputX7ShapeRange = context->GetInputShapeRange(7);
        const gert::Range<gert::Shape> *inputX8ShapeRange = context->GetInputShapeRange(8);
        const gert::Range<gert::Shape> *inputX9ShapeRange = context->GetInputShapeRange(9);
        const gert::Range<gert::Shape> *inputX10ShapeRange = context->GetInputShapeRange(10);
        const gert::Range<gert::Shape> *inputX11ShapeRange = context->GetInputShapeRange(11);
        const gert::Range<gert::Shape> *inputX12ShapeRange = context->GetInputShapeRange(12);
        const gert::Range<gert::Shape> *inputX13ShapeRange = context->GetInputShapeRange(13);
        const gert::Range<gert::Shape> *inputX14ShapeRange = context->GetInputShapeRange(14);
        const gert::Range<gert::Shape> *inputX15ShapeRange = context->GetInputShapeRange(15);
        const gert::Range<gert::Shape> *inputX16ShapeRange = context->GetInputShapeRange(16);
        const gert::Range<gert::Shape> *inputX17ShapeRange = context->GetInputShapeRange(17);

        gert::Range<gert::Shape> *y0ShapeRange = context->GetOutputShapeRange(0);
        gert::Range<gert::Shape> *y1ShapeRange = context->GetOutputShapeRange(1);
        gert::Range<gert::Shape> *y2ShapeRange = context->GetOutputShapeRange(2);
        gert::Range<gert::Shape> *y3ShapeRange = context->GetOutputShapeRange(3);
        gert::Range<gert::Shape> *y4ShapeRange = context->GetOutputShapeRange(4);
        gert::Range<gert::Shape> *y5ShapeRange = context->GetOutputShapeRange(5);
        gert::Range<gert::Shape> *y6ShapeRange = context->GetOutputShapeRange(6);
        gert::Range<gert::Shape> *y7ShapeRange = context->GetOutputShapeRange(7);
        gert::Range<gert::Shape> *y8ShapeRange = context->GetOutputShapeRange(8);
        gert::Range<gert::Shape> *y9ShapeRange = context->GetOutputShapeRange(9);
        gert::Range<gert::Shape> *y10ShapeRange = context->GetOutputShapeRange(10);
        gert::Range<gert::Shape> *y11ShapeRange = context->GetOutputShapeRange(11);
        gert::Range<gert::Shape> *y12ShapeRange = context->GetOutputShapeRange(12);
        gert::Range<gert::Shape> *y13ShapeRange = context->GetOutputShapeRange(13);
        gert::Range<gert::Shape> *y14ShapeRange = context->GetOutputShapeRange(14);
        gert::Range<gert::Shape> *y15ShapeRange = context->GetOutputShapeRange(15);
        gert::Range<gert::Shape> *y16ShapeRange = context->GetOutputShapeRange(16);

        *y0ShapeRange = *inputX0ShapeRange;
        *y1ShapeRange = *inputX1ShapeRange;
        *y2ShapeRange = *inputX3ShapeRange;
        *y3ShapeRange = *inputX4ShapeRange;
        *y4ShapeRange = *inputX5ShapeRange;
        *y5ShapeRange = *inputX6ShapeRange;
        *y6ShapeRange = *inputX7ShapeRange;
        *y7ShapeRange = *inputX8ShapeRange;
        *y8ShapeRange = *inputX9ShapeRange;
        *y9ShapeRange = *inputX10ShapeRange;
        *y10ShapeRange = *inputX11ShapeRange;
        *y11ShapeRange = *inputX12ShapeRange;
        *y12ShapeRange = *inputX13ShapeRange;
        *y13ShapeRange = *inputX14ShapeRange;
        *y14ShapeRange = *inputX15ShapeRange;
        *y15ShapeRange = *inputX16ShapeRange;
        *y16ShapeRange = *inputX17ShapeRange;

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus StepPaddleInferDataType(gert::InferDataTypeContext *context)
    {
        const ge::DataType x0DataType = context->GetInputDataType(0);
        const ge::DataType x1DataType = context->GetInputDataType(1);
        const ge::DataType x3DataType = context->GetInputDataType(3);
        const ge::DataType x4DataType = context->GetInputDataType(4);
        const ge::DataType x5DataType = context->GetInputDataType(5);
        const ge::DataType x6DataType = context->GetInputDataType(6);
        const ge::DataType x7DataType = context->GetInputDataType(7);
        const ge::DataType x8DataType = context->GetInputDataType(8);
        const ge::DataType x9DataType = context->GetInputDataType(9);
        const ge::DataType x10DataType = context->GetInputDataType(10);
        const ge::DataType x11DataType = context->GetInputDataType(11);
        const ge::DataType x12DataType = context->GetInputDataType(12);
        const ge::DataType x13DataType = context->GetInputDataType(13);
        const ge::DataType x14DataType = context->GetInputDataType(14);
        const ge::DataType x15DataType = context->GetInputDataType(15);
        const ge::DataType x16DataType = context->GetInputDataType(16);
        const ge::DataType x17DataType = context->GetInputDataType(17);

        context->SetOutputDataType(0, x0DataType);
        context->SetOutputDataType(1, x1DataType);
        context->SetOutputDataType(2, x3DataType);
        context->SetOutputDataType(3, x4DataType);
        context->SetOutputDataType(4, x5DataType);
        context->SetOutputDataType(5, x6DataType);
        context->SetOutputDataType(6, x7DataType);
        context->SetOutputDataType(7, x8DataType);
        context->SetOutputDataType(8, x9DataType);
        context->SetOutputDataType(9, x10DataType);
        context->SetOutputDataType(10, x11DataType);
        context->SetOutputDataType(11, x12DataType);
        context->SetOutputDataType(12, x13DataType);
        context->SetOutputDataType(13, x14DataType);
        context->SetOutputDataType(14, x15DataType);
        context->SetOutputDataType(15, x16DataType);
        context->SetOutputDataType(16, x17DataType);

        return GRAPH_SUCCESS;
    }
}

namespace ops {
    class StepPaddle : public OpDef {
    public:
        explicit StepPaddle(const char *name) : OpDef(name)
        {            
            this->Input("stop_flags")
                .ParamType(REQUIRED)
                .DataType({ge::DT_BOOL})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("seq_lens_this_time")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("ori_seq_lens_encoder")
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
            
            this->Input("block_tables")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("encoder_block_lens")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("is_block_step")
                .ParamType(REQUIRED)
                .DataType({ge::DT_BOOL})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("step_block_list")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("step_lens")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("recover_block_list")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("recover_lens")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("need_block_list")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("need_block_len")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("used_list_len")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("free_list")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("free_list_len")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("input_ids")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("pre_ids")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("step_idx")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Input("next_tokens")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Output("stop_flags_out")
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
            
            this->Output("block_tables_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Output("encoder_block_lens_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Output("is_block_step_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_BOOL})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Output("step_block_list_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Output("step_lens_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Output("recover_block_list_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Output("recover_lens_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Output("need_block_list_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Output("need_block_len_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Output("used_list_len_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Output("free_list_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Output("free_list_len_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT32})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->Output("input_ids_out")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->SetInferShape(ge::StepPaddleInferShape)
                .SetInferShapeRange(ge::StepPaddleInferShapeRange)
                .SetInferDataType(ge::StepPaddleInferDataType);

            this->Attr("block_size").AttrType(REQUIRED).Int();
            this->Attr("encoder_decoder_block_num").AttrType(REQUIRED).Int();
            this->Attr("first_token_id").AttrType(REQUIRED).Int(0);

            this->AICore()
                .SetTiling(optiling::StepPaddleTilingFunc);

            this->AICore().AddConfig("ascend310p");
            this->AICore().AddConfig("ascend910");
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(StepPaddle);
}
