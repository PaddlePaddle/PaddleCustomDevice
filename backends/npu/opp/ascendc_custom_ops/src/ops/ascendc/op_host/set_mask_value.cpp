#include "set_mask_value_tiling.h"
#include "register/op_def_registry.h"
namespace {
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
}

namespace optiling {
static ge::graphStatus SetMaskValueTilingFunc(gert::TilingContext *context)
{
    SetMaskValueTilingData tiling;

    const gert::StorageShape *input_data_shape = context->GetInputShape(0);
    const gert::StorageShape *seq_lens_shape = context->GetInputShape(2);


    int32_t inputBs = input_data_shape->GetStorageShape().GetDim(0);
    int32_t length = input_data_shape->GetStorageShape().GetDim(3);
    int32_t seqBs = seq_lens_shape->GetStorageShape().GetDim(0);

    int32_t blockSize = 1;

    tiling.set_seqBs(seqBs);
    tiling.set_length(length);

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
static ge::graphStatus SetMaskValueInferShape(gert::InferShapeContext *context)
{
    const gert::Shape *x3_shape = context->GetInputShape(2);
    gert::Shape *y0_shape = context->GetOutputShape(0);

    *y0_shape = *x3_shape;
    return GRAPH_SUCCESS;
}

ge::graphStatus SetMaskValueInferShapeRange(gert::InferShapeRangeContext *context)
{
    const gert::Range<gert::Shape> *inputXShapeRange = context->GetInputShapeRange(2);
    gert::Range<gert::Shape> *outputShapeRange = context->GetOutputShapeRange(0);
    *outputShapeRange = *inputXShapeRange;

    return GRAPH_SUCCESS;
}

ge::graphStatus SetMaskValueInferDataType(gert::InferDataTypeContext *context)
{
    const ge::DataType Q = context->GetInputDataType(2);

    context->SetOutputDataType(0, Q);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class SetMaskValue : public OpDef {
public:
    SetMaskValue(const char *name) : OpDef(name)
    {
        this->Input("input_data")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_FLOAT16 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("stop_flags")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_BOOL  })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Input("seq_lens")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });
        this->Output("sequence_lengths")
            .ParamType(REQUIRED)
            .DataType({ ge::DT_INT32 })
            .Format({ ge::FORMAT_ND })
            .UnknownShapeFormat({ ge::FORMAT_ND });

        this->SetInferShape(ge::SetMaskValueInferShape)
            .SetInferShapeRange(ge::SetMaskValueInferShapeRange)
            .SetInferDataType(ge::SetMaskValueInferDataType);

        this->AICore().SetTiling(optiling::SetMaskValueTilingFunc);

        this->AICore().AddConfig("ascend310p");
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SetMaskValue);
}
