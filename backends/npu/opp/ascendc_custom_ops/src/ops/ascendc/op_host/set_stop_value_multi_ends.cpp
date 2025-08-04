
#include "set_stop_value_multi_ends_tiling.h"
#include "register/op_def_registry.h"

namespace {
    constexpr int32_t MODE = 2;
    constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
}
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context)
{
    SetStopValueMultiEndsTilingData tiling;

    const gert::StorageShape *input_data_shape = context->GetInputShape(0);
    const gert::StorageShape *seq_lens_shape = context->GetInputShape(2);

    int32_t inputBs = input_data_shape->GetStorageShape().GetDim(0);
    int32_t length = seq_lens_shape->GetStorageShape().GetDim(0);
    int32_t seqBs = input_data_shape->GetStorageShape().GetDim(0);

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
static ge::graphStatus InferShape(gert::InferShapeContext* context)
{
    const gert::Shape* x1_shape = context->GetInputShape(0);
    gert::Shape* y_shape = context->GetOutputShape(0);
    *y_shape = *x1_shape;
    return GRAPH_SUCCESS;
}
}


namespace ops {
class SetStopValueMultiEnds : public OpDef {
public:
    explicit SetStopValueMultiEnds(const char* name) : OpDef(name)
    {
        this->Input("topk_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("stop_flags")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Input("end_ids")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("topk_ids_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_INT64})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("stop_flags_out")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BOOL})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});

        this->SetInferShape(ge::InferShape);

        this->AICore()
            .SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend310p");
        this->AICore().AddConfig("ascend910");
        this->AICore().AddConfig("ascend910b");
    }
};

OP_ADD(SetStopValueMultiEnds);
}
