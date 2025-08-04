#include "set_value_by_flags_and_idx_tiling.h"
#include "register/op_def_registry.h"

using namespace std;

namespace {
constexpr uint32_t MINIMAL_WORKSPACE = 16 * 1024 * 1024;
}

namespace optiling {
    static ge::graphStatus SetValueByFlagsAndIdxTilingFunc(gert::TilingContext *context)
    {
        SetValueByFlagsAndIdxTilingData tiling;
        const gert::StorageShape* preIdsAllShape = context->GetInputShape(0);
        const gert::StorageShape* stopFlagsShape = context->GetInputShape(3);

        int bs = stopFlagsShape->GetStorageShape().GetDim(0);
        int length = preIdsAllShape->GetStorageShape().GetDim(1);

        tiling.set_bs(bs);
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
    static ge::graphStatus SetValueByFlagsAndIdxInferShape(gert::InferShapeContext *context)
    {
        const gert::Shape* x3Shape = context->GetInputShape(3);
        
        gert::Shape* y0Shape = context->GetOutputShape(0);

        *y0Shape = *x3Shape;

        int num = y0Shape->GetDimNum();

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus SetValueByFlagsAndIdxInferShapeRange(gert::InferShapeRangeContext *context)
    {
        const gert::Range<gert::Shape> *inputXShapeRange = context->GetInputShapeRange(3);
        gert::Range<gert::Shape> *y0ShapeRange = context->GetOutputShapeRange(0);

        *y0ShapeRange = *inputXShapeRange;

        return GRAPH_SUCCESS;
    }

    static ge::graphStatus SetValueByFlagsAndIdxInferDataType(gert::InferDataTypeContext *context)
    {
        const ge::DataType x3DataType = context->GetInputDataType(3);

        context->SetOutputDataType(0, x3DataType);

        return GRAPH_SUCCESS;
    }
}

namespace ops {
    class SetValueByFlagsAndIdx : public OpDef {
    public:
        explicit SetValueByFlagsAndIdx(const char *name) : OpDef(name)
        {            
            this->Input("preIdsAll")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});
            
            this->Input("preIdsNow")
                .ParamType(REQUIRED)
                .DataType({ge::DT_INT64})
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

            this->Output("stopFlagsOut")
                .ParamType(REQUIRED)
                .DataType({ge::DT_BOOL})
                .Format({ge::FORMAT_ND})
                .UnknownShapeFormat({ge::FORMAT_ND});

            this->SetInferShape(ge::SetValueByFlagsAndIdxInferShape)
                .SetInferShapeRange(ge::SetValueByFlagsAndIdxInferShapeRange)
                .SetInferDataType(ge::SetValueByFlagsAndIdxInferDataType);

            this->AICore()
                .SetTiling(optiling::SetValueByFlagsAndIdxTilingFunc);

            this->AICore().AddConfig("ascend310p");
            this->AICore().AddConfig("ascend910");
            this->AICore().AddConfig("ascend910b");
        }
    };

    OP_ADD(SetValueByFlagsAndIdx);
}
