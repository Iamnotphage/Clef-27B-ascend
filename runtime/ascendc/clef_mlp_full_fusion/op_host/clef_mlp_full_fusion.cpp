#include "../op_kernel/clef_mlp_full_fusion_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace {
bool IsMatrix(const gert::Shape& shape, int64_t rows, int64_t cols)
{
    return shape.GetDimNum() == 2 && shape.GetDim(0) == rows && shape.GetDim(1) == cols;
}
}  // namespace

namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext* ctx)
{
    using namespace ClefMlp;
    const auto& x = ctx->GetInputShape(0)->GetOriginShape();
    const auto& gate = ctx->GetInputShape(1)->GetOriginShape();
    const auto& down = ctx->GetInputShape(2)->GetOriginShape();
    if (x.GetDimNum() != 2 || x.GetDim(0) < 1 || x.GetDim(0) > 1595 ||
        x.GetDim(1) != H || !IsMatrix(gate, 2 * I, H) || !IsMatrix(down, H, I)) {
        return ge::GRAPH_FAILED;
    }
    const auto* diagnostic = ctx->GetAttrs()->GetAttrPointer<int64_t>(0);
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    if (diagnostic == nullptr || (*diagnostic != 0 && *diagnostic != 1) ||
        platform.GetCoreNumAic() != Groups || platform.GetCoreNumAiv() < 2 * Groups) {
        return ge::GRAPH_FAILED;
    }
    auto* data = ctx->GetTilingData<TilingData>();
    data->tokens = static_cast<uint32_t>(x.GetDim(0));
    data->diagnostic = static_cast<uint32_t>(*diagnostic);
    ctx->SetBlockDim(Groups);
    ctx->GetWorkspaceSizes(1)[0] = platform.GetLibApiWorkSpaceSize() +
        (RingElements + static_cast<size_t>(data->tokens) * I) * sizeof(uint16_t);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* ctx)
{
    const auto* x = ctx->GetInputShape(0);
    const auto* diagnostic = ctx->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (x->GetDimNum() != 2 || diagnostic == nullptr || (*diagnostic != 0 && *diagnostic != 1)) {
        return GRAPH_FAILED;
    }
    gert::Shape y;
    y.AppendDim(x->GetDim(0));
    y.AppendDim(ClefMlp::H + (*diagnostic ? 3 * ClefMlp::I : 0));
    *ctx->GetOutputShape(0) = y;
    return GRAPH_SUCCESS;
}

static graphStatus InferDtype(gert::InferDataTypeContext* ctx)
{
    ctx->SetOutputDataType(0, DT_BF16);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class ClefMlpFullFusion : public OpDef {
public:
    explicit ClefMlpFullFusion(const char* name) : OpDef(name)
    {
        Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        Input("gateWeight").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_FRACTAL_NZ}).UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
        Input("downWeight").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_FRACTAL_NZ}).UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
        Output("y").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        Attr("diagnostic").AttrType(OPTIONAL).Int(0);
        SetInferShape(ge::InferShape).SetInferDataType(ge::InferDtype);
        AICore().SetTiling(optiling::Tiling);
        AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(ClefMlpFullFusion);
}  // namespace ops
