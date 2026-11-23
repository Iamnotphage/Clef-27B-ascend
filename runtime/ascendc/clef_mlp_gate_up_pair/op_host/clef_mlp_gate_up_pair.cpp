#include "../op_kernel/clef_mlp_gate_up_pair_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"

namespace optiling {
static ge::graphStatus Tiling(gert::TilingContext* ctx)
{
    using namespace ClefGateUp;
    const auto& x = ctx->GetInputShape(0)->GetOriginShape();
    const auto& weights = ctx->GetInputShape(1)->GetOriginShape();
    const auto* mode = ctx->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (x.GetDimNum() != 2 || x.GetDim(0) < 1 || x.GetDim(0) > 1595 ||
        x.GetDim(1) != Hidden || weights.GetDimNum() != 2 ||
        weights.GetDim(0) != 2 * Intermediate || weights.GetDim(1) != Hidden ||
        mode == nullptr || *mode < Raw || *mode > ShortFused ||
        (IsShort(*mode) && x.GetDim(0) > 384)) {
        return ge::GRAPH_FAILED;
    }
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    if (platform.GetCoreNumAic() != Groups || platform.GetCoreNumAiv() < 2 * Groups) {
        return ge::GRAPH_FAILED;
    }
    auto* data = ctx->GetTilingData<TilingData>();
    data->tokens = static_cast<uint32_t>(x.GetDim(0));
    data->mode = static_cast<uint32_t>(*mode);
    ctx->SetBlockDim(Groups);
    const size_t ringBytes = Groups * RingSlots * 128 * 256 * sizeof(uint16_t);
    ctx->GetWorkspaceSizes(1)[0] =
        platform.GetLibApiWorkSpaceSize() + (IsFused(data->mode) ? ringBytes : 0);
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* ctx)
{
    const auto* x = ctx->GetInputShape(0);
    const auto* mode = ctx->GetAttrs()->GetAttrPointer<int64_t>(0);
    if (x->GetDimNum() != 2 || mode == nullptr || *mode < ClefGateUp::Raw ||
        *mode > ClefGateUp::ShortFused) {
        return GRAPH_FAILED;
    }
    gert::Shape y;
    y.AppendDim(x->GetDim(0));
    y.AppendDim(ClefGateUp::Intermediate * (ClefGateUp::IsFused(*mode) ? 1 : 2));
    *ctx->GetOutputShape(0) = y;
    return GRAPH_SUCCESS;
}

}  // namespace ge

namespace ops {
class ClefMlpGateUpPair : public OpDef {
public:
    explicit ClefMlpGateUpPair(const char* name) : OpDef(name)
    {
        Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        Input("pairedWeight").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_FRACTAL_NZ}).UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
        Output("y").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        Attr("mode").AttrType(OPTIONAL).Int(0);
        SetInferShape(ge::InferShape);
        AICore().SetTiling(optiling::Tiling);
        AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(ClefMlpGateUpPair);
}  // namespace ops
