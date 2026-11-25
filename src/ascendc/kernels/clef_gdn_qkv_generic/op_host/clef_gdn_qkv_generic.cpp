#include "../op_kernel/clef_gdn_qkv_generic_tiling.h"
#include "register/op_def_registry.h"
#include <cmath>
#include <limits>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto& dims = context->GetInputShape(0)->GetStorageShape();
    if (dims.GetDimNum() != 3 || (dims.GetDim(0) != 40 && dims.GetDim(0) != 80) ||
        dims.GetDim(2) != 128 || dims.GetDim(1) <= 0 ||
        dims.GetDim(1) > std::numeric_limits<uint32_t>::max() / 80) {
        return ge::GRAPH_FAILED;
    }
    const float* eps = context->GetAttrs()->GetAttrPointer<float>(0);
    const int64_t* tile = context->GetAttrs()->GetAttrPointer<int64_t>(1);
    if (eps == nullptr || !std::isfinite(*eps) || *eps <= 0 || tile == nullptr ||
        (*tile != 16 && *tile != 32 && *tile != 64)) {
        return ge::GRAPH_FAILED;
    }
    auto* t = context->GetTilingData<ClefGdnQkvGenericTilingData>();
    t->totalTokens = static_cast<uint32_t>(dims.GetDim(1));
    t->tileTokens = static_cast<uint32_t>(*tile);
    t->tilesPerHead = (t->totalTokens + t->tileTokens - 1) / t->tileTokens;
    t->keyHeads = static_cast<uint32_t>(dims.GetDim(0)) / 5;
    t->valueHeads = 3 * t->keyHeads;
    t->totalTiles = static_cast<uint32_t>(dims.GetDim(0)) * t->tilesPerHead;
    t->eps = *eps;
    context->SetBlockDim(24);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const auto* x = context->GetInputShape(0);
    if (x->GetDimNum() != 3 || (x->GetDim(0) != 40 && x->GetDim(0) != 80) ||
        x->GetDim(2) != 128 || x->GetDim(1) <= 0) {
        return GRAPH_FAILED;
    }
    const int64_t hk = x->GetDim(0) / 5;
    gert::Shape q, v;
    q.AppendDim(1); q.AppendDim(x->GetDim(1)); q.AppendDim(hk); q.AppendDim(128);
    v.AppendDim(1); v.AppendDim(x->GetDim(1)); v.AppendDim(3 * hk); v.AppendDim(128);
    *context->GetOutputShape(0) = q;
    *context->GetOutputShape(1) = q;
    *context->GetOutputShape(2) = v;
    return GRAPH_SUCCESS;
}
static ge::graphStatus InferDataType(gert::InferDataTypeContext* context) {
    const auto dtype = context->GetInputDataType(0);
    context->SetOutputDataType(0, dtype);
    context->SetOutputDataType(1, dtype);
    context->SetOutputDataType(2, dtype);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class ClefGdnQkvGeneric : public OpDef {
public:
    explicit ClefGdnQkvGeneric(const char* name) : OpDef(name) {
        this->Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("q").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("k").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("v").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("eps").Float();
        this->Attr("tile_tokens").Int();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(ClefGdnQkvGeneric);
}  // namespace ops
