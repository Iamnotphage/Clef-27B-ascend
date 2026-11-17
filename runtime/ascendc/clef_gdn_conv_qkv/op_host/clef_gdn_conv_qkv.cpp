#include "../op_kernel/clef_gdn_conv_qkv_tiling.h"
#include "register/op_def_registry.h"
#include <initializer_list>

namespace {
constexpr uint32_t kHeads = 40;
constexpr uint32_t kDim = 128;
constexpr uint32_t kChannels = kHeads * kDim;
constexpr uint32_t kCores = 24;
uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto& x = context->GetInputShape(0)->GetStorageShape();
    const auto& w = context->GetInputShape(1)->GetStorageShape();
    if (x.GetDimNum() != 2 || x.GetDim(0) <= 0 || x.GetDim(0) > 0x7fffffff ||
        x.GetDim(1) != kChannels || w.GetDimNum() != 2 ||
        w.GetDim(0) != 4 || w.GetDim(1) != kChannels) {
        return ge::GRAPH_FAILED;
    }
    const auto* tileAttr = context->GetAttrs()->GetAttrPointer<int64_t>(1);
    const uint32_t tile = tileAttr == nullptr ? 32 : static_cast<uint32_t>(*tileAttr);
    if (tile != 16 && tile != 32 && tile != 64) { return ge::GRAPH_FAILED; }
    const auto* eps = context->GetAttrs()->GetAttrPointer<float>(0);
    auto* t = context->GetTilingData<ClefGdnConvQkvTilingData>();
    t->totalTokens = static_cast<uint32_t>(x.GetDim(0));
    t->tileTokens = tile;
    t->tilesPerHead = CeilDiv(t->totalTokens, tile);
    t->totalTiles = kHeads * t->tilesPerHead;
    t->tilesPerCore = CeilDiv(t->totalTiles, kCores);
    t->coreNum = kCores;
    t->eps = eps == nullptr ? 1e-6f : *eps;
    context->SetBlockDim(kCores);
    context->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    const auto* x = context->GetInputShape(0);
    if (x->GetDimNum() != 2 || x->GetDim(1) != kChannels) { return GRAPH_FAILED; }
    gert::Shape q, v;
    q.AppendDim(1); q.AppendDim(x->GetDim(0)); q.AppendDim(8); q.AppendDim(kDim);
    v.AppendDim(1); v.AppendDim(x->GetDim(0)); v.AppendDim(24); v.AppendDim(kDim);
    *context->GetOutputShape(0) = q;
    *context->GetOutputShape(1) = q;
    *context->GetOutputShape(2) = v;
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    for (uint32_t i = 0; i < 3; ++i) {
        context->SetOutputDataType(i, context->GetInputDataType(0));
    }
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class ClefGdnConvQkv : public OpDef {
public:
    explicit ClefGdnConvQkv(const char* name) : OpDef(name) {
        for (const char* input : {"x", "weight"}) {
            this->Input(input).ParamType(REQUIRED).DataType({ge::DT_BF16})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        }
        for (const char* output : {"q", "k", "v"}) {
            this->Output(output).ParamType(REQUIRED).DataType({ge::DT_BF16})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->Attr("eps").Float();
        this->Attr("tile_tokens").Int(32);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(ClefGdnConvQkv);
}  // namespace ops
