#include "../op_kernel/clef_gdn_qkv_norm_tiling.h"
#include "register/op_def_registry.h"

namespace {
constexpr uint32_t kHeads = 40;
constexpr uint32_t kKHeads = 8;
constexpr uint32_t kVHeads = 24;
constexpr uint32_t kDim = 128;
constexpr uint32_t kTileTokens = 64;
constexpr uint32_t kCoreNum = 24;

uint32_t CeilDiv(uint32_t value, uint32_t divisor) {
    return (value + divisor - 1) / divisor;
}
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    auto* tiling = context->GetTilingData<ClefGdnQkvNormTilingData>();
    const gert::StorageShape* xShape = context->GetInputShape(0);
    const auto& dims = xShape->GetStorageShape();
    if (dims.GetDimNum() != 3 || dims.GetDim(0) != kHeads ||
        dims.GetDim(2) != kDim || dims.GetDim(1) == 0) {
        return ge::GRAPH_FAILED;
    }
    uint32_t tokens = static_cast<uint32_t>(dims.GetDim(1));
    tiling->totalTokens = tokens;
    tiling->tileTokens = kTileTokens;
    tiling->tilesPerHead = CeilDiv(tokens, kTileTokens);
    tiling->totalTiles = kHeads * tiling->tilesPerHead;
    tiling->tilesPerCore = CeilDiv(tiling->totalTiles, kCoreNum);
    tiling->coreNum = kCoreNum;
    const float* eps = context->GetAttrs()->GetAttrPointer<float>(0);
    tiling->eps = (eps == nullptr) ? 1e-6f : *eps;
    context->SetBlockDim(kCoreNum);
    size_t* workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static ge::graphStatus InferShape(gert::InferShapeContext* context) {
    const gert::Shape* xShape = context->GetInputShape(0);
    uint32_t tokens = static_cast<uint32_t>(xShape->GetDim(1));
    gert::Shape qShape;
    qShape.AppendDim(1);
    qShape.AppendDim(tokens);
    qShape.AppendDim(kKHeads);
    qShape.AppendDim(kDim);
    *context->GetOutputShape(0) = qShape;
    *context->GetOutputShape(1) = qShape;
    gert::Shape vShape;
    vShape.AppendDim(1);
    vShape.AppendDim(tokens);
    vShape.AppendDim(kVHeads);
    vShape.AppendDim(kDim);
    *context->GetOutputShape(2) = vShape;
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
class ClefGdnQkvNorm : public OpDef {
public:
    explicit ClefGdnQkvNorm(const char* name) : OpDef(name) {
        this->Input("x")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("q")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("k")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("v")
            .ParamType(REQUIRED)
            .DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND})
            .UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("eps").Float();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(ClefGdnQkvNorm);
}  // namespace ops
