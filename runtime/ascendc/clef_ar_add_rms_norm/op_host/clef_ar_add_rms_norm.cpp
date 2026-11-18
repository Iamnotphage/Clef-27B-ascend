#include "../op_kernel/clef_ar_add_rms_norm_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <hccl/hccl_mc2.h>
#include <adv_api/hccl/hccl_tiling.h>
#include <adv_api/hccl/hccl_tilingdata.h>
#include <cstddef>
#include <cstring>

namespace {
constexpr const char* kAlg = "AllReduce=level0:ring";
constexpr uint8_t kOpTypeAllReduce = 2;   // HCCL_CMD_ALLREDUCE
constexpr uint8_t kReduceSum = 0;         // HCCL_REDUCE_SUM
constexpr uint8_t kBf16 = 11;             // HCCL_DATA_TYPE_BFP16
constexpr uint8_t kEngineAicpu = 2;       // COMM_ENGINE_AICPU
}  // namespace

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* context) {
    const auto& x = context->GetInputShape(0)->GetStorageShape();
    if (x.GetDimNum() != 2 || x.GetDim(0) <= 0 || x.GetDim(1) <= 0 ||
        x.GetDim(0) > 0x7fffffff || x.GetDim(1) > 0x7fffffff) {
        return ge::GRAPH_FAILED;
    }
    auto* t = context->GetTilingData<ClefArAddRmsNormTilingData>();
    context->GetRawTilingData()->SetDataSize(sizeof(ClefArAddRmsNormTilingData));
    t->m = static_cast<uint32_t>(x.GetDim(0));
    t->n = static_cast<uint32_t>(x.GetDim(1));
    t->mPad = (t->m + 7u) / 8u * 8u;
    const auto* eps = context->GetAttrs()->GetAttrPointer<float>(0);
    t->eps = eps == nullptr ? 1e-6f : *eps;
    t->invN = 1.0f / static_cast<float>(t->n);
    const auto* repeat = context->GetAttrs()->GetAttrPointer<int64_t>(1);
    t->repeat = repeat == nullptr ? 1u : static_cast<uint32_t>(*repeat);
    if (t->repeat == 0 || t->repeat > 255) { return ge::GRAPH_FAILED; }
    // The group must be the real HCCL comm name (from the actual PG), never a
    // hardcoded string; the op def marks it REQUIRED and binds it to MC2.
    const char* group = context->GetAttrs()->GetAttrPointer<char>(2);
    if (group == nullptr || group[0] == '\0') { return ge::GRAPH_FAILED; }
    t->commEngine = kEngineAicpu;
    t->opType = kOpTypeAllReduce;
    t->useHccl = 1;

    // mc2InitTiling is the first member and mc2CcTiling follows it, so the
    // kernel can use offsetof() as the SetCcTilingV2 offset.
    static_assert(sizeof(::Mc2InitTiling) == kMc2InitBytes, "init tiling size");
    static_assert(sizeof(::Mc2CcTiling) == kMc2CcBytes, "cc tiling size");
    std::memset(t->mc2InitTiling, 0, kMc2InitBytes);
    std::memset(t->mc2CcTiling, 0, kMc2CcBytes);
    AscendC::Mc2CcTilingConfig cfg(group, kOpTypeAllReduce, kAlg, kReduceSum,
                                   kBf16, kBf16, kEngineAicpu);
    t->initTilingRc = cfg.GetTiling(*reinterpret_cast<::Mc2InitTiling*>(t->mc2InitTiling));
    t->ccTilingRc = cfg.GetTiling(*reinterpret_cast<::Mc2CcTiling*>(t->mc2CcTiling));

    // The AR smoke runs the collective from a single block: only core 0 may
    // write hcclMsgArea, and cross-block handle sharing is still unverified.
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t cores = platform.GetCoreNumAiv();
    if (cores == 0) cores = 8;
    context->SetBlockDim(cores);
    // Out-of-place reduction destination (in-place was observed to leave the
    // source unchanged even though Wait returned success).
    context->GetWorkspaceSizes(1)[0] = static_cast<size_t>(t->m) * t->n * 2 + 128;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    const auto* x = context->GetInputShape(0);
    if (x->GetDimNum() != 2) { return GRAPH_FAILED; }
    *context->GetOutputShape(0) = *x;              // out
    *context->GetOutputShape(1) = *x;              // residual_out
    gert::Shape rstd;
    rstd.AppendDim(x->GetDim(0));
    *context->GetOutputShape(2) = rstd;            // rstd [M]
    gert::Shape diag;
    diag.AppendDim(8);
    *context->GetOutputShape(3) = diag;
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(0));
    context->SetOutputDataType(2, ge::DT_FLOAT);
    context->SetOutputDataType(3, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class ClefArAddRmsNorm : public OpDef {
public:
    explicit ClefArAddRmsNorm(const char* name) : OpDef(name) {
        for (const char* input : {"x", "residual", "gamma"}) {
            this->Input(input).ParamType(REQUIRED).DataType({ge::DT_BF16})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        }
        for (const char* output : {"out", "residual_out"}) {
            this->Output(output).ParamType(REQUIRED).DataType({ge::DT_BF16})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->Output("rstd").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Output("diag").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("eps").Float();
        this->Attr("repeat").Int(1);
        this->Attr("group").String();
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_93");
        this->MC2().HcclGroup("group");
    }
};
OP_ADD(ClefArAddRmsNorm);
}  // namespace ops
