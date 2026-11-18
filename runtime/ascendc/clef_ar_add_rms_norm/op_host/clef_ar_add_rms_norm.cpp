#include "../op_kernel/clef_ar_add_rms_norm_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <hccl/hccl_mc2.h>
#include <adv_api/hccl/hccl_tiling.h>
#include <cstring>

// Gate-B scaffold constants.  The real smoke will lift group/alg into attrs or
// a builder flag once the official build proves the tiling path links.
namespace {
constexpr const char* kGroup = "hccl_world_group";
constexpr const char* kAlg = "AllReduce=level0:ring";
constexpr uint8_t kOpTypeAllReduce = 2;   // HCCL_CMD_ALLREDUCE
constexpr uint8_t kReduceSum = 0;         // HCCL_REDUCE_SUM
constexpr uint8_t kBf16 = 11;             // HCCL_DATA_TYPE_BFP16
constexpr uint8_t kEngineAicpu = 2;       // COMM_ENGINE_AICPU
// core count comes from the platform at tiling time
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
    t->commEngine = kEngineAicpu;
    t->opType = kOpTypeAllReduce;

    // The whole point of gate B: prove Mc2CcTilingConfig resolves and links in
    // the official npu_op_package op_host flow (it cannot link standalone).
    AscendC::Mc2CcTilingConfig cfg(kGroup, kOpTypeAllReduce, kAlg, kReduceSum,
                                   kBf16, kBf16, kEngineAicpu);
    auto* initTiling = reinterpret_cast<::Mc2InitTiling*>(t->mc2InitTiling);
    auto* ccTiling = reinterpret_cast<::Mc2CcTiling*>(t->mc2CcTiling);
    static_assert(sizeof(::Mc2InitTiling) <= kMc2TilingBytes, "init tiling too large");
    static_assert(sizeof(::Mc2CcTiling) <= kMc2TilingBytes, "cc tiling too large");
    std::memset(t->mc2InitTiling, 0, kMc2TilingBytes);
    std::memset(t->mc2CcTiling, 0, kMc2TilingBytes);
    t->initTilingRc = cfg.GetTiling(*initTiling);
    t->ccTilingRc = cfg.GetTiling(*ccTiling);
    t->initTilingBytes = sizeof(::Mc2InitTiling);
    t->ccTilingBytes = sizeof(::Mc2CcTiling);
    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    uint32_t cores = platform.GetCoreNumAiv();
    if (cores == 0) cores = 8;
    context->SetBlockDim(cores);
    context->GetWorkspaceSizes(1)[0] = 0;
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
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, context->GetInputDataType(0));
    context->SetOutputDataType(1, context->GetInputDataType(0));
    context->SetOutputDataType(2, ge::DT_FLOAT);
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
        this->Input("gamma").DataType({ge::DT_BF16});
        for (const char* output : {"out", "residual_out"}) {
            this->Output(output).ParamType(REQUIRED).DataType({ge::DT_BF16})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        }
        this->Output("rstd").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        this->Attr("eps").Float();
        this->Attr("repeat").Int(1);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_93");
    }
};
OP_ADD(ClefArAddRmsNorm);
}  // namespace ops
