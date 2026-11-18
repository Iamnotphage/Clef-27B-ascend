#include "../op_kernel/clef_matmul_ar_add_rms_norm_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <hccl/hccl_mc2.h>
#include <adv_api/hccl/hccl_tiling.h>
#include <adv_api/hccl/hccl_tilingdata.h>
#include <adv_api/matmul/bmm_tiling.h>
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
    // X[M,K], W[K,N], residual[M,N], gamma[N]
    const auto& xShape = context->GetInputShape(0)->GetStorageShape();
    const auto& wShape = context->GetInputShape(1)->GetStorageShape();
    const auto& rShape = context->GetInputShape(2)->GetStorageShape();
    const auto& gShape = context->GetInputShape(3)->GetStorageShape();
    if (xShape.GetDimNum() != 2 || wShape.GetDimNum() != 2 || rShape.GetDimNum() != 2 ||
        gShape.GetDimNum() != 1) {
        return ge::GRAPH_FAILED;
    }
    const int64_t m = xShape.GetDim(0);
    const int64_t k = xShape.GetDim(1);
    const int64_t n = wShape.GetDim(1);
    if (m <= 0 || k <= 0 || n <= 0 || wShape.GetDim(0) != k ||
        rShape.GetDim(0) != m || rShape.GetDim(1) != n || gShape.GetDim(0) != n) {
        return ge::GRAPH_FAILED;
    }
    if (m > 0x7fffffff || n > 0x7fffffff || k > 0x7fffffff) {
        return ge::GRAPH_FAILED;
    }

    auto* t = context->GetTilingData<ClefMatmulArAddRmsNormTilingData>();
    context->GetRawTilingData()->SetDataSize(sizeof(ClefMatmulArAddRmsNormTilingData));
    t->m = static_cast<uint32_t>(m);
    t->n = static_cast<uint32_t>(n);
    t->k = static_cast<uint32_t>(k);
    t->mPad = (t->m + 7u) / 8u * 8u;

    const auto* eps = context->GetAttrs()->GetAttrPointer<float>(0);
    t->eps = (eps == nullptr) ? 1e-6f : *eps;
    t->invN = 1.0f / static_cast<float>(t->n);

    const auto* repeat = context->GetAttrs()->GetAttrPointer<int64_t>(1);
    t->repeat = (repeat == nullptr) ? 1u : static_cast<uint32_t>(*repeat);
    if (t->repeat == 0 || t->repeat > 255) { return ge::GRAPH_FAILED; }

    // use_hccl is tri-state: 0 = skip the whole HCCL section, 1 = full
    // (lifecycle + AllReduce/Wait), 2 = lifecycle only (InitV2 /
    // SetCcTilingV2 / Finalize, no collective).  Mode 2 exists to test whether
    // the host-injected KFC service requires the device-side lifecycle to be
    // paired even when nothing is published.
    const auto* useHccl = context->GetAttrs()->GetAttrPointer<int64_t>(3);
    const int64_t hcclAttr = (useHccl == nullptr) ? 1 : *useHccl;
    if (hcclAttr < 0 || hcclAttr > 2) { return ge::GRAPH_FAILED; }
    const bool wantHccl = (hcclAttr >= 1);
    const char* group = context->GetAttrs()->GetAttrPointer<char>(2);
    if (wantHccl && (group == nullptr || group[0] == '\0')) { return ge::GRAPH_FAILED; }
    if (group == nullptr) group = "";
    t->commEngine = kEngineAicpu;
    t->opType = kOpTypeAllReduce;
    t->useHccl = static_cast<uint32_t>(hcclAttr);

    // mc2InitTiling is the first member and mc2CcTiling follows it, so the
    // kernel can use offsetof() as the SetCcTilingV2 offset.
    static_assert(sizeof(::Mc2InitTiling) == kClefMmMc2InitBytes, "init tiling size");
    static_assert(sizeof(::Mc2CcTiling) == kClefMmMc2CcBytes, "cc tiling size");
    std::memset(t->mc2InitTiling, 0, kClefMmMc2InitBytes);
    std::memset(t->mc2CcTiling, 0, kClefMmMc2CcBytes);
    if (wantHccl) {
        AscendC::Mc2CcTilingConfig cfg(group, kOpTypeAllReduce, kAlg, kReduceSum,
                                       kBf16, kBf16, kEngineAicpu);
        t->initTilingRc = cfg.GetTiling(*reinterpret_cast<::Mc2InitTiling*>(t->mc2InitTiling));
        t->ccTilingRc = cfg.GetTiling(*reinterpret_cast<::Mc2CcTiling*>(t->mc2CcTiling));
    } else {
        t->initTilingRc = 0;
        t->ccTilingRc = 0;
    }

    // mm_enable=0 skips the whole cube stage (bisect arm: does the
    // scalar-access fault live in the matmul at all?).  mm_dim only drives
    // MultiCoreMatmulTiling::SetDim and therefore TCubeTiling.usedCoreNum; the
    // kernel guards the matmul by usedCoreNum, so it can differ from block_dim.
    const auto* mmEnableAttr = context->GetAttrs()->GetAttrPointer<int64_t>(5);
    t->mmEnable = ((mmEnableAttr == nullptr) || (*mmEnableAttr != 0)) ? 1u : 0u;
    const auto* mmDimAttr = context->GetAttrs()->GetAttrPointer<int64_t>(6);
    const int64_t mmDimReq = (mmDimAttr == nullptr) ? 0 : *mmDimAttr;
    // mm_mode: 2 = full (default), 1 = stop after Init/SetOrgShape/SetTensorA/B
    // and never call IterateAll.  That splits a tiling/init fault from an
    // iterate (L1/L0C streaming) fault.
    const auto* mmModeAttr = context->GetAttrs()->GetAttrPointer<int64_t>(7);
    const int64_t mmModeReq = (mmModeAttr == nullptr) ? 2 : *mmModeAttr;
    if (mmModeReq < 1 || mmModeReq > 2) { return ge::GRAPH_FAILED; }
    t->mmMode = static_cast<uint32_t>(mmModeReq);
    // mm_bufspace: 0 = leave the tiling's own buffer model (default),
    // 1 = constrain L1/L0C to the platform's real sizes, 2 = also reserve the
    // normalisation pass's UB footprint.  The fault is a scalar access past an
    // AICore internal buffer, so an over-allocated tiling is the prime suspect.
    const auto* bufSpaceAttr = context->GetAttrs()->GetAttrPointer<int64_t>(8);
    const int64_t bufSpaceReq = (bufSpaceAttr == nullptr) ? 0 : *bufSpaceAttr;
    if (bufSpaceReq < 0 || bufSpaceReq > 2) { return ge::GRAPH_FAILED; }
    // norm_ub_late=1 (default) defers the normalisation pass's ~120 KB of UB
    // until after the cube stage, so the matmul's own InitBuffer cannot find
    // the pool already spent.  0 restores the previous allocation order.
    const auto* normLateAttr = context->GetAttrs()->GetAttrPointer<int64_t>(9);
    t->normLate = ((normLateAttr == nullptr) || (*normLateAttr != 0)) ? 1u : 0u;
    // mm_btrans=1: B is physically column-major.  The host tiling flag and the
    // kernel's SetTensorB flag are driven by this one attribute so they cannot
    // disagree.
    const auto* bTransAttr = context->GetAttrs()->GetAttrPointer<int64_t>(10);
    const bool bTrans = (bTransAttr != nullptr) && (*bTransAttr != 0);
    t->mmBTrans = bTrans ? 1u : 0u;
    // mm_step gates the cube calls one at a time (1 = Init only ... 5 = full);
    // 0 keeps the mm_mode meaning.  Used only to attribute the init-stage fault
    // to a single call, never to change behaviour on its own.
    const auto* mmStepAttr = context->GetAttrs()->GetAttrPointer<int64_t>(11);
    const int64_t mmStepReq = (mmStepAttr == nullptr) ? 0 : *mmStepAttr;
    if (mmStepReq < 0 || mmStepReq > 5) { return ge::GRAPH_FAILED; }
    t->mmStep = static_cast<uint32_t>(mmStepReq);

    auto platform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    // block_dim: 0 = auto (AIV count, the value that crashed with
    // "scalar access to the internal buffer of AICore is out of bounds"),
    // -1 = AIC count, >0 = explicit.  Both 0 and -1 were observed to crash
    // with the same fault and the same blk value, so this is now only a
    // secondary knob.
    const auto* blockDimAttr = context->GetAttrs()->GetAttrPointer<int64_t>(4);
    const int64_t blockDimReq = (blockDimAttr == nullptr) ? 0 : *blockDimAttr;
    uint32_t cores = 0;
    if (blockDimReq == 0) {
        cores = platform.GetCoreNumAiv();
    } else if (blockDimReq == -1) {
        cores = platform.GetCoreNumAic();
    } else if (blockDimReq > 0) {
        cores = static_cast<uint32_t>(blockDimReq);
    } else {
        return ge::GRAPH_FAILED;
    }
    if (cores == 0) cores = 1;

    uint32_t mmDim = cores;
    if (mmDimReq == -1) {
        mmDim = platform.GetCoreNumAic();
    } else if (mmDimReq > 0) {
        mmDim = static_cast<uint32_t>(mmDimReq);
    }
    if (mmDim == 0) mmDim = 1;
    t->mmDim = mmDim;

    // Cube tiling for X[M,K] x W[K,N] -> partial[M,N].  Cube/Vector are 1:2 on
    // this SoC, so the same block count drives both the matmul split and the
    // row split of the normalisation pass.
    matmul_tiling::MultiCoreMatmulTiling mmTiling(platform);
    mmTiling.SetDim(static_cast<int32_t>(mmDim));
    mmTiling.SetAType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_BFLOAT16, false);
    // B is ND in this diagnostic build: the block-dim sweep must isolate a
    // single variable.  The service contract is FRACTAL_NZ and the NZ build is
    // produced once the block-dim / sub-block question is settled.
    mmTiling.SetBType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_BFLOAT16, bTrans);
    mmTiling.SetCType(matmul_tiling::TPosition::GM, matmul_tiling::CubeFormat::ND,
                      matmul_tiling::DataType::DT_BFLOAT16);
    mmTiling.SetBias(false);
    mmTiling.SetShape(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k));
    mmTiling.SetOrgShape(static_cast<int32_t>(m), static_cast<int32_t>(n), static_cast<int32_t>(k));
    if (bufSpaceReq >= 1) {
        uint64_t l1 = 0, l0c = 0, ub = 0;
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L1, l1);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::L0_C, l0c);
        platform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ub);
        int32_t ubForMm = -1;
        if (bufSpaceReq >= 2) {
            // Reserve the normalisation pass's UB footprint so the cube stage
            // cannot plan into buffers the vector stage already owns.
            const int32_t reserved = 128 * 1024;
            ubForMm = (static_cast<int32_t>(ub) > reserved)
                          ? static_cast<int32_t>(ub) - reserved : 0;
        }
        mmTiling.SetBufferSpace(static_cast<int32_t>(l1), static_cast<int32_t>(l0c),
                                ubForMm, -1);
    }
    t->mmTilingRc = static_cast<uint32_t>(mmTiling.GetTiling(t->cubeTiling));
    if (t->mmTilingRc != 0) { return ge::GRAPH_FAILED; }

    context->SetBlockDim(cores);
    // workspace: partial[M,N] (matmul out) | reduced[M,N] (AllReduce out) | flag
    context->GetWorkspaceSizes(1)[0] = static_cast<size_t>(m) * n * 2 * 2 + 128;
    return ge::GRAPH_SUCCESS;
}
}  // namespace optiling

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* context) {
    const auto* x = context->GetInputShape(0);
    const auto* r = context->GetInputShape(2);
    if (x->GetDimNum() != 2 || r->GetDimNum() != 2) { return GRAPH_FAILED; }
    *context->GetOutputShape(0) = *x;              // out      [M,N]
    *context->GetOutputShape(1) = *r;              // residual_out [M,N]
    gert::Shape rstd;
    rstd.AppendDim(x->GetDim(0));
    *context->GetOutputShape(2) = rstd;            // rstd [M]
    gert::Shape diag;
    diag.AppendDim(8);
    *context->GetOutputShape(3) = diag;
    return GRAPH_SUCCESS;
}
static graphStatus InferDataType(gert::InferDataTypeContext* context) {
    context->SetOutputDataType(0, ge::DT_BF16);
    context->SetOutputDataType(1, ge::DT_BF16);
    context->SetOutputDataType(2, ge::DT_FLOAT);
    context->SetOutputDataType(3, ge::DT_FLOAT);
    return GRAPH_SUCCESS;
}
}  // namespace ge

namespace ops {
class ClefMatmulArAddRmsNorm : public OpDef {
public:
    explicit ClefMatmulArAddRmsNorm(const char* name) : OpDef(name) {
        for (const char* input : {"x", "w", "residual", "gamma"}) {
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
        this->Attr("use_hccl").Int(1);
        this->Attr("block_dim").Int(0);
        this->Attr("mm_enable").Int(1);
        this->Attr("mm_dim").Int(0);
        this->Attr("mm_mode").Int(2);
        this->Attr("mm_bufspace").Int(0);
        this->Attr("norm_ub_late").Int(1);
        this->Attr("mm_btrans").Int(0);
        this->Attr("mm_step").Int(0);
        this->SetInferShape(ge::InferShape).SetInferDataType(ge::InferDataType);
        this->AICore().SetTiling(optiling::TilingFunc);
        this->AICore().AddConfig("ascend910_93");
        this->MC2().HcclGroup("group");
    }
};
OP_ADD(ClefMatmulArAddRmsNorm);
}  // namespace ops
