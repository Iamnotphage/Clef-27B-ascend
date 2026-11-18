#include "../op_kernel/clef_native_ar_norm25_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
#include <adv_api/hccl/hccl_tiling.h>
#include <adv_api/hccl/hccl_tilingdata.h>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* c) {
    const auto& x = c->GetInputShape(0)->GetStorageShape();
    const auto& r = c->GetInputShape(1)->GetStorageShape();
    const auto& g = c->GetInputShape(2)->GetStorageShape();
    if (x.GetDimNum() != 2 || x.GetDim(0) < 1 || x.GetDim(0) > 16384 ||
        x.GetDim(1) != 5120 || r.GetDimNum() != 2 ||
        r.GetDim(0) != x.GetDim(0) || r.GetDim(1) != 5120 ||
        g.GetDimNum() != 1 || g.GetDim(0) != 5120) return ge::GRAPH_FAILED;
    const float eps = *c->GetAttrs()->GetAttrPointer<float>(0);
    const char* group = c->GetAttrs()->GetAttrPointer<char>(1);
    const int64_t use = *c->GetAttrs()->GetAttrPointer<int64_t>(2);
    const int64_t requested = *c->GetAttrs()->GetAttrPointer<int64_t>(3);
    if (!std::isfinite(eps) || eps <= 0 || (use != 0 && use != 1) ||
        requested < 1 || requested > 48 ||
        !group || !group[0]) return ge::GRAPH_FAILED;
    auto p = platform_ascendc::PlatformAscendC(c->GetPlatformInfo());
    if (requested > p.GetCoreNumAiv()) return ge::GRAPH_FAILED;
    auto* t = c->GetTilingData<ClefNativeArNorm25TilingData>();
    std::memset(t, 0, sizeof(*t));
    t->m = x.GetDim(0); t->n = 5120; t->cores = requested;
    t->useHccl = use; t->eps = eps; t->invN = 1.0f / 5120.0f;
    static_assert(sizeof(::Mc2InitTiling) == 64, "MC2 init ABI");
    static_assert(sizeof(::Mc2CcTiling) == 512, "MC2 cc ABI");
    // The generated API marks both arms MC2. ARoff retains Init/Finalize.
    {
        AscendC::Mc2CcTilingConfig cfg(group, 2, "AllReduce=level0:ring", 0, 11, 11, 2);
        if (cfg.GetTiling(*reinterpret_cast<::Mc2InitTiling*>(t->mc2InitTiling)) != 0 ||
            cfg.GetTiling(*reinterpret_cast<::Mc2CcTiling*>(t->mc2CcTiling)) != 0)
            return ge::GRAPH_FAILED;
    }
    c->GetRawTilingData()->SetDataSize(sizeof(*t));
    c->SetBlockDim(t->cores);
    // Required by SyncAll: reserve the full launch as one scheduling batch.
    c->SetScheduleMode(1);
    // AllReduce writes the explicit reduced output. No user workspace or GM flags.
    c->GetWorkspaceSizes(1)[0] = 0;
    return ge::GRAPH_SUCCESS;
}
}

namespace ge {
static graphStatus InferShape(gert::InferShapeContext* c) {
    const auto* x = c->GetInputShape(0);
    if (x->GetDimNum() != 2) return GRAPH_FAILED;
    *c->GetOutputShape(0) = *x;
    *c->GetOutputShape(1) = *x;
    gert::Shape rs; rs.AppendDim(x->GetDim(0));
    *c->GetOutputShape(2) = rs;
    *c->GetOutputShape(3) = *x;
    gert::Shape diag; diag.AppendDim(48); diag.AppendDim(32);
    *c->GetOutputShape(4) = diag;
    return GRAPH_SUCCESS;
}
static graphStatus InferType(gert::InferDataTypeContext* c) {
    for (int i : {0, 1, 3}) c->SetOutputDataType(i, ge::DT_BF16);
    c->SetOutputDataType(2, ge::DT_FLOAT);
    c->SetOutputDataType(4, ge::DT_UINT32);
    return GRAPH_SUCCESS;
}
}

namespace ops {
class ClefNativeArNorm25 : public OpDef {
public:
    explicit ClefNativeArNorm25(const char* name) : OpDef(name) {
        for (const char* s : {"partial", "residual", "gamma"})
            Input(s).ParamType(REQUIRED).DataType({ge::DT_BF16})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        for (const char* s : {"y", "residual_out"})
            Output(s).ParamType(REQUIRED).DataType({ge::DT_BF16})
                .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("rstd").ParamType(REQUIRED).DataType({ge::DT_FLOAT})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("reduced").ParamType(REQUIRED).DataType({ge::DT_BF16})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Output("diag").ParamType(REQUIRED).DataType({ge::DT_UINT32})
            .Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
        Attr("eps").Float();
        Attr("group").String();
        Attr("use_hccl").Int(0);
        Attr("cores").Int(1);
        SetInferShape(ge::InferShape).SetInferDataType(ge::InferType);
        AICore().SetTiling(optiling::TilingFunc).AddConfig("ascend910_93");
        MC2().HcclGroup("group");
    }
};
OP_ADD(ClefNativeArNorm25);
}
