#include "../op_kernel/clef_mlp_swiglu_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* ctx) {
    const auto& x = ctx->GetInputShape(0)->GetOriginShape();
    const auto& w = ctx->GetInputShape(1)->GetOriginShape();
    if (x.GetDimNum()!=2 || x.GetDim(0)<1 || x.GetDim(0)>1595 || x.GetDim(1)!=5120 ||
        w.GetDimNum()!=2 || w.GetDim(0)!=17408 || w.GetDim(1)!=5120) return ge::GRAPH_FAILED;
    auto* t = ctx->GetTilingData<ClefMlpSwigluTilingData>();
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    matmul_tiling::MatmulApiTiling mm(platform);
    using namespace matmul_tiling;
    mm.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_BFLOAT16);
    mm.SetBType(TPosition::GM, CubeFormat::NZ, DataType::DT_BFLOAT16, true);
    mm.SetCType(TPosition::VECIN, CubeFormat::ND, DataType::DT_BFLOAT16);
    mm.SetShape(64,256,5120);
    mm.SetOrgShape(x.GetDim(0),17408,5120);
    mm.SetFixSplit(64,256,64);
    mm.SetBufferSpace(-1,-1,-1);
    if (mm.GetTiling(t->cube)!=0) return ge::GRAPH_FAILED;
    const auto* stage=ctx->GetAttrs()->GetAttrPointer<int64_t>(0);
    if(stage && (*stage<0 || *stage>2)) return ge::GRAPH_FAILED;
    t->stage=stage ? static_cast<uint32_t>(*stage) : 0;
    t->tokens=x.GetDim(0); t->tasks=((t->tokens+63)/64)*68; t->cores=48;
    ctx->SetBlockDim(24);
    ctx->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext* ctx) {
    const auto* x=ctx->GetInputShape(0);
    if(x->GetDimNum()!=2) return GRAPH_FAILED;
    gert::Shape y; y.AppendDim(x->GetDim(0)); y.AppendDim(8704);
    *ctx->GetOutputShape(0)=y; return GRAPH_SUCCESS;
}
static graphStatus InferDtype(gert::InferDataTypeContext* ctx) {
    ctx->SetOutputDataType(0,DT_BF16); return GRAPH_SUCCESS;
}
}
namespace ops {
class ClefMlpSwiglu: public OpDef {
public:
 explicit ClefMlpSwiglu(const char* name):OpDef(name) {
  Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  Input("weight").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_FRACTAL_NZ}).UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
  Output("y").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  Attr("stage").Int(0);
  SetInferShape(ge::InferShape).SetInferDataType(ge::InferDtype);
  AICore().SetTiling(optiling::TilingFunc); AICore().AddConfig("ascend910_93");
 }
};
OP_ADD(ClefMlpSwiglu);
}
