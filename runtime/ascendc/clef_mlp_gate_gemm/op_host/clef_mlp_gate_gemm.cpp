// Gate-only diagnostic; validation evidence is recorded in docs/mlp-bf16-screen-2026-11-21.md.
#include "../op_kernel/clef_mlp_gate_gemm_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* ctx) {
    const auto& x=ctx->GetInputShape(0)->GetOriginShape();
    const auto& w=ctx->GetInputShape(1)->GetOriginShape();
    if(x.GetDimNum()!=2 || x.GetDim(0)<1 || x.GetDim(0)>1595 || x.GetDim(1)!=5120 ||
       w.GetDimNum()!=2 || w.GetDim(0)!=17408 || w.GetDim(1)!=5120) return ge::GRAPH_FAILED;
    auto* t=ctx->GetTilingData<ClefMlpGateGemmTilingData>();
    platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
    matmul_tiling::MultiCoreMatmulTiling mm(platform);
    using namespace matmul_tiling;
    mm.SetDim(platform.GetCoreNumAic());
    mm.SetAType(TPosition::GM,CubeFormat::ND,DataType::DT_BFLOAT16);
    mm.SetBType(TPosition::GM,CubeFormat::NZ,DataType::DT_BFLOAT16,true);
    mm.SetCType(TPosition::GM,CubeFormat::ND,DataType::DT_BFLOAT16);
    mm.SetShape(x.GetDim(0),17408,5120);
    mm.SetOrgShape(x.GetDim(0),17408,5120);
    mm.EnableBias(false);
    mm.SetBufferSpace(-1,-1,-1);
    // Preserve the API's complete L1/L0 schedule; no fixed base tile or padding sweep.
    if(mm.GetTiling(t->cube)!=0 || t->cube.singleCoreM==0 || t->cube.singleCoreN==0 ||
       t->cube.usedCoreNum==0 || t->cube.baseN==0 || t->cube.baseN%16!=0) return ge::GRAPH_FAILED;
    // MultiCore tiling may return an unaligned N partition (967: 5803).
    // NZ-transposed B offsets require a whole 16-row block. Align only the
    // partition boundary; base tiles and the number of base blocks stay fixed.
    const uint32_t oldN=t->cube.singleCoreN;
    const uint32_t alignedN=(oldN+15)/16*16;
    if((oldN+t->cube.baseN-1)/t->cube.baseN !=
       (alignedN+t->cube.baseN-1)/t->cube.baseN) return ge::GRAPH_FAILED;
    t->cube.singleCoreN=alignedN;
    t->tokens=x.GetDim(0);
    t->mTasks=(t->tokens+t->cube.singleCoreM-1)/t->cube.singleCoreM;
    t->nTasks=(17408+t->cube.singleCoreN-1)/t->cube.singleCoreN;
    ctx->SetBlockDim(t->cube.usedCoreNum);
    ctx->GetWorkspaceSizes(1)[0]=platform.GetLibApiWorkSpaceSize();
    return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext* ctx) {
    const auto* x=ctx->GetInputShape(0);if(x->GetDimNum()!=2) return GRAPH_FAILED;
    gert::Shape y;y.AppendDim(x->GetDim(0));y.AppendDim(17408);
    *ctx->GetOutputShape(0)=y;return GRAPH_SUCCESS;
}
static graphStatus InferDtype(gert::InferDataTypeContext* ctx) {
    ctx->SetOutputDataType(0,DT_BF16);return GRAPH_SUCCESS;
}
}
namespace ops {
class ClefMlpGateGemm:public OpDef {
public:
 explicit ClefMlpGateGemm(const char* name):OpDef(name) {
  Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  Input("weight").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_FRACTAL_NZ}).UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
  Output("y").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  SetInferShape(ge::InferShape).SetInferDataType(ge::InferDtype);
  AICore().SetTiling(optiling::TilingFunc);AICore().AddConfig("ascend910_93");
 }
};
OP_ADD(ClefMlpGateGemm);
}
