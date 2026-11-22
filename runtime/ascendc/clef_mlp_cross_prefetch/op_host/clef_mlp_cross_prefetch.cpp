#include "../op_kernel/clef_mlp_cross_prefetch_tiling.h"
#include "register/op_def_registry.h"
#include "tiling/platform/platform_ascendc.h"
namespace optiling {
static ge::graphStatus TilingFunc(gert::TilingContext* ctx) {
 const auto& x=ctx->GetInputShape(0)->GetOriginShape();
 const auto& w=ctx->GetInputShape(1)->GetOriginShape();
 if(x.GetDimNum()!=2 || x.GetDim(0)<1 || x.GetDim(0)>1595 || x.GetDim(1)!=5120 ||
    w.GetDimNum()!=2 || w.GetDim(0)!=17408 || w.GetDim(1)!=5120) return ge::GRAPH_FAILED;
 const auto* value=ctx->GetAttrs()->GetInt(0);
 if(value==nullptr || (*value!=0 && *value!=1)) return ge::GRAPH_FAILED;
 auto* t=ctx->GetTilingData<ClefMlpCrossPrefetchTilingData>();
 platform_ascendc::PlatformAscendC platform(ctx->GetPlatformInfo());
 t->tokens=x.GetDim(0);t->cores=platform.GetCoreNumAic();t->prefetch=*value;t->version=1;
 if(t->cores==0) return ge::GRAPH_FAILED;
 ctx->SetBlockDim(t->cores);ctx->GetWorkspaceSizes(1)[0]=0;
 return ge::GRAPH_SUCCESS;
}
}
namespace ge {
static graphStatus InferShape(gert::InferShapeContext* ctx) {
 const auto* x=ctx->GetInputShape(0);if(x->GetDimNum()!=2) return GRAPH_FAILED;
 gert::Shape y;y.AppendDim(x->GetDim(0));y.AppendDim(17408);*ctx->GetOutputShape(0)=y;return GRAPH_SUCCESS;
}
static graphStatus InferDtype(gert::InferDataTypeContext* ctx) {ctx->SetOutputDataType(0,DT_BF16);return GRAPH_SUCCESS;}
}
namespace ops {
class ClefMlpCrossPrefetch:public OpDef {
public:
 explicit ClefMlpCrossPrefetch(const char* name):OpDef(name) {
  Input("x").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  Input("weight").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_FRACTAL_NZ}).UnknownShapeFormat({ge::FORMAT_FRACTAL_NZ});
  Output("y").ParamType(REQUIRED).DataType({ge::DT_BF16}).Format({ge::FORMAT_ND}).UnknownShapeFormat({ge::FORMAT_ND});
  Attr("prefetch").AttrType(OPTIONAL).Int(0);
  SetInferShape(ge::InferShape).SetInferDataType(ge::InferDtype);
  AICore().SetTiling(optiling::TilingFunc);AICore().AddConfig("ascend910_93");
 }
};
OP_ADD(ClefMlpCrossPrefetch);
}
