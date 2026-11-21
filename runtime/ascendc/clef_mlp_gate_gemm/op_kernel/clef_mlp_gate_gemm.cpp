// Gate GEMM-only control: canonical BF16 output, original NZ weights.
// CPU preparation only. No claim of SDK compilation, numerical or speed acceptance.
#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"
#include "clef_mlp_gate_gemm_tiling.h"
using namespace AscendC;
__aicore__ inline constexpr MatmulConfig GateConfig() {auto cfg=CFG_MDL;cfg.enUnitFlag=true;return cfg;}
constexpr MatmulConfig kGateConfig=GateConfig();
using GateMM=Matmul<MatmulType<TPosition::GM,CubeFormat::ND,bfloat16_t>,
                        MatmulType<TPosition::GM,CubeFormat::NZ,bfloat16_t,true>,
                        MatmulType<TPosition::GM,CubeFormat::ND,bfloat16_t>,
                        MatmulType<TPosition::GM,CubeFormat::ND,bfloat16_t>,kGateConfig>;
extern "C" __global__ __aicore__ void clef_mlp_gate_gemm(
 GM_ADDR x,GM_ADDR w,GM_ADDR y,GM_ADDR workspace,GM_ADDR tiling) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
 if ASCEND_IS_AIV {return;}
 REGISTER_TILING_DEFAULT(ClefMlpGateGemmTilingData);
 GET_TILING_DATA(t,tiling);
 TPipe pipe;GateMM mm;
 GlobalTensor<bfloat16_t> a,b,c;
 a.SetGlobalBuffer((__gm__ bfloat16_t*)x,static_cast<uint64_t>(t.tokens)*5120);
 b.SetGlobalBuffer((__gm__ bfloat16_t*)w,static_cast<uint64_t>(17408)*5120);
 c.SetGlobalBuffer((__gm__ bfloat16_t*)y,static_cast<uint64_t>(t.tokens)*17408);
 mm.SetSubBlockIdx(0);
 REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),mm,&t.cube);
 for(uint32_t task=GetBlockIdx();task<t.mTasks*t.nTasks;task+=GetBlockNum()) {
  uint32_t m=(task%t.mTasks)*t.cube.singleCoreM;
  uint32_t n=(task/t.mTasks)*t.cube.singleCoreN;
  uint32_t rows=t.tokens-m<t.cube.singleCoreM?t.tokens-m:t.cube.singleCoreM;
  uint32_t cols=17408-n<t.cube.singleCoreN?17408-n:t.cube.singleCoreN;
  mm.SetOrgShape(t.tokens,17408,5120,5120,17408);
  mm.SetSingleShape(rows,cols,5120);
  mm.SetTensorA(a[static_cast<uint64_t>(m)*5120]);
  mm.SetTensorB(b[static_cast<uint64_t>(n)*16],true);
  // One complete core task, with API-managed base blocks/L1 reuse; no AIV client loop.
  mm.IterateAll(c[static_cast<uint64_t>(m)*17408+n]);
  mm.End();
 }
}
