// Fixed CATLASS eec373fb official preload block; only cross-output first-K preload
// is switched. Its K-slice ping-pong/MMAD/FP32 accumulator/Fixpipe are unmodified.
#include "kernel_operator.h"
#include "clef_mlp_cross_prefetch_tiling.h"
#include "catlass/gemm/block/block_mmad.hpp"
#include "catlass/gemm/block/block_swizzle.hpp"
using namespace Catlass;
using AType=Gemm::GemmType<bfloat16_t,layout::RowMajor>;
using BType=Gemm::GemmType<bfloat16_t,layout::nZ>;
using CType=Gemm::GemmType<bfloat16_t,layout::RowMajor>;
using Policy=Gemm::MmadAtlasA2Preload<true,false>; // unit flag ON, K shuffle OFF
using L1Shape=GemmShape<128,256,256>;
using L0Shape=GemmShape<128,256,64>;
using Block=Gemm::Block::BlockMmad<Policy,L1Shape,L0Shape,AType,BType,CType>;
using Scheduler=Gemm::Block::GemmIdentityBlockSwizzle<3,1>;
static_assert(std::is_same_v<typename Block::ElementAccumulator,float>);
extern "C" __global__ __aicore__ void clef_mlp_cross_prefetch(
 GM_ADDR x,GM_ADDR weight,GM_ADDR y,GM_ADDR workspace,GM_ADDR tiling) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
 if ASCEND_IS_AIV {return;}
 REGISTER_TILING_DEFAULT(ClefMlpCrossPrefetchTilingData);
 GET_TILING_DATA(t,tiling);
 Arch::Resource<Arch::AtlasA2> resource;
 Block block(resource);
 auto la=layout::RowMajor(t.tokens,5120);
 auto lb=layout::nZ::MakeLayout<bfloat16_t>(5120,17408);
 auto lc=layout::RowMajor(t.tokens,17408);
 AscendC::GlobalTensor<bfloat16_t> a,b,c;
 a.SetGlobalBuffer((__gm__ bfloat16_t*)x);
 b.SetGlobalBuffer((__gm__ bfloat16_t*)weight);
 c.SetGlobalBuffer((__gm__ bfloat16_t*)y);
 Scheduler schedule(GemmCoord(t.tokens,17408,5120),MatrixCoord(128U,256U));
 const uint32_t count=schedule.GetCoreLoops();
 const bool preload=t.prefetch!=0;
 for(uint32_t id=AscendC::GetBlockIdx();id<count;id+=AscendC::GetBlockNum()) {
  auto coord=schedule.GetBlockCoord(id);
  auto shape=schedule.GetActualBlockShape(coord);
  // Keep next shape valid even on the final block: official block computes
  // firstTileIdxNext unconditionally. No invalid address/modulo-zero sentinel.
  auto nextCoord=coord;auto nextShape=shape;
  const bool hasNext=id+AscendC::GetBlockNum()<count;
  if(hasNext) {nextCoord=schedule.GetBlockCoord(id+AscendC::GetBlockNum());nextShape=schedule.GetActualBlockShape(nextCoord);}
  const auto oa=la.GetOffset(MatrixCoord(coord.m()*128,0));
  const auto ob=lb.GetOffset(MatrixCoord(0,coord.n()*256));
  const auto oc=lc.GetOffset(MatrixCoord(coord.m()*128,coord.n()*256));
  const auto na=la.GetOffset(MatrixCoord(nextCoord.m()*128,0));
  const auto nb=lb.GetOffset(MatrixCoord(0,nextCoord.n()*256));
  // OFF: first-K loads at every block entry, no cross-block preload.
  // ON: first block loads at entry; last K slice preloads next block's first K.
  // Both arms retain all official per-buffer MTE1/MTE2/M/FIX handshakes.
  block(a[oa],la,b[ob],lb,c[oc],lc,a[na],b[nb],shape,nextShape,
        !preload || id==AscendC::GetBlockIdx(),preload && hasNext);
 }
 AscendC::PipeBarrier<PIPE_ALL>();
}
