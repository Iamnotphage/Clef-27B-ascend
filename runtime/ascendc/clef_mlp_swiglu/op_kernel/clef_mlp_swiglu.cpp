#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "clef_mlp_swiglu_tiling.h"
using namespace AscendC;
using MM = Matmul<MatmulType<TPosition::GM,CubeFormat::ND,bfloat16_t>,
                  MatmulType<TPosition::GM,CubeFormat::NZ,bfloat16_t,true>,
                  MatmulType<TPosition::VECIN,CubeFormat::ND,bfloat16_t>>;
class Kernel {
public:
 MM mm;
 __aicore__ inline void Init(GM_ADDR x,GM_ADDR w,GM_ADDR y,TPipe* p,const ClefMlpSwigluTilingData* t) {
  t_=t;
  x_.SetGlobalBuffer((__gm__ bfloat16_t*)x); w_.SetGlobalBuffer((__gm__ bfloat16_t*)w);
  y_.SetGlobalBuffer((__gm__ bfloat16_t*)y);
  p->InitBuffer(c_,64*256*2); p->InitBuffer(g_,64*128*4);
  p->InitBuffer(u_,64*128*4); p->InitBuffer(v_,64*128*4);
  p->InitBuffer(out_,1,64*128*2);
 }
 __aicore__ inline void Process() {
  // Both AIV clients participate; each owns disjoint compact output tiles.
  const uint32_t core=GetBlockIdx();
  for(uint32_t task=core;task<t_->tasks;task+=t_->cores) {
   uint32_t m=(task/68)*64,n=(task%68)*256;
   uint32_t rows=t_->tokens-m<64?t_->tokens-m:64;
   // B retains its full NZ leading dimension; local C is dense [64,256].
   // End closes each tile, so reapply the logical strides for every tile.
   mm.SetOrgShape(t_->tokens,17408,5120,5120,256);
   mm.SetSingleShape(rows,256,5120);
   mm.SetTensorA(x_[static_cast<uint64_t>(m)*5120]);
   // FRACTAL_NZ [K/16,N/16,N0,K0], row offset in transposed B.
   mm.SetTensorB(w_[static_cast<uint64_t>(n)*16],true);
   mm.Iterate();
   auto c=c_.Get<bfloat16_t>(); mm.GetTensorC(c,0,true);
   auto out=out_.AllocTensor<bfloat16_t>();
   auto g=g_.Get<float>(),u=u_.Get<float>(),v=v_.Get<float>();
   // Diagnostic stage 1/2 writes the compact gate/up GEMM result directly.
   // stage=0 executes the intended BF16-rounded SwiGLU epilogue.
   DataCopy(out,c,{64,8,8,0}); PipeBarrier<PIPE_V>();
   if(t_->stage!=1) {
    if(t_->stage==0) {
     Cast(g,out,RoundMode::CAST_NONE,8192); PipeBarrier<PIPE_V>();
    }
    DataCopy(out,c[128],{64,8,8,0}); PipeBarrier<PIPE_V>();
    if(t_->stage==0) {
     Cast(u,out,RoundMode::CAST_NONE,8192); PipeBarrier<PIPE_V>();
     Silu(v,g,8192); PipeBarrier<PIPE_V>();
     Mul(v,v,u,8192); PipeBarrier<PIPE_V>();
     Cast(out,v,RoundMode::CAST_RINT,8192); PipeBarrier<PIPE_V>();
    }
   }
   out_.EnQue(out); auto result=out_.DeQue<bfloat16_t>();
   DataCopyExtParams copy{static_cast<uint16_t>(rows),256,0,(8704-128)*2,0};
   DataCopyPad(y_[static_cast<uint64_t>(m)*8704+n/2],result,copy);
   out_.FreeTensor(result);
   mm.End();
  }
 }
private:
 const ClefMlpSwigluTilingData* t_;
 GlobalTensor<bfloat16_t> x_,w_,y_;
 TBuf<TPosition::VECIN> c_;
 TBuf<TPosition::VECCALC> g_,u_,v_;
 TQue<TPosition::VECOUT,1> out_;
};
extern "C" __global__ __aicore__ void clef_mlp_swiglu(GM_ADDR x,GM_ADDR w,GM_ADDR y,GM_ADDR workspace,GM_ADDR tiling) {
 KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
 REGISTER_TILING_DEFAULT(ClefMlpSwigluTilingData);
 GET_TILING_DATA(tilingData,tiling);
 TPipe pipe; Kernel op;
 REGIST_MATMUL_OBJ(&pipe,GetSysWorkSpacePtr(),op.mm,&tilingData.cube);
 if ASCEND_IS_AIV { op.Init(x,w,y,&pipe,&tilingData); op.Process(); }
}
