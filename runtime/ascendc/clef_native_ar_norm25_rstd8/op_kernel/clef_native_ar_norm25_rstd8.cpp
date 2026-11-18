#include "kernel_operator.h"
#include "adv_api/hccl/hccl.h"
#include "clef_native_ar_norm25_rstd8_tiling.h"

using namespace AscendC;

// N1 boundary: native GEMM -> explicit BF16 partial -> this AIV-only kernel.
// No Matmul/KFC client, shared GM scalar stores, or persistent completion flags.
class NativeArNorm25Rstd8 {
public:
    __aicore__ inline void Init(GM_ADDR partial, GM_ADDR residual, GM_ADDR gamma,
        GM_ADDR y, GM_ADDR resOut, GM_ADDR rstd, GM_ADDR reduced, GM_ADDR diag,
        const ClefNativeArNorm25Rstd8TilingData* t) {
        t_ = t; partial_ = partial; reduced_ = reduced;
        const uint64_t count = static_cast<uint64_t>(t->m) * t->n;
        p_.SetGlobalBuffer((__gm__ bfloat16_t*)partial, count);
        x_.SetGlobalBuffer((__gm__ bfloat16_t*)(t->useHccl ? reduced : partial), count);
        r_.SetGlobalBuffer((__gm__ bfloat16_t*)residual, count);
        g_.SetGlobalBuffer((__gm__ bfloat16_t*)gamma, t->n);
        y_.SetGlobalBuffer((__gm__ bfloat16_t*)y, count);
        ro_.SetGlobalBuffer((__gm__ bfloat16_t*)resOut, count);
        red_.SetGlobalBuffer((__gm__ bfloat16_t*)reduced, count);
        rs_.SetGlobalBuffer((__gm__ float*)rstd, t->m);
        diag_.SetGlobalBuffer((__gm__ uint32_t*)diag, 48 * 32);
        // 103072 bytes < 192 KiB UB: v2 buffers plus one 32B rstd staging tile.
        pipe_.InitBuffer(bx_, 5120 * 2);
        pipe_.InitBuffer(br_, 5120 * 2);
        pipe_.InitBuffer(bg_, 5120 * 2);
        pipe_.InitBuffer(bo_, 5120 * 2);
        pipe_.InitBuffer(bz_, 5120 * 4);
        pipe_.InitBuffer(bt_, 5120 * 4);
        pipe_.InitBuffer(bgf_, 5120 * 4);
        pipe_.InitBuffer(bsum_, 64 * 4);
        pipe_.InitBuffer(bstat_, 64 * 4);
        pipe_.InitBuffer(bdiag_, 128);
        pipe_.InitBuffer(brstd_, 8 * sizeof(float));
    }

    __aicore__ inline void Process() {
        int32_t rank = -1, ranks = -1, setRc = 0, waitRc = 0;
        if (GetBlockIdx() == 0) {
            Hccl hccl;
            hccl.InitV2(GetHcclContext<HCCL_GROUP_ID_0>(), &t_->mc2InitTiling);
            setRc = hccl.SetCcTilingV2(offsetof(ClefNativeArNorm25Rstd8TilingData, mc2CcTiling));
            rank = hccl.GetRankId(); ranks = hccl.GetRankDim();
            if (t_->useHccl && setRc == 0) {
                auto handle = hccl.AllReduce<true>(partial_, reduced_,
                    static_cast<uint64_t>(t_->m) * t_->n,
                    HCCL_DATA_TYPE_BFP16, HCCL_REDUCE_SUM, 1);
                waitRc = hccl.Wait(handle);
            }
            // Generated NnopbaseSetMc2 is unconditional: the ARoff arm
            // closes the same communication lifecycle without issuing AR.
            hccl.Finalize();
        }
        // All launched AIVs reach exactly one FFTS barrier, including idle rows.
        // Wait completion and GM visibility still require the two-rank device gate.
        SyncAll<true>();
        if (t_->useHccl) {
            DataCacheCleanAndInvalid<bfloat16_t, CacheLine::ENTIRE_DATA_CACHE,
                                     DcciDst::CACHELINE_OUT>(x_);
            PipeBarrier<PIPE_ALL>();
        }

        auto gl = bg_.Get<bfloat16_t>();
        auto gf = bgf_.Get<float>();
        DataCopy(gl, g_, 5120);
        PipeBarrier<PIPE_ALL>();
        Cast(gf, gl, RoundMode::CAST_NONE, 5120);
        PipeBarrier<PIPE_V>();

        const uint32_t start = static_cast<uint64_t>(t_->m) * GetBlockIdx() / t_->cores;
        const uint32_t end = static_cast<uint64_t>(t_->m) * (GetBlockIdx() + 1) / t_->cores;
        for (uint32_t row = start; row < end; ++row)
            Row(row, gf, (row - start) % 8, row + 1 == end);
        // Only UB scalar writes; all 32 words get a single 128B DMA publication.
        auto d = bdiag_.Get<uint32_t>();
        for (uint32_t j = 0; j < 32; ++j) d.SetValue(j, 0u);
        d.SetValue(0, 0x4e312519u); d.SetValue(1, 1u);
        d.SetValue(2, GetBlockIdx()); d.SetValue(3, GetBlockNum());
        d.SetValue(4, start); d.SetValue(5, end);
        d.SetValue(6, t_->useHccl); d.SetValue(7, static_cast<uint32_t>(rank));
        d.SetValue(8, static_cast<uint32_t>(ranks));
        d.SetValue(9, static_cast<uint32_t>(setRc));
        d.SetValue(10, static_cast<uint32_t>(waitRc));
        d.SetValue(11, t_->m); d.SetValue(12, t_->n);
        uint32_t hash = 2166136261u;
        for (uint32_t j = 0; j < 31; ++j) hash = (hash ^ d.GetValue(j)) * 16777619u;
        d.SetValue(31, hash);
        PipeBarrier<PIPE_ALL>();
        DataCopy(diag_[GetBlockIdx() * 32], d, 32);
        PipeBarrier<PIPE_ALL>();
    }

private:
    __aicore__ inline void Row(uint32_t row, LocalTensor<float> gf,
        uint32_t rstdSlot, bool lastRow) {
        const uint64_t base = static_cast<uint64_t>(row) * 5120;
        auto xl = bx_.Get<bfloat16_t>(); auto rl = br_.Get<bfloat16_t>();
        auto ol = bo_.Get<bfloat16_t>();
        auto z = bz_.Get<float>(); auto tmp = bt_.Get<float>();
        auto lanes = bsum_.Get<float>(); auto stat = bstat_.Get<float>();
        DataCopy(xl, x_[base], 5120); DataCopy(rl, r_[base], 5120);
        PipeBarrier<PIPE_ALL>();
        if (!t_->useHccl) {
            DataCopy(red_[base], xl, 5120);
            PipeBarrier<PIPE_ALL>();
        }
        Cast(z, xl, RoundMode::CAST_NONE, 5120);
        Cast(tmp, rl, RoundMode::CAST_NONE, 5120);
        PipeBarrier<PIPE_V>();
        Add(z, z, tmp, 5120);
        PipeBarrier<PIPE_V>();
        Cast(ol, z, RoundMode::CAST_RINT, 5120);
        PipeBarrier<PIPE_ALL>();
        DataCopy(ro_[base], ol, 5120);
        PipeBarrier<PIPE_ALL>();

        // CANN 9.2.1 AddRmsNorm BF16 order: square -> scale EACH term -> reduce.
        Mul(tmp, z, z, 5120);
        PipeBarrier<PIPE_V>();
        Muls(tmp, tmp, t_->invN, 5120);
        PipeBarrier<PIPE_V>();
        // ReduceSumFP32 from the SDK rms_norm_base.h: 80 repeats accumulating
        // 64 lanes, followed by one WholeReduceSum (same native reduction order).
        Duplicate(lanes, 0.0f, 64);
        PipeBarrier<PIPE_V>();
        BinaryRepeatParams rp;
        rp.src0RepStride = 8; rp.src0BlkStride = 1;
        rp.src1RepStride = 0; rp.src1BlkStride = 1;
        rp.dstRepStride = 0; rp.dstBlkStride = 1;
        Add(lanes, tmp, lanes, 64, 80, rp);
        PipeBarrier<PIPE_V>();
        WholeReduceSum(stat, lanes, 64, 1, 0, 1, 0);
        PipeBarrier<PIPE_V>();
        Adds(stat, stat, t_->eps, 1);
        PipeBarrier<PIPE_V>();
        Sqrt(stat, stat, 1);
        Duplicate(lanes, 1.0f, 1);
        PipeBarrier<PIPE_V>();
        Div(stat, lanes, stat, 1);
        PipeBarrier<PIPE_ALL>();
        const float inv = stat.GetValue(0);
        // Preserve the computed FP32 bit pattern; only GM publication is grouped.
        // Tile origin is this core's start, so even an unaligned core boundary
        // writes exactly its own rows. UB source always starts at the 32B base.
        auto rstdTile = brstd_.Get<float>();
        rstdTile.SetValue(rstdSlot, inv);
        if (rstdSlot == 7 || lastRow) {
            const auto publish = static_cast<event_t>(pipe_.FetchEventID(HardEvent::S_MTE3));
            SetFlag<HardEvent::S_MTE3>(publish);
            WaitFlag<HardEvent::S_MTE3>(publish);
            DataCopyExtParams cp{1, static_cast<uint32_t>((rstdSlot + 1) * sizeof(float)), 0, 0, 0};
            DataCopyPad(rs_[row - rstdSlot], rstdTile, cp);
            // Finish DMA before the next tile reuses any scalar-written UB word.
            const auto reuse = static_cast<event_t>(pipe_.FetchEventID(HardEvent::MTE3_S));
            SetFlag<HardEvent::MTE3_S>(reuse);
            WaitFlag<HardEvent::MTE3_S>(reuse);
        }
        PipeBarrier<PIPE_ALL>();
        Muls(z, z, inv, 5120);
        PipeBarrier<PIPE_V>();
        Cast(ol, z, RoundMode::CAST_RINT, 5120);
        PipeBarrier<PIPE_V>();
        Cast(z, ol, RoundMode::CAST_NONE, 5120);
        PipeBarrier<PIPE_V>();
        Mul(z, z, gf, 5120);
        PipeBarrier<PIPE_V>();
        Cast(ol, z, RoundMode::CAST_RINT, 5120);
        PipeBarrier<PIPE_ALL>();
        DataCopy(y_[base], ol, 5120);
        PipeBarrier<PIPE_ALL>();
    }

    TPipe pipe_;
    TBuf<TPosition::VECCALC> bx_, br_, bg_, bo_, bz_, bt_, bgf_, bsum_, bstat_, bdiag_, brstd_;
    GlobalTensor<bfloat16_t> p_, x_, r_, g_, y_, ro_, red_;
    GlobalTensor<float> rs_;
    GlobalTensor<uint32_t> diag_;
    GM_ADDR partial_;
    GM_ADDR reduced_;
    const ClefNativeArNorm25Rstd8TilingData* t_;
};

extern "C" __global__ __aicore__ void clef_native_ar_norm25_rstd8(
    GM_ADDR partial, GM_ADDR residual, GM_ADDR gamma,
    GM_ADDR y, GM_ADDR residualOut, GM_ADDR rstd, GM_ADDR reduced, GM_ADDR diag,
    GM_ADDR workspace, GM_ADDR tiling) {
    // FFTS hard synchronization requires this vector-only mix launch type.
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIV_1_0);
    REGISTER_TILING_DEFAULT(ClefNativeArNorm25Rstd8TilingData);
    GET_TILING_DATA(t, tiling);
    NativeArNorm25Rstd8 op;
    op.Init(partial, residual, gamma, y, residualOut, rstd, reduced, diag, &t);
    op.Process();
}
