#include "kernel_operator.h"
#include "clef_gdn_qkv_norm_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kHeads = 40;
constexpr uint32_t kKHeads = 8;
constexpr uint32_t kVHeads = 24;
constexpr uint32_t kDim = 128;

class ClefGdnQkvNormKernel {
public:
    __aicore__ inline ClefGdnQkvNormKernel() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                const ClefGdnQkvNormTilingData* tiling) {
        totalTokens_ = tiling->totalTokens;
        tileTokens_ = tiling->tileTokens;
        tilesPerHead_ = tiling->tilesPerHead;
        totalTiles_ = tiling->totalTiles;
        tilesPerCore_ = tiling->tilesPerCore;
        eps_ = tiling->eps;
        xGm_.SetGlobalBuffer((__gm__ bfloat16_t*)x, (uint64_t)kHeads * totalTokens_ * kDim);
        qGm_.SetGlobalBuffer((__gm__ bfloat16_t*)q, (uint64_t)kKHeads * totalTokens_ * kDim);
        kGm_.SetGlobalBuffer((__gm__ bfloat16_t*)k, (uint64_t)kKHeads * totalTokens_ * kDim);
        vGm_.SetGlobalBuffer((__gm__ bfloat16_t*)v, (uint64_t)kVHeads * totalTokens_ * kDim);
        pipe_.InitBuffer(inQueue_, 1, tileTokens_ * kDim * sizeof(bfloat16_t));
        pipe_.InitBuffer(outQueue_, 1, tileTokens_ * kDim * sizeof(bfloat16_t));
        pipe_.InitBuffer(fp32Buf_, tileTokens_ * kDim * sizeof(float) + 512);
        pipe_.InitBuffer(sqBuf_, tileTokens_ * kDim * sizeof(float) + 512);
        pipe_.InitBuffer(sumBuf_, tileTokens_ * 2 * sizeof(float) + 512);
        pipe_.InitBuffer(tmpBuf_, 16 * 1024 + 512);
    }

    __aicore__ inline void Process() {
        uint32_t block = GetBlockIdx();
        uint32_t begin = block * tilesPerCore_;
        if (begin >= totalTiles_) {
            return;
        }
        uint32_t end = begin + tilesPerCore_;
        if (end > totalTiles_) {
            end = totalTiles_;
        }
        for (uint32_t tile = begin; tile < end; ++tile) {
            uint32_t head = tile / tilesPerHead_;
            uint32_t tileInHead = tile - head * tilesPerHead_;
            uint32_t t0 = tileInHead * tileTokens_;
            uint32_t rows = totalTokens_ - t0;
            if (rows > tileTokens_) {
                rows = tileTokens_;
            }
            uint32_t inOff = head * totalTokens_ * kDim + t0 * kDim;
            if (head < kKHeads) {
                ProcessTile(qGm_, inOff, head, kKHeads, t0, rows, true);
            } else if (head < 2 * kKHeads) {
                ProcessTile(kGm_, inOff, head - kKHeads, kKHeads, t0, rows, true);
            } else {
                ProcessTile(vGm_, inOff, head - 2 * kKHeads, kVHeads, t0, rows, false);
            }
        }
    }

private:
    __aicore__ inline void ProcessTile(const GlobalTensor<bfloat16_t>& out, uint32_t inOff,
                                       uint32_t outHead, uint32_t outHeads,
                                       uint32_t t0, uint32_t rows, bool normalize) {
        LocalTensor<bfloat16_t> xLocal = inQueue_.AllocTensor<bfloat16_t>();
        DataCopy(xLocal, xGm_[inOff], rows * kDim);
        inQueue_.EnQue(xLocal);
        LocalTensor<bfloat16_t> xin = inQueue_.DeQue<bfloat16_t>();
        LocalTensor<float> xfRaw = fp32Buf_.Get<float>();
        LocalTensor<float> xf =
            xfRaw[static_cast<uint32_t>((512 - (xfRaw.GetPhyAddr() & 511)) & 511) / sizeof(float)];
        Cast(xf, xin, RoundMode::CAST_NONE, rows * kDim);
        inQueue_.FreeTensor(xin);
        PipeBarrier<PIPE_V>();

        if (normalize) {
            LocalTensor<float> sqRaw = sqBuf_.Get<float>();
            LocalTensor<float> sq =
                sqRaw[static_cast<uint32_t>((512 - (sqRaw.GetPhyAddr() & 511)) & 511) / sizeof(float)];
            Mul(sq, xf, xf, rows * kDim);
            PipeBarrier<PIPE_V>();
            LocalTensor<float> sumsRaw = sumBuf_.Get<float>();
            LocalTensor<float> sums =
                sumsRaw[static_cast<uint32_t>((512 - (sumsRaw.GetPhyAddr() & 511)) & 511) / sizeof(float)];
            LocalTensor<float> workRaw = tmpBuf_.Get<float>();
            LocalTensor<float> work =
                workRaw[static_cast<uint32_t>((512 - (workRaw.GetPhyAddr() & 511)) & 511) / sizeof(float)];
            LocalTensor<float> sums2 =
                sumsRaw[static_cast<uint32_t>((512 - (sumsRaw.GetPhyAddr() & 511)) & 511) / sizeof(float) + 128];
            for (uint32_t row = 0; row < rows; ++row) {
                ReduceSum(sums[row], sq[row * kDim], work, 64);
                ReduceSum(sums2[row], sq[row * kDim + 64], work, 64);
            }
            PipeBarrier<PIPE_V>();
            Add(sums, sums, sums2, rows);
            PipeBarrier<PIPE_V>();
            Adds(sums, sums, eps_, rows);
            PipeBarrier<PIPE_V>();
            Rsqrt(sums, sums, rows);
            PipeBarrier<PIPE_V>();
            event_t eventVS = static_cast<event_t>(
                GetTPipePtr()->FetchEventID(HardEvent::V_S));
            SetFlag<HardEvent::V_S>(eventVS);
            WaitFlag<HardEvent::V_S>(eventVS);
            for (uint32_t row = 0; row < rows; ++row) {
                float scale = sums.GetValue(row);
                Muls(xf[row * kDim], xf[row * kDim], scale, kDim);
            }
            PipeBarrier<PIPE_V>();
        }

        LocalTensor<bfloat16_t> yLocal = outQueue_.AllocTensor<bfloat16_t>();
        Cast(yLocal, xf, RoundMode::CAST_RINT, rows * kDim);
        outQueue_.EnQue(yLocal);
        LocalTensor<bfloat16_t> yout = outQueue_.DeQue<bfloat16_t>();
        DataCopyExtParams params;
        params.blockCount = static_cast<uint16_t>(rows);
        params.blockLen = kDim * sizeof(bfloat16_t);
        params.srcStride = 0;
        params.dstStride = (outHeads - 1) * kDim * sizeof(bfloat16_t);
        params.rsv = 0;
        uint32_t outOff = t0 * outHeads * kDim + outHead * kDim;
        DataCopyPad(out[outOff], yout, params);
        outQueue_.FreeTensor(yout);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 1> inQueue_;
    TQue<TPosition::VECOUT, 1> outQueue_;
    TBuf<TPosition::VECCALC> fp32Buf_;
    TBuf<TPosition::VECCALC> sqBuf_;
    TBuf<TPosition::VECCALC> sumBuf_;
    TBuf<TPosition::VECCALC> tmpBuf_;
    GlobalTensor<bfloat16_t> xGm_;
    GlobalTensor<bfloat16_t> qGm_;
    GlobalTensor<bfloat16_t> kGm_;
    GlobalTensor<bfloat16_t> vGm_;
    uint32_t totalTokens_;
    uint32_t tileTokens_;
    uint32_t tilesPerHead_;
    uint32_t totalTiles_;
    uint32_t tilesPerCore_;
    float eps_;
};
}  // namespace

extern "C" __global__ __aicore__ void clef_gdn_qkv_norm(
    GM_ADDR x, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClefGdnQkvNormTilingData);
    GET_TILING_DATA(tilingData, tiling);
    ClefGdnQkvNormKernel op;
    op.Init(x, q, k, v, &tilingData);
    op.Process();
}
