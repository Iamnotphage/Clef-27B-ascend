#include "kernel_operator.h"
#include "clef_gdn_conv_qkv_varlen_tiling.h"

using namespace AscendC;
namespace {
constexpr uint32_t kHeads = 40;
constexpr uint32_t kDim = 128;
constexpr uint32_t kChannels = kHeads * kDim;
constexpr uint32_t kMaxSeqs = 64;

// Per-sequence tile decomposition: every task owns one head and a contiguous
// token tile inside one sequence.  Cross-task data comes only from the
// immutable input, so a poisoned or NaN neighbour can never leak into another
// sequence.  The vector body below is the accepted v4 body, unchanged.
class ClefGdnConvQkvVarlenKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR cu_seqlens,
                                GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                const ClefGdnConvQkvVarlenTilingData* t) {
        tokens_ = t->totalTokens;
        tile_ = t->tileTokens;
        numSeqs_ = t->numSeqs > kMaxSeqs ? kMaxSeqs : t->numSeqs;
        eps_ = t->eps;
        inputChannels_ = t->inputChannels;
        x_.SetGlobalBuffer((__gm__ bfloat16_t*)x,
                           static_cast<uint64_t>(tokens_) * inputChannels_);
        weight_.SetGlobalBuffer((__gm__ bfloat16_t*)weight, 4 * kChannels);
        cuGm_.SetGlobalBuffer((__gm__ int32_t*)cu_seqlens, numSeqs_ + 1);
        q_.SetGlobalBuffer((__gm__ bfloat16_t*)q,
                           static_cast<uint64_t>(tokens_) * 8 * kDim);
        k_.SetGlobalBuffer((__gm__ bfloat16_t*)k,
                           static_cast<uint64_t>(tokens_) * 8 * kDim);
        v_.SetGlobalBuffer((__gm__ bfloat16_t*)v,
                           static_cast<uint64_t>(tokens_) * 24 * kDim);
        pipe_.InitBuffer(in_, 1, (tile_ + 7) * kDim * sizeof(bfloat16_t));
        pipe_.InitBuffer(out_, 1, tile_ * kDim * sizeof(bfloat16_t));
        pipe_.InitBuffer(xf_, (tile_ + 3) * kDim * sizeof(float) + 512);
        pipe_.InitBuffer(wf_, 4 * kDim * sizeof(float) + 512);
        pipe_.InitBuffer(acc_, tile_ * kDim * sizeof(float) + 512);
        pipe_.InitBuffer(tmp_, tile_ * kDim * sizeof(float) + 512);
        pipe_.InitBuffer(sums_, tile_ * 2 * sizeof(float) + 512);
        pipe_.InitBuffer(work_, 16 * 1024 + 512);
        pipe_.InitBuffer(cuBuf_, (kMaxSeqs + 2) * sizeof(int32_t) + 512);
        LoadSeqBoundaries();
    }

    __aicore__ inline void Process() {
        for (uint32_t task = GetBlockIdx(); task < totalTiles_;
             task += GetBlockNum()) {
            const uint32_t head = task % kHeads;
            const uint32_t seqTile = task / kHeads;
            uint32_t seq = 0;
            while (seq + 1 <= numSeqs_ && seqTile >= tileOff_[seq + 1]) {
                ++seq;
            }
            if (seq >= numSeqs_) {
                continue;
            }
            const uint32_t seqStart = cu_[seq];
            const uint32_t seqEnd = cu_[seq + 1];
            if (seqEnd <= seqStart) {
                continue;
            }
            const uint32_t tileStart = seqStart + (seqTile - tileOff_[seq]) * tile_;
            if (tileStart >= seqEnd) {
                continue;
            }
            uint32_t rows = seqEnd - tileStart;
            if (rows > tile_) {
                rows = tile_;
            }
            ProcessTile(head, seqStart, tileStart, rows);
        }
    }

private:
    // MTE2 -> Scalar is a distinct hardware dependency: PipeBarrier<PIPE_MTE2>
    // is not enough.  Copy the <= 65 boundaries into UB and wait on the
    // explicit MTE2_S event before GetValue.
    __aicore__ inline void LoadSeqBoundaries() {
        auto cuLocal = cuBuf_.Get<int32_t>();
        DataCopyExtParams cp{1, static_cast<uint32_t>((numSeqs_ + 1) * sizeof(int32_t)),
                             0, 0, 0};
        DataCopyPadExtParams<int32_t> pad{false, 0, 0, 0};
        DataCopyPad(cuLocal, cuGm_, cp, pad);
        event_t eventId = static_cast<event_t>(
            pipe_.AllocEventID<HardEvent::MTE2_S>());
        SetFlag<HardEvent::MTE2_S>(eventId);
        WaitFlag<HardEvent::MTE2_S>(eventId);
        for (uint32_t i = 0; i <= numSeqs_; ++i) {
            cu_[i] = static_cast<uint32_t>(cuLocal.GetValue(i));
        }
        pipe_.ReleaseEventID<HardEvent::MTE2_S>(eventId);

        uint32_t tiles = 0;
        tileOff_[0] = 0;
        for (uint32_t s = 0; s < numSeqs_; ++s) {
            uint32_t len = cu_[s + 1] > cu_[s] ? cu_[s + 1] - cu_[s] : 0;
            tiles += (len + tile_ - 1) / tile_;
            tileOff_[s + 1] = tiles;
        }
        totalTiles_ = kHeads * tiles;
    }

    __aicore__ inline LocalTensor<float> Aligned(TBuf<TPosition::VECCALC>& buf) {
        auto raw = buf.Get<float>();
        return raw[static_cast<uint32_t>((512 - (raw.GetPhyAddr() & 511)) & 511) / sizeof(float)];
    }

    __aicore__ inline void ProcessTile(uint32_t head, uint32_t seqStart,
                                       uint32_t tileStart, uint32_t rows) {
        const uint32_t localT0 = tileStart - seqStart;
        const uint32_t history = localT0 < 3 ? localT0 : 3;
        const uint32_t leading = 3 - history;
        auto input = in_.AllocTensor<bfloat16_t>();
        DataCopyExtParams xp{static_cast<uint16_t>(rows + history),
                             kDim * sizeof(bfloat16_t),
                             static_cast<uint32_t>((inputChannels_ - kDim) * sizeof(bfloat16_t)), 0, 0};
        DataCopyPadExtParams<bfloat16_t> noPad{false, 0, 0, static_cast<bfloat16_t>(0)};
        const uint64_t xoff = static_cast<uint64_t>(tileStart - history) * inputChannels_ + head * kDim;
        DataCopyPad(input[leading * kDim], x_[xoff], xp, noPad);
        DataCopyExtParams wp{4, kDim * sizeof(bfloat16_t),
                             (kChannels - kDim) * sizeof(bfloat16_t), 0, 0};
        DataCopyPad(input[(tile_ + 3) * kDim], weight_[head * kDim], wp, noPad);
        in_.EnQue(input);
        auto xin = in_.DeQue<bfloat16_t>();
        auto xf = Aligned(xf_); auto wf = Aligned(wf_);
        auto acc = Aligned(acc_); auto tmp = Aligned(tmp_);
        if (leading) { Duplicate(xf, 0.0f, leading * kDim); }
        Cast(xf[leading * kDim], xin[leading * kDim], RoundMode::CAST_NONE, (rows + history) * kDim);
        Cast(wf, xin[(tile_ + 3) * kDim], RoundMode::CAST_NONE, 4 * kDim);
        in_.FreeTensor(xin);
        PipeBarrier<PIPE_V>();

        const BinaryRepeatParams rp{1, 1, 1, 16, 16, 0};
        for (uint32_t j = 0; j < 4; ++j) {
            for (uint32_t half = 0; half < 2; ++half) {
                const uint32_t offset = half * 64;
                if (j == 0) {
                    Mul(acc[offset], xf[j * kDim + offset], wf[j * kDim + offset],
                        64, static_cast<uint8_t>(rows), rp);
                } else {
                    MulAddDst(acc[offset], xf[j * kDim + offset], wf[j * kDim + offset],
                              64, static_cast<uint8_t>(rows), rp);
                }
            }
            PipeBarrier<PIPE_V>();
        }
        Silu(tmp, acc, rows * kDim);
        PipeBarrier<PIPE_V>();
        auto y = out_.AllocTensor<bfloat16_t>();
        Cast(y, tmp, RoundMode::CAST_RINT, rows * kDim);
        PipeBarrier<PIPE_V>();
        if (head < 16) {
            Cast(xf, y, RoundMode::CAST_NONE, rows * kDim);
            PipeBarrier<PIPE_V>();
            Mul(acc, xf, xf, rows * kDim);
            PipeBarrier<PIPE_V>();
            auto sums = Aligned(sums_); auto work = Aligned(work_);
            SetMaskNorm();
            Add(work, acc, acc[64], 64, static_cast<uint8_t>(rows),
                {1, 1, 1, 8, 16, 16});
            PipeBarrier<PIPE_V>();
            SetMaskNorm();
            WholeReduceSum<float>(sums, work, 64, rows, 1, 1, 8);
            PipeBarrier<PIPE_V>();
            Adds(sums, sums, eps_, rows);
            PipeBarrier<PIPE_V>();
            Rsqrt(sums, sums, rows);
            PipeBarrier<PIPE_V>();
            uint32_t dstShape[2] = {rows, kDim};
            uint32_t srcShape[2] = {rows, 1};
            Broadcast<float, 2, 1>(tmp, sums, dstShape, srcShape);
            PipeBarrier<PIPE_V>();
            Mul(xf, xf, tmp, rows * kDim);
            PipeBarrier<PIPE_V>();
            Cast(y, xf, RoundMode::CAST_RINT, rows * kDim);
        }
        out_.EnQue(y);
        auto result = out_.DeQue<bfloat16_t>();
        const uint32_t outHeads = head < 16 ? 8 : 24;
        const uint32_t outHead = head < 8 ? head : (head < 16 ? head - 8 : head - 16);
        DataCopyExtParams yp{static_cast<uint16_t>(rows), kDim * sizeof(bfloat16_t),
                             0, static_cast<uint32_t>((outHeads - 1) * kDim * sizeof(bfloat16_t)), 0};
        const uint64_t outOff = static_cast<uint64_t>(tileStart) * outHeads * kDim + outHead * kDim;
        if (head < 8) { DataCopyPad(q_[outOff], result, yp); }
        else if (head < 16) { DataCopyPad(k_[outOff], result, yp); }
        else { DataCopyPad(v_[outOff], result, yp); }
        out_.FreeTensor(result);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 1> in_;
    TQue<TPosition::VECOUT, 1> out_;
    TBuf<TPosition::VECCALC> xf_, wf_, acc_, tmp_, sums_, work_, cuBuf_;
    GlobalTensor<bfloat16_t> x_, weight_, q_, k_, v_;
    GlobalTensor<int32_t> cuGm_;
    uint32_t tokens_, tile_, numSeqs_, totalTiles_, inputChannels_;
    uint32_t cu_[kMaxSeqs + 1];
    uint32_t tileOff_[kMaxSeqs + 1];
    float eps_;
};
}  // namespace

extern "C" __global__ __aicore__ void clef_gdn_conv_qkv_varlen(
    GM_ADDR x, GM_ADDR weight, GM_ADDR cu_seqlens, GM_ADDR q, GM_ADDR k,
    GM_ADDR v, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClefGdnConvQkvVarlenTilingData);
    GET_TILING_DATA(tilingData, tiling);
    ClefGdnConvQkvVarlenKernel op;
    op.Init(x, weight, cu_seqlens, q, k, v, &tilingData);
    op.Process();
}
