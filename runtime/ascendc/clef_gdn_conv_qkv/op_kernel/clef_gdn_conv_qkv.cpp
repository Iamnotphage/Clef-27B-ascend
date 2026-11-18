#include "kernel_operator.h"
#include "clef_gdn_conv_qkv_tiling.h"

using namespace AscendC;
namespace {
constexpr uint32_t kHeads = 40;
constexpr uint32_t kDim = 128;
constexpr uint32_t kChannels = kHeads * kDim;

// Each task owns one head and a contiguous token tile.  All cross-task data
// comes from the immutable input: no recurrent state or inter-core flags.
class ClefGdnConvQkvKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR weight, GM_ADDR q, GM_ADDR k,
                                GM_ADDR v, const ClefGdnConvQkvTilingData* t) {
        tokens_ = t->totalTokens; tile_ = t->tileTokens;
        perHead_ = t->tilesPerHead; total_ = t->totalTiles;
        perCore_ = t->tilesPerCore; eps_ = t->eps;
        inputChannels_ = t->inputChannels;
        x_.SetGlobalBuffer((__gm__ bfloat16_t*)x, static_cast<uint64_t>(tokens_) * inputChannels_);
        weight_.SetGlobalBuffer((__gm__ bfloat16_t*)weight, 4 * kChannels);
        q_.SetGlobalBuffer((__gm__ bfloat16_t*)q, static_cast<uint64_t>(tokens_) * 8 * kDim);
        k_.SetGlobalBuffer((__gm__ bfloat16_t*)k, static_cast<uint64_t>(tokens_) * 8 * kDim);
        v_.SetGlobalBuffer((__gm__ bfloat16_t*)v, static_cast<uint64_t>(tokens_) * 24 * kDim);
        pipe_.InitBuffer(in_, 1, (tile_ + 7) * kDim * sizeof(bfloat16_t));
        pipe_.InitBuffer(out_, 1, tile_ * kDim * sizeof(bfloat16_t));
        pipe_.InitBuffer(xf_, (tile_ + 3) * kDim * sizeof(float) + 512);
        pipe_.InitBuffer(wf_, 4 * kDim * sizeof(float) + 512);
        pipe_.InitBuffer(acc_, tile_ * kDim * sizeof(float) + 512);
        pipe_.InitBuffer(tmp_, tile_ * kDim * sizeof(float) + 512);
        pipe_.InitBuffer(sums_, tile_ * 2 * sizeof(float) + 512);
        pipe_.InitBuffer(work_, 16 * 1024 + 512);
    }

    __aicore__ inline void Process() {
        // Q/K include reductions while V only does convolution. A contiguous
        // head-major partition puts every expensive head on the first cores.
        // Round-robin tiles spread both kinds across all vector cores.
        for (uint32_t task = GetBlockIdx(); task < total_; task += GetBlockNum()) {
            const uint32_t head = task / perHead_;
            const uint32_t t0 = (task % perHead_) * tile_;
            const uint32_t rows = tokens_ - t0 < tile_ ? tokens_ - t0 : tile_;
            ProcessTile(head, t0, rows);
        }
    }

private:
    __aicore__ inline LocalTensor<float> Aligned(TBuf<TPosition::VECCALC>& buf) {
        auto raw = buf.Get<float>();
        return raw[static_cast<uint32_t>((512 - (raw.GetPhyAddr() & 511)) & 511) / sizeof(float)];
    }

    __aicore__ inline void ProcessTile(uint32_t head, uint32_t t0, uint32_t rows) {
        const uint32_t history = t0 < 3 ? t0 : 3;
        const uint32_t leading = 3 - history;
        auto input = in_.AllocTensor<bfloat16_t>();
        // GM strides are bytes for DataCopyPad Ext.  128 BF16 values per
        // row are contiguous; adjacent rows start 5120 values apart.
        DataCopyExtParams xp{static_cast<uint16_t>(rows + history),
                             kDim * sizeof(bfloat16_t),
                             static_cast<uint32_t>((inputChannels_ - kDim) * sizeof(bfloat16_t)), 0, 0};
        DataCopyPadExtParams<bfloat16_t> noPad{false, 0, 0, static_cast<bfloat16_t>(0)};
        const uint64_t xoff = static_cast<uint64_t>(t0 - history) * inputChannels_ + head * kDim;
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

        // Each repeat covers one 64-float half-row; repeat strides are
        // 32-byte blocks.  Source1 repeat stride zero broadcasts one tap's
        // weights across tokens.  Both 64-element halves are explicit.
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
            // Preserve the original convolution's BF16 output rounding
            // before L2Norm rather than normalizing its FP32 intermediate.
            Cast(xf, y, RoundMode::CAST_NONE, rows * kDim);
            PipeBarrier<PIPE_V>();
            Mul(acc, xf, xf, rows * kDim);
            PipeBarrier<PIPE_V>();
            auto sums = Aligned(sums_); auto work = Aligned(work_);
            // FLA 08d21f7 arch22 prepare-vector norm: pair the two 64-column
            // halves first, then reduce the packed [rows,64] tile. The
            // independent work buffer avoids overlapping source rows.
            // dst repeat stride=8 x32B=256B; src strides=16 x32B=512B.
            SetMaskNorm();
            Add(work, acc, acc[64], 64, static_cast<uint8_t>(rows),
                {1, 1, 1, 8, 16, 16});
            PipeBarrier<PIPE_V>();
            // WholeReduceSum sets the vector mask; normal mask mode is
            // explicit after count-mode arithmetic. dst stride=1 float.
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
        const uint64_t outOff = static_cast<uint64_t>(t0) * outHeads * kDim + outHead * kDim;
        if (head < 8) { DataCopyPad(q_[outOff], result, yp); }
        else if (head < 16) { DataCopyPad(k_[outOff], result, yp); }
        else { DataCopyPad(v_[outOff], result, yp); }
        out_.FreeTensor(result);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 1> in_;
    TQue<TPosition::VECOUT, 1> out_;
    TBuf<TPosition::VECCALC> xf_, wf_, acc_, tmp_, sums_, work_;
    GlobalTensor<bfloat16_t> x_, weight_, q_, k_, v_;
    uint32_t tokens_, tile_, perHead_, total_, perCore_, inputChannels_;
    float eps_;
};
}  // namespace

extern "C" __global__ __aicore__ void clef_gdn_conv_qkv(
    GM_ADDR x, GM_ADDR weight, GM_ADDR q, GM_ADDR k, GM_ADDR v,
    GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClefGdnConvQkvTilingData);
    GET_TILING_DATA(tilingData, tiling);
    ClefGdnConvQkvKernel op;
    op.Init(x, weight, q, k, v, &tilingData);
    op.Process();
}
