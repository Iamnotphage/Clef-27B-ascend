#include "kernel_operator.h"
#include "clef_gdn_qkv_generic_tiling.h"

using namespace AscendC;
namespace {
constexpr uint32_t kDim = 128;

class ClefGdnQkvGenericKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR q, GM_ADDR k, GM_ADDR v,
                                const ClefGdnQkvGenericTilingData* t) {
        tokens_ = t->totalTokens; tile_ = t->tileTokens;
        perHead_ = t->tilesPerHead; total_ = t->totalTiles;
        hk_ = t->keyHeads; hv_ = t->valueHeads; eps_ = t->eps;
        x_.SetGlobalBuffer((__gm__ bfloat16_t*)x, static_cast<uint64_t>(5 * hk_) * tokens_ * kDim);
        q_.SetGlobalBuffer((__gm__ bfloat16_t*)q, static_cast<uint64_t>(hk_) * tokens_ * kDim);
        k_.SetGlobalBuffer((__gm__ bfloat16_t*)k, static_cast<uint64_t>(hk_) * tokens_ * kDim);
        v_.SetGlobalBuffer((__gm__ bfloat16_t*)v, static_cast<uint64_t>(hv_) * tokens_ * kDim);
        const uint32_t bytes = tile_ * kDim * sizeof(bfloat16_t);
        pipe_.InitBuffer(in_, 1, bytes);
        pipe_.InitBuffer(out_, 1, bytes);
        pipe_.InitBuffer(copy_, 1, bytes);
        pipe_.InitBuffer(values_, tile_ * kDim * sizeof(float) + 512);
        pipe_.InitBuffer(squares_, tile_ * kDim * sizeof(float) + 512);
        pipe_.InitBuffer(scales_, tile_ * kDim * sizeof(float) + 512);
        pipe_.InitBuffer(sum0_, tile_ * sizeof(float) + 512);
        pipe_.InitBuffer(sum1_, tile_ * sizeof(float) + 512);
    }

    __aicore__ inline void Process() {
        // Cyclic assignment distributes heavy Q/K and copy-only V tiles
        // across all cores instead of leaving Q/K on a subset of cores.
        for (uint32_t task = GetBlockIdx(); task < total_; task += GetBlockNum()) {
            const uint32_t head = task / perHead_;
            const uint32_t start = (task % perHead_) * tile_;
            const uint32_t rows = tokens_ - start < tile_ ? tokens_ - start : tile_;
            const uint64_t src = (static_cast<uint64_t>(head) * tokens_ + start) * kDim;
            if (head < 2 * hk_) {
                auto output = head < hk_ ? q_ : k_;
                Normalize(output, src, head < hk_ ? head : head - hk_, hk_, start, rows);
            } else {
                CopyValue(src, head - 2 * hk_, start, rows);
            }
        }
    }

private:
    __aicore__ inline LocalTensor<float> Aligned(TBuf<TPosition::VECCALC>& buffer) {
        auto raw = buffer.Get<float>();
        return raw[static_cast<uint32_t>((512 - (raw.GetPhyAddr() & 511)) & 511) / sizeof(float)];
    }

    __aicore__ inline void CopyOut(const GlobalTensor<bfloat16_t>& output,
                                   const LocalTensor<bfloat16_t>& local,
                                   uint32_t head, uint32_t heads, uint32_t start, uint32_t rows) {
        DataCopyExtParams p{static_cast<uint16_t>(rows), kDim * sizeof(bfloat16_t),
                             0, static_cast<uint32_t>((heads - 1) * kDim * sizeof(bfloat16_t)), 0};
        const uint64_t offset = (static_cast<uint64_t>(start) * heads + head) * kDim;
        DataCopyPad(output[offset], local, p);
    }

    __aicore__ inline void CopyValue(uint64_t src, uint32_t head, uint32_t start, uint32_t rows) {
        // TQueBind establishes MTE2->MTE3 ordering and buffer reuse for the
        // pure movement path. No BF16/FP32 casts or V instructions are used.
        auto local = copy_.AllocTensor<bfloat16_t>();
        DataCopy(local, x_[src], rows * kDim);
        copy_.EnQue(local);
        auto ready = copy_.DeQue<bfloat16_t>();
        CopyOut(v_, ready, head, hv_, start, rows);
        copy_.FreeTensor(ready);
    }

    __aicore__ inline void Normalize(const GlobalTensor<bfloat16_t>& output, uint64_t src,
                                     uint32_t head, uint32_t heads, uint32_t start, uint32_t rows) {
        auto input = in_.AllocTensor<bfloat16_t>();
        DataCopy(input, x_[src], rows * kDim);
        in_.EnQue(input);
        auto ready = in_.DeQue<bfloat16_t>();
        auto values = Aligned(values_); auto squares = Aligned(squares_);
        auto sum0 = Aligned(sum0_); auto sum1 = Aligned(sum1_);
        auto scales = Aligned(scales_);
        Cast(values, ready, RoundMode::CAST_NONE, rows * kDim);
        in_.FreeTensor(ready);
        PipeBarrier<PIPE_V>();
        Mul(squares, values, values, rows * kDim);
        PipeBarrier<PIPE_V>();
        // Each instruction reduces one 64-float half-row over the whole
        // tile. srcRepStride=16 x32B=512B; dstRepStride=1 FLOAT element.
        // WholeReduceSum's source offset 64 FLOATs is 256B and satisfies
        // its 32B requirement; all buffer bases are aligned to 512B.
        WholeReduceSum<float>(sum0, squares, 64, rows, 1, 1, 16);
        WholeReduceSum<float>(sum1, squares[64], 64, rows, 1, 1, 16);
        PipeBarrier<PIPE_V>();
        Add(sum0, sum0, sum1, rows);
        PipeBarrier<PIPE_V>();
        Adds(sum0, sum0, eps_, rows);
        PipeBarrier<PIPE_V>();
        Rsqrt(sum0, sum0, rows);
        PipeBarrier<PIPE_V>();
        uint32_t dstShape[2] = {rows, kDim}, srcShape[2] = {rows, 1};
        Broadcast<float, 2, 1>(scales, sum0, dstShape, srcShape);
        PipeBarrier<PIPE_V>();
        Mul(values, values, scales, rows * kDim);
        PipeBarrier<PIPE_V>();
        auto result = out_.AllocTensor<bfloat16_t>();
        Cast(result, values, RoundMode::CAST_RINT, rows * kDim);
        out_.EnQue(result);
        auto copied = out_.DeQue<bfloat16_t>();
        CopyOut(output, copied, head, heads, start, rows);
        out_.FreeTensor(copied);
    }

    TPipe pipe_;
    TQue<TPosition::VECIN, 1> in_;
    TQue<TPosition::VECOUT, 1> out_;
    TQueBind<TPosition::VECIN, TPosition::VECOUT, 1> copy_;
    TBuf<TPosition::VECCALC> values_, squares_, scales_, sum0_, sum1_;
    GlobalTensor<bfloat16_t> x_, q_, k_, v_;
    uint32_t tokens_, tile_, perHead_, total_, hk_, hv_;
    float eps_;
};
}  // namespace

extern "C" __global__ __aicore__ void clef_gdn_qkv_generic(
    GM_ADDR x, GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(ClefGdnQkvGenericTilingData);
    GET_TILING_DATA(tilingData, tiling);
    ClefGdnQkvGenericKernel op;
    op.Init(x, q, k, v, &tilingData);
    op.Process();
}
