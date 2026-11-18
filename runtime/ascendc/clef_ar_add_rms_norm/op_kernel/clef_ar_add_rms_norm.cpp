#include "kernel_operator.h"
#include "clef_ar_add_rms_norm_tiling.h"

using namespace AscendC;

// Norm-only variant used to validate Add+RMSNorm semantics on device before the
// HCCL context wiring is solved.  No all-reduce: x is treated as an
// already-reduced partial.  z = x + residual; rstd = rsqrt(mean(z^2)+eps);
// out = z*rstd*gamma; residual_out = z (the stream the service carries).
class ClefArAddRmsNormKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR residual, GM_ADDR gamma,
                                GM_ADDR out, GM_ADDR residualOut, GM_ADDR rstd,
                                const ClefArAddRmsNormTilingData* t) {
        m_ = t->m;
        n_ = t->n;
        eps_ = t->eps;
        invN_ = t->invN;
        mPad_ = t->mPad;
        x_.SetGlobalBuffer((__gm__ bfloat16_t*)x, static_cast<uint64_t>(m_) * n_);
        r_.SetGlobalBuffer((__gm__ bfloat16_t*)residual, static_cast<uint64_t>(m_) * n_);
        g_.SetGlobalBuffer((__gm__ bfloat16_t*)gamma, n_);
        o_.SetGlobalBuffer((__gm__ bfloat16_t*)out, static_cast<uint64_t>(m_) * n_);
        ro_.SetGlobalBuffer((__gm__ bfloat16_t*)residualOut, static_cast<uint64_t>(m_) * n_);
        rs_.SetGlobalBuffer((__gm__ float*)rstd, m_);
        pipe_.InitBuffer(bx_, n_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(br_, n_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(bg_, n_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(bo_, n_ * sizeof(bfloat16_t));
        pipe_.InitBuffer(bz_, n_ * sizeof(float));
        pipe_.InitBuffer(bt_, n_ * sizeof(float));
        pipe_.InitBuffer(bw_, n_ * sizeof(float));
        pipe_.InitBuffer(bs_, 64 * sizeof(float));
        pipe_.InitBuffer(bwr_, 512 * sizeof(float));
        pipe_.InitBuffer(bgst_, 32);
    }

    __aicore__ inline void Process() {
        LocalTensor<bfloat16_t> gl = bg_.Get<bfloat16_t>();
        LocalTensor<float> gf = bt_.Get<float>();
        DataCopy(gl, g_[0], n_);
        Cast(gf, gl, RoundMode::CAST_NONE, n_);

        const uint32_t blocks = GetBlockNum();
        const uint32_t perCoreRaw = (m_ + blocks - 1) / blocks;
        const uint32_t perCore = (perCoreRaw + 7u) / 8u * 8u;
        const uint32_t start = GetBlockIdx() * perCore;
        if (start >= m_) return;
        const uint32_t end = (start + perCore < m_) ? (start + perCore) : m_;

        LocalTensor<float> sums = bs_.Get<float>();
        LocalTensor<float> work = bwr_.Get<float>();
        uint32_t buffered = 0;
        uint32_t batchStart = start;

        for (uint32_t row = start; row < end; ++row) {
            const uint64_t base = static_cast<uint64_t>(row) * n_;
            LocalTensor<bfloat16_t> xl = bx_.Get<bfloat16_t>();
            LocalTensor<bfloat16_t> rl = br_.Get<bfloat16_t>();
            DataCopy(xl, x_[base], n_);
            DataCopy(rl, r_[base], n_);
            PipeBarrier<PIPE_ALL>();

            LocalTensor<float> z = bz_.Get<float>();
            LocalTensor<float> tmp = bw_.Get<float>();
            Cast(z, xl, RoundMode::CAST_NONE, n_);
            Cast(tmp, rl, RoundMode::CAST_NONE, n_);
            Add(z, z, tmp, n_);

            LocalTensor<bfloat16_t> rol = bo_.Get<bfloat16_t>();
            Cast(rol, z, RoundMode::CAST_RINT, n_);
            PipeBarrier<PIPE_ALL>();
            DataCopy(ro_[base], rol, n_);
            PipeBarrier<PIPE_ALL>();

            Mul(tmp, z, z, n_);
            float acc = 0.0f;
            for (uint32_t off = 0; off < n_; off += 512) {
                const uint32_t cnt = (n_ - off < 512) ? (n_ - off) : 512;
                LocalTensor<float> part = bgst_.Get<float>();
                ReduceSum<float>(part, tmp[off], work, cnt);
                acc += part.GetValue(0);
            }
            const float rstd = 1.0f / sqrt(acc * invN_ + eps_);
            sums.SetValue(buffered, rstd);
            ++buffered;

            Muls(z, z, rstd, n_);
            Mul(z, z, gf, n_);
            LocalTensor<bfloat16_t> ol = bo_.Get<bfloat16_t>();
            Cast(ol, z, RoundMode::CAST_RINT, n_);
            PipeBarrier<PIPE_ALL>();
            DataCopy(o_[base], ol, n_);
            PipeBarrier<PIPE_ALL>();

            if (buffered == 8 || row + 1 == end) {
                for (uint32_t k = buffered; k < 8; ++k) sums.SetValue(k, 0.0f);
                uint32_t cnt = 8;
                if (batchStart + cnt > mPad_) cnt = (mPad_ > batchStart) ? (mPad_ - batchStart) : 0;
                PipeBarrier<PIPE_ALL>();
                if (cnt > 0) DataCopy(rs_[batchStart], sums, cnt);
                PipeBarrier<PIPE_ALL>();
                batchStart += buffered;
                buffered = 0;
            }
        }
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECCALC> bx_, br_, bg_, bo_, bz_, bt_, bw_, bs_, bwr_, bgst_;
    GlobalTensor<bfloat16_t> x_, r_, g_, o_, ro_;
    GlobalTensor<float> rs_;
    uint32_t m_{0}, n_{0};
    float eps_{1e-6f};
    float invN_{0.0f};
    uint32_t mPad_{0};
};

extern "C" __global__ __aicore__ void clef_ar_add_rms_norm(
    GM_ADDR x, GM_ADDR residual, GM_ADDR gamma,
    GM_ADDR out, GM_ADDR residualOut, GM_ADDR rstd,
    GM_ADDR workspace, GM_ADDR tilingGm) {
    (void)workspace;
    REGISTER_TILING_DEFAULT(ClefArAddRmsNormTilingData);
    GET_TILING_DATA(tilingData, tilingGm);
    ClefArAddRmsNormKernel kernel;
    kernel.Init(x, residual, gamma, out, residualOut, rstd, &tilingData);
    kernel.Process();
}
