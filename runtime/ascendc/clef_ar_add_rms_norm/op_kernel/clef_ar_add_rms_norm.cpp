#include "kernel_operator.h"
#include "clef_ar_add_rms_norm_tiling.h"

using namespace AscendC;

// Gate-B scaffold kernel: byte-copy x into out so the official package builds.
// It performs NO communication and NO normalization; the real KFC kernel lands
// only after the tiling link and the device-side repeat=1 correctness smoke.
class ClefArAddRmsNormKernel {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR residual, GM_ADDR gamma,
                                GM_ADDR out, GM_ADDR residualOut, GM_ADDR rstd,
                                const ClefArAddRmsNormTilingData* t) {
        (void)x; (void)residual; (void)gamma; (void)out; (void)residualOut;
        m_ = t->m;
        n_ = t->n;
        rstd_.SetGlobalBuffer((__gm__ float*)rstd, m_);
    }

    // Gate-B scaffold: only writes rstd so the package builds.  No copy, no
    // communication, no normalization; the real kernel replaces this.
    __aicore__ inline void Process() {
        if (GetBlockIdx() == 0) {
            for (uint32_t row = 0; row < m_; ++row) {
                rstd_.SetValue(row, 0.0f);
            }
        }
    }

private:
    GlobalTensor<float> rstd_;
    uint32_t m_{0}, n_{0};
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
