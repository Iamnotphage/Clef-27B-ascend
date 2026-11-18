#ifndef CLEF_MATMUL_AR_ADD_RMS_NORM_TILING_H
#define CLEF_MATMUL_AR_ADD_RMS_NORM_TILING_H
#include <cstdint>
#include "adv_api/kernel_tiling.h"

// Device-safe tiling payload.  The two MC2 blocks are opaque byte arrays sized
// to the official structures (Mc2InitTiling 64 B, Mc2CcTiling 512 B; the host
// asserts the sizes).  mc2InitTiling MUST stay first and mc2CcTiling must
// follow it immediately: SetCcTilingV2 takes an offset relative to the
// initTiling pointer, and the kernel passes offsetof(this struct, mc2CcTiling).
static const uint32_t kClefMmMc2InitBytes = 64;
static const uint32_t kClefMmMc2CcBytes = 512;

struct ClefMatmulArAddRmsNormTilingData {
    uint8_t mc2InitTiling[kClefMmMc2InitBytes];
    uint8_t mc2CcTiling[kClefMmMc2CcBytes];
    // Cube tiling for X[M,K] x W[K,N] -> partial[M,N], produced on the host by
    // matmul_tiling::MultiCoreMatmulTiling.
    AscendC::tiling::TCubeTiling cubeTiling;
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t mPad;
    uint32_t repeat;
    uint32_t commEngine;
    uint32_t opType;
    uint32_t initTilingRc;
    uint32_t ccTilingRc;
    uint32_t mmTilingRc;
    uint32_t useHccl;
    uint32_t mmEnable;
    uint32_t mmDim;
    uint32_t mmMode;
    uint32_t normLate;
    uint32_t mmBTrans;
    uint32_t mmStage;
    uint32_t normFromResidual;
    uint32_t normFromX;
    float eps;
    float invN;
};
#endif
