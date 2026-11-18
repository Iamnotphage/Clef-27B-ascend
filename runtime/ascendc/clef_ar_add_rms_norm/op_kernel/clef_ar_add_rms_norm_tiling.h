#ifndef CLEF_AR_ADD_RMS_NORM_TILING_H
#define CLEF_AR_ADD_RMS_NORM_TILING_H
#include <cstdint>

// Device-safe tiling payload: the two MC2 blocks are opaque byte arrays sized
// to the official structures (Mc2InitTiling 64 B, Mc2CcTiling 512 B; the host
// asserts the sizes).  mc2InitTiling MUST be first and mc2CcTiling must follow
// immediately: SetCcTilingV2 takes an offset relative to the initTiling
// pointer, which the host computes with offsetof on this struct.
static const uint32_t kMc2InitBytes = 64;
static const uint32_t kMc2CcBytes = 512;

struct ClefArAddRmsNormTilingData {
    uint8_t mc2InitTiling[kMc2InitBytes];
    uint8_t mc2CcTiling[kMc2CcBytes];
    uint32_t m;
    uint32_t n;
    uint32_t mPad;
    uint32_t repeat;
    uint32_t commEngine;
    uint32_t opType;
    uint32_t initTilingRc;
    uint32_t ccTilingRc;
    uint32_t useHccl;
    float eps;
    float invN;
};
#endif
