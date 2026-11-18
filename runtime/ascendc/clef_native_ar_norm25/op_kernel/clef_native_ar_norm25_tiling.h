#ifndef CLEF_NATIVE_AR_NORM25_TILING_H
#define CLEF_NATIVE_AR_NORM25_TILING_H
#include <cstdint>

struct ClefNativeArNorm25TilingData {
    // SetCcTilingV2 offset is relative to mc2InitTiling.
    uint8_t mc2InitTiling[64];
    uint8_t mc2CcTiling[512];
    uint32_t m;
    uint32_t n;
    uint32_t cores;
    uint32_t useHccl;
    float eps;
    float invN;
};
#endif
