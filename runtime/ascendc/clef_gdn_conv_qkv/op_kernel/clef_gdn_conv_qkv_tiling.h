#ifndef CLEF_GDN_CONV_QKV_TILING_H
#define CLEF_GDN_CONV_QKV_TILING_H
#include <cstdint>

struct ClefGdnConvQkvTilingData {
    uint32_t totalTokens;
    uint32_t tileTokens;
    uint32_t tilesPerHead;
    uint32_t totalTiles;
    uint32_t tilesPerCore;
    uint32_t coreNum;
    float eps;
};
#endif
