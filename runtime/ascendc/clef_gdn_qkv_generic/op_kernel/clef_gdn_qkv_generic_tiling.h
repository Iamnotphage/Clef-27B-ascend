#ifndef CLEF_GDN_QKV_GENERIC_TILING_H
#define CLEF_GDN_QKV_GENERIC_TILING_H
#include <cstdint>

struct ClefGdnQkvGenericTilingData {
    uint32_t totalTokens;
    uint32_t tileTokens;
    uint32_t tilesPerHead;
    uint32_t totalTiles;
    uint32_t keyHeads;
    uint32_t valueHeads;
    float eps;
};
#endif
