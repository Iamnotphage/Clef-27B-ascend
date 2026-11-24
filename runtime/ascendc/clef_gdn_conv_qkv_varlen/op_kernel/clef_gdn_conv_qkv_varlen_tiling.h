#ifndef CLEF_GDN_CONV_QKV_VARLEN_TILING_H
#define CLEF_GDN_CONV_QKV_VARLEN_TILING_H
#include <cstdint>

struct ClefGdnConvQkvVarlenTilingData {
    uint32_t totalTokens;
    uint32_t numSeqs;
    uint32_t tileTokens;
    uint32_t coreNum;
    uint32_t inputChannels;
    float eps;
};
#endif
