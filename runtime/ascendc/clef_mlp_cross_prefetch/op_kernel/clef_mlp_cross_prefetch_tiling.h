#pragma once
#include <cstdint>
// Identical explicit tiles and one binary for both arms. No SDK automatic tiler.
struct ClefMlpCrossPrefetchTilingData {
    uint32_t tokens, cores, prefetch, version;
};
