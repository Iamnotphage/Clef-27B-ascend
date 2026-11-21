#pragma once
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
struct ClefMlpSwigluTilingData {
    AscendC::tiling::TCubeTiling cube;
    uint32_t tokens, tasks, cores, stage;
};
