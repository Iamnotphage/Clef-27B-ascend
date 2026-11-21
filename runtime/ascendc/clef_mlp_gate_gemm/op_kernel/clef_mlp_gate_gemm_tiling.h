#pragma once
#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"
struct ClefMlpGateGemmTilingData {
    AscendC::tiling::TCubeTiling cube;
    uint32_t tokens, mTasks, nTasks;
};
