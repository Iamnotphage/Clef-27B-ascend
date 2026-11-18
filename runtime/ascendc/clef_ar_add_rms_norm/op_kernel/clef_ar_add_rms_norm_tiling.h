#ifndef CLEF_AR_ADD_RMS_NORM_TILING_H
#define CLEF_AR_ADD_RMS_NORM_TILING_H
#include <cstdint>

// Gate-B scaffold: the MC2 tiling blobs are kept as opaque byte buffers so the
// kernel can stay trivial while the host proves Mc2CcTilingConfig links in the
// official npu_op_package flow.  The real kernel will reinterpret these after
// including adv_api/hccl/hccl_tilingdata.h.
static const uint32_t kMc2TilingBytes = 4096;

struct ClefArAddRmsNormTilingData {
    uint32_t m;                 // rows (967)
    uint32_t n;                 // hidden (5120)
    uint32_t repeat;            // Hccl repeat (uint8 range, 1..255)
    uint32_t commEngine;        // 2 = AICPU
    uint32_t opType;            // 2 = HCCL_CMD_ALLREDUCE
    uint32_t initTilingRc;      // GetTiling(Mc2InitTiling) return code
    uint32_t ccTilingRc;        // GetTiling(Mc2CcTiling) return code
    uint32_t initTilingBytes;
    uint32_t ccTilingBytes;
    float eps;
    uint8_t mc2InitTiling[kMc2TilingBytes];
    uint8_t mc2CcTiling[kMc2TilingBytes];
};
#endif
