// Host-only observation bridge, compiled against the installed CANN MSTX SDK.
// No kernels, tensor operations, runtime initialization or synchronization.
#include <cstdint>
#include <mstx/ms_tools_ext.h>
extern "C" __attribute__((visibility("default")))
uint64_t clef40_range_start(const char* message, void* stream) {
    return mstxRangeStartA(message, stream);
}
extern "C" __attribute__((visibility("default")))
void clef40_range_end(uint64_t id) { mstxRangeEnd(id); }
// 26.2's injection backend numbers ranges from zero. Registration readiness,
// not the numeric ID, distinguishes a disconnected SDK from a bound callback.
// A successful BasicInfo export is still required to establish a target hit.
extern "C" __attribute__((visibility("default")))
int clef40_range_ready() {
    return g_mstxContext.initState == MSTX_INIT_STATE_COMPLETE &&
           g_mstxContext.mstxRangeStartAPtr != nullptr &&
           g_mstxContext.mstxRangeStartAPtr != mstxRangeStartAInit &&
           g_mstxContext.mstxRangeEndPtr != nullptr &&
           g_mstxContext.mstxRangeEndPtr != mstxRangeEndInit;
}
